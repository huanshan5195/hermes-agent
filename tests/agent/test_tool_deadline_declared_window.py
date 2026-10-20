"""A declared deadline floor covers the tool's own window, not the setup in front of it.

``terminal`` declares ``timeout`` (+ grace) as its floor, but the executor arms that deadline at
dispatch, and a foreground call first creates its sandbox (Docker/Modal/SSH) and runs the pre-exec
guards. When that setup took longer than the grace, the generic guard fired while the command's own
timeout still had time left, and the call returned ``timed out after Ns`` with no output.
"""

import json
import threading
import time

import pytest

from agent import tool_executor
from agent.tool_executor import (
    _ManagedToolResult,
    _ToolTimeoutResult,
    _run_sequential_tool_execution_middleware,
)
from tools import terminal_tool as tt

_COMMAND_TIMEOUT = 1  # terminal's declared floor: 1s + the 0.2s grace below


class _FakeAgent:
    def __init__(self):
        self._tool_worker_threads = set()
        self._tool_worker_threads_lock = threading.Lock()
        self._interrupt_requested = False

    def _touch_activity(self, msg):
        pass


class _SlowEnv:
    """A sandbox whose command runs ``work`` seconds, inside its own timeout."""

    def __init__(self, work):
        self.work = work

    def execute(self, command, timeout=None, **kwargs):
        assert self.work < timeout
        # health: allow HX013 -- the command's runtime against the executor's deadline IS the behaviour under test
        time.sleep(self.work)
        return {"output": "command output", "returncode": 0}


@pytest.fixture(autouse=True)
def _terminal_without_side_effects(monkeypatch):
    # The worker's exit imports run_agent (``_registered_tool_worker``); a cold import inside a
    # sub-second deadline is the test's cost, not the tool's.
    import run_agent
    monkeypatch.setattr(tool_executor, "_SEQUENTIAL_INTERRUPT_POLL_SECONDS", 0.05)
    monkeypatch.setattr(tool_executor, "_DECLARED_DEADLINE_GRACE_S", 0.2)
    monkeypatch.setattr(tool_executor, "_resolve_sequential_tool_timeout", lambda: 0.3)
    monkeypatch.setattr(tool_executor, "_emit_terminal_post_tool_call", lambda agent, **kw: None)
    monkeypatch.setattr(tt, "_pre_exec_block", lambda *a, **kw: None)
    monkeypatch.setattr(tt, "_resolve_command_cwd", lambda **kw: kw["default_cwd"])
    monkeypatch.setattr(tt, "finalize_foreground_result", lambda **kw: json.dumps({"output": kw["result"]["output"]}))
    monkeypatch.setattr("hermes_cli.observability.shared_metrics_harness.record_terminal_outcome", lambda *a, **kw: None)
    monkeypatch.setattr("hermes_cli.observability.shared_metrics_loop.record_terminal_backend", lambda plan, result, **kw: result)


def _run_terminal(monkeypatch, acquire_env):
    """One foreground ``terminal`` call through the real sequential runner and terminal_tool.
    Returns once its worker has finished too, so nothing outlives the patches."""
    monkeypatch.setattr(tt, "_acquire_env", acquire_env)
    finished = threading.Event()

    def _middleware(agent, **kwargs):
        try:
            result = tt.terminal_tool(command="echo hi", timeout=_COMMAND_TIMEOUT, force=True)
            return _ManagedToolResult(result=result, args={}, middleware_trace=[], blocked=False, dispatched=True)
        finally:
            finished.set()

    monkeypatch.setattr(tool_executor, "_run_agent_tool_execution_middleware", _middleware)
    managed = _run_sequential_tool_execution_middleware(
        _FakeAgent(), function_name="terminal", function_args={"command": "echo hi", "timeout": _COMMAND_TIMEOUT},
        effective_task_id="t", tool_call_id="c1", execute=None,
    )
    return managed.result, finished


def _env_after(setup, work):
    def _acquire(plan, task_id):
        # health: allow HX013 -- sandbox creation time before the command window IS the behaviour under test
        time.sleep(setup)
        return _SlowEnv(work)
    return _acquire


def test_slow_sandbox_setup_does_not_eat_the_command_window(monkeypatch):
    """The regression: 0.6s of setup + a 0.8s command, all inside its 1s timeout, used to die
    at the 1.2s deadline armed at dispatch."""
    result, finished = _run_terminal(monkeypatch, _env_after(setup=0.6, work=0.8))
    assert finished.wait(5)
    assert not isinstance(result, _ToolTimeoutResult), result
    assert json.loads(result)["output"] == "command output"


def test_a_command_that_overruns_its_own_window_still_dies(monkeypatch):
    """The re-armed deadline is the command's window, not an open-ended extension."""
    release = threading.Event()

    def _acquire(plan, task_id):
        # health: allow HX013 -- sandbox creation time before the command window IS the behaviour under test
        time.sleep(0.6)
        env = _SlowEnv(0)
        env.execute = lambda command, **kw: release.wait(10) and {"output": "late", "returncode": 0}
        return env

    started = time.monotonic()
    result, finished = _run_terminal(monkeypatch, _acquire)
    elapsed = time.monotonic() - started
    release.set()
    assert finished.wait(5)
    assert isinstance(result, _ToolTimeoutResult)
    assert 1.6 <= elapsed < 3.0, elapsed


def test_a_wedged_setup_is_still_bounded_by_the_dispatch_deadline(monkeypatch):
    """Setup that never reaches the command never opens the window, so nothing is re-armed."""
    release = threading.Event()

    def _wedged(plan, task_id):
        release.wait(10)
        raise RuntimeError("released by the test")

    started = time.monotonic()
    result, finished = _run_terminal(monkeypatch, _wedged)
    elapsed = time.monotonic() - started
    release.set()
    assert finished.wait(5)
    assert isinstance(result, _ToolTimeoutResult)
    assert "timed out after 1.2s" in str(result)
    assert elapsed < 1.6, elapsed
