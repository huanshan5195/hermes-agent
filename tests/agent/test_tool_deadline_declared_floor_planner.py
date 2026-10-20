"""A call that declares a deadline floor never joins a parallel segment.

A parallel segment arms one deadline when the batch is submitted, and runs at most
``_MAX_TOOL_WORKERS`` calls at a time. A declared window (``ToolEntry.deadline_floor``) is how long
ONE call may block from when it starts, so a declaring call queued behind the worker cap would see
its window cut short by however long it waited. Today's declarers (``terminal``, ``process_manage``)
are barriers only because they are not on the parallel-safe list; this pins it to the declaration.
"""

import json
import uuid
from types import SimpleNamespace

import pytest

from agent import tool_dispatch_helpers
from agent.tool_dispatch_helpers import _plan_tool_batch_segments
from agent.tool_executor import _MAX_TOOL_WORKERS
from tools.registry import registry

_NAME = "spec_parallel_safe_waiter"


@pytest.fixture()
def declared():
    floor = {"seconds": None}
    registry.register(
        name=_NAME, toolset="testing", schema={"name": _NAME, "parameters": {}},
        handler=lambda **kw: "unused", emoji="🧪",
        deadline_floor=lambda args: floor["seconds"] if args.get("action") == "wait" else None,
    )
    try:
        yield floor
    finally:
        registry.deregister(_NAME)


@pytest.fixture(autouse=True)
def _parallel_safe(monkeypatch):
    monkeypatch.setattr(tool_dispatch_helpers, "_PARALLEL_SAFE_TOOLS", tool_dispatch_helpers._PARALLEL_SAFE_TOOLS | {_NAME})


def _calls(count, action):
    return [
        SimpleNamespace(id=f"call_{uuid.uuid4().hex[:8]}", type="function",
                        function=SimpleNamespace(name=_NAME, arguments=json.dumps({"action": action})))
        for _ in range(count)
    ]


def test_more_declaring_calls_than_workers_run_one_at_a_time(declared):
    declared["seconds"] = 600.0
    segments = _plan_tool_batch_segments(_calls(_MAX_TOOL_WORKERS + 1, "wait"))
    assert [kind for kind, _ in segments] == ["sequential"]


def test_calls_that_declare_nothing_still_run_in_parallel(declared):
    declared["seconds"] = 600.0
    segments = _plan_tool_batch_segments(_calls(_MAX_TOOL_WORKERS + 1, "poll"))
    assert [kind for kind, _ in segments] == ["parallel"]
