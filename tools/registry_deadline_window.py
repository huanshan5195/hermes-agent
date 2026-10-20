"""When the window a tool declared through ``ToolEntry.deadline_floor`` actually opens.

The executor raises its generic deadline to a call's declared floor, but arms that deadline at
dispatch. Whatever the tool does before its own window opens (creating a Docker/Modal/SSH
sandbox, the pre-exec guards) is charged to the same clock, so a command that started late was cut
off by the generic guard while its own timeout still had time left. A tool marks the moment its
window opens; the executor then holds the deadline to at least that moment plus the floor. A tool
that wedges before marking is still bounded by the deadline armed at dispatch.
"""

import threading
import time

_opened_at: dict[int, float] = {}
_lock = threading.Lock()


def mark_declared_window_open() -> None:
    """Record that the blocking work this call's floor describes starts now, on this thread."""
    with _lock:
        _opened_at[threading.get_ident()] = time.monotonic()


def forget_declared_windows(tids) -> None:
    with _lock:
        for tid in tids:
            _opened_at.pop(tid, None)


def poll_through_declared_window(poll, deadline: float | None, floor: float | None, tids, excluded_seconds):
    """``poll(deadline)``, re-armed once at ``opened + floor`` when one of *tids* (the call's
    worker threads) opened its declared window too late for *deadline* to cover it.

    ``poll`` adds the human-approval exclusion to whatever deadline it is given; an approval
    wait happens before the window opens and is already behind ``opened``, so it is taken back
    out of the re-armed deadline rather than granted twice.
    """
    state, result = poll(deadline)
    if state != "timeout" or deadline is None or floor is None:
        return state, result
    with _lock:
        opened = [_opened_at[tid] for tid in tids if tid in _opened_at]
    if not opened:
        return state, result
    rearmed = max(opened) + floor - excluded_seconds()
    return poll(rearmed) if rearmed > deadline else (state, result)
