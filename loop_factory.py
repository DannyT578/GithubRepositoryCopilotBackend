"""
Custom uvicorn loop factory for Windows compatibility.

uvicorn's built-in asyncio loop factory deliberately returns SelectorEventLoop
when `use_subprocess=True` (i.e. when --reload spawns worker subprocesses).
SelectorEventLoop cannot spawn subprocesses, which breaks gitingest's
`git ls-remote` calls.

When a loop factory is given as a dotted import path, uvicorn passes it
directly to asyncio.Runner(loop_factory=...).  asyncio.Runner calls
loop_factory() with NO arguments and expects a fully-constructed event loop
INSTANCE back (not the class itself).
"""

import asyncio
import sys


def proactor_loop_factory() -> asyncio.AbstractEventLoop:
    if sys.platform == "win32":
        return asyncio.ProactorEventLoop()
    return asyncio.SelectorEventLoop()
