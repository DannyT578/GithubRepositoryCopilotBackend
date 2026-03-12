"""
Dev/prod entrypoint for the FastAPI server.

Run with:  python run.py

Why this exists:
  uvicorn's built-in asyncio loop factory explicitly returns SelectorEventLoop
  when use_subprocess=True (i.e. when --reload spawns worker processes).
  SelectorEventLoop cannot spawn subprocesses on Windows, which breaks gitingest's
  `git ls-remote` calls.

  We pass a custom loop factory via loop="loop_factory:proactor_loop_factory"
  that always uses ProactorEventLoop on Windows.  uvicorn pickles the Config
  (including the loop= string) into the reload subprocess, so the factory is
  re-imported and applied there too — fixing the issue end-to-end.
"""

import uvicorn

if __name__ == "__main__":
    uvicorn.run(
        "main:app",
        host="0.0.0.0",
        port=8000,
        reload=True,
        loop="loop_factory:proactor_loop_factory",
    )
