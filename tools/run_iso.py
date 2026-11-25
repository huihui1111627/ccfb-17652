#!/usr/bin/env python3
"""Isolated-mode bootstrap.

Runs the project under `python3 -I` (no cwd / script dir on sys.path, no
user site-packages, no env vars) so that data directories containing
untrusted files cannot shadow stdlib modules. Trusted third-party packages
(numpy / sklearn, installed in the user site) are added by absolute path.

Usage:
    python3 -I tools/run_iso.py some_script.py [args...]
    python3 -I tools/run_iso.py -m package.module [args...]
"""
import os
import site
import sys

# Explicit, absolute, trusted package location only.
USER_SITE = os.path.expanduser("~/Library/Python/3.9/lib/python/site-packages")
if os.path.isdir(USER_SITE) and USER_SITE not in sys.path:
    sys.path.insert(0, USER_SITE)
site.removeduppaths()

WORKSPACE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

if len(sys.argv) < 2:
    raise SystemExit("usage: run_iso.py (-m module | script.py) [args...]")

if sys.argv[1] == "-m":
    mod = sys.argv[2]
    sys.argv = [mod] + sys.argv[3:]
    sys.path.insert(0, WORKSPACE)
    import runpy
    runpy.run_module(mod, run_name="__main__")
else:
    script = os.path.abspath(sys.argv[1])
    sys.argv = [script] + sys.argv[2:]
    # Project root (not the script's dir, not cwd) is the only project entry.
    sys.path.insert(0, WORKSPACE)
    import runpy
    runpy.run_path(script, run_name="__main__")
