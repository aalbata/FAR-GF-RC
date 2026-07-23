#!/usr/bin/env python3
"""Frozen processed artifacts ship with the repository; this script verifies rather than rebuilds.

Rebuilding from raw would change nothing if the protocol is followed, and risks silent
divergence if it is not. Regeneration recipes live in docs/data_protocol.md."""
import subprocess, sys
sys.exit(subprocess.call([sys.executable, "scripts/audit_data.py", *sys.argv[1:]]))
