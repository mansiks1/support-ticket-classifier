"""Check staged file sizes, forbidden paths and common secret patterns without printing contents."""

import re
import subprocess
from pathlib import Path

files = subprocess.check_output(
    ["git", "diff", "--cached", "--name-only", "--diff-filter=ACM"], text=True
).splitlines()
patterns = [
    rb"gh[pousr]_[A-Za-z0-9]{30,}",
    rb"github_pat_[A-Za-z0-9_]{30,}",
    rb"-----BEGIN (?:RSA |OPENSSH )?PRIVATE KEY-----",
]
for name in files:
    path = Path(name)
    assert not any(
        part in {".venv", "raw", "processed", "mlruns", "__pycache__"} for part in path.parts
    ), name
    assert path.name != ".env", name
    assert path.stat().st_size < 5_000_000, f"Oversized Git file: {name}"
    content = path.read_bytes()
    assert not any(re.search(pattern, content) for pattern in patterns), f"Potential secret: {name}"
print(f"Publication scan passed for {len(files)} staged files (not a comprehensive PII audit)")
