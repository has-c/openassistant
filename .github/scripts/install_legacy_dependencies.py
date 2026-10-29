"""Install the legacy dependencies changed by Dependabot for import smoke tests."""
import subprocess
import sys
from pathlib import Path

wanted = {"certifi", "llama-cpp-python", "pillow"}
requirements = []
for raw in Path("legacy/requirements.txt").read_text().splitlines():
    line = raw.strip()
    if not line or line.startswith("#"):
        continue
    package = line.split("=", 1)[0].rstrip("<>!~ ").lower()
    if package in wanted:
        requirements.append(line)
        wanted.remove(package)
if wanted:
    raise SystemExit(f"Missing expected requirements: {', '.join(sorted(wanted))}")
subprocess.run([sys.executable, "-m", "pip", "install", *requirements], check=True)
