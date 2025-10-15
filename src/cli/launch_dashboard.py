from __future__ import annotations
import argparse, sys, subprocess
from pathlib import Path

def main():
    parser = argparse.ArgumentParser(description="Launch Streamlit dashboard.")
    parser.add_argument("--port", type=int, default=8501)
    parser.add_argument("--address", default="0.0.0.0")
    parser.add_argument("--headless", action="store_true", help="Run without opening a browser")
    args, extra = parser.parse_known_args()

    here = Path(__file__).resolve()
    repo_root = here.parents[2]                 # <repo>/ 
    app_path  = repo_root / "src" / "chronic_risk" / "dashboard.py"

    if not app_path.exists():
        raise SystemExit(f"Dashboard file not found: {app_path}")

    cmd = [
        sys.executable, "-m", "streamlit", "run", str(app_path),
        "--server.port", str(args.port),
        "--server.address", args.address,
    ]
    if args.headless:
        cmd += ["--server.headless", "true"]
    cmd += extra  # forward any extra Streamlit flags

    # Ensure relative paths inside the app are resolved from repo root
    raise SystemExit(subprocess.call(cmd, cwd=str(repo_root)))

if __name__ == "__main__":
    main()
