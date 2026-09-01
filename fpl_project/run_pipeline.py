"""One-command refresh: results -> fixtures -> predictions."""

from __future__ import annotations

import subprocess
import sys
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parent


def run(script_name: str) -> None:
    print("\n" + "=" * 70)
    print(f"RUNNING {script_name}")
    print("=" * 70)
    subprocess.run(
        [sys.executable, script_name],
        cwd=PROJECT_ROOT,
        check=True,
    )


def main() -> None:
    run("update_results.py")
    run("fetch_fixtures.py")
    run("predict_week.py")
    print("\nPipeline complete. The web app can now read the refreshed CSV files.")


if __name__ == "__main__":
    main()
