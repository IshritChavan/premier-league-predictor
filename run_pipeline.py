"""One-command refresh: results -> fixtures -> predictions.

Each stage now runs independently: if one stage fails, the others still
run, and whatever data they produced still gets committed by the workflow.
The script exits non-zero at the end if anything failed, so the run is
still correctly flagged red in the Actions tab -- but partial progress is
never lost, and the printed traceback tells you exactly what broke without
needing to sign in to view raw logs.
"""

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
    stages = ["update_results.py", "fetch_fixtures.py", "predict_week.py"]
    failures: list[str] = []

    for script in stages:
        try:
            run(script)
        except subprocess.CalledProcessError as exc:
            print(f"\n!!! {script} FAILED (exit code {exc.returncode}) !!!")
            print("!!! Continuing to the next stage so other data still refreshes. !!!")
            failures.append(script)
        except Exception as exc:  # noqa: BLE001 - want to surface anything and keep going
            print(f"\n!!! {script} FAILED with an unexpected error: {exc!r} !!!")
            failures.append(script)

    print("\n" + "=" * 70)
    if failures:
        print(f"Pipeline finished WITH FAILURES in: {', '.join(failures)}")
        print("Any stage that succeeded above still has fresh data ready to commit.")
        sys.exit(1)

    print("Pipeline complete. The web app can now read the refreshed CSV files.")


if __name__ == "__main__":
    main()
