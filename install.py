from pathlib import Path
from subprocess import check_output

import launch


REPO_ROOT = Path(__file__).parent
COMMIT_FILE_PATH = REPO_ROOT / "commit.txt"
COMMIT = check_output(
    [launch.git, "-C", REPO_ROOT.as_posix(), "rev-parse", "HEAD"],
    shell = False,
    encoding = "utf-8",
).strip()

if not COMMIT_FILE_PATH.exists() or COMMIT_FILE_PATH.read_text() != COMMIT:
    launch.run_pip(f"install -r {(REPO_ROOT / 'requirements.txt').as_posix()}", "Temporal requirements")

    COMMIT_FILE_PATH.write_text(COMMIT)
