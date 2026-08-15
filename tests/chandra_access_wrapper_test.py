import json
import os
from pathlib import Path
import subprocess
import sys


SCRIPT = Path(__file__).parents[1] / "scripts" / "with_chandra_access"


def _write_credentials(path: Path, *, mode: int = 0o600) -> None:
    path.write_text(
        json.dumps({"client_id": "test-client", "client_secret": "test-secret"}),
        encoding="utf-8",
    )
    path.chmod(mode)


def test_wrapper_injects_credentials_only_into_child(tmp_path):
    credentials = tmp_path / "chandra-access.json"
    _write_credentials(credentials)
    parent_id = os.environ.get("CHANDRA_CF_ACCESS_CLIENT_ID")
    parent_secret = os.environ.get("CHANDRA_CF_ACCESS_CLIENT_SECRET")

    result = subprocess.run(
        [
            sys.executable,
            str(SCRIPT),
            "--credentials",
            str(credentials),
            sys.executable,
            "-c",
            (
                "import os; "
                "assert os.environ['CHANDRA_CF_ACCESS_CLIENT_ID'] == 'test-client'; "
                "assert os.environ['CHANDRA_CF_ACCESS_CLIENT_SECRET'] == 'test-secret'"
            ),
        ],
        check=False,
        capture_output=True,
        text=True,
    )

    assert result.returncode == 0, result.stderr
    assert os.environ.get("CHANDRA_CF_ACCESS_CLIENT_ID") == parent_id
    assert os.environ.get("CHANDRA_CF_ACCESS_CLIENT_SECRET") == parent_secret


def test_wrapper_rejects_credentials_readable_by_other_users(tmp_path):
    credentials = tmp_path / "chandra-access.json"
    _write_credentials(credentials, mode=0o644)

    result = subprocess.run(
        [
            sys.executable,
            str(SCRIPT),
            "--credentials",
            str(credentials),
            sys.executable,
            "-c",
            "raise SystemExit('command must not run')",
        ],
        check=False,
        capture_output=True,
        text=True,
    )

    assert result.returncode != 0
    assert "must not be accessible by group/others" in result.stderr
    assert "test-secret" not in result.stderr
