import subprocess
import sys
from pathlib import Path
from zipfile import ZipFile

ROOT = Path(__file__).parents[1]


def test_wheel_contains_report_assets_and_font_license(tmp_path: Path) -> None:
    subprocess.run(
        [
            sys.executable,
            "-m",
            "pip",
            "wheel",
            str(ROOT),
            "--no-deps",
            "--wheel-dir",
            str(tmp_path),
        ],
        check=True,
        capture_output=True,
        text=True,
    )
    wheel = next(tmp_path.glob("*.whl"))
    with ZipFile(wheel) as archive:
        names = archive.namelist()
    assert any(name.endswith("NotoSans-Variable.ttf") for name in names)
    assert any(name.endswith("OFL-NotoSans.txt") for name in names)
    assert any(name.endswith("locales/en.json") for name in names)
    assert any(name.endswith("locales/pl.json") for name in names)
