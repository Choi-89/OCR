from __future__ import annotations

from pathlib import Path


def build_lmdb_manifest(data_dir: str | Path, label_file: str | Path, output_dir: str | Path) -> dict[str, str]:
    output = Path(output_dir)
    output.mkdir(parents=True, exist_ok=True)
    manifest = output / "manifest.txt"
    manifest.write_text(
        f"data_dir={Path(data_dir)}\nlabel_file={Path(label_file)}\nstatus=placeholder\n",
        encoding="utf-8",
    )
    return {
        "data_dir": str(Path(data_dir)),
        "label_file": str(Path(label_file)),
        "output_dir": str(output),
        "manifest": str(manifest),
    }
