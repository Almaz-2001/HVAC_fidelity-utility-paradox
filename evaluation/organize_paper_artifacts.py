from __future__ import annotations

import csv
import shutil
from datetime import datetime
from pathlib import Path

from docx import Document


ROOT = Path(__file__).resolve().parents[1]
REPORTS = ROOT / "reports"
FIG_SRC = REPORTS / "figures" / "article_real"
ARTIFACTS = ROOT / "paper_artifacts"
MAIN_MANIFEST = REPORTS / "final_q1_12_engineering_figures_manifest.csv"
DOCX = ROOT / "docs" / "hvac_paper_final_q1.docx"


def ensure_dirs() -> None:
    for path in [
        ARTIFACTS / "figures" / "main",
        ARTIFACTS / "figures" / "supplementary",
        ARTIFACTS / "tables" / "main",
        ARTIFACTS / "csv" / "reports",
        ARTIFACTS / "manifests",
    ]:
        path.mkdir(parents=True, exist_ok=True)


def copy_file(src: Path, dst: Path, inventory: list[dict[str, str]], role: str) -> None:
    if not src.exists():
        inventory.append(
            {
                "role": role,
                "status": "missing",
                "source": str(src.relative_to(ROOT)),
                "destination": str(dst.relative_to(ROOT)),
                "bytes": "",
            }
        )
        return
    dst.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(src, dst)
    inventory.append(
        {
            "role": role,
            "status": "copied",
            "source": str(src.relative_to(ROOT)),
            "destination": str(dst.relative_to(ROOT)),
            "bytes": str(dst.stat().st_size),
        }
    )


def read_main_manifest() -> list[dict[str, str]]:
    if not MAIN_MANIFEST.exists():
        raise FileNotFoundError(MAIN_MANIFEST)
    with MAIN_MANIFEST.open(newline="", encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


def copy_main_figures(inventory: list[dict[str, str]]) -> set[str]:
    main_rows = read_main_manifest()
    main_basenames: set[str] = set()

    for row in main_rows:
        file_name = row["file"].strip()
        src_png = FIG_SRC / file_name
        stem = Path(file_name).stem
        main_basenames.add(stem)

        for ext in [".png", ".pdf"]:
            src = FIG_SRC / f"{stem}{ext}"
            dst = ARTIFACTS / "figures" / "main" / f"{int(row['figure']):02d}_{stem}{ext}"
            copy_file(src, dst, inventory, "main_figure")

    copy_file(
        MAIN_MANIFEST,
        ARTIFACTS / "manifests" / "main_figures_manifest.csv",
        inventory,
        "manifest",
    )
    return main_basenames


def copy_supplementary_figures(main_basenames: set[str], inventory: list[dict[str, str]]) -> None:
    if not FIG_SRC.exists():
        return
    for src in sorted(FIG_SRC.iterdir()):
        if not src.is_file() or src.suffix.lower() not in {".png", ".pdf"}:
            continue
        if src.stem in main_basenames:
            continue
        dst = ARTIFACTS / "figures" / "supplementary" / src.name
        copy_file(src, dst, inventory, "supplementary_figure")


def copy_reports_csv(inventory: list[dict[str, str]]) -> None:
    for src in sorted(REPORTS.glob("*.csv")):
        role = "csv_manifest" if "manifest" in src.name else "source_csv"
        target_dir = ARTIFACTS / "manifests" if role == "csv_manifest" else ARTIFACTS / "csv" / "reports"
        copy_file(src, target_dir / src.name, inventory, role)


def export_docx_tables(inventory: list[dict[str, str]]) -> None:
    if not DOCX.exists():
        inventory.append(
            {
                "role": "main_table",
                "status": "missing_docx",
                "source": str(DOCX.relative_to(ROOT)),
                "destination": "",
                "bytes": "",
            }
        )
        return

    doc = Document(DOCX)
    for index, table in enumerate(doc.tables, start=1):
        dst = ARTIFACTS / "tables" / "main" / f"table_{index:02d}.csv"
        with dst.open("w", newline="", encoding="utf-8") as handle:
            writer = csv.writer(handle)
            for row in table.rows:
                writer.writerow([cell.text.replace("\n", " ").strip() for cell in row.cells])
        inventory.append(
            {
                "role": "main_table",
                "status": "exported",
                "source": str(DOCX.relative_to(ROOT)),
                "destination": str(dst.relative_to(ROOT)),
                "bytes": str(dst.stat().st_size),
            }
        )


def write_inventory(inventory: list[dict[str, str]]) -> None:
    dst = ARTIFACTS / "manifests" / "paper_artifacts_inventory.csv"
    fields = ["role", "status", "source", "destination", "bytes"]
    with dst.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(inventory)


def write_readme() -> None:
    readme = ARTIFACTS / "README.md"
    readme.write_text(
        """# Paper Artifacts

Canonical paper-facing artifact directory for the HVAC DRL/MORL Q1 article.

## Structure

- `figures/main/`: final main-paper figures selected from `reports/final_q1_12_engineering_figures_manifest.csv`.
- `figures/supplementary/`: retained diagnostic figures not selected for the main paper.
- `tables/main/`: main-paper tables exported from `docs/hvac_paper_final_q1.docx`.
- `csv/reports/`: report-level CSV evidence used to build tables and figures.
- `manifests/`: figure manifests and the generated artifact inventory.

Large training outputs, raw corpora, model checkpoints, and BOPTEST binaries remain outside this directory and are ignored by Git according to `.gitignore`.
""",
        encoding="utf-8",
    )


def archive_old_article_real_dir(inventory: list[dict[str, str]]) -> None:
    if not FIG_SRC.exists():
        return
    stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    dst = ROOT / "draft" / "legacy_archive" / "figure_variants_archive" / f"article_real_{stamp}"
    dst.parent.mkdir(parents=True, exist_ok=True)
    shutil.move(str(FIG_SRC), str(dst))
    inventory.append(
        {
            "role": "legacy_archive",
            "status": "moved",
            "source": str(FIG_SRC.relative_to(ROOT)),
            "destination": str(dst.relative_to(ROOT)),
            "bytes": "",
        }
    )


def main() -> None:
    ensure_dirs()
    inventory: list[dict[str, str]] = []

    main_basenames = copy_main_figures(inventory)
    copy_supplementary_figures(main_basenames, inventory)
    copy_reports_csv(inventory)
    export_docx_tables(inventory)
    write_readme()
    archive_old_article_real_dir(inventory)
    write_inventory(inventory)

    print(f"Created canonical paper artifacts at: {ARTIFACTS}")
    print(f"Inventory rows: {len(inventory)}")


if __name__ == "__main__":
    main()
