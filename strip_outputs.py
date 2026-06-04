"""for pushing on gh
"""
import json
import sys
from pathlib import Path


def strip(nb_path: Path) -> tuple[int, int]:
    size_before = nb_path.stat().st_size
    with nb_path.open("r", encoding="utf-8") as f:
        nb = json.load(f)
    for cell in nb.get("cells", []):
        if cell.get("cell_type") == "code":
            cell["outputs"] = []
            cell["execution_count"] = None
        md = cell.get("metadata", {})
        md.pop("execution", None)
        md.pop("colab", None)
        md.pop("outputId", None)
        md.pop("executionInfo", None)
    nb.get("metadata", {}).pop("widgets", None)
    with nb_path.open("w", encoding="utf-8") as f:
        json.dump(nb, f, ensure_ascii=False, indent=1)
        f.write("\n")
    return size_before, nb_path.stat().st_size


def main(argv):
    targets = []
    for arg in argv:
        p = Path(arg)
        if p.is_dir():
            targets.extend(p.rglob("*.ipynb"))
        elif p.suffix == ".ipynb":
            targets.append(p)
    total_before = total_after = 0
    for nb in targets:
        try:
            before, after = strip(nb)
            total_before += before
            total_after += after
            print(f"{nb}  {before:>10,} -> {after:>10,}")
        except Exception as e:
            print(f"FAIL {nb}: {e}")
    if targets:
        print(f"\nTotal: {total_before:,} -> {total_after:,} "
              f"({100*total_after/total_before:.1f}%)")


if __name__ == "__main__":
    main(sys.argv[1:])
