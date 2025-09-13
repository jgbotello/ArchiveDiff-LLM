import argparse
import json
from pathlib import Path
from typing import List, Tuple

def count_mementos_in_file(path: Path) -> int:
    """Returns the number of mementos in the JSON file.
    It expects an array (list). If it finds a dict with 'mementos' (list), it also supports it."""
    try:
        with path.open("r", encoding="utf-8") as f:
            data = json.load(f)
        if isinstance(data, list):
            return len(data)
        if isinstance(data, dict) and isinstance(data.get("mementos"), list):
            return len(data["mementos"])
        print(f"⚠ Unexpected format (no list ni dict['mementos']): {path.name}")
        return 0
    except Exception as e:
        print(f"✗ Could not read {path.name}: {e}")
        return 0

def print_table(rows: List[Tuple[str, int]]) -> None:
    """Prints a simple aligned table."""
    if not rows:
        print("No .json files were found.")
        return
    name_w = max(len("File"), max(len(r[0]) for r in rows))
    cnt_w  = max(len("Mementos"), max(len(str(r[1])) for r in rows))
    sep = "-" * (name_w + cnt_w + 3)
    print(sep)
    print(f"{'File'.ljust(name_w)} | {'Mementos'.rjust(cnt_w)}")
    print(sep)
    for name, cnt in rows:
        print(f"{name.ljust(name_w)} | {str(cnt).rjust(cnt_w)}")
    print(sep)

def main():
    ap = argparse.ArgumentParser(description="Count items by file and total.")
    ap.add_argument("dataset_dir", nargs="?", default="dataset",
                    help="Folder with .json files (default: dataset)")
    ap.add_argument("--csv", metavar="PATH", help="Save summary as CSV (filename,mementos,total_pairs)")
    args = ap.parse_args()

    ds = Path(args.dataset_dir)
    if not ds.is_dir():
        raise SystemExit(f"The directory does not exist: {ds}")

    files = sorted([p for p in ds.iterdir() if p.is_file() and p.suffix.lower() == ".json"])
    rows: List[Tuple[str, int]] = []
    total_mementos = 0
    total_pairs = 0 

    for p in files:
        n = count_mementos_in_file(p)
        rows.append((p.name, n))
        total_mementos += n
        total_pairs += max(0, n - 1)

    # print table
    print_table(rows)
    print(f"Total mementos: {total_mementos}")
    print(f"Total consecutive pairs (n-1 per file): {total_pairs}")

    if args.csv:
        import csv
        with open(args.csv, "w", newline="", encoding="utf-8") as f:
            w = csv.writer(f)
            w.writerow(["filename", "mementos", "consecutive_pairs"])
            for name, n in rows:
                w.writerow([name, n, max(0, n - 1)])
            w.writerow(["__TOTAL__", total_mementos, total_pairs])
        print(f"✓ CSV saved in: {args.csv}")

if __name__ == "__main__":
    main()
