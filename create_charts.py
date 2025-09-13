import os
import re
import json
from typing import Any, Dict, List, Optional, Tuple
from datetime import datetime, timezone
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.ticker import MaxNLocator

DATASET_DIR = os.getenv("DATASET_DIR", "dataset")
ANALYSIS_DIR = os.getenv("ANALYSIS_DIR", "analysis_all_output")
METRICS_DIRNAME = "metrics"
OUT_BASENAME = "importance_over_time_daily.png"

def slugify(s: str, max_len: int = 120) -> str:
    s = (s or "").strip().lower()
    s = re.sub(r"https?://", "", s)
    s = re.sub(r"[^a-z0-9\-_\.]+", "_", s)
    s = re.sub(r"_+", "_", s).strip("_")
    return s[:max_len] if s else "unknown_link"

def get_url_from_meta(meta: Dict[str, Any]) -> Optional[str]:
    return meta.get("warc-target-uri") or meta.get("url") or meta.get("target-uri")

def parse_warc_date(s: Optional[str]) -> Optional[datetime]:
    if not s or not isinstance(s, str):
        return None
    try:
        if s.endswith("Z"):
            dt = datetime.strptime(s, "%Y-%m-%dT%H:%M:%SZ")
            return dt.replace(tzinfo=timezone.utc)
    except Exception:
        pass
    for fmt in ("%Y-%m-%dT%H:%M:%S%z", "%Y-%m-%d %H:%M:%S%z",
                "%Y-%m-%dT%H:%M:%S", "%Y-%m-%d %H:%M:%S"):
        try:
            dt = datetime.strptime(s, fmt)
            if dt.tzinfo is None:
                dt = dt.replace(tzinfo=timezone.utc)
            return dt.astimezone(timezone.utc)
        except Exception:
            continue
    return None

def iso_or_none(dt: Optional[datetime]) -> Optional[str]:
    if not dt:
        return None
    return dt.astimezone(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")

def textual_diff_is_change(v: str) -> bool:
    if not isinstance(v, str):
        return False
    t = v.strip().lower()
    return t in {"yes", "yes (addition)", "yes addition", "yes (deletion)", "yes deletion"}

def starts_with_important(s: Optional[str]) -> bool:
    return isinstance(s, str) and s.strip().lower().startswith("important -")

def starts_with_not_important(s: Optional[str]) -> bool:
    return isinstance(s, str) and s.strip().lower().startswith("not important -")


def index_dataset_by_slug(dataset_dir: str) -> Dict[str, Dict[str, Any]]:
    """
    Returns {slug: {"pair_timestamps": [ISO M2 per pair],
                     "pair_dates":      ["YYYY-MM-DD" per pair]}}
    """
    out: Dict[str, Dict[str, Any]] = {}
    for fname in os.listdir(dataset_dir):
        if not fname.endswith(".json"):
            continue
        path = os.path.join(dataset_dir, fname)
        try:
            with open(path, "r", encoding="utf-8") as f:
                arr = json.load(f)
        except Exception:
            continue
        if not isinstance(arr, list) or not arr:
            continue

        def sort_key(m: Dict[str, Any]) -> str:
            return m.get("metadata", {}).get("warc-date", "")
        m_sorted = sorted(arr, key=sort_key)

        first_url = None
        for m in m_sorted:
            first_url = get_url_from_meta(m.get("metadata", {}))
            if first_url:
                break
        slug = slugify(first_url if first_url else fname.replace(".json", ""))

        pair_ts: List[str] = []
        pair_dates: List[str] = []
        for i in range(len(m_sorted) - 1):
            m2_meta = m_sorted[i + 1].get("metadata", {})
            dt = parse_warc_date(m2_meta.get("warc-date"))
            iso_ts = iso_or_none(dt) or f"pair_{i+1}"
            pair_ts.append(iso_ts)
            if isinstance(iso_ts, str) and len(iso_ts) >= 10:
                pair_dates.append(iso_ts[:10])
            else:
                pair_dates.append(str(iso_ts))

        out[slug] = {"pair_timestamps": pair_ts, "pair_dates": pair_dates}
    return out


def load_importance_counts(article_dir: str) -> Dict[int, Dict[str, int]]:
    """
    Reads <article_dir>/<pi>.json (LLM outputs) and returns:
      {pair_index: {"important": I, "not_important": N}}
    ONLY for units with change.
    """
    results: Dict[int, Dict[str, int]] = {}
    for fname in os.listdir(article_dir):
        if not fname.endswith(".json"):
            continue
        name_wo_ext = os.path.splitext(fname)[0]
        if not name_wo_ext.isdigit():
            continue
        pi = int(name_wo_ext)

        path = os.path.join(article_dir, fname)
        try:
            with open(path, "r", encoding="utf-8") as f:
                doc = json.load(f)
        except Exception:
            continue

        items = doc.get("items", []) or []
        imp = 0
        nimp = 0
        for it in items:
            a = it.get("assessment", {}) or {}
            if not textual_diff_is_change(a.get("textual differences", "")):
                continue
            oi = a.get("overall importance of the change")
            if starts_with_important(oi):
                imp += 1
            elif starts_with_not_important(oi):
                nimp += 1

        results[pi] = {"important": imp, "not_important": nimp}
    return results

def pick_one_pair_per_day(pair_dates: List[str],
                          llm_counts: Dict[int, Dict[str, int]]) -> List[int]:
    """
    Receives the list of dates per pair (aligned with pair_index) and a dict of counts from the LLM.
    Returns the list of pair_index chosen (one per day), in chronological order.

    Rule:
      - For each date, choose the pair with highest (important + not_important).
      - If all are 0 (or the pair does not exist in llm_counts), choose the LAST pair of the day.
    """
    by_date: Dict[str, List[int]] = {}
    for i, d in enumerate(pair_dates):
        by_date.setdefault(d, []).append(i)

    selected_indices: List[int] = []
    for d in sorted(by_date.keys()):
        candidates = by_date[d]
        best = (-1, 0)  # (score, -pi)
        best_idx: int = candidates[-1]
        for pi in candidates:
            cnt = llm_counts.get(pi, {"important": 0, "not_important": 0})
            score = int(cnt.get("important", 0)) + int(cnt.get("not_important", 0))
            cur = (score, -pi)
            if cur > best:
                best = cur
                best_idx = pi
        selected_indices.append(best_idx)
    return selected_indices

def style_axes(ax):
    ax.grid(True, axis="y", linestyle="--", linewidth=0.6, alpha=0.5)
    for spine in ["top", "right"]:
        ax.spines[spine].set_visible(False)
    ax.yaxis.set_major_locator(MaxNLocator(integer=True))

def add_value_labels(ax, x, heights_bottom, heights_top):
    totals = [b + t for b, t in zip(heights_bottom, heights_top)]
    for xi, total in zip(x, totals):
        if total > 0:
            ax.text(xi, total + 0.1, str(int(total)), ha="center", va="bottom", fontsize=8)

def plot_stacked_importance_daily(dates: List[str],
                                  important_counts: List[int],
                                  not_important_counts: List[int],
                                  out_path: str,
                                  title: str,
                                  subtitle: Optional[str] = None) -> None:
    n = len(dates)
    x = np.arange(n)

    fig_w = max(10, n * 0.6)
    fig, ax = plt.subplots(figsize=(fig_w, 5))
    palette = ["#1f4e79", "#5b9bd5"] 

    ax.bar(x, important_counts, label="Important", color=palette[0])
    ax.bar(x, not_important_counts, bottom=important_counts, label="Not important", color=palette[1])

    if subtitle:
        ax.set_title(title + "\n" + subtitle, fontsize=12, loc="left")
    else:
        ax.set_title(title, fontsize=12, loc="left")

    ax.set_xlabel("Date")
    ax.set_ylabel("Changed units")
    ax.set_xticks(x, dates, rotation=45, ha="right")

    style_axes(ax)
    ax.legend(ncol=2, frameon=False, loc="upper left")
    add_value_labels(ax, x, important_counts, not_important_counts)

    fig.tight_layout()
    fig.savefig(out_path, dpi=180)
    plt.close(fig)


def main() -> None:
    ds_index = index_dataset_by_slug(DATASET_DIR)
    if not os.path.isdir(ANALYSIS_DIR):
        raise RuntimeError(f"No existe ANALYSIS_DIR: {ANALYSIS_DIR}")

    subdirs = [d for d in os.listdir(ANALYSIS_DIR)
               if os.path.isdir(os.path.join(ANALYSIS_DIR, d))]
    subdirs.sort()

    for slug in subdirs:
        article_dir = os.path.join(ANALYSIS_DIR, slug)
        if slug not in ds_index:
            print(f"⚠ No dataset match for slug '{slug}'. Skipping.")
            continue

        pair_dates = ds_index[slug]["pair_dates"]
        pair_ts    = ds_index[slug]["pair_timestamps"]
        llm_counts = load_importance_counts(article_dir)

        picked = pick_one_pair_per_day(pair_dates, llm_counts)

        daily_dates: List[str] = []
        imp_series: List[int] = []
        nimp_series: List[int] = []
        for pi in picked:
            d = pair_dates[pi]
            cnt = llm_counts.get(pi, {"important": 0, "not_important": 0})
            daily_dates.append(d)
            imp_series.append(int(cnt.get("important", 0)))
            nimp_series.append(int(cnt.get("not_important", 0)))

        metrics_dir = os.path.join(article_dir, METRICS_DIRNAME)
        os.makedirs(metrics_dir, exist_ok=True)
        out_path = os.path.join(metrics_dir, OUT_BASENAME)

        subtitle = None
        if pair_ts:
            first = pair_ts[0][:10] if isinstance(pair_ts[0], str) else str(pair_ts[0])
            last  = pair_ts[-1][:10] if isinstance(pair_ts[-1], str) else str(pair_ts[-1])
            subtitle = f"Coverage: {first} → {last} (one pair of consecutive mementos per day)"

        title = f"Importance of changes over time"
        plot_stacked_importance_daily(daily_dates, imp_series, nimp_series, out_path, title, subtitle)

        print(f"✓ Saved chart: {out_path}")

    print("✅ All daily importance charts generated.")

if __name__ == "__main__":
    main()

