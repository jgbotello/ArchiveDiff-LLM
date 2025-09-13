# run through analysis_all_output/<news>/*.json and type metrics/metrics.json
# with {"summary": {...}, "per_pair": [...]} sorted chronologically.
# all quantifiable metrics (semantic, sentiment, POS/NER, grammar, verbals, rewritten, relevance)
# are calculated **only among units that CHANGED** (textual differences in {"yes", "yes (addition)", "yes (deletion)"}).

import os
import json
from typing import Any, Dict, List, Optional, Tuple
from collections import Counter
from datetime import datetime, timezone

# ================= Config =================
ANALYSIS_ROOT = os.getenv("ANALYSIS_DIR", "analysis_all_output")
PAIR_FILE_SUFFIX = ".json"          
METRICS_DIRNAME = "metrics"          
METRICS_FILENAME = "metrics.json"    

# ================ Helpers =================
def is_pair_file(article_dir: str, fname: str) -> bool:
    if not fname.endswith(PAIR_FILE_SUFFIX):
        return False
    if fname.lower() == METRICS_FILENAME.lower():
        return False
    path = os.path.join(article_dir, fname)
    return os.path.isfile(path)

def safe_get(d: Dict[str, Any], *keys, default=None):
    cur = d
    for k in keys:
        if not isinstance(cur, dict) or k not in cur:
            return default
        cur = cur[k]
    return cur

def parse_warc_date(s: Optional[str]) -> Optional[datetime]:
    """Parsea '2012-06-21T02:46:46Z' a datetime UTC."""
    if not s or not isinstance(s, str):
        return None
    if s.endswith("Z"):
        try:
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

def to_list(value) -> List[str]:
    """Normaliza a lista: None->[], 'none'->[], str->[str], list->list filtrada."""
    if value is None:
        return []
    if isinstance(value, list):
        return [v for v in value if isinstance(v, str)]
    if isinstance(value, str):
        if value.strip().lower() in {"none", ""}:
            return []
    return [str(value)]

def starts_with_important(s: Optional[str]) -> bool:
    return isinstance(s, str) and s.strip().lower().startswith("important -")

def starts_with_not_important(s: Optional[str]) -> bool:
    return isinstance(s, str) and s.strip().lower().startswith("not important -")

def count_units(items: List[Dict[str, Any]]) -> Tuple[int, int, int]:
    """(total_units, units_M1_non_null, units_M2_non_null)"""
    total = len(items)
    m1_nonnull = 0
    m2_nonnull = 0
    for it in items:
        s = it.get("sentences", {})
        if isinstance(s.get("M1", None), str) and s["M1"].strip():
            m1_nonnull += 1
        if isinstance(s.get("M2", None), str) and s["M2"].strip():
            m2_nonnull += 1
    return total, m1_nonnull, m2_nonnull

# ============== Metrics per pair ==============
def compute_pair_metrics(doc: Dict[str, Any]) -> Dict[str, Any]:
    items = doc.get("items", []) or []
    total_units, units_m1, units_m2 = count_units(items)

    # timestamps
    t_old = parse_warc_date(safe_get(doc, "metadata_old", "warc-date"))
    t_new = parse_warc_date(safe_get(doc, "metadata_new", "warc-date"))
    delta_hours = None
    if t_old and t_new:
        delta_hours = (t_new - t_old).total_seconds() / 3600.0

    # counters per pair
    td_yes = td_yes_add = td_yes_del = td_no = 0
    important_cnt = 0
    not_important_cnt = 0
    unknown_importance_changed = 0

    # Type counts (total and changed only)
    item_type_counts_total = Counter({"match":0,"insert":0,"delete":0})
    changed_type_counts_mid = Counter({"match":0,"insert":0,"delete":0})

    sem_counts = Counter()                # NA/low/moderate/high
    sent_before_counts = Counter()        # Very Negative..Very Positive
    sent_after_counts = Counter()
    sent_dir_counts = Counter()           # more positive/no change/more negative
    pos_counts = Counter({"VERB":0,"NOUN":0,"PROPN":0,"ADJ":0,"NUM":0,"ADV":0})
    ner_counts = Counter({"PERSON":0,"ORG":0,"GPE":0,"LOC":0,"DATE":0,"MONEY":0,"PERCENT":0})
    grammar_counts = Counter({"yes":0,"no":0})
    verbal_counts = Counter({"tense":0,"aspect":0,"voice":0,"modality":0})
    rewritten_counts = Counter({True:0, False:0})
    importance_category_counts = Counter()

    textual_diff_counts = Counter()       # yes/no/yes (addition)/yes (deletion)

    changed_units_total = 0

    for it in items:
        it_type = str(it.get("type", "")).strip().lower()
        if it_type in item_type_counts_total:
            item_type_counts_total[it_type] += 1

        a = it.get("assessment", {}) or {}

        tdl_raw = str(a.get("textual differences", "")).strip()
        tdl = tdl_raw.lower()
        textual_diff_counts[tdl_raw] += 1

        is_changed = False
        if tdl in {"yes (addition)", "yes addition"}:
            td_yes_add += 1; is_changed = True
        elif tdl in {"yes (deletion)", "yes deletion"}:
            td_yes_del += 1; is_changed = True
        elif tdl == "yes":
            td_yes += 1; is_changed = True
        else:
            td_no += 1

        if not is_changed:
            continue

        changed_units_total += 1

        oi = a.get("overall importance of the change")
        if starts_with_important(oi):
            important_cnt += 1
        elif starts_with_not_important(oi):
            not_important_cnt += 1
        else:
            unknown_importance_changed += 1

        if it_type in changed_type_counts_mid:
            changed_type_counts_mid[it_type] += 1

        sil_raw = str(a.get("semantic impact", "")).strip()
        if sil_raw:
            sem_counts[sil_raw] += 1

        sb = a.get("sentiment before")
        sa = a.get("sentiment after")
        sd = a.get("sentiment change direction")
        if isinstance(sb, str) and sb:
            sent_before_counts[sb] += 1
        if isinstance(sa, str) and sa:
            sent_after_counts[sa] += 1
        if isinstance(sd, str) and sd:
            sent_dir_counts[sd] += 1

        for p in to_list(a.get("POS category changed")):
            if p in pos_counts:
                pos_counts[p] += 1
        for n in to_list(a.get("NER category changed")):
            if n in ner_counts:
                ner_counts[n] += 1

        gc = str(a.get("grammar change", "")).strip().lower()
        if gc in {"yes", "no"}:
            grammar_counts[gc] += 1

        for v in to_list(a.get("verbal changes")):
            v_norm = v.strip().lower()
            if v_norm in verbal_counts:
                verbal_counts[v_norm] += 1

        rw = a.get("rewritten", None)
        if isinstance(rw, bool):
            rewritten_counts[rw] += 1

        ic = a.get("importance category")
        if isinstance(ic, str) and ic.strip():
            importance_category_counts[ic.strip().lower()] += 1

    changes_total = td_yes + td_yes_add + td_yes_del

    v_flags = {
        "tense": 1 if verbal_counts.get("tense", 0) > 0 else 0,
        "aspect": 1 if verbal_counts.get("aspect", 0) > 0 else 0,
        "voice": 1 if verbal_counts.get("voice", 0) > 0 else 0,
        "modality": 1 if verbal_counts.get("modality", 0) > 0 else 0,
    }

    return {
        "pair_index": doc.get("pair_index"),
        "timestamp_old": iso_or_none(t_old),
        "timestamp_new": iso_or_none(t_new),
        "delta_hours": delta_hours,
        "units_total": total_units,
        "units_m1_nonnull": units_m1,
        "units_m2_nonnull": units_m2,
        # Pair counts (units)
        "units_by_type": dict(item_type_counts_total),          # match/insert/delete (totales)
        "changed_units_by_type": dict(changed_type_counts_mid), # match/insert/delete ONLY changed
        "changed_units_total": changed_units_total,             # quantifiable metrics base
        # Changes and importance
        "changes": {
            "total_changed": changes_total,
            "yes": td_yes,
            "yes_addition": td_yes_add,
            "yes_deletion": td_yes_del,
            "no": td_no,
            "important": important_cnt,              
            "not_important": not_important_cnt,    
            "unknown_importance_among_changed": unknown_importance_changed,
        },
        "semantic": dict(sem_counts),
        "verbal_flags": v_flags,
        "llm_fields": {
            "textual_differences": dict(textual_diff_counts),  
            "semantic_impact": dict(sem_counts),              
            "sentiment_before": dict(sent_before_counts),      
            "sentiment_after": dict(sent_after_counts),        
            "sentiment_change_direction": dict(sent_dir_counts),
            "importance_overall": {
                "Important": important_cnt,
                "Not important": not_important_cnt,
                "Unknown (among changed)": unknown_importance_changed,
            },
            "importance_category_freq": dict(importance_category_counts),
            "pos_category_changed": dict(pos_counts),
            "ner_category_changed": dict(ner_counts),
            "grammar_change": dict(grammar_counts),
            "verbal_changes": dict(verbal_counts),
            "rewritten": {"true": rewritten_counts[True], "false": rewritten_counts[False]},
            
            "bases": {
                "units_total": total_units,
                "changed_units_total": changed_units_total
            },
        },
    }

def _sum_dict_of_ints(a: Dict[str, int], b: Dict[str, int]) -> Dict[str, int]:
    out = dict(a)
    for k, v in b.items():
        out[k] = out.get(k, 0) + int(v)
    return out

def _merge_llm_fields(acc: Dict[str, Any], cur: Dict[str, Any]) -> Dict[str, Any]:
    """Sum all counters inside llm_fields (all metrics are already 'just changed')."""
    merged = {}
    keys_counter = {
        "textual_differences", "semantic_impact", "sentiment_before",
        "sentiment_after", "sentiment_change_direction", "importance_category_freq",
        "pos_category_changed", "ner_category_changed", "grammar_change",
        "verbal_changes",
    }
    for k in keys_counter:
        merged[k] = _sum_dict_of_ints(acc.get(k, {}), cur.get(k, {}))

    imp_acc = acc.get("importance_overall", {})
    imp_cur = cur.get("importance_overall", {})
    merged["importance_overall"] = {
        "Important": int(imp_acc.get("Important", 0)) + int(imp_cur.get("Important", 0)),
        "Not important": int(imp_acc.get("Not important", 0)) + int(imp_cur.get("Not important", 0)),
        "Unknown (among changed)": int(imp_acc.get("Unknown (among changed)", 0)) + int(imp_cur.get("Unknown (among changed)", 0)),
    }
    rw_acc = acc.get("rewritten", {"true": 0, "false": 0})
    rw_cur = cur.get("rewritten", {"true": 0, "false": 0})
    merged["rewritten"] = {
        "true": int(rw_acc.get("true", 0)) + int(rw_cur.get("true", 0)),
        "false": int(rw_acc.get("false", 0)) + int(rw_cur.get("false", 0)),
    }

    base_acc = acc.get("bases", {})
    base_cur = cur.get("bases", {})
    merged["bases"] = {
        "units_total": int(base_acc.get("units_total", 0)) + int(base_cur.get("units_total", 0)),
        "changed_units_total": int(base_acc.get("changed_units_total", 0)) + int(base_cur.get("changed_units_total", 0)),
    }
    return merged

def build_summary(per_pair_metrics: List[Dict[str, Any]]) -> Dict[str, Any]:
    """Suma directa de los outputs por par."""
    summary = {
        "pairs_total": len(per_pair_metrics),
        "units_total": 0,
        "units_m1_nonnull": 0,
        "units_m2_nonnull": 0,
        "units_by_type": {"match":0,"insert":0,"delete":0},
        "changed_units_by_type": {"match":0,"insert":0,"delete":0},
        "changed_units_total": 0, 
        "changes": {
            "total_changed": 0,
            "yes": 0,
            "yes_addition": 0,
            "yes_deletion": 0,
            "no": 0,
            "important": 0,
            "not_important": 0,
            "unknown_importance_among_changed": 0,
        },
        "semantic": {}, 
        "llm_fields": {
            "textual_differences": {},
            "semantic_impact": {},
            "sentiment_before": {},
            "sentiment_after": {},
            "sentiment_change_direction": {},
            "importance_overall": {"Important": 0, "Not important": 0, "Unknown (among changed)": 0},
            "importance_category_freq": {},
            "pos_category_changed": {},
            "ner_category_changed": {},
            "grammar_change": {},
            "verbal_changes": {},
            "rewritten": {"true": 0, "false": 0},
            "bases": {"units_total": 0, "changed_units_total": 0},
        },
    }

    for p in per_pair_metrics:
        summary["units_total"] += int(p.get("units_total", 0))
        summary["units_m1_nonnull"] += int(p.get("units_m1_nonnull", 0))
        summary["units_m2_nonnull"] += int(p.get("units_m2_nonnull", 0))
        summary["units_by_type"] = _sum_dict_of_ints(summary["units_by_type"], p.get("units_by_type", {}))
        summary["changed_units_by_type"] = _sum_dict_of_ints(summary["changed_units_by_type"], p.get("changed_units_by_type", {}))
        summary["changed_units_total"] += int(p.get("changed_units_total", 0))

        ch = p.get("changes", {})
        summary["changes"]["total_changed"] += int(ch.get("total_changed", 0))
        summary["changes"]["yes"] += int(ch.get("yes", 0))
        summary["changes"]["yes_addition"] += int(ch.get("yes_addition", 0))
        summary["changes"]["yes_deletion"] += int(ch.get("yes_deletion", 0))
        summary["changes"]["no"] += int(ch.get("no", 0))
        summary["changes"]["important"] += int(ch.get("important", 0))
        summary["changes"]["not_important"] += int(ch.get("not_important", 0))
        summary["changes"]["unknown_importance_among_changed"] += int(ch.get("unknown_importance_among_changed", 0))

        summary["semantic"] = _sum_dict_of_ints(summary["semantic"], p.get("semantic", {}))

        summary["llm_fields"] = _merge_llm_fields(summary["llm_fields"], p.get("llm_fields", {}))

    return summary

# ============== Main ==============
def main() -> None:
    if not os.path.isdir(ANALYSIS_ROOT):
        raise RuntimeError(f"The directory does not exist: {ANALYSIS_ROOT}")

    subdirs = [os.path.join(ANALYSIS_ROOT, d)
               for d in os.listdir(ANALYSIS_ROOT)
               if os.path.isdir(os.path.join(ANALYSIS_ROOT, d))]
    subdirs.sort()

    for article_dir in subdirs:
        pair_files = [f for f in os.listdir(article_dir) if is_pair_file(article_dir, f)]
        pair_files.sort()

        per_pair_metrics: List[Dict[str, Any]] = []
        for fname in pair_files:
            path = os.path.join(article_dir, fname)
            try:
                with open(path, "r", encoding="utf-8") as f:
                    doc = json.load(f)
            except Exception as e:
                print(f"✗ Skipping unreadable file: {path} ({e})")
                continue
            if not isinstance(doc, dict) or "items" not in doc:
                print(f"✗ Unexpected format (no 'items'): {path}")
                continue
            per_pair_metrics.append(compute_pair_metrics(doc))

        if not per_pair_metrics:
            continue

        def sort_key(p):
            tn = p.get("timestamp_new")
            pi = p.get("pair_index")
            return (tn or "", pi if isinstance(pi, int) else 10**9)

        per_pair_metrics = sorted(per_pair_metrics, key=sort_key)
        summary = build_summary(per_pair_metrics)

        metrics_dir = os.path.join(article_dir, METRICS_DIRNAME)
        os.makedirs(metrics_dir, exist_ok=True)
        metrics_json_path = os.path.join(metrics_dir, METRICS_FILENAME)
        with open(metrics_json_path, "w", encoding="utf-8") as outf:
            json.dump({"summary": summary, "per_pair": per_pair_metrics}, outf, ensure_ascii=False, indent=2)
        print(f"✓ Saved {metrics_json_path}")

    print("🏁 Done: metrics/metrics.json created in each news folder.")

if __name__ == "__main__":
    main()
