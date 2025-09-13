import os
import time
import json
import re
import random
from typing import Any, Dict, List, Optional, Tuple
from openai import OpenAI

# =========================
# Configuration
# =========================
API_KEY = os.getenv("OPENAI_API_KEY") or "Paste_KEY_HERE"
if not API_KEY or API_KEY == "Paste_KEY_HERE":
    raise RuntimeError("Set OPENAI_API_KEY or hard-code it in API_KEY before running.")

MODEL = "gpt-4o-mini"

DATASET_DIR  = "dataset"                   
ANALYSIS_DIR = "analysis_all_output"      
START_FROM_PAIR_INDEX = 0                 

# Rate limiting / retries
OPENAI_RPM = float(os.getenv("OPENAI_RPM", "20"))
REQ_JITTER = float(os.getenv("OPENAI_REQ_JITTER", "0.2"))
OPENAI_MAX_RETRIES = int(os.getenv("OPENAI_MAX_RETRIES", "5"))
OPENAI_BASE_BACKOFF = float(os.getenv("OPENAI_BASE_BACKOFF", "2.0"))
FILE_PAUSE_SECONDS = float(os.getenv("FILE_PAUSE_SECONDS", "2.0"))

os.makedirs(ANALYSIS_DIR, exist_ok=True)
client = OpenAI(api_key=API_KEY)

# =========================
# JSON schema
# =========================
JSON_SCHEMA: Dict[str, Any] = {
    "name": "change_assessments",
    "schema": {
        "type": "array",
        "items": {
            "type": "object",
            "properties": {
                "type": { "type": "string", "enum": ["match","insert","delete","split","merge"] },
                "sentences": {
                    "type": "object",
                    "properties": {
                        "M1": {"type": ["string", "null"]},
                        "M2": {"type": ["string", "null"]}
                    },
                    "required": ["M1", "M2"],
                    "additionalProperties": False
                },
                "assessment": {
                    "type": "object",
                    "properties": {
                        "textual differences": {
                            "type": "string",
                            "enum": ["yes", "no", "yes (addition)", "yes (deletion)"]
                        },
                        "semantic impact": { "type": "string", "enum": ["NA", "low", "moderate", "high"] },
                        "sentiment before": { "type": "string", "enum": ["Very Negative", "Negative", "Neutral","Positive","Very Positive"] },
                        "sentiment after": { "type": "string", "enum": ["Very Negative", "Negative", "Neutral","Positive","Very Positive"] },
                        "sentiment change direction": { "type": "string", "enum": ["more positive","no change","more negative"] },
                        "overall importance of the change": {
                            "type": "string",
                            "pattern": r"^(?:[Ii]mportant|[Nn]ot important)\s-\s.+$"
                        },
                        "importance category": { "type": "string" },
                        "importance reason": { "type": "string" },
                        "literature rationale": { "type": "string" },
                        "version diff summary": { "type": "string" },
                        "overall assessment": { "type": "string" },
                        "POS category changed": {
                            "type": "array",
                            "items": { "type": "string", "enum": ["VERB","NOUN","PROPN","ADJ","NUM","ADV"] },
                            "uniqueItems": True
                        },
                        "NER category changed": {
                            "type": "array",
                            "items": { "type": "string", "enum": ["PERSON","ORG","GPE","LOC","DATE","MONEY","PERCENT"] },
                            "uniqueItems": True
                        },
                        "grammar change": { "type": "string", "enum": ["yes","no"] },
                        "verbal changes": {
                            "type": "array",
                            "items": { "type": "string", "enum": ["tense","aspect","voice","modality"] },
                            "uniqueItems": True
                        },
                        "rewritten": { "type": "boolean" }
                    },
                    "required": [
                        "textual differences",
                        "semantic impact",
                        "sentiment before",
                        "sentiment after",
                        "sentiment change direction",
                        "overall importance of the change",
                        "importance category",
                        "importance reason",
                        "literature rationale",
                        "version diff summary",
                        "overall assessment",
                        "POS category changed",
                        "NER category changed",
                        "grammar change",
                        "verbal changes",
                        "rewritten"
                    ],
                    "additionalProperties": False
                }
            },
            "required": ["sentences", "assessment"],
            "additionalProperties": False
        }
    }
}

# =========================
# Rate limiter
# =========================
class RateLimiter:
    def __init__(self, rpm: float, jitter: float = 0.2):
        self.min_delay = 60.0 / max(1.0, rpm)
        self.jitter = max(0.0, jitter)
        self._last_ts = 0.0

    def wait(self):
        now = time.time()
        delay = self.min_delay * (1.0 + random.uniform(-self.jitter, self.jitter))
        elapsed = now - self._last_ts
        sleep_for = delay - elapsed
        if sleep_for > 0:
            time.sleep(sleep_for)
        self._last_ts = time.time()

rate_limiter = RateLimiter(OPENAI_RPM, REQ_JITTER)

# =========================
# Helpers
# =========================
def slugify(s: str, max_len: int = 120) -> str:
    s = (s or "").strip().lower()
    s = re.sub(r"https?://", "", s)
    s = re.sub(r"[^a-z0-9\-_\.]+", "_", s)
    s = re.sub(r"_+", "_", s).strip("_")
    return s[:max_len] if s else "unknown_link"

def rough_sentence_split(txt: str) -> List[str]:
    if not isinstance(txt, str):
        return []
    parts = re.split(r'(?<=[\.\?\!])\s+|\n+', (txt or "").strip())
    return [p.strip() for p in parts if p.strip()]

def rough_sentence_count(txt: str) -> int:
    return len(rough_sentence_split(txt))

def extract_first_json_array(text: str) -> List[Dict[str, Any]]:
    text = text.strip()
    if text.startswith('['):
        return json.loads(text)
    start = text.find('[')
    if start == -1:
        raise ValueError("No '[' found in model output.")
    depth = 0
    for i in range(start, len(text)):
        c = text[i]
        if c == '[':
            depth += 1
        elif c == ']':
            depth -= 1
            if depth == 0:
                return json.loads(text[start:i+1])
    raise ValueError("Could not find a complete top-level JSON array in model output.")

def validate_array_range(arr: List[Dict[str, Any]], min_len: int, max_len: int) -> Tuple[bool, str]:
    if not isinstance(arr, list):
        return False, "Output root is not an array."
    if len(arr) < min_len:
        return False, f"Too few items: got {len(arr)}, need at least {min_len}."
    if len(arr) > max_len:
        return False, f"Too many items: got {len(arr)}, must be <= {max_len}."
    for idx, item in enumerate(arr):
        if not isinstance(item, dict):
            return False, f"Item {idx} is not an object."
        s = item.get("sentences")
        if not isinstance(s, dict):
            return False, f"Item {idx}: 'sentences' missing or not object."
        m1 = s.get("M1", None)
        m2 = s.get("M2", None)
        if m1 is None and m2 is None:
            return False, f"Item {idx}: both M1 and M2 are null."
        a = item.get("assessment")
        if not isinstance(a, dict):
            return False, f"Item {idx}: 'assessment' missing or not object."
        # Requeridos según el schema de codigo 2
        for key in [
            "textual differences",
            "semantic impact",
            "sentiment before",
            "sentiment after",
            "sentiment change direction",
            "overall importance of the change",
            "importance category",
            "importance reason",
            "literature rationale",
            "version diff summary",
            "overall assessment",
            "POS category changed",
            "NER category changed",
            "grammar change",
            "verbal changes",
            "rewritten",
        ]:
            if key not in a:
                return False, f"Item {idx}: missing '{key}' in assessment."
    return True, ""

def get_url_from_meta(meta: Dict[str, Any]) -> Optional[str]:
    return (
        meta.get("warc-target-uri")
        or meta.get("url")
        or meta.get("target-uri")
    )

def get_text_and_meta(m: Dict[str, Any]) -> Tuple[str, Dict[str, Any]]:
    # Supports two common formats: {"article":{"text":...}, "metadata":{...}} or {"text":..., "metadata":{...}}
    text = ""
    if isinstance(m.get("article"), dict):
        text = m["article"].get("text") or ""
    if not text:
        text = m.get("text") or ""
    meta = m.get("metadata", {})
    return text, meta

def _present(x: Any) -> bool:
    if x is None:
        return False
    if isinstance(x, str):
        s = x.strip()
        if not s:
            return False
        if s.lower() in {"null", "none"}:
            return False
    return True

def normalize_text(text: str) -> str:
    if text is None:
        return ""
    t = text.lower()
    t = re.sub(r'\s+', ' ', t)
    return t.strip()

def has_change(old_text: str, new_text: str) -> bool:
    """True si difieren tras normalizar."""
    return normalize_text(old_text) != normalize_text(new_text)

# =========================
# Prompt for segmenting + aligning + evaluating changes
# =========================
def build_user_prompt_align_and_assess(m1_text: str, m2_text: str, min_items: int, max_items: int) -> str:
    return (
        "You will receive two mementos (M1, M2). Let's Think Step by Step\n"
        "Perform these tasks internally (do NOT reveal steps):\n"
        "1) Split both mementos into sentences.\n"
        "2) Align sentences into operations: match / insert / delete\n"
        "3) For each aligned unit, output ONE JSON object with:\n"
        "   - 'type': one of ['match','insert','delete','split','merge']\n"
        "   - 'sentences': { M1: <string|null>, M2: <string|null> } (copy verbatim; use null for missing side)\n"
        "   - 'assessment': {\n"
        "       'textual differences': 'yes' | 'no' | 'yes (addition)' | 'yes (deletion)',\n"
        "       'POS category changed': [], or any of ['VERB','NOUN','PROPN','ADJ','NUM','ADV'],\n"
        "       'NER category changed': [], or any of ['PERSON','ORG','GPE','LOC','DATE','MONEY','PERCENT'],\n"
        "       'grammar change': 'yes' | 'no',\n"
        "       'verbal changes': [], or any of ['tense','aspect','voice','modality'],\n"
        "       'rewritten': true | false,\n"
        "       'semantic impact': 'NA' | 'low' | 'moderate' | 'high',\n"
        "       'sentiment before': one of ['Very Negative','Negative','Neutral','Positive','Very Positive'],\n"
        "       'sentiment after': one of ['Very Negative','Negative','Neutral','Positive','Very Positive'],\n"
        "       'sentiment change direction': one of ['more positive','no change','more negative'],\n"
        "       'overall importance of the change': 'Important - <why>' or 'Not important - <why>',\n"
        "       'importance category': short free-text label (e.g., 'major wording','temporal anchoring','numerical update','named-entity change','attribution/hedging','policy implication', etc.),\n"
        "       'importance reason': 1–2 sentences explaining why the change matters,\n"
        "       'literature rationale': brief paragraph (2–4 sentences) grounding the category in relevant scholarly areas (journalism studies, credibility, HCI change perception, fact-checking, temporal reasoning),\n"
        "       'version diff summary': concise 1–2 sentence summary of what changed between M1 and M2,\n"
        "       'overall assessment': a concise 2–4 sentence synthesis combining textual differences, POS/NER changes, grammar/verb changes, rewritten, semantic impact, sentiments and direction, and overall importance.\n"
        "     }\n"
        "\n"
        "DEFINITIONS & RULES (be strict):\n"
        "• 'textual differences':\n"
        "   - 'yes (addition)' if M1 is null and M2 has text.\n"
        "   - 'yes (deletion)' if M2 is null and M1 has text.\n"
        "   - 'yes' if both sides exist but wording differs; 'no' only if strings are identical (ignoring trivial whitespace).\n"
        "\n"
        "• 'POS category changed' (select ALL that apply; use [] if none):\n"
        "   - VERB: change in main verb(s) or auxiliaries that alter verb lexeme or form (e.g., 'was'→'is', 'support'→'oppose').\n"
        "   - NOUN: change/add/remove a common noun head or key nominal (e.g., 'protests'→'riots').\n"
        "   - PROPN: addition/removal/change of a proper name/title (e.g., 'Mr. Mubarak', 'Apple').\n"
        "   - ADJ: change in an attributive/predicative adjective affecting description/intensity (e.g., 'severe'→'mild').\n"
        "   - NUM: numbers or quantities (e.g., '3'→'5', 'thousands'→'dozens').\n"
        "   - ADV: adverbs that change manner, time, degree or negation (e.g., 'allegedly', 'no longer').\n"
        "\n"
        "• 'NER category changed' (select ALL that apply; use [] if none):\n"
        "   - PERSON (named individual), ORG (organization), GPE (countries/cities/polities), LOC (non-GPE locations),\n"
        "     DATE (dates/periods like 'June 5', 'beginning of the month'), MONEY, PERCENT.\n"
        "\n"
        "• 'grammar change': 'yes' when there are grammatical/orthographic edits NOT captured by POS/NER lists, e.g.,\n"
        "   articles/determiners, prepositions, conjunctions, agreement/number (singular/plural), word order micro-edits,\n"
        "   punctuation/capitalization/hyphenation, and spelling variants. If POS/NER changes also occurred, still set 'yes'.\n"
        "\n"
        "• 'verbal changes' (pick all that apply; [] if none):\n"
        "   - tense (time: present/past/future; e.g., 'have been'→'had been'),\n"
        "   - aspect (progressive/perfect; e.g., 'is investigating'→'has investigated'),\n"
        "   - voice (active↔passive),\n"
        "   - modality (may/might/should/must/likely/allegedly).\n"
        "\n"
        "• 'rewritten': true if structure/order is substantially rephrased beyond minor edits.\n"
        "   False when changes are local without structural rephrasing.\n"
        "\n"
        "• 'semantic impact' (does the *meaning, interpretation or factual content* change?):\n"
        "   - NA: Not applicable if there are no changes.\n"
        "   - low: equivalent meaning; no new facts/specificity.\n"
        "   - moderate: added specificity/hedging/attribution that refines but does not contradict core facts.\n"
        "   - high: Meaning is not equivalent or facts/claims/entities/numbers/dates change; reversals/contradictions; materially new information.\n"
        "\n"
        "• SENTIMENT CATEGORIES (analyze each side independently):\n"
        "   - Very Negative: highly unfavorable, strongly critical, or alarming polarity with clear negative intensity.\n"
        "   - Negative: unfavorable, critical, or disapproving polarity, but less intense than Very Negative.\n"
        "   - Neutral: factual, descriptive, or balanced expression with no clear positive or negative polarity.\n"
        "   - Positive: favorable, approving, or supportive polarity, but less intense than Very Positive.\n"
        "   - Very Positive: highly favorable, strongly approving, or enthusiastic polarity with clear positive intensity.\n"
        "  Then set 'sentiment change direction' by comparing M1 vs M2: 'more positive', 'no change', or 'more negative'.\n"
        "\n"
        "• 'overall importance of the change':\n"
        "   Mark 'Important' if the change affects meaning in any way, Otherwise 'Not important'\n"
        "   Use exactly 'Important - …' or 'Not important - …' followed by a brief explanation of why.\n"
        "\n"
        "• 'importance category': provide a compact, human-readable label (e.g., 'major wording', 'hedging/attribution', etc.).\n"
        "\n"
        "• 'literature rationale': briefly connect the category to relevant scholarly areas and literature. Do not cite specific papers; keep it general but academically grounded.\n"
        "\n"
        f"Constraints: Return a JSON ARRAY ONLY (no prose). The array length must be BETWEEN {min_items} and {max_items}; preserve order; never create an item where both M1 and M2 are null.\n"
        "\nMemento 1 (M1):\n" + m1_text + "\n\nMemento 2 (M2):\n" + m2_text + "\n"
    )

def _is_transient_error(err: Exception) -> bool:
    s = str(err).lower()
    return any(k in s for k in [
        "rate", "429", "too many", "overload", "temporarily", "timeout",
        "timed out", "connection reset", "502", "503", "504"
    ])

def _chat_create_with_rate_limit(messages: List[Dict[str, str]],
                                 max_tokens: int,
                                 use_schema: bool = False) -> Any:
    last_exc = None
    for attempt in range(OPENAI_MAX_RETRIES):
        try:
            rate_limiter.wait()
            if use_schema:
                resp = client.chat.completions.create(
                    model=MODEL,
                    messages=messages,
                    temperature=0.0,
                    max_tokens=max_tokens,
                    response_format={"type": "json_schema", "json_schema": JSON_SCHEMA},
                )
            else:
                resp = client.chat.completions.create(
                    model=MODEL,
                    messages=messages,
                    temperature=0.0,
                    max_tokens=max_tokens,
                )
            return resp
        except Exception as e:
            last_exc = e
            if _is_transient_error(e):
                backoff = OPENAI_BASE_BACKOFF * (2 ** attempt) + random.uniform(0, 1.0)
                print(f"   ↻ Transient error (attempt {attempt+1}/{OPENAI_MAX_RETRIES}). Backing off {backoff:.1f}s")
                time.sleep(backoff)
                continue
            else:
                raise
    raise RuntimeError(f"Max retries exceeded: {last_exc}")

def call_llm_align_and_assess(m1_text: str, m2_text: str) -> List[Dict[str, Any]]:
    n1 = rough_sentence_count(m1_text)
    n2 = rough_sentence_count(m2_text)
    min_items = max(n1, n2)
    max_items = n1 + n2

    prompt = build_user_prompt_align_and_assess(m1_text, m2_text, min_items, max_items)

    # Intent 1: json_schema
    try:
        messages = [
            {"role": "system", "content": "You are a precise, factual news-change analyst. Think silently. Output MUST be valid JSON only."},
            {"role": "user", "content": prompt}
        ]
        resp = _chat_create_with_rate_limit(messages, max_tokens=16000, use_schema=True)
        content = (resp.choices[0].message.content or "").strip()
        arr = json.loads(content)
        ok, why = validate_array_range(arr, min_items, max_items)
        if ok:
            return arr
        retry_note = f"Previous output failed validation: {why}"
    except Exception as e:
        retry_note = f"Previous attempt failed: {e}"

    # Intent 2: no pattern
    messages = [
        {"role": "system", "content": "You are a precise, factual news-change analyst. Think silently. Output MUST be ONLY a JSON array, nothing else."},
        {"role": "user", "content": prompt + f"\nIMPORTANT: {retry_note}\n"}
    ]
    resp = _chat_create_with_rate_limit(messages, max_tokens=16000, use_schema=False)
    content = (resp.choices[0].message.content or "")
    arr = extract_first_json_array(content)
    ok, why = validate_array_range(arr, min_items, max_items)
    if not ok:
        print(f"⚠ WARNING: Model output failed validation: {why}")
    return arr

# =========================
# Main
# =========================
def main() -> None:
    for fname in sorted(os.listdir(DATASET_DIR)):
        if not fname.endswith(".json"):
            continue
        in_path = os.path.join(DATASET_DIR, fname)

        try:
            with open(in_path, "r", encoding="utf-8") as f:
                mementos = json.load(f)
        except Exception as e:
            print(f"✗ Skipping {in_path}: {e}")
            continue

        if not isinstance(mementos, list) or len(mementos) < 2:
            continue

        def sort_key(m: Dict[str, Any]) -> str:
            md = m.get("metadata", {})
            return md.get("warc-date", "")
        mementos_sorted = sorted(mementos, key=sort_key)

        first_url = None
        for m in mementos_sorted:
            _, meta_tmp = get_text_and_meta(m)
            first_url = get_url_from_meta(meta_tmp)
            if first_url:
                break
        subfolder_name = slugify(first_url if first_url else fname.replace(".json", ""))
        out_folder = os.path.join(ANALYSIS_DIR, subfolder_name)
        os.makedirs(out_folder, exist_ok=True)

        print(f"Processing {in_path} -> {out_folder}")

        for i in range(len(mementos_sorted) - 1):
            pi = i  
            if pi < START_FROM_PAIR_INDEX:
                continue

            m_old = mementos_sorted[i]
            m_new = mementos_sorted[i + 1]

            old_text, meta_old = get_text_and_meta(m_old)
            new_text, meta_new = get_text_and_meta(m_new)

            if not (isinstance(old_text, str) and isinstance(new_text, str)):
                print(f"  ✗ pair_index {pi}: missing text fields.")
                continue

            if not has_change(old_text, new_text):
                continue
            
            try:
                items = call_llm_align_and_assess(old_text, new_text)
            except Exception as e:
                print(f"  ✗ pair_index {pi}: LLM call failed: {e}")
                continue

            n_sentences_old = sum(
                1 for it in items
                if _present(it.get("sentences", {}).get("M1"))
            )
            n_sentences_new = sum(
                1 for it in items
                if _present(it.get("sentences", {}).get("M2"))
            )

            file_payload = {
                "pair_index": pi,
                "n_sentences_old": n_sentences_old,
                "n_sentences_new": n_sentences_new,
                "metadata_old": meta_old,
                "metadata_new": meta_new,
                "items": items  
            }

            out_file = os.path.join(out_folder, f"{pi:04d}.json")
            with open(out_file, "w", encoding="utf-8") as outf:
                json.dump(file_payload, outf, ensure_ascii=False, indent=2)
            print(f"  ✓ saved analysis -> {out_file}")

        # Pause between files to be cautious with rate
        if FILE_PAUSE_SECONDS > 0:
            time.sleep(FILE_PAUSE_SECONDS)

        print(f"✔ Finished {in_path} -> folder {out_folder}")

    print(f"\n✅ All analyses saved under {ANALYSIS_DIR}/<link_slug>/<pair_index>.json")

if __name__ == "__main__":
    main()
