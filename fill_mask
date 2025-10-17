import os
import io
import json
from typing import Dict, List, Tuple

import streamlit as st
from transformers import pipeline, AutoTokenizer, AutoModelForMaskedLM
import pandas as pd

# --- Basic setup ---
os.environ["TOKENIZERS_PARALLELISM"] = "false"
st.set_page_config(page_title="Fill‑Mask Labeler (DE)", page_icon="🧩", layout="wide")

# --------------------
# Utility functions
# --------------------
MASK_TOKENS = {
    "bert-base-german-cased": "[MASK]",
    "bert-base-multilingual-cased": "[MASK]",
    "xlm-roberta-base": "<mask>",
}

DEFAULT_MODEL = "bert-base-german-cased"
SUPPORTED_MODELS = [
    "bert-base-german-cased",
    "xlm-roberta-base",
    "bert-base-multilingual-cased",
]

@st.cache_resource(show_spinner="Lade Sprachmodell …")
def load_fill_mask(model_name: str):
    """Load a fill-mask pipeline with a safe tokenizer fallback."""
    try:
        tok = AutoTokenizer.from_pretrained(model_name, use_fast=True)
    except Exception:
        tok = AutoTokenizer.from_pretrained(model_name, use_fast=False)
    mdl = AutoModelForMaskedLM.from_pretrained(model_name)
    return pipeline("fill-mask", model=mdl, tokenizer=tok, device=-1)


def parse_allowed_labels(raw: str) -> List[str]:
    return [x.strip() for x in raw.split(",") if x.strip()]


def parse_seed_synonyms(raw: str) -> Dict[str, List[str]]:
    """Parse textarea format: one label per line: label: syn1, syn2"""
    out: Dict[str, List[str]] = {}
    for line in raw.splitlines():
        if not line.strip():
            continue
        if ":" in line:
            label, rest = line.split(":", 1)
            syns = [s.strip() for s in rest.split(",") if s.strip()]
            out[label.strip()] = syns
        else:
            # If no colon, treat entire line as a label with no extra synonyms
            out[line.strip()] = []
    return out


def expand_synonyms_for_label(
    label: str,
    fm,  # fill-mask pipeline
    top_k: int,
    templates: List[str]
) -> List[str]:
    """Generate synonym candidates for a single label using masked-LM prompts."""
    mask_token = MASK_TOKENS.get(fm.model.name_or_path, "[MASK]")

    candidates = []
    seen = set()

    for tpl in templates:
        prompt = tpl.format(label=label, mask=mask_token)
        try:
            preds = fm(prompt, top_k=top_k)
        except Exception:
            continue
        for p in preds:
            token = p.get("token_str", "").strip()
            # basic cleaning / filtering
            if not token:
                continue
            if token.lower() == label.lower():
                continue
            if len(token) < 2:
                continue
            if any(ch.isdigit() for ch in token):
                continue
            # keep alphabetic-ish tokens
            if token not in seen:
                candidates.append(token)
                seen.add(token)

    return candidates


def build_label_lexicon(
    allowed_labels: List[str],
    seed_syns: Dict[str, List[str]],
    fm,
    top_k: int,
) -> Dict[str, List[str]]:
    """Create a dictionary {label: [synonyms…]} using seeds + fill-mask expansion."""
    # German-centric & generic templates
    templates = [
        "{label} wird auch {mask} genannt.",
        "{label} ist ein Begriff für {mask}.",
        "{mask} ist ähnlich wie {label}.",
        "{label} gehört zur Kategorie {mask}.",
    ]

    lex: Dict[str, List[str]] = {}
    for lab in allowed_labels:
        seeds = list({lab, *(seed_syns.get(lab, []))})  # include the label itself as a seed
        # Expand with masked LM
        expanded = expand_synonyms_for_label(lab, fm, top_k=top_k, templates=templates)
        # Merge & dedup (case-insensitive)
        merged = []
        seen_ci = set()
        for w in seeds + expanded:
            key = w.lower()
            if key not in seen_ci:
                merged.append(w)
                seen_ci.add(key)
        lex[lab] = merged
    return lex


def score_label(text: str, lexicon: Dict[str, List[str]]) -> Tuple[str, Dict[str, int]]:
    """Very simple lexical scoring: count hits of synonyms per label; return best label.
    Returns (best_label, counts_per_label)."""
    t = f" {text.lower()} "
    counts: Dict[str, int] = {}
    for label, words in lexicon.items():
        cnt = 0
        for w in words:
            w_norm = f" {w.lower()} "
            if w_norm in t:
                cnt += 1
        counts[label] = cnt
    # choose best label (ties -> first max)
    best = max(counts, key=counts.get) if counts else None
    return best, counts


def classify_dataframe(df: pd.DataFrame, text_col: str, lexicon: Dict[str, List[str]]):
    rows = []
    for _, row in df.iterrows():
        text = str(row.get(text_col, ""))
        best, counts = score_label(text, lexicon)
        rows.append({
            **row.to_dict(),
            "predicted_label": best,
            "label_counts": json.dumps(counts, ensure_ascii=False),
        })
    return pd.DataFrame(rows)


# --------------------
# UI
# --------------------
st.title("🧩 Fill‑Mask basierte Synonym‑Erweiterung & Ticket‑Labeling")

with st.sidebar:
    st.header("⚙️ Einstellungen")
    model_name = st.selectbox("HF‑Modell (Fill‑Mask)", SUPPORTED_MODELS, index=SUPPORTED_MODELS.index(DEFAULT_MODEL))
    top_k = st.slider("Synonym‑Vorschläge pro Prompt (top_k)", 1, 50, 10)
    text_column = st.text_input("Spaltenname für Ticket‑Text (CSV)", value="text")

    st.markdown("---")
    st.caption("Erlaube Labels (Komma‑getrennt)")
    allowed_raw = st.text_input("allowed_labels", value="Rechnung, Vertrag, Support, Termin, Lieferung")

    st.caption("Optionale Seeds: je Zeile `Label: syn1, syn2`")
    seeds_raw = st.text_area(
        "Seed‑Synonyme",
        value=(
            "Rechnung: Rechnung, Rechnungstellung, Rechnungsbetrag\n"
            "Vertrag: Vertrag, Kontrakt, Vereinbarung\n"
            "Support: Support, Hilfe, Kundendienst\n"
        ), height=120
    )

# Load model once
fm = load_fill_mask(model_name)

st.subheader("1) Synonym‑Lexikon erzeugen")
col1, col2 = st.columns([1, 1])
allowed_labels = parse_allowed_labels(allowed_raw)
seed_syns = parse_seed_synonyms(seeds_raw)

with col1:
    if st.button("Synonyme erweitern", type="primary"):
        lexicon = build_label_lexicon(allowed_labels, seed_syns, fm, top_k)
        st.session_state["lexicon"] = lexicon

with col2:
    if st.button("Nur Seeds verwenden (ohne Erweiterung)"):
        lex = {}
        for lab in allowed_labels:
            lex[lab] = list({lab, *(seed_syns.get(lab, []))})
        st.session_state["lexicon"] = lex

lexicon = st.session_state.get("lexicon")
if lexicon:
    st.success("Lexikon erstellt.")
    with st.expander("Lexikon anzeigen"):
        st.json(lexicon, expanded=False)
else:
    st.info("Erzeuge zuerst ein Lexikon (links Button klicken).")

st.subheader("2) Tickets klassifizieren")

data_mode = st.radio("Eingabeart", ["Einzeltext", "CSV hochladen"], horizontal=True)

if data_mode == "Einzeltext":
    sample_text = st.text_area("Ticket‑Text", value="Die Lieferung ist verspätet, bitte um Support.", height=140)
    if st.button("Einzeltext klassifizieren"):
        if not lexicon:
            st.error("Bitte zuerst ein Lexikon erzeugen.")
        else:
            best, counts = score_label(sample_text, lexicon)
            st.write({"predicted_label": best, "label_counts": counts})

else:
    up = st.file_uploader("CSV hochladen", type=["csv"]) 
    if up is not None:
        try:
            df = pd.read_csv(up)
        except Exception:
            up.seek(0)
            df = pd.read_csv(io.BytesIO(up.read()), sep=";")
        st.write("Erkannte Spalten:", list(df.columns))
        if text_column not in df.columns:
            st.error(f"Spalte '{text_column}' nicht gefunden. Passe den Spaltennamen in den Einstellungen an.")
        else:
            if st.button("CSV klassifizieren", type="primary"):
                if not lexicon:
                    st.error("Bitte zuerst ein Lexikon erzeugen.")
                else:
                    out = classify_dataframe(df, text_column, lexicon)
                    st.dataframe(out.head(100))
                    # Download
                    csv = out.to_csv(index=False).encode("utf-8")
                    st.download_button("Ergebnis als CSV herunterladen", data=csv, file_name="labeled_tickets.csv", mime="text/csv")

st.markdown("---")
st.caption("Hinweis: Dieses Demo verwendet einfache lexikalische Trefferzählung pro Label. Für präzisere Klassifikation kann man zusätzlich eine Zero‑Shot‑Pipeline verwenden und Synonym‑Treffer als Features kombinieren.")
