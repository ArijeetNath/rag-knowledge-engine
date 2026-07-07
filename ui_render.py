"""Display helpers for the KIS Streamlit UI, split out so they can be unit-tested
without booting the whole app (importing app.py runs the entire Streamlit page).
"""
from __future__ import annotations

import re

import streamlit as st

import kis


# A "figure-label" token: the short symbolic tokens that PDF diagrams and tables
# are built from — [CLS], [SEP], E[CLS], E1, T1', EN, TN, C, Tok, B-PER, and
# numbers/table cells (2048, 4.92, 0.1, 100K, ×106). In real prose these appear
# only in isolation (an acronym, a lone number); in a figure or table they come
# in long dense strings. We key on the *run*, not the token.
_FIG_LABEL = re.compile(
    r"[ETC]?\[[A-Z/]{1,6}\]|[A-Z]{1,3}\d*['’]?|[A-Z]-[A-Z]+|Tok|[×x]?\d+(?:[.,]\d+)*[KMB%]?")
_FIG_RUN = 4  # this many label/cell tokens in a row = a diagram or table, not a sentence


def _strip_figure_labels(words):
    """Drop maximal runs of >= _FIG_RUN consecutive figure-label tokens. A real
    sentence never stacks four of these; a figure is almost nothing but."""
    out, i, n = [], 0, len(words)
    while i < n:
        j = i
        while j < n and _FIG_LABEL.fullmatch(words[j].strip(".,")):
            j += 1
        if j - i >= _FIG_RUN:      # dense label soup -> drop the whole run
            i = j
        else:                      # keep isolated tokens (acronyms, numbers)
            out.append(words[i])
            i += 1
    return out


def reflow(text):
    """Clean a raw document passage for display. PDF extraction hard-wraps
    mid-sentence, figure labels come out one word per line littered with
    tokenizer markers (<EOS>/<pad>) and the sentence repeated once per attention
    head, with a space before every punctuation mark; diagram figures dump long
    strings of symbolic labels (E[CLS] E1 T1 [SEP] Tok 1 ...). So: strip the
    markers, join intra-paragraph line breaks into spaces, drop diagram-label
    runs, drop repeated identical paragraphs, and remove the space before
    punctuation — leaving normal prose. Blank-line paragraph breaks are kept."""
    # Only the known model special tokens — not arbitrary <tags>, which are real
    # content in HTML/code source passages.
    text = re.sub(r"<(?:EOS|BOS|PAD|UNK|SEP|CLS|MASK|s|/s|unk|pad)>", " ",
                  text, flags=re.IGNORECASE)
    paras = [" ".join(_strip_figure_labels(p.split()))
             for p in re.split(r"\n\s*\n", text)]
    paras = list(dict.fromkeys(p for p in paras if p))  # order-preserving dedup
    out = "\n\n".join(paras)
    return re.sub(r"\s+([.,;:!?])", r"\1", out)          # un-tokenize punctuation


def similarity_badge(sim):
    """Colored confidence label for a retrieval hit. Uses :color[...] colored
    text (supported broadly across Streamlit versions), not newer badge syntax."""
    if sim is None:
        return ":gray[● keyword match]"
    if sim >= 0.6:
        return f":green[● similarity {sim:.2f}]"
    if sim >= kis.MIN_SIMILARITY:
        return f":orange[● similarity {sim:.2f}]"
    return f":red[● similarity {sim:.2f}]"


def render_sources(hits):
    """Expandable, cited source passages under an answer."""
    st.markdown("**Sources**")
    for i, h in enumerate(hits, 1):
        loc = f" · {h['loc']}" if h["loc"] else ""
        header = f"[{i}] {h['source']}{loc}  —  {similarity_badge(h['similarity'])}"
        with st.expander(header):
            # st.text (not st.write/markdown): raw document text contains *, _,
            # [n], # etc. that Markdown would interpret — scrambling the passage.
            # Plain text renders it verbatim; reflow() undoes PDF line-wrapping.
            st.text(reflow(h["text"]))
