from __future__ import annotations

import re

import streamlit as st

import kis


_FIG_LABEL = re.compile(
    r"[ETC]?\[[A-Z/]{1,6}\]|[A-Z]{1,3}\d*['’]?|[A-Z]-[A-Z]+|Tok|[×x]?\d+(?:[.,]\d+)*[KMB%]?")
_FIG_RUN = 4


def _strip_figure_labels(words):
    out, i, n = [], 0, len(words)
    while i < n:
        j = i
        while j < n and _FIG_LABEL.fullmatch(words[j].strip(".,")):
            j += 1
        if j - i >= _FIG_RUN:
            i = j
        else:
            out.append(words[i])
            i += 1
    return out


def reflow(text):
    text = re.sub(r"<(?:EOS|BOS|PAD|UNK|SEP|CLS|MASK|s|/s|unk|pad)>", " ",
                  text, flags=re.IGNORECASE)
    paras = [" ".join(_strip_figure_labels(p.split()))
             for p in re.split(r"\n\s*\n", text)]
    paras = list(dict.fromkeys(p for p in paras if p))
    out = "\n\n".join(paras)
    return re.sub(r"\s+([.,;:!?])", r"\1", out)


def similarity_badge(sim):
    if sim is None:
        return ":gray[● keyword match]"
    if sim >= 0.6:
        return f":green[● similarity {sim:.2f}]"
    if sim >= kis.MIN_SIMILARITY:
        return f":orange[● similarity {sim:.2f}]"
    return f":red[● similarity {sim:.2f}]"


def render_sources(hits):
    st.markdown("**Sources**")
    for i, h in enumerate(hits, 1):
        loc = f" · {h['loc']}" if h["loc"] else ""
        header = f"[{i}] {h['source']}{loc}  —  {similarity_badge(h['similarity'])}"
        with st.expander(header):
            st.text(reflow(h["text"]))
