from __future__ import annotations

import time
from pathlib import Path

import streamlit as st

import kis
from ui_render import render_sources, reflow

st.set_page_config(page_title="KIS · Local Document Intelligence",
                   page_icon="📚", layout="wide")


@st.cache_data(show_spinner=False, ttl=10)
def cached_stats():
    return kis.index_stats()


def refresh_stats():
    cached_stats.clear()


with st.sidebar:
    st.title("📚 KIS")
    st.caption("Local Knowledge Intelligence System")

    ok, detail = kis.ollama_status()
    (st.success if ok else st.warning)(detail, icon="✅" if ok else "⚠️")

    st.divider()
    st.subheader("Index")
    stats = cached_stats()
    if stats is None:
        st.info("No documents indexed yet. Use **Manage index** below.")
        total, counts = 0, {}
    else:
        total, counts = stats
        c1, c2 = st.columns(2)
        c1.metric("Files", len(counts))
        c2.metric("Chunks", total)
        for src in sorted(counts):
            st.caption(f"📄 {src}  ·  {counts[src]} chunks")

    st.divider()
    with st.expander("⚙️ Configuration"):
        st.write(f"**Embedding model**\n\n`{kis.EMBED_MODEL}`")
        st.write(f"**LLM (via Ollama)**\n\n`{kis.LLM_MODEL}`")
        st.write(f"**Chunk size**: {kis.CHUNK_CHARS} chars · "
                 f"overlap {kis.CHUNK_OVERLAP}")
        st.write(f"**Top-K**: {kis.TOP_K} · "
                 f"**min similarity**: {kis.MIN_SIMILARITY}")
        st.write(f"**Index dir**: `{kis.DB_DIR}`")

    with st.expander("📥 Manage index"):
        folder = st.text_input("Folder to ingest", value="sample_docs")
        if st.button("Ingest folder", use_container_width=True):
            try:
                root, files = kis.find_ingestable(folder)
            except ValueError as e:
                st.error(str(e))
            else:
                bar = st.progress(0.0, text=f"Found {len(files)} files…")

                def on_progress(n, tot, source, n_chunks):
                    label = (f"[{n}/{tot}] +{n_chunks} chunks · {source}"
                             if n_chunks else f"[{n}/{tot}] unchanged · {source}")
                    bar.progress(n / tot, text=label)

                with st.spinner("Parsing, embedding and indexing…"):
                    result = kis.ingest_files(files, root, progress=on_progress)
                bar.empty()
                if result["empty"]:
                    st.error("Nothing could be extracted from those files.")
                else:
                    st.success(f"Done · +{result['added']} chunks "
                               f"({result['updated']} updated, "
                               f"{result['skipped']} unchanged)")
                    refresh_stats()
                    st.rerun()

        st.divider()
        st.caption("Maintenance")
        if st.button("Prune missing files", use_container_width=True):
            missing = kis.find_missing(folder)
            if not missing:
                st.info("Nothing to prune — every indexed file still exists.")
            else:
                chunks = sum(n for _, n in missing)
                kis.delete_sources([s for s, _ in missing])
                st.success(f"Pruned {chunks} chunks from {len(missing)} "
                           f"missing file(s).")
                refresh_stats()
                st.rerun()
        sure = st.checkbox("Confirm full reset")
        if st.button("Reset index", use_container_width=True, disabled=not sure):
            if kis.reset_index():
                st.success("Index reset — all chunks removed.")
            else:
                st.info("No index to reset.")
            refresh_stats()
            st.rerun()


st.title("Local Knowledge Intelligence System")
st.markdown(
    "Ask questions about your own documents and get answers grounded **only** in "
    "their contents, with citations. Runs fully offline — a private, zero-cost "
    "alternative to cloud RAG assistants."
)

with st.expander("ℹ️  About this project — architecture & design decisions"):
    st.markdown(
        """
**What it does.** Point it at a folder (PDF, DOCX, PPTX, XLSX, email, code, text).
It parses, chunks, embeds and indexes every file, then answers natural-language
questions using retrieval-augmented generation (RAG) — citing the exact passages
it used.

**Pipeline**
"""
    )
    st.markdown(
        """\
```
Documents → Parse → Chunk (overlap) → Embed (fastembed / BGE)
                                          │
                                   LanceDB (on-disk)
                              vector index + full-text (BM25)
                                          │
Question → Hybrid retrieve → Confidence gate → LLM (Ollama) → Cited answer
```"""
    )
    col_a, col_b = st.columns(2)
    with col_a:
        st.markdown(
            """
**Engineering decisions**
- **Hybrid retrieval** — semantic vectors *plus* keyword/BM25, so exact names,
  IDs and acronyms aren't missed by embeddings alone.
- **Confidence gate** — if the best match is below a similarity threshold, it
  answers *"I couldn't find this"* instead of hallucinating.
- **Idempotent ingest** — files are skipped unless their mtime changed; edited
  files are re-indexed in place.
"""
        )
    with col_b:
        st.markdown(
            """
**Runs on modest hardware**
- Tuned for **8 GB RAM, CPU-only** — small chunks and top-K keep prompts inside
  a 1.5B model's context window.
- **100% local & private** — no API keys, no data leaves the machine.
- **Shared core** — this UI and the CLI call the exact same functions in
  `kis.py`; no logic is duplicated.
"""
        )
    st.info("Tech: Python · Streamlit · LanceDB · fastembed (BAAI/BGE) · "
            "Ollama · PyMuPDF / python-docx / python-pptx / openpyxl", icon="🛠️")

if cached_stats() is None:
    st.warning("No documents are indexed yet. Open **Manage index** in the "
               "sidebar and ingest a folder (try the bundled `sample_docs`).",
               icon="👈")
    st.stop()

ask_tab, about_tab, where_tab = st.tabs(
    ["💬  Ask", "🧭  Explore a topic", "🔎  Find a term"])

with ask_tab:
    st.subheader("Ask a question")
    st.caption("Answers are generated only from your indexed documents, with sources.")
    examples = "  ·  ".join(["What is this document about?",
                             "Summarize the key points", "Who is mentioned?"])
    st.caption(f"Try: {examples}")

    question = st.text_input("Your question", key="ask_q",
                             placeholder="e.g. What are the main findings?")
    go = st.button("Ask", type="primary", key="ask_go")

    if go and question.strip():
        if not kis.ollama_status()[0]:
            st.error("The LLM isn't available — see the status banner in the "
                     "sidebar. Retrieval still works via the other tabs.")
        else:
            with st.spinner("Retrieving relevant passages…"):
                table, hits = kis.retrieve(question)
            if not hits or hits[0]["similarity"] is None \
                    or hits[0]["similarity"] < kis.MIN_SIMILARITY:
                st.warning("I couldn't find this in your documents.")
                if hits and hits[0]["similarity"] is not None:
                    st.caption(f"Closest match ~{hits[0]['similarity']:.2f}, "
                               f"below threshold {kis.MIN_SIMILARITY}.")
            else:
                t0 = time.time()
                with st.spinner(f"Generating grounded answer with "
                                f"{kis.LLM_MODEL}…"):
                    ctx = kis.format_context(hits)
                    answer = kis.ollama_chat(
                        kis.SYSTEM_PROMPT,
                        kis.build_user_prompt(ctx, question))
                st.markdown("### Answer")
                st.write(answer)
                st.caption(f"Grounded in {len(hits)} passages · "
                           f"{time.time() - t0:.1f}s")
                st.divider()
                render_sources(hits)

with about_tab:
    st.subheader("Explore a topic across documents")
    st.caption("A short synthesis of what your documents say about a topic.")
    topic = st.text_input("Topic", key="about_t",
                          placeholder="e.g. machine learning, budget, timeline")
    if st.button("Summarize", type="primary", key="about_go") and topic.strip():
        if not kis.ollama_status()[0]:
            st.error("The LLM isn't available — see the sidebar status banner.")
        else:
            with st.spinner("Retrieving…"):
                table, hits = kis.retrieve(topic, k=5)
            if not hits or hits[0]["similarity"] is None \
                    or hits[0]["similarity"] < kis.MIN_SIMILARITY:
                st.warning(f"I couldn't find '{topic}' in your documents.")
            else:
                hits = hits[:6]
                with st.spinner(f"Synthesizing with {kis.LLM_MODEL}…"):
                    summary = kis.ollama_chat(
                        kis.INSIGHTS_PROMPT,
                        f"Topic: {topic}\n\nContext:\n{kis.format_context(hits)}\n\n"
                        "Summarize what the context says about the topic, "
                        "citing passages like [1].\nSummary:")
                st.markdown("### Summary")
                st.write(summary)
                docs = list(dict.fromkeys(h["source"] for h in hits))
                st.caption(f"Across {len(docs)} document(s): {', '.join(docs)}")
                st.divider()
                render_sources(hits)

with where_tab:
    st.subheader("Find where a term appears")
    st.caption("Exact keyword (full-text) search — no LLM, instant.")
    term = st.text_input("Term or phrase", key="where_t",
                         placeholder="e.g. a name, ID, or acronym")
    if st.button("Search", type="primary", key="where_go") and term.strip():
        table = kis.open_table()
        try:
            hits = table.search(term, query_type="fts").limit(20).to_list()
        except Exception:
            hits = []
            st.warning("No keyword index. Re-ingest to build it.")
        if term.strip() and not hits:
            st.info(f"'{term}' not found in your documents.")
        for h in hits:
            loc = f" · {h['loc']}" if h["loc"] else ""
            snippet = reflow(h["text"])[:240]
            st.markdown(f"**{h['source']}{loc}**")
            st.caption(f"…{snippet}…")

st.divider()
st.caption("Built with Streamlit · fully local RAG · shares its core with the "
           "`kis.py` CLI. Educational / portfolio project.")
