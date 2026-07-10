# KIS — Local Knowledge Intelligence System

Offline "chat with your documents". Ingest local files, ask questions in plain
English, get answers grounded **only** in your data with citations. Nothing
leaves your machine; no API keys; no cost.

Built for modest hardware (tested on an i5 / 8GB RAM / no GPU, Windows).

## What it does
- Ingests **PDF, Word, PowerPoint, Excel, text/Markdown/code, and email (.eml/.mbox)**.
- Finds the passages that answer your question (semantic search) and shows
  **where** they came from (file + page/slide/sheet).
- Answers using a small **local** LLM, and **refuses to answer** ("I couldn't
  find this in your documents") when your files don't contain the answer —
  it does not fall back on general knowledge.
- Usable two ways: a **command-line tool** (`kis.py`) and a **local web UI**
  (`app.py`, Streamlit). Both share the same core engine — no duplicated logic.

## Startup
1. **Install Python deps:**
   ```
   pip install -r requirements.txt
   ```
2. **Install Ollama** (the local LLM runtime): https://ollama.com
3. **Pull the model** (small, fits 8GB RAM):
   ```
   ollama pull qwen2.5:1.5b
   ```
   After this, everything runs fully offline. The first `ingest`/`ask` also
   downloads the ~130MB embedding model once.

## Web UI
For a point-and-click experience, launch the Streamlit app:
```
streamlit run app.py
```
It opens in your browser (default http://localhost:8501) and provides:
- **Ask**, **Explore a topic**, and **Find a term** tabs (the `ask`/`about`/`where`
  commands, with cited sources and per-passage similarity scores).
- A sidebar showing **Ollama/model health**, **index contents**, the active
  config, and a folder **ingest** control with a live progress bar.
- A built-in project explainer (architecture, design decisions, tech stack).

Everything still runs 100% locally. `streamlit` is included in
`requirements.txt`; the CLI does not need it.

## CLI usage
```
python kis.py ingest <folder>       # index every supported file under a folder
python kis.py ask "your question"   # grounded answer + sources (hybrid search)
python kis.py about "a topic"       # cross-document summary of a topic
python kis.py where "term"          # find where a term/phrase appears (keyword)
python kis.py status                # what's in the index
python kis.py prune <folder>        # remove chunks for files deleted from <folder>
python kis.py reset                 # wipe the whole index (asks to confirm)
python kis.py selftest              # offline check of the chunker
```

**Clearing old data:** `ingest` is additive, so files from earlier runs stay
indexed. Use `prune <folder>` to drop chunks for files you've since deleted, or
`reset` to wipe the index entirely (both are also buttons in the web UI's
sidebar, under *Manage index → Maintenance*). Add `--yes` to skip the prompt.

`ask` uses **hybrid retrieval**: semantic (vector) search plus keyword recall,
so exact names/IDs/acronyms aren't missed. If a question is clearly ambiguous
the model may ask a clarifying question — best-effort, and more reliable on a
larger model (`KIS_MODEL`).

**Re-running `ingest` is safe and incremental**: unchanged files are skipped,
edited files are re-indexed in place (no duplicates), so you can point it at the
same folder again after adding or editing documents.

Example:
```
python kis.py ingest sample_docs
python kis.py ask "What is the approved budget for Aurora and who leads it?"
```

## Tuning knobs (environment variables)
| Var | Default | Meaning |
|-----|---------|---------|
| `KIS_MODEL` | `qwen2.5:1.5b` | Ollama model. Upgrade to `qwen2.5:3b`/`llama3.2:3b` with more RAM. |
| `KIS_MINSIM` | `0.40` | Similarity gate. Raise if answers wander; lower if it wrongly says "not found". |
| `KIS_TOPK` | `4` | How many passages to feed the model. |
| `KIS_CHUNK` / `KIS_OVERLAP` | `1200` / `150` | Chunk size (chars) and overlap. |
| `KIS_DB` | `.kis_db` | Where the on-disk index lives. |

## Hardware notes (8GB RAM)
- 3B models need ~7.9GB and won't load once the OS takes its share — hence the
  1.5B default. It fits with headroom.
- Ollama's default context window (~2048 tokens) is the ceiling here; passing a
  larger `num_ctx` crashes the runner on low RAM, so KIS keeps prompts small
  via chunk size + top-K instead.

## Not yet built (planned)
- OCR for scanned images (excluded — too heavy for 8GB)
