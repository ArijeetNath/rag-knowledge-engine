"""Regression test for source-passage rendering (run: python test_sources.py).

Bug: raw document passages contain Markdown-active characters (*, _, [n], #).
When rendered with st.write/st.markdown, Markdown reinterprets them — emphasis
runs, dropped chars, collapsed newlines — so the Sources panel looked jumbled.
Fix: render_sources() emits the passage as plain text (st.text), reflowed so
PDF line-wrapping (and one-word-per-line figure labels) reads as normal text.
"""
from streamlit.testing.v1 import AppTest

# A passage packed with the characters that Markdown would mangle, plus the
# hard newlines that document extraction produces. After reflow() the soft
# line-wraps become spaces, but every Markdown-active character survives.
RAW = ("Vaswani* and Shazeer* propose *attention* [1] with weight_1 and\n"
       "weight_2 in section #3; see refs [12] [15] and email a_b@x.com\n"
       "line continues here mid-sentence")
REFLOWED = ("Vaswani* and Shazeer* propose *attention* [1] with weight_1 and "
            "weight_2 in section #3; see refs [12] [15] and email a_b@x.com "
            "line continues here mid-sentence")


def _script():
    from ui_render import render_sources
    hits = [{"source": "paper.txt", "loc": "p.1", "similarity": 0.83,
             "text": ("Vaswani* and Shazeer* propose *attention* [1] with weight_1 and\n"
                      "weight_2 in section #3; see refs [12] [15] and email a_b@x.com\n"
                      "line continues here mid-sentence")}]
    render_sources(hits)


def test_passage_rendered_verbatim_as_plaintext():
    at = AppTest.from_function(_script).run()
    assert not at.exception, at.exception
    # The passage must land in a plain-text element, reflowed but with every
    # Markdown-active character intact...
    texts = [t.value for t in at.text]
    assert REFLOWED in texts, f"passage not rendered as reflowed st.text; got {texts!r}"
    # ...and must NOT have been fed through Markdown (only our "Sources" heading is).
    markdowns = [m.value for m in at.markdown]
    assert all("attention" not in m for m in markdowns), \
        f"passage leaked into a Markdown element (would be reinterpreted): {markdowns!r}"


if __name__ == "__main__":
    test_passage_rendered_verbatim_as_plaintext()
    print("test_sources: OK — passages render verbatim as plain text")
