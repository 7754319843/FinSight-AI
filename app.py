"""FinSight AI — polished Streamlit frontend for the existing FinSightRAG backend.

Replace the original app.py with this file; keep the existing src/ folder.
"""
import html
import os
import tempfile
from pathlib import Path

import streamlit as st
from dotenv import load_dotenv
from src.rag_pipeline import FinSightRAG

load_dotenv()
st.set_page_config(page_title="FinSight AI | Financial Intelligence", page_icon="📈", layout="wide", initial_sidebar_state="expanded")

st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=DM+Sans:wght@400;500;600;700&family=Manrope:wght@400;500;600;700;800&display=swap');
:root { color-scheme: light; }
html, body, [class*="css"], [data-testid="stAppViewContainer"] {font-family:'DM Sans',sans-serif;}
.stApp {background:linear-gradient(140deg,#f6f9ff 0%,#fbfcff 46%,#f8fafc 100%); color:#24344b;}
[data-testid="stSidebar"] {background:#ffffff; border-right:1px solid #e9eef6;}
[data-testid="stSidebar"] [data-testid="stMarkdownContainer"] p {color:#53627b;}
.block-container {max-width:1240px; padding-top:2rem; padding-bottom:3rem;}
h1,h2,h3,h4 {font-family:'Manrope',sans-serif !important;letter-spacing:-.035em;color:#192c48;}
h2 {font-size:1.3rem !important;} h3 {font-size:1.1rem !important;}
[data-testid="stMetric"] {background:#fff;border:1px solid #e8eef8;border-radius:18px;padding:18px 20px;box-shadow:0 8px 28px rgba(39,75,129,.045);}
[data-testid="stMetricLabel"] {color:#71829b;}
[data-testid="stMetricValue"] {color:#1c365b;font-family:'Manrope',sans-serif;font-size:1.6rem;}
.stButton > button[kind="primary"] {background:#426bcb;color:#fff;border:1px solid #426bcb;}
.stButton > button {border-radius:11px;min-height:42px;font-weight:600;border-color:#dce6f3;}
.stButton > button:hover {border-color:#6687d6;color:#335bb5;}
[data-baseweb="input"] {border-radius:12px;}
[data-testid="stFileUploader"] {border-radius:14px;}
[data-baseweb="tab-list"] {gap:10px;border-bottom:1px solid #e4eaf4;}
button[data-baseweb="tab"] {font-weight:600;border-radius:9px 9px 0 0;}
[data-testid="stExpander"] {border:1px solid #e4ebf6;border-radius:12px;background:#fff;}
.fs-hero {background:linear-gradient(110deg,#fff 12%,#f0f6ff 100%);border:1px solid #e7eef9;border-radius:24px;padding:27px 30px;margin-bottom:20px;box-shadow:0 10px 40px rgba(52,85,139,.05);}
.fs-brand {font-family:'Manrope',sans-serif;font-size:1.9rem;font-weight:800;color:#1b3150;letter-spacing:-.055em;line-height:1.15;}
.fs-eyebrow {color:#5577b7;font-weight:800;letter-spacing:.12em;font-size:.69rem;margin-bottom:10px;}
.fs-sub {color:#71819a;font-size:.96rem;margin-top:8px;line-height:1.6;}
.fs-pill {display:inline-block;border-radius:100px;padding:6px 12px;font-size:.76rem;font-weight:700;background:#eaf8f1;color:#198463;margin-top:14px;}
.fs-pill-wait {background:#fff5e9;color:#a36c20;}
.fs-section {font-family:'Manrope',sans-serif;font-weight:800;color:#223a5a;font-size:1.08rem;margin:25px 0 12px;letter-spacing:-.03em;}
.fs-report {display:flex;align-items:center;gap:12px;background:white;border:1px solid #e7edf7;border-radius:14px;padding:13px 16px;margin:8px 0;box-shadow:0 5px 18px rgba(40,70,120,.035);}
.fs-file-icon {display:flex;align-items:center;justify-content:center;background:#edf3ff;color:#4670c5;border-radius:10px;width:39px;height:39px;font-size:1.25rem;flex-shrink:0;}
.fs-report-title {font-weight:700;color:#2b405d;overflow-wrap:anywhere;}
.fs-report-note {font-size:.78rem;color:#8593a8;}
.fs-check {margin-left:auto;color:#20a477;font-size:1.15rem;}
.fs-answer {background:#fff;border:1px solid #e7eef8;border-left:4px solid #5a81d4;border-radius:14px;padding:18px 20px;margin:12px 0;}
.fs-small {font-size:.83rem;color:#8090a7;}
.fs-sidebrand {font-family:'Manrope',sans-serif;color:#1e3657;font-weight:800;font-size:1.25rem;letter-spacing:-.04em;}
.fs-sidehint {font-size:.83rem;color:#8090a7;line-height:1.55;margin:7px 0 18px;}
</style>
""", unsafe_allow_html=True)

if "rag" not in st.session_state:
    st.session_state.rag = FinSightRAG()
if "ingested" not in st.session_state:
    st.session_state.ingested = False
if "stats" not in st.session_state:
    st.session_state.stats = None
if "last_answer" not in st.session_state:
    st.session_state.last_answer = None
if "last_summary" not in st.session_state:
    st.session_state.last_summary = None


def temp_paths_for(files):
    temp_dir = tempfile.TemporaryDirectory()
    paths = []
    for file in files:
        target = Path(temp_dir.name) / Path(file.name).name
        target.write_bytes(file.getvalue())
        paths.append(str(target))
    return paths, temp_dir


def show_sources(sources, heading="Source evidence"):
    st.markdown(f"#### {heading}")
    if not sources:
        st.caption("No source passages were returned.")
        return
    for i, source in enumerate(sources, start=1):
        name = str(source.get("source", "Unknown document"))
        page = str(source.get("page", "N/A"))
        section = str(source.get("section", ""))
        with st.expander(f"{i}. {name} · Page {page}" + (f" · {section}" if section else "")):
            st.write(source.get("text", "No extracted passage available."))


with st.sidebar:
    st.markdown('<div class="fs-sidebrand">◈ FinSight AI</div><div class="fs-sidehint">Your financial research workspace</div>', unsafe_allow_html=True)
    st.markdown("### 📁 Document manager")
    uploaded_files = st.file_uploader("Upload PDF or TXT reports", type=["pdf", "txt"], accept_multiple_files=True)
    build_temp_button = st.button("⚡ Build temporary index", use_container_width=True)
    add_update_button = st.button("＋ Add / update saved index", type="primary", use_container_width=True)
    load_button = st.button("↻ Load saved index", use_container_width=True)
    with st.expander("Advanced database controls"):
        clear_button = st.button("Clear saved database", use_container_width=True)
        clear_memory_button = st.button("Clear chat memory", use_container_width=True)
    st.divider()
    st.markdown("### ⚙️ Model settings")
    provider = os.getenv("LLM_PROVIDER", "offline")
    ollama_model = os.getenv("OLLAMA_MODEL", "not set")
    st.caption(f"Provider: **{provider}**")
    if provider.lower() == "ollama":
        st.caption(f"Local model: **{ollama_model}**")
    st.info("Temporary indexes are for testing. Save an index to reuse it after restarting.", icon="💡")

if build_temp_button or add_update_button:
    if not uploaded_files:
        st.warning("Upload at least one PDF or TXT document first.")
    else:
        paths, temp_dir = temp_paths_for(uploaded_files)
        try:
            with st.spinner("Reading reports and building the search index..."):
                if build_temp_button:
                    stats = st.session_state.rag.ingest_documents(paths, save_index=False)
                else:
                    stats = st.session_state.rag.add_documents(paths)
            st.session_state.ingested = True
            st.session_state.stats = stats
            st.session_state.last_answer = None
            st.session_state.last_summary = None
            st.success("Temporary knowledge base created." if build_temp_button else "Saved knowledge base updated successfully.")
            with st.expander("View processing details"):
                st.json(stats)
        except Exception as exc:
            st.error(f"Could not process documents: {exc}")
        finally:
            temp_dir.cleanup()

if load_button:
    try:
        with st.spinner("Loading the saved knowledge base..."):
            stats = st.session_state.rag.load_existing_index()
        st.session_state.ingested = True
        st.session_state.stats = stats
        st.session_state.last_answer = None
        st.session_state.last_summary = None
        st.success("Saved knowledge base loaded.")
    except Exception as exc:
        st.error(f"Failed to load the saved index: {exc}")

if clear_button:
    try:
        st.session_state.rag.clear_database()
        st.session_state.ingested = False
        st.session_state.stats = None
        st.session_state.last_answer = None
        st.session_state.last_summary = None
        st.success("Saved database cleared.")
    except Exception as exc:
        st.error(f"Failed to clear the database: {exc}")

if clear_memory_button:
    try:
        st.session_state.rag.clear_memory()
        st.session_state.last_answer = None
        st.success("Chat memory cleared.")
    except Exception as exc:
        st.error(f"Failed to clear chat memory: {exc}")

status = "Knowledge base ready" if st.session_state.ingested else "Load or build an index to begin"
status_class = "fs-pill" if st.session_state.ingested else "fs-pill fs-pill-wait"
st.markdown(f"""<div class="fs-hero"><div class="fs-eyebrow">FINANCIAL INTELLIGENCE WORKSPACE</div>
<div class="fs-brand">◈ FinSight AI</div><div class="fs-sub">Explore financial reports, uncover insights and ask questions backed by document evidence.</div>
<span class="{status_class}">{html.escape(status)}</span></div>""", unsafe_allow_html=True)

stats = st.session_state.stats or {}
docs = stats.get("documents_processed", "—")
chunks = stats.get("chunks_created", "—")
a, b, c = st.columns(3)
a.metric("📄  Reports indexed", docs)
b.metric("🧩  Search chunks", chunks)
c.metric("🔎  Retrieval engine", "FAISS + keywords")

sources = stats.get("sources", [])
if sources or uploaded_files:
    st.markdown('<div class="fs-section">Your financial reports</div>', unsafe_allow_html=True)
    shown = sources if sources else [f.name for f in uploaded_files]
    for source in shown:
        filename = html.escape(str(source))
        st.markdown(f'<div class="fs-report"><div class="fs-file-icon">▤</div><div><div class="fs-report-title">{filename}</div><div class="fs-report-note">Financial document</div></div><div class="fs-check">✓</div></div>', unsafe_allow_html=True)

st.markdown('<div class="fs-section">Research assistant</div>', unsafe_allow_html=True)
tab1, tab2, tab3, tab4 = st.tabs(["✦ Ask FinSight", "▤ Summarize", "💡 Example questions", "◷ Chat history"])

with tab1:
    st.markdown("#### Ask a financial question")
    st.caption("Ask about a company's revenue, net income, risks, trends or comparisons across uploaded reports.")
    with st.form("question_form", clear_on_submit=False):
        question = st.text_area("Your question", placeholder="e.g. Compare Netflix and Microsoft revenue in 2024.", height=95)
        submit_question = st.form_submit_button("✦ Analyze financial reports", type="primary", use_container_width=True)
    if submit_question:
        if not st.session_state.ingested:
            st.warning("Build or load a knowledge base first.")
        elif not question.strip():
            st.warning("Enter a question first.")
        else:
            try:
                with st.spinner("Retrieving evidence and preparing your answer..."):
                    st.session_state.last_answer = (question, st.session_state.rag.ask(question))
            except Exception as exc:
                st.error(f"Could not answer the question: {exc}")
    if st.session_state.last_answer:
        last_question, result = st.session_state.last_answer
        st.markdown("#### ✨ AI analysis")
        st.caption(last_question)
        st.markdown(result.get("answer", "No answer returned."))
        top_sources = []
        for source in result.get("sources", [])[:5]:
            label = f"{source.get('source','Document')} · p. {source.get('page','?')}"
            if label not in top_sources:
                top_sources.append(label)
        if top_sources:
            st.info("Sources: " + "  •  ".join(top_sources), icon="📚")
        metrics = result.get("metrics", {})
        st.markdown("#### Financial mentions found")
        m1, m2, m3, m4 = st.columns(4)
        for col, title, key in [(m1,"Money values","money_values"),(m2,"Percentages","percentages"),(m3,"Years","years"),(m4,"Quarters","quarters")]:
            col.metric(title, len(metrics.get(key, [])))
        with st.expander("View extracted metric details"):
            st.json(metrics)
        show_sources(result.get("sources", []), "Retrieved evidence & page citations")

with tab2:
    st.markdown("#### Document summary")
    st.caption("Generate a concise summary from your indexed financial reports.")
    if st.button("✦ Generate summary", type="primary", use_container_width=True):
        if not st.session_state.ingested:
            st.warning("Build or load a knowledge base first.")
        else:
            try:
                with st.spinner("Preparing document summary..."):
                    st.session_state.last_summary = st.session_state.rag.summarize()
            except Exception as exc:
                st.error(f"Could not generate summary: {exc}")
    if st.session_state.last_summary:
        st.markdown("#### Summary")
        st.markdown(st.session_state.last_summary.get("summary", "No summary returned."))
        show_sources(st.session_state.last_summary.get("sources", []), "Evidence used")

with tab3:
    st.markdown("#### Questions to try")
    st.caption("These questions work best when the corresponding reports are indexed.")
    examples = [
        "What was Netflix's total revenue in 2024?",
        "What was Microsoft's total revenue in fiscal year 2024?",
        "Compare Netflix and Microsoft revenue in 2024.",
        "What were the major financial risks described in Netflix's 2024 report?",
        "How did Microsoft's net income change compared with the previous year?",
        "Summarize the key financial differences between Netflix and Microsoft."
    ]
    for example in examples:
        st.code(example, language=None)

with tab4:
    st.markdown("#### Recent conversations")
    history = st.session_state.rag.chat_history
    if not history:
        st.info("No conversations yet. Ask a question to start your research history.")
    else:
        for i, (q, answer) in enumerate(reversed(history[-10:]), start=1):
            with st.expander(f"{i}. {q}"):
                st.markdown(answer)

st.markdown('<div style="text-align:center;color:#97a4b7;font-size:.76rem;padding-top:36px">FinSight AI · Financial document intelligence powered by retrieval-augmented generation</div>', unsafe_allow_html=True)
