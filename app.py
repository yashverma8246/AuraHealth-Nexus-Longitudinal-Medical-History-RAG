import os
import time
import logging
from dotenv import load_dotenv

# Load env before any local module imports
load_dotenv(override=True)

import streamlit as st

from rag_pipeline import RAGPipeline
from loader import Document, extract_text_from_bytes

# Configure logger
logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)

# --- PAGE CONFIGURATION ---
st.set_page_config(
    page_title="AuraHealth Nexus | SYN-PAT-001 Longitudinal Medical RAG",
    page_icon="🧬",
    layout="wide",
    initial_sidebar_state="expanded"
)

# --- CUSTOM LIGHT THEME CSS (Clinical Minimalism) ---
st.markdown("""
<style>
/* Clinical Minimalism Design System – Light Theme */
:root {
  /* Primary teal palette */
  --color-primary: #0F766E;
  --color-primary-dark: #115E59;
  --color-primary-light: #E8F5F1;

  /* Backgrounds */
  --color-background: #F7F9F8;
  --color-surface: #FFFFFF;
  --color-surface-secondary: #F1F5F4;

  /* Text */
  --color-text-primary: #172B2A;
  --color-text-secondary: #475569;
  --color-muted: #64748B;

  /* Borders */
  --color-border: #DFE7E4;

  /* Semantic */
  --color-success: #15803D;
  --color-warning: #B45309;
  --color-error: #B91C1C;
  --color-info: #2563EB;

  /* Spacing */
  --space-0: 0rem;
  --space-1: 0.25rem;
  --space-2: 0.5rem;
  --space-3: 0.75rem;
  --space-4: 1rem;
  --space-5: 1.5rem;
  --space-6: 2rem;
  --space-8: 2.5rem;
  --space-10: 3rem;

  /* Border radius */
  --radius-sm: 4px;
  --radius-md: 8px;
  --radius-lg: 12px;
  --radius-xl: 16px;

  /* Font sizes */
  --font-xs: 0.75rem;
  --font-sm: 0.875rem;
  --font-base: 1rem;
  --font-lg: 1.125rem;
  --font-xl: 1.25rem;
  --font-2xl: 1.5rem;
  --font-3xl: 1.875rem;
  --font-4xl: 2.25rem;
  --font-5xl: 3rem;
}

/* ===== Global Reset ===== */
html, body, .stApp {
  background: var(--color-background);
  color: var(--color-text-primary);
  font-family: 'Inter', 'Helvetica', 'Arial', sans-serif;
  margin: 0;
  padding: 0;
}

/* Ensure all Streamlit markdown text inherits proper dark text */
.stMarkdown, .stMarkdown p, .stMarkdown span, .stMarkdown li,
.stMarkdown h1, .stMarkdown h2, .stMarkdown h3, .stMarkdown h4,
.stMarkdown h5, .stMarkdown h6 {
  color: var(--color-text-primary);
}

/* ===== Main Titles ===== */
.main-title {
  background: linear-gradient(90deg, var(--color-primary), var(--color-primary-dark));
  -webkit-background-clip: text;
  -webkit-text-fill-color: transparent;
  font-size: var(--font-4xl);
  font-weight: 800;
  margin-bottom: var(--space-3);
}

.sub-title {
  color: var(--color-text-secondary);
  font-size: var(--font-base);
  margin-bottom: var(--space-5);
  border-bottom: 1px solid var(--color-border);
  padding-bottom: var(--space-3);
}

/* ===== Demo‑Query Panel ===== */
.demo-query-panel {
  background: #FFFFFF;
  border: 1px solid #DDE7E3;
  border-radius: 14px;
  padding: 22px;
  margin-bottom: 1.5rem;
}
.demo-query-panel h4 {
  margin-top: 0;
  color: #115E59 !important;
  -webkit-text-fill-color: #115E59 !important;
  font-size: 1.1rem;
  font-weight: 700;
}
.demo-query-panel p {
  color: #475569 !important;
  -webkit-text-fill-color: #475569 !important;
  font-size: 0.9rem;
  margin-bottom: 0;
  line-height: 1.5;
}

/* ===== Sidebar ===== */
section[data-testid="stSidebar"] {
  background: var(--color-surface-secondary);
  border-right: 1px solid var(--color-border);
}
section[data-testid="stSidebar"] h1,
section[data-testid="stSidebar"] h2,
section[data-testid="stSidebar"] h3,
section[data-testid="stSidebar"] h4 {
  color: var(--color-text-primary) !important;
}
section[data-testid="stSidebar"] p:not(button *),
section[data-testid="stSidebar"] span:not(button *),
section[data-testid="stSidebar"] label:not(button *) {
  color: var(--color-text-secondary) !important;
}
section[data-testid="stSidebar"] .stCaption,
section[data-testid="stSidebar"] caption,
section[data-testid="stSidebar"] small {
  color: var(--color-muted) !important;
}

/* ===== Metric Cards ===== */
div[data-testid="stMetric"] {
  background: var(--color-surface);
  border: 1px solid var(--color-border);
  border-radius: var(--radius-md);
  padding: var(--space-4) var(--space-5);
  box-shadow: 0 1px 3px rgba(0,0,0,0.05);
  transition: transform 0.1s ease, border-color 0.1s ease;
}
div[data-testid="stMetric"]:hover {
  transform: translateY(-2px);
  border-color: var(--color-primary);
}
div[data-testid="stMetric"] label {
  color: var(--color-muted) !important;
  font-size: var(--font-xs) !important;
  font-weight: 500 !important;
}
div[data-testid="stMetric"] div[data-testid="stMetricValue"] {
  color: var(--color-primary-dark) !important;
  font-size: var(--font-lg) !important;
  font-weight: 700 !important;
}

/* ===== Response Card ===== */
.response-card {
  background: var(--color-surface);
  border: 1px solid var(--color-border);
  border-radius: var(--radius-lg);
  padding: var(--space-5);
  margin-top: var(--space-3);
  margin-bottom: var(--space-4);
  box-shadow: 0 2px 6px -2px rgba(0,0,0,0.08);
}
.response-card span, .response-card div, .response-card b {
  color: inherit;
}

/* ===== Badges ===== */
.badge-high, .badge-medium, .badge-low, .badge-info {
  padding: var(--space-1) var(--space-3);
  border-radius: 9999px;
  font-size: var(--font-xs);
  font-weight: 600;
  display: inline-block;
}
.badge-high   { background: rgba(21,128,61,0.12); color: #15803D; border: 1px solid #15803D; }
.badge-medium { background: rgba(180,83,9,0.12);  color: #B45309; border: 1px solid #B45309; }
.badge-low    { background: rgba(185,28,28,0.12);  color: #B91C1C; border: 1px solid #B91C1C; }
.badge-info   { background: rgba(37,99,235,0.12);  color: #2563EB; border: 1px solid #2563EB; }

/* ===== Chat Messages ===== */
div[data-testid="stChatMessage"] {
  background: var(--color-surface);
  border: 1px solid var(--color-border);
  border-radius: var(--radius-lg);
  padding: var(--space-3);
  margin-bottom: var(--space-3);
}
div[data-testid="stChatMessage"] p,
div[data-testid="stChatMessage"] span,
div[data-testid="stChatMessage"] li {
  color: var(--color-text-primary) !important;
}

/* ===== Expanders ===== */
div[data-testid="stExpander"] {
  border: 1px solid var(--color-border) !important;
  border-radius: var(--radius-md) !important;
  background: var(--color-surface-secondary) !important;
}
div[data-testid="stExpander"] summary,
div[data-testid="stExpander"] summary span {
  color: var(--color-text-primary) !important;
}
div[data-testid="stExpander"] .stMarkdown p,
div[data-testid="stExpander"] .stMarkdown span {
  color: var(--color-text-secondary) !important;
}

/* ===== Buttons (Enabled, Hover, Focus, Disabled) ===== */
div.stButton > button,
section[data-testid="stSidebar"] div.stButton > button,
button[data-testid="baseButton-secondary"],
button[data-testid="baseButton-primary"],
button[kind="secondary"],
button[kind="primary"],
div[data-testid="stBaseButton-secondary"] > button,
div[data-testid="stBaseButton-primary"] > button {
  background-color: #0F766E !important;
  color: #FFFFFF !important;
  font-weight: 600 !important;
  border: 1px solid #0F766E !important;
  border-radius: var(--radius-md) !important;
  padding: var(--space-2) var(--space-4) !important;
  transition: all 0.2s ease !important;
  cursor: pointer !important;
}

/* Ensure white text on all inner elements of enabled buttons */
div.stButton > button *,
section[data-testid="stSidebar"] div.stButton > button *,
button[data-testid="baseButton-secondary"] *,
button[data-testid="baseButton-primary"] *,
button[kind="secondary"] *,
button[kind="primary"] *,
div.stButton > button p,
section[data-testid="stSidebar"] div.stButton > button p,
div.stButton > button span,
section[data-testid="stSidebar"] div.stButton > button span,
div.stButton > button div,
section[data-testid="stSidebar"] div.stButton > button div {
  color: #FFFFFF !important;
  -webkit-text-fill-color: #FFFFFF !important;
}

/* Button Hover State */
div.stButton > button:hover:not(:disabled),
section[data-testid="stSidebar"] div.stButton > button:hover:not(:disabled),
button[data-testid="baseButton-secondary"]:hover:not(:disabled),
button[data-testid="baseButton-primary"]:hover:not(:disabled),
button[kind="secondary"]:hover:not(:disabled),
button[kind="primary"]:hover:not(:disabled) {
  background-color: #115E59 !important;
  border-color: #115E59 !important;
  color: #FFFFFF !important;
  box-shadow: 0 2px 6px rgba(15, 118, 110, 0.3) !important;
}

div.stButton > button:hover:not(:disabled) *,
section[data-testid="stSidebar"] div.stButton > button:hover:not(:disabled) *,
button[data-testid="baseButton-secondary"]:hover:not(:disabled) *,
button[data-testid="baseButton-primary"]:hover:not(:disabled) * {
  color: #FFFFFF !important;
  -webkit-text-fill-color: #FFFFFF !important;
}

/* Button Active State */
div.stButton > button:active:not(:disabled),
section[data-testid="stSidebar"] div.stButton > button:active:not(:disabled) {
  background-color: #0D5F58 !important;
  border-color: #0D5F58 !important;
  color: #FFFFFF !important;
}

div.stButton > button:active:not(:disabled) *,
section[data-testid="stSidebar"] div.stButton > button:active:not(:disabled) * {
  color: #FFFFFF !important;
  -webkit-text-fill-color: #FFFFFF !important;
}

/* Button Focus State */
div.stButton > button:focus,
div.stButton > button:focus-visible,
section[data-testid="stSidebar"] div.stButton > button:focus,
section[data-testid="stSidebar"] div.stButton > button:focus-visible,
button[data-testid="baseButton-secondary"]:focus,
button[data-testid="baseButton-primary"]:focus {
  outline: 2px solid #0F766E !important;
  outline-offset: 2px !important;
  box-shadow: 0 0 0 3px rgba(15, 118, 110, 0.25) !important;
}

/* Disabled State for all buttons */
div.stButton > button:disabled,
div.stButton > button[disabled],
section[data-testid="stSidebar"] div.stButton > button:disabled,
section[data-testid="stSidebar"] div.stButton > button[disabled],
button[data-testid="baseButton-secondary"]:disabled,
button[data-testid="baseButton-secondary"][disabled],
button[data-testid="baseButton-primary"]:disabled,
button[data-testid="baseButton-primary"][disabled],
button[kind="secondary"]:disabled,
button[kind="secondary"][disabled],
button[kind="primary"]:disabled,
button[kind="primary"][disabled] {
  background-color: #E2E8F0 !important;
  color: #64748B !important;
  border: 1px solid #CBD5E1 !important;
  cursor: not-allowed !important;
  opacity: 1 !important;
  box-shadow: none !important;
}

div.stButton > button:disabled *,
div.stButton > button[disabled] *,
section[data-testid="stSidebar"] div.stButton > button:disabled *,
section[data-testid="stSidebar"] div.stButton > button[disabled] *,
button[data-testid="baseButton-secondary"]:disabled *,
button[data-testid="baseButton-secondary"][disabled] *,
button[data-testid="baseButton-primary"]:disabled *,
button[data-testid="baseButton-primary"][disabled] *,
button:disabled p,
button:disabled span,
button[disabled] p,
button[disabled] span {
  color: #64748B !important;
  -webkit-text-fill-color: #64748B !important;
}

/* ===== Text Inputs & Chat Input ===== */
div[data-testid="stTextInput"] input,
div[data-testid="stChatInput"] textarea,
textarea {
  background: var(--color-surface) !important;
  color: var(--color-text-primary) !important;
  border: 1px solid var(--color-border) !important;
  border-radius: var(--radius-md) !important;
}
div[data-testid="stTextInput"] input:focus,
div[data-testid="stChatInput"] textarea:focus,
textarea:focus {
  border-color: var(--color-primary) !important;
  box-shadow: 0 0 0 2px rgba(15, 118, 110, 0.2) !important;
  outline: none !important;
}
div[data-testid="stTextInput"] input::placeholder,
div[data-testid="stChatInput"] textarea::placeholder,
textarea::placeholder {
  color: var(--color-muted) !important;
}

/* Chat Input Submit Button */
button[data-testid="stChatInputSubmitButton"],
div[data-testid="stChatInput"] button {
  background-color: #0F766E !important;
  border: 1px solid #0F766E !important;
  border-radius: var(--radius-md) !important;
  color: #FFFFFF !important;
  transition: all 0.2s ease !important;
}
button[data-testid="stChatInputSubmitButton"] svg,
div[data-testid="stChatInput"] button svg {
  fill: #FFFFFF !important;
  stroke: #FFFFFF !important;
  color: #FFFFFF !important;
}
button[data-testid="stChatInputSubmitButton"]:hover:not(:disabled),
div[data-testid="stChatInput"] button:hover:not(:disabled) {
  background-color: #115E59 !important;
  border-color: #115E59 !important;
}
button[data-testid="stChatInputSubmitButton"]:disabled,
button[data-testid="stChatInputSubmitButton"][disabled],
div[data-testid="stChatInput"] button:disabled,
div[data-testid="stChatInput"] button[disabled] {
  background-color: #E2E8F0 !important;
  border-color: #CBD5E1 !important;
  cursor: not-allowed !important;
  opacity: 1 !important;
}
button[data-testid="stChatInputSubmitButton"]:disabled svg,
button[data-testid="stChatInputSubmitButton"][disabled] svg,
div[data-testid="stChatInput"] button:disabled svg,
div[data-testid="stChatInput"] button[disabled] svg {
  fill: #64748B !important;
  stroke: #64748B !important;
  color: #64748B !important;
}

/* ===== File Uploader ===== */
div[data-testid="stFileUploader"] {
  color: var(--color-text-secondary) !important;
}
div[data-testid="stFileUploader"] label {
  color: var(--color-text-primary) !important;
}

/* ===== Status Widgets ===== */
div[data-testid="stStatus"],
div[data-testid="stStatusWidget"] {
  background: var(--color-surface) !important;
  color: var(--color-text-primary) !important;
}

/* ===== Select Boxes ===== */
div[data-testid="stSelectbox"] label {
  color: var(--color-text-primary) !important;
}

/* ===== Alert Components ===== */
div[data-testid="stAlert"] {
  border-radius: var(--radius-md) !important;
}

/* ===== Tables ===== */
table, th, td {
  color: var(--color-text-primary) !important;
  border-color: var(--color-border) !important;
}
th {
  background: var(--color-surface-secondary) !important;
  font-weight: 600;
}

/* ===== Code Blocks ===== */
pre, code {
  color: var(--color-text-primary);
}

/* ===== Links ===== */
a {
  color: var(--color-primary-dark);
  text-decoration: none;
}
a:hover {
  text-decoration: underline;
  color: var(--color-primary);
}

/* ===== Responsive ===== */
@media (max-width: 768px) {
  .main-title { font-size: var(--font-3xl); }
  .sub-title  { font-size: var(--font-sm); }
  .response-card    { padding: var(--space-4); }
  .demo-query-panel { padding: var(--space-4); }
}

/* ===== Accessibility ===== */
button:focus, a:focus {
  outline: 2px solid var(--color-primary);
  outline-offset: 2px;
}
</style>
""", unsafe_allow_html=True)

# --- SESSION STATE INITIALIZATION ---
if "pipeline" not in st.session_state:
    with st.spinner("🚀 Initializing AuraHealth Nexus Longitudinal RAG Engine..."):
        pipeline = RAGPipeline(top_k=6)
        try:
            pipeline.index()
        except Exception as e:
            logger.error(f"Error initializing index: {e}")
        st.session_state.pipeline = pipeline

if "chat_history" not in st.session_state:
    st.session_state.chat_history = []

if "last_response_time" not in st.session_state:
    st.session_state.last_response_time = 0.0

if "notification" not in st.session_state:
    st.session_state.notification = None

# --- SIDEBAR: STATISTICS & MANAGEMENT ---
with st.sidebar:
    st.markdown("### 🧬 AuraHealth Nexus")
    st.markdown("*Single-Patient Longitudinal RAG System*")
    st.markdown("---")
    
    # Statistics Dashboard
    st.markdown("#### 📊 Patient & System Stats")
    stats = st.session_state.pipeline.get_stats()
    
    col1, col2 = st.columns(2)
    with col1:
        st.metric(label="Patient ID", value=stats["patient_id"])
    with col2:
        st.metric(label="Time Span", value=stats["years_span"])
        
    col3, col4 = st.columns(2)
    with col3:
        st.metric(label="Yearly Docs", value=stats["indexed_documents"])
    with col4:
        st.metric(label="Total Chunks", value=stats["total_chunks"])
        
    resp_time = f"{st.session_state.last_response_time:.2f}s" if st.session_state.last_response_time > 0 else "--"
    st.metric(label="Last Query Latency", value=resp_time)
        
    st.markdown(f"**🧠 Embedding Model:** `{stats['embedding_model']}`")
    st.markdown(f"**🗄️ Vector Store:** `{stats['vector_store']}`")
    st.markdown(f"**🤖 LLM Engine:** `{stats['llm_model']}`")
    st.markdown("---")
    
    # Document Upload & Ingestion
    st.markdown("#### 📤 Ingest Yearly Records")
    uploaded_files = st.file_uploader(
        "Upload patient records (.txt, .pdf, .docx)",
        type=["txt", "pdf", "docx"],
        accept_multiple_files=True,
        help="Upload new yearly records to automatically extract patient metadata and index into vector store."
    )
    
    if uploaded_files:
        if st.button("⚡ Ingest & Update Patient Index", use_container_width=True):
            data_dir = st.session_state.pipeline.data_dir
            os.makedirs(data_dir, exist_ok=True)
            
            with st.status("🔄 Processing uploaded records...", expanded=True) as status:
                success_count = 0
                for ufile in uploaded_files:
                    try:
                        st.write(f"📖 Reading `{ufile.name}`...")
                        file_bytes = ufile.read()
                        text = extract_text_from_bytes(ufile.name, file_bytes)
                        if not text.strip():
                            st.warning(f"⚠️ `{ufile.name}` is empty or unreadable.")
                            continue
                            
                        save_path = os.path.join(data_dir, ufile.name)
                        with open(save_path, "wb") as f:
                            f.write(file_bytes)
                        success_count += 1
                    except Exception as e:
                        st.error(f"❌ Failed to ingest `{ufile.name}`: {str(e)}")
                        
                if success_count > 0:
                    st.write("🧠 Re-chunking and rebuilding patient FAISS index...")
                    try:
                        st.session_state.pipeline.index(force_rebuild=True)
                        status.update(label=f"✅ Ingested {success_count} record(s) and rebuilt index!", state="complete")
                        st.session_state.notification = ("success", f"Ingested {success_count} file(s) and updated index!")
                        time.sleep(1)
                        st.rerun()
                    except Exception as e:
                        status.update(label=f"❌ Indexing failed: {str(e)}", state="error")
                else:
                    status.update(label="⚠️ No valid files were ingested.", state="error")

    st.markdown("---")
    
    # Action Buttons
    st.markdown("#### ⚙️ System Controls")
    if st.button("🔄 Force Rebuild Patient Index", use_container_width=True):
        with st.spinner("Rebuilding patient vector index from scratch..."):
            try:
                st.session_state.pipeline.index(force_rebuild=True)
                st.success("✅ Patient index rebuilt successfully!")
                time.sleep(1)
                st.rerun()
            except Exception as e:
                st.error(f"Error rebuilding index: {e}")
                
    if st.button("🗑️ Clear Chat History", use_container_width=True):
        st.session_state.pipeline.reset_history()
        st.session_state.chat_history = []
        st.session_state.last_response_time = 0.0
        st.success("💬 Conversation cleared!")
        time.sleep(0.5)
        st.rerun()

    st.markdown("---")
    st.caption("AuraHealth Nexus | Longitudinal Medical RAG v3.0")

# --- MAIN PAGE: HEADER & DESCRIPTION ---
st.markdown('<div class="main-title">🧬 AuraHealth Nexus</div>', unsafe_allow_html=True)
st.markdown('<div class="sub-title"><b>SYN-PAT-001</b> Longitudinal Medical History RAG (2010 – 2024) | <i>Synthetic Clinical Dataset</i></div>', unsafe_allow_html=True)

# Display toast/notification if present
if st.session_state.notification:
    ntype, nmsg = st.session_state.notification
    if ntype == "success":
        st.success(nmsg)
    elif ntype == "error":
        st.error(nmsg)
    st.session_state.notification = None

# --- ERROR HANDLING & ENVIRONMENT WARNINGS ---
groq_key = os.getenv("GROQ_API_KEY")
if not groq_key:
    st.warning("⚠️ **GROQ_API_KEY not detected!** Live generation via Groq API will fail. Please configure `GROQ_API_KEY` in your `.env` file.")

if stats["total_chunks"] == 0:
    st.error(f"🚨 **Vector Store is Empty!** No documents found in `{st.session_state.pipeline.data_dir}`. Please check data directory or rebuild index.")

# --- EMPTY STATE: SUGGESTED DEMO QUERIES ---
demo_query = None
if not st.session_state.chat_history:
    st.markdown("""
    <div class="demo-query-panel">
        <h4>💡 Longitudinal Medical Reasoning Demo Queries</h4>
        <p>Click any sample question below to test timeline synthesis, medication history, and lab trends across 15 years:</p>
    </div>
    """, unsafe_allow_html=True)
    
    col_a, col_b = st.columns(2)
    with col_a:
        if st.button("📉 How did kidney function (eGFR) change over time?", use_container_width=True):
            demo_query = "How did eGFR and kidney function change from baseline in 2010 to 2024?"
        if st.button("💊 When was Losartan added and what are the 2024 medications?", use_container_width=True):
            demo_query = "When was losartan added and what medications were documented in 2024?"
    with col_b:
        if st.button("🩺 When did prediabetes progress to Type 2 diabetes?", use_container_width=True):
            demo_query = "When was type 2 diabetes first documented and what was the progression from prediabetes?"
        if st.button("📅 Give a chronological summary of major medical events", use_container_width=True):
            demo_query = "Give a chronological summary of the patient's major medical events from 2010 to 2024."

# --- MAIN CHAT AREA ---
for msg in st.session_state.chat_history:
    role = msg["role"]
    avatar = "👤" if role == "user" else "🧬"
    
    with st.chat_message(role, avatar=avatar):
        st.markdown(msg["content"])
        
        # Render Response Card & Citations for assistant answers
        if role == "assistant" and "metadata" in msg and msg["metadata"]:
            meta = msg["metadata"]
            conf = meta.get("confidence", "Low")
            score = meta.get("top_score", 0.0)
            elapsed = meta.get("time_taken", 0.0)
            sources = meta.get("sources", [])
            years = meta.get("years", [])
            strategy = meta.get("strategy", "STANDARD_SEMANTIC")
            retrieved_chunks = meta.get("retrieved", [])
            tokens = meta.get("total_tokens")
            
            badge_class = f"badge-{conf.lower()}"
            badge_icon = "🟢" if conf == "High" else "🟡" if conf == "Medium" else "🔴"
            token_display = f" | 🔤 Tokens: <b>{tokens}</b>" if tokens else ""
            years_display = f"📅 Years Retrieved: <b>{', '.join(str(y) for y in years)}</b>" if years else "📅 Years: <i>None</i>"
            
            st.markdown(f"""
            <div class="response-card">
                <div style="display: flex; justify-content: space-between; align-items: center; margin-bottom: 10px; flex-wrap: wrap; gap: 8px;">
                    <div>
                        <span class="{badge_class}">{badge_icon} {conf} Confidence ({score:.1%})</span>
                        <span class="badge-info" style="margin-left: 6px;">🎯 Strategy: {strategy.replace('_', ' ').title()}</span>
                    </div>
                    <span style="color: #64748B; font-size: 0.85rem;">⏱️ Latency: <b>{elapsed:.2f}s</b>{token_display}</span>
                </div>
                <div style="font-size: 0.88rem; color: #475569; margin-bottom: 4px;">
                    {years_display}
                </div>
                <div style="font-size: 0.88rem; color: #64748B;">
                    <b>📑 Primary Source Records:</b> {', '.join([f'`{s}`' for s in sources]) if sources else '<i>None retrieved</i>'}
                </div>
            </div>
            """, unsafe_allow_html=True)
            
            # Expandable Source Viewer
            if retrieved_chunks:
                with st.expander(f"🔍 Inspect Chronological Evidence Chunks ({len(retrieved_chunks)} Chunks Retrieved)"):
                    for idx, (chk, sim) in enumerate(retrieved_chunks, 1):
                        yr_str = str(chk.year) if chk.year is not None else "Unknown"
                        st.markdown(f"""
                        **Rank #{idx}** | 📅 **Year:** `{yr_str}` | 📄 **Source:** `{chk.source}` | 🔢 **Chunk ID:** `{chk.id}` | 🎯 **Relevance:** `{sim:.3f}`
                        """)
                        st.code(chk.text, language="markdown")
                        if idx < len(retrieved_chunks):
                            st.divider()

# --- CHAT INPUT & EXECUTION ---
input_query = st.chat_input("Ask about SYN-PAT-001 (e.g., 'How did HbA1c change from 2010 to 2024?')")
active_query = demo_query or input_query

if active_query:
    if stats["total_chunks"] == 0:
        st.error("Please index patient documents before asking questions.")
    else:
        # Render user question immediately
        with st.chat_message("user", avatar="👤"):
            st.markdown(active_query)
        
        st.session_state.chat_history.append({
            "role": "user",
            "content": active_query,
            "timestamp": time.strftime("%H:%M:%S")
        })
        
        with st.chat_message("assistant", avatar="🧬"):
            with st.status("🧠 Analyzing patient longitudinal medical history...", expanded=True) as status:
                try:
                    st.write("🔎 Performing chronological temporal retrieval across patient records...")
                    res = st.session_state.pipeline.ask_with_metadata(active_query, use_history=True)
                    
                    st.write("⚙️ Synthesizing grounded medical response...")
                    time.sleep(0.3)
                    
                    status.update(label="✅ Response synthesized successfully!", state="complete", expanded=False)
                    
                    st.markdown(res["answer"])
                    
                    st.session_state.last_response_time = res["time_taken"]
                    
                    st.session_state.chat_history.append({
                        "role": "assistant",
                        "content": res["answer"],
                        "timestamp": time.strftime("%H:%M:%S"),
                        "metadata": res
                    })
                    
                    st.rerun()
                    
                except Exception as e:
                    logger.error(f"Error during RAG execution: {e}")
                    status.update(label="❌ Error generating response", state="error", expanded=True)
                    err_msg = f"An error occurred while processing your request: `{str(e)}`. Please verify your network and API configuration."
                    st.error(err_msg)
                    st.session_state.chat_history.append({
                        "role": "assistant",
                        "content": err_msg,
                        "timestamp": time.strftime("%H:%M:%S"),
                        "metadata": {}
                    })
