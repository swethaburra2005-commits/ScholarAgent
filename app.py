from pathlib import Path
import json
import re

import streamlit as st
import torch
from sentence_transformers import SentenceTransformer, CrossEncoder
from sentence_transformers.util import semantic_search

from multipaper.multi_reranker import load_embeddings
from llm.synthesizer import synthesize_answer
from pdf_ingest import index_pdf


# ============================================================
# PAGE CONFIG
# ============================================================

st.set_page_config(
    page_title="ScholarAgent | Research Intelligence",
    page_icon="🔬",
    layout="wide",
    initial_sidebar_state="expanded",
)


# ============================================================
# PROJECT PATHS
# ============================================================

PROJECT_ROOT = Path(__file__).resolve().parent
EMBEDDING_FILE = PROJECT_ROOT / "papers" / "multi_paper_embeddings.json"

EMBEDDING_MODEL_NAME = "sentence-transformers/all-MiniLM-L6-v2"
RERANKER_MODEL_NAME = "cross-encoder/ms-marco-MiniLM-L6-v2"


# ============================================================
# DESIGN SYSTEM
# ============================================================

PRIMARY = "#7C3AED"
PRIMARY_LIGHT = "#A78BFA"
CYAN = "#22D3EE"
BG = "#070B14"
SURFACE = "#0D1321"
SURFACE_2 = "#111827"
BORDER = "#1F2937"
TEXT = "#F8FAFC"
MUTED = "#94A3B8"
SUCCESS = "#34D399"
WARNING = "#FBBF24"


# ============================================================
# CUSTOM CSS
# ============================================================

st.html(
    f"""
    <style>
    .stApp {{
        background:
            radial-gradient(circle at 85% 5%,
                rgba(124,58,237,.13), transparent 28%),
            radial-gradient(circle at 10% 20%,
                rgba(34,211,238,.06), transparent 24%),
            {BG};
    }}

    .main .block-container {{
        max-width: 1450px;
        padding-top: 2rem;
        padding-bottom: 4rem;
    }}

    [data-testid="stSidebar"] {{
        background: #080D18;
        border-right: 1px solid {BORDER};
    }}

    h1, h2, h3 {{
        color: {TEXT} !important;
        letter-spacing: -.025em;
    }}

    p, label, span {{
        color: {TEXT};
    }}

    .hero {{
        padding: 2.7rem 2.8rem;
        margin-bottom: 1.5rem;
        border: 1px solid rgba(124,58,237,.25);
        border-radius: 24px;
        background:
            linear-gradient(135deg,
                rgba(124,58,237,.16),
                rgba(34,211,238,.05)),
            {SURFACE};
        position: relative;
        overflow: hidden;
    }}

    .hero::after {{
        content: "";
        position: absolute;
        width: 260px;
        height: 260px;
        right: -90px;
        top: -100px;
        border-radius: 50%;
        background: rgba(124,58,237,.10);
        filter: blur(8px);
    }}

    .eyebrow {{
        color: {PRIMARY_LIGHT};
        font-size: .76rem;
        font-weight: 700;
        letter-spacing: .16em;
        text-transform: uppercase;
        margin-bottom: .7rem;
    }}

    .hero-title {{
        font-size: 2.8rem;
        line-height: 1.05;
        font-weight: 800;
        margin-bottom: .8rem;
    }}

    .hero-subtitle {{
        color: {MUTED};
        font-size: 1.02rem;
        line-height: 1.7;
        max-width: 820px;
    }}

    .brand {{
        display: flex;
        align-items: center;
        gap: 12px;
        padding: .6rem .4rem 1.3rem;
    }}

    .brand-icon {{
        width: 42px;
        height: 42px;
        border-radius: 12px;
        display: flex;
        align-items: center;
        justify-content: center;
        background: linear-gradient(135deg,{PRIMARY},#4F46E5);
        font-size: 1.35rem;
        box-shadow: 0 8px 25px rgba(124,58,237,.28);
    }}

    .brand-name {{
        font-size: 1.15rem;
        font-weight: 800;
        color: {TEXT};
    }}

    .brand-caption {{
        color: {MUTED};
        font-size: .72rem;
    }}

    .section-label {{
        color: {MUTED};
        font-size: .72rem;
        text-transform: uppercase;
        letter-spacing: .14em;
        font-weight: 700;
        margin: 1.7rem 0 .7rem;
    }}

    .metric-card {{
        background: {SURFACE};
        border: 1px solid {BORDER};
        border-radius: 17px;
        padding: 1.25rem 1.35rem;
        min-height: 115px;
    }}

    .metric-label {{
        color: {MUTED};
        font-size: .76rem;
        font-weight: 600;
        text-transform: uppercase;
        letter-spacing: .08em;
    }}

    .metric-value {{
        color: {TEXT};
        font-size: 1.85rem;
        font-weight: 800;
        margin-top: .45rem;
    }}

    .metric-detail {{
        color: {MUTED};
        font-size: .74rem;
        margin-top: .25rem;
    }}

    .pipeline {{
        display: flex;
        align-items: center;
        gap: 8px;
        margin: 1rem 0 1.8rem;
    }}

    .pipeline-step {{
        flex: 1;
        background: {SURFACE};
        border: 1px solid {BORDER};
        border-radius: 14px;
        padding: .85rem 1rem;
    }}

    .pipeline-step.active {{
        border-color: rgba(124,58,237,.5);
        background: linear-gradient(135deg,
            rgba(124,58,237,.13), {SURFACE});
    }}

    .pipeline-number {{
        color: {PRIMARY_LIGHT};
        font-size: .7rem;
        font-weight: 800;
    }}

    .pipeline-name {{
        color: {TEXT};
        font-size: .83rem;
        font-weight: 700;
        margin-top: .2rem;
    }}

    .pipeline-status {{
        color: {SUCCESS};
        font-size: .68rem;
        margin-top: .3rem;
    }}

    .pipeline-arrow {{
        color: #475569;
        font-size: 1rem;
    }}

    .query-heading {{
        color: {TEXT};
        font-size: 1.15rem;
        font-weight: 750;
        margin-bottom: .35rem;
    }}

    .query-caption {{
        color: {MUTED};
        font-size: .86rem;
        margin-bottom: .9rem;
    }}

    textarea {{
        background: {SURFACE_2} !important;
        border: 1px solid {BORDER} !important;
        border-radius: 16px !important;
        color: {TEXT} !important;
    }}

    textarea:focus {{
        border-color: {PRIMARY} !important;
        box-shadow: 0 0 0 1px {PRIMARY},
                    0 0 25px rgba(124,58,237,.13) !important;
    }}

    .answer-card {{
        background: linear-gradient(145deg,
            rgba(124,58,237,.09),
            rgba(17,24,39,.95));
        border: 1px solid rgba(124,58,237,.25);
        border-radius: 20px;
        padding: 1.7rem 1.8rem;
        margin-top: 1rem;
    }}

    .answer-badge {{
        display: inline-block;
        padding: .3rem .65rem;
        border-radius: 999px;
        background: rgba(52,211,153,.10);
        border: 1px solid rgba(52,211,153,.25);
        color: {SUCCESS};
        font-size: .68rem;
        font-weight: 800;
        letter-spacing: .08em;
        text-transform: uppercase;
        margin-bottom: .9rem;
    }}

    .answer-title {{
        color: {TEXT};
        font-size: 1.35rem;
        font-weight: 800;
    }}

    .evidence-card {{
        background: {SURFACE};
        border: 1px solid {BORDER};
        border-radius: 16px;
        padding: 1.15rem;
        height: 100%;
    }}

    .evidence-rank {{
        color: {PRIMARY_LIGHT};
        font-size: .7rem;
        font-weight: 800;
        letter-spacing: .1em;
    }}

    .evidence-title {{
        color: {TEXT};
        font-size: .9rem;
        font-weight: 750;
        line-height: 1.45;
        margin-top: .5rem;
    }}

    .evidence-meta {{
        color: {MUTED};
        font-size: .72rem;
        margin-top: .6rem;
    }}

    .score-pill {{
        display: inline-block;
        margin-top: .75rem;
        padding: .25rem .55rem;
        border-radius: 8px;
        background: rgba(124,58,237,.12);
        color: {PRIMARY_LIGHT};
        font-size: .68rem;
        font-weight: 750;
    }}

    .sidebar-status {{
        background: {SURFACE};
        border: 1px solid {BORDER};
        border-radius: 14px;
        padding: .9rem;
        margin-top: .65rem;
    }}

    .status-row {{
        display: flex;
        align-items: center;
        gap: 8px;
        margin: .45rem 0;
        color: #CBD5E1;
        font-size: .78rem;
    }}

    .status-dot {{
        width: 7px;
        height: 7px;
        border-radius: 50%;
        background: {SUCCESS};
        box-shadow: 0 0 8px rgba(52,211,153,.5);
    }}

    .footer {{
        margin-top: 4rem;
        padding-top: 1.3rem;
        border-top: 1px solid {BORDER};
        color: #64748B;
        font-size: .72rem;
        text-align: center;
    }}
    </style>
    """
)


# ============================================================
# SESSION STATE
# ============================================================

defaults = {
    "history": [],
    "last_question": "",
    "last_results": [],
    "last_answer": "",
    "show_evaluation": False,
}

for key, value in defaults.items():
    if key not in st.session_state:
        st.session_state[key] = value


# ============================================================
# CACHED RESOURCES
# ============================================================
# Streamlit reruns the app from top to bottom after interactions.
# ML models are therefore cached as resources instead of being
# loaded repeatedly.

@st.cache_resource(show_spinner="Loading semantic embedding model...")
def get_embedding_model():
    return SentenceTransformer(EMBEDDING_MODEL_NAME)


@st.cache_resource(show_spinner="Loading Cross-Encoder reranker...")
def get_reranker_model():
    return CrossEncoder(
        RERANKER_MODEL_NAME,
        activation_fn=torch.nn.Sigmoid(),
    )


@st.cache_data(show_spinner=False)
def get_corpus():
    return load_embeddings()


data = get_corpus()


# ============================================================
# RETRIEVAL
# ============================================================

def retrieve_candidates_cached(question, top_k=10):
    model = get_embedding_model()

    corpus_embeddings = torch.tensor(
        [item["embedding"] for item in data],
        dtype=torch.float32,
    )

    query_embedding = model.encode_query(
        question,
        convert_to_tensor=True,
    )

    results = semantic_search(
        query_embedding,
        corpus_embeddings,
        top_k=top_k,
    )

    candidates = []

    for result in results[0]:
        item = data[result["corpus_id"]]

        candidates.append(
            {
                "paper_id": item["paper_id"],
                "chunk_id": item["chunk_id"],
                "title": item["title"],
                "arxiv_url": item["arxiv_url"],
                "text": item["text"],
                "retrieval_score": float(result["score"]),
            }
        )

    return candidates


def rerank_candidates_cached(question, candidates, top_k=5):
    reranker = get_reranker_model()

    pairs = [
        (question, candidate["text"])
        for candidate in candidates
    ]

    scores = reranker.predict(pairs)

    results = []

    for candidate, score in zip(candidates, scores):
        results.append(
            {
                **candidate,
                "rerank_score": float(score),
            }
        )

    results.sort(
        key=lambda item: item["rerank_score"],
        reverse=True,
    )

    return results[:top_k]


# ============================================================
# LOCAL LLM
# ============================================================

def render_citation_map(answer, results):
    """Show which retrieved evidence items are explicitly cited by the LLM."""

    cited_numbers = []
    for match in re.findall(r"\[Evidence\s+(\d+)\]", answer or "", flags=re.IGNORECASE):
        number = int(match)
        if 1 <= number <= len(results) and number not in cited_numbers:
            cited_numbers.append(number)

    st.markdown(
        '<div class="section-label">Citation Map</div>',
        unsafe_allow_html=True,
    )

    if not cited_numbers:
        st.info(
            "The local model did not emit explicit evidence markers in this answer. "
            "The five reranked evidence sources are shown below for verification."
        )
        return

    st.caption(
        "Each [Evidence N] marker in the synthesis maps to the corresponding "
        "reranked research chunk below."
    )

    columns = st.columns(min(3, len(cited_numbers)))

    for position, number in enumerate(cited_numbers):
        result = results[number - 1]
        with columns[position % len(columns)]:
            title = result.get("title", "Untitled paper")
            paper_id = result.get("paper_id", "unknown")
            chunk_id = result.get("chunk_id", "unknown")
            rerank_score = result.get("rerank_score", 0)
            arxiv_url = result.get("arxiv_url", "")

            st.markdown(
                f"**[Evidence {number}]**  \n"
                f"**{title}**  \n"
                f"`{paper_id} • Chunk {chunk_id}`  \n"
                f"Cross-Encoder: `{rerank_score:.4f}`"
            )

            if arxiv_url:
                st.link_button(
                    "Open source ↗",
                    arxiv_url,
                    key=f"citation_source_{number}",
                )


def generate_local_answer(question, evidence):
    """Generate a grounded answer directly from the reranked evidence."""

    try:
        return synthesize_answer(
            question=question,
            evidence=evidence,
        )
    except Exception as error:
        return (
            "The local LLM encountered an error.\n\n"
            f"```text\n{error}\n```"
        )



# ============================================================
# EVALUATION
# ============================================================

EVALUATION_QUESTIONS_FILE = (
    PROJECT_ROOT / "multipaper" / "evaluation_questions.json"
)
EVALUATION_LABELS_FILE = (
    PROJECT_ROOT / "multipaper" / "evaluation_labels.json"
)


def _load_json_file(path):
    if not path.exists():
        raise FileNotFoundError(f"Missing evaluation file: {path}")

    with path.open("r", encoding="utf-8") as handle:
        return json.load(handle)


def _question_id_from_item(item, fallback):
    if isinstance(item, dict):
        for key in ("question_id", "id", "qid"):
            if item.get(key):
                return str(item[key])
    return fallback


def _question_text_from_item(item):
    if isinstance(item, str):
        return item

    if isinstance(item, dict):
        for key in ("question", "text", "query", "prompt"):
            value = item.get(key)
            if value:
                return str(value)

    return ""


def _extract_question_map(raw):
    result = {}

    if isinstance(raw, dict):
        # Common format:
        # {"q1": "question text", "q2": "question text"}
        if all(isinstance(value, str) for value in raw.values()):
            for key, value in raw.items():
                result[str(key)] = str(value)
            return result

        # Wrapped formats.
        for wrapper in ("questions", "data", "items"):
            if wrapper in raw:
                return _extract_question_map(raw[wrapper])

        # {"q1": {"question": "..."}}
        for key, value in raw.items():
            question = _question_text_from_item(value)
            if question:
                result[str(key)] = question
        return result

    if isinstance(raw, list):
        for index, item in enumerate(raw, start=1):
            qid = _question_id_from_item(item, f"q{index}")
            question = _question_text_from_item(item)
            if question:
                result[qid] = question

    return result


def _extract_label_map(raw):
    result = {}

    if isinstance(raw, dict):
        for wrapper in ("labels", "data", "items", "questions"):
            if wrapper in raw and isinstance(raw[wrapper], (dict, list)):
                return _extract_label_map(raw[wrapper])

        for key, value in raw.items():
            result[str(key)] = value

        return result

    if isinstance(raw, list):
        for index, item in enumerate(raw, start=1):
            if isinstance(item, dict):
                qid = _question_id_from_item(item, f"q{index}")

                value = None
                for key in (
                    "relevant",
                    "relevant_chunks",
                    "relevant_papers",
                    "ground_truth",
                    "labels",
                    "answers",
                ):
                    if key in item:
                        value = item[key]
                        break

                if value is not None:
                    result[qid] = value

    return result


def _normalize_targets(value):
    """
    Convert several reasonable label formats into a common representation.

    Supported examples:
      "paper_001"
      "paper_001:38"
      {"paper_id": "paper_001"}
      {"paper_id": "paper_001", "chunk_id": "38"}
      [{"paper_id": "paper_001", "chunk_id": "38"}]
    """
    if value is None:
        return []

    if isinstance(value, dict):
        for wrapper in (
            "relevant",
            "relevant_chunks",
            "relevant_papers",
            "ground_truth",
            "labels",
        ):
            if wrapper in value:
                return _normalize_targets(value[wrapper])

        paper_id = value.get("paper_id")
        chunk_id = value.get("chunk_id")

        if paper_id is not None and chunk_id is not None:
            return [("chunk", str(paper_id), str(chunk_id))]

        if paper_id is not None:
            return [("paper", str(paper_id))]

        return []

    if isinstance(value, (list, tuple, set)):
        targets = []
        for item in value:
            targets.extend(_normalize_targets(item))
        return targets

    if isinstance(value, str):
        value = value.strip()

        if not value:
            return []

        # paper_001:38 / paper_001#38
        match = re.match(
            r"^(paper_\d+)[\:#](\d+)$",
            value,
            flags=re.IGNORECASE,
        )

        if match:
            return [(
                "chunk",
                match.group(1),
                match.group(2),
            )]

        return [("paper", value)]

    return []


def _matches_target(result, targets):
    paper_id = str(result.get("paper_id", ""))
    chunk_id = str(result.get("chunk_id", ""))

    for target in targets:
        if target[0] == "paper" and paper_id == target[1]:
            return True

        if (
            target[0] == "chunk"
            and paper_id == target[1]
            and chunk_id == target[2]
        ):
            return True

    return False


def _first_relevant_rank(results, targets):
    for rank, result in enumerate(results, start=1):
        if _matches_target(result, targets):
            return rank

    return None


def run_evaluation():
    questions_raw = _load_json_file(
        EVALUATION_QUESTIONS_FILE
    )
    labels_raw = _load_json_file(
        EVALUATION_LABELS_FILE
    )

    questions = _extract_question_map(
        questions_raw
    )

    labels = _extract_label_map(
        labels_raw
    )

    if not questions:
        raise ValueError(
            "No evaluation questions could be read from "
            "evaluation_questions.json."
        )

    if not labels:
        raise ValueError(
            "No evaluation labels could be read from "
            "evaluation_labels.json."
        )

    results = []

    for qid, question in questions.items():

        if qid not in labels:
            continue

        targets = _normalize_targets(
            labels[qid]
        )

        if not targets:
            continue

        candidates = retrieve_candidates_cached(
            question,
            top_k=10,
        )

        reranked = rerank_candidates_cached(
            question,
            candidates,
            top_k=5,
        )

        retrieval_rank = _first_relevant_rank(
            candidates,
            targets,
        )

        rerank_rank = _first_relevant_rank(
            reranked,
            targets,
        )

        results.append(
            {
                "question_id": qid,
                "question": question,
                "retrieval_rank": retrieval_rank,
                "rerank_rank": rerank_rank,
                "retrieval_hit": retrieval_rank is not None,
                "rerank_hit": rerank_rank is not None,
            }
        )

    if not results:
        raise ValueError(
            "Evaluation files were found, but their label format "
            "could not be matched to the retrieved paper/chunk IDs."
        )

    total = len(results)

    retrieval_hits = sum(
        item["retrieval_hit"]
        for item in results
    )

    rerank_hits = sum(
        item["rerank_hit"]
        for item in results
    )

    retrieval_recall = (
        retrieval_hits / total
    )

    rerank_recall = (
        rerank_hits / total
    )

    retrieval_rr = [
        1 / item["retrieval_rank"]
        for item in results
        if item["retrieval_rank"] is not None
    ]

    rerank_rr = [
        1 / item["rerank_rank"]
        for item in results
        if item["rerank_rank"] is not None
    ]

    return {
        "total": total,
        "retrieval_recall": retrieval_recall,
        "rerank_recall": rerank_recall,
        "retrieval_mrr": (
            sum(retrieval_rr) / total
            if retrieval_rr else 0.0
        ),
        "rerank_mrr": (
            sum(rerank_rr) / total
            if rerank_rr else 0.0
        ),
        "results": results,
    }


def render_evaluation_dashboard():
    st.markdown(
        '<div class="section-label">Evaluation Dashboard</div>',
        unsafe_allow_html=True,
    )

    st.caption(
        "Measures whether labeled relevant evidence is retrieved "
        "and retained after Cross-Encoder reranking."
    )

    try:
        evaluation = run_evaluation()
    except Exception as error:
        st.error(
            f"Evaluation could not be completed: {error}"
        )
        st.info(
            "The dashboard expects evaluation_questions.json and "
            "evaluation_labels.json inside the multipaper folder."
        )
        return

    c1, c2, c3, c4 = st.columns(4)

    with c1:
        st.metric(
            "Questions",
            evaluation["total"],
        )

    with c2:
        st.metric(
            "Retrieval Recall@10",
            f"{evaluation['retrieval_recall']:.1%}",
        )

    with c3:
        st.metric(
            "Reranking Recall@5",
            f"{evaluation['rerank_recall']:.1%}",
        )

    with c4:
        st.metric(
            "Reranking MRR@5",
            f"{evaluation['rerank_mrr']:.3f}",
        )

    st.markdown("### Evaluation results")

    for item in evaluation["results"]:
        retrieval_status = (
            "✓" if item["retrieval_hit"] else "✗"
        )
        rerank_status = (
            "✓" if item["rerank_hit"] else "✗"
        )

        retrieval_rank = (
            str(item["retrieval_rank"])
            if item["retrieval_rank"] is not None
            else "—"
        )

        rerank_rank = (
            str(item["rerank_rank"])
            if item["rerank_rank"] is not None
            else "—"
        )

        st.markdown(
            f"**{item['question_id']}** — "
            f"{item['question']}"
        )

        st.caption(
            f"Retrieval@10: {retrieval_status} "
            f"(rank {retrieval_rank})  •  "
            f"Reranking@5: {rerank_status} "
            f"(rank {rerank_rank})"
        )


# SIDEBAR
# ============================================================

with st.sidebar:
    st.html(
        """
        <div class="brand">
            <div class="brand-icon">🔬</div>
            <div>
                <div class="brand-name">ScholarAgent</div>
                <div class="brand-caption">
                    Research Intelligence Platform
                </div>
            </div>
        </div>
        """
    )

    if st.button(
        "＋  New Research",
        use_container_width=True,
    ):
        st.session_state.last_question = ""
        st.session_state.last_results = []
        st.session_state.last_answer = ""
        st.rerun()

    st.markdown(
        '<div class="section-label">Add Research Paper</div>',
        unsafe_allow_html=True,
    )

    uploaded_pdf = st.file_uploader(
        "Upload a research PDF",
        type=["pdf"],
        help="Add a PDF to the ScholarAgent knowledge base.",
        key="research_pdf_uploader",
    )

    if uploaded_pdf is not None:
        if st.button(
            "＋ Index Paper",
            use_container_width=True,
            key="index_paper_button",
        ):
            with st.spinner("Extracting, chunking, and embedding the paper..."):
                try:
                    embedding_model = get_embedding_model()

                    result = index_pdf(
                        pdf_bytes=uploaded_pdf.getvalue(),
                        filename=uploaded_pdf.name,
                        embedding_model=embedding_model,
                        embedding_file=EMBEDDING_FILE,
                    )

                    if result["status"] == "duplicate":
                        st.warning(result["message"])
                    else:
                        st.success(
                            f"✓ {result['message']} "
                            f"({result['chunks_added']} chunks added)"
                        )

                    get_corpus.clear()
                    st.session_state.last_results = []
                    st.session_state.last_answer = ""
                    st.rerun()

                except Exception as error:
                    st.error(f"Could not index the PDF: {error}")

    if st.button(
        "▦  Evaluation",
        use_container_width=True,
        key="evaluation_button",
    ):
        st.session_state.show_evaluation = True
        st.session_state.last_results = []
        st.session_state.last_answer = ""
        st.rerun()

    st.markdown(
        '<div class="section-label">Knowledge Base</div>',
        unsafe_allow_html=True,
    )

    unique_papers = len(
        {item["paper_id"] for item in data}
    )

    st.html(
        f"""
        <div class="sidebar-status">
            <div class="status-row">
                <span class="status-dot"></span>
                <b>{len(data)} chunks indexed</b>
            </div>
            <div class="status-row">
                <span class="status-dot"></span>
                {unique_papers}-paper research corpus
            </div>
            <div class="status-row">
                <span class="status-dot"></span>
                Dense semantic search
            </div>
        </div>
        """
    )

    st.markdown(
        '<div class="section-label">Pipeline</div>',
        unsafe_allow_html=True,
    )

    st.html(
        """
        <div class="sidebar-status">
            <div class="status-row">
                <span class="status-dot"></span>
                Embedding Retrieval
            </div>
            <div class="status-row">
                <span class="status-dot"></span>
                Cross-Encoder Reranking
            </div>
            <div class="status-row">
                <span class="status-dot"></span>
                Local LLM Synthesis
            </div>
        </div>
        """
    )

    st.markdown(
        '<div class="section-label">Session</div>',
        unsafe_allow_html=True,
    )

    if st.session_state.history:
        st.caption(
            f"{len(st.session_state.history)} research queries this session"
        )

        for item in reversed(st.session_state.history[-5:]):
            st.caption("• " + item[:70])
    else:
        st.caption("No research queries yet.")

    st.markdown(
        '<div class="section-label">System</div>',
        unsafe_allow_html=True,
    )

    st.caption("Local inference • No API key required")
    st.caption("Dense retrieval • Cross-Encoder reranking")


# ============================================================
# EVALUATION VIEW
# ============================================================

if st.session_state.show_evaluation:
    st.markdown(
        '<div class="hero">'
        '<div class="eyebrow">System Evaluation</div>'
        '<div class="hero-title">Measure retrieval quality.</div>'
        '<div class="hero-subtitle">'
        'Evaluate the current ScholarAgent retrieval and reranking pipeline '
        'against the project\'s labeled research questions.'
        '</div>'
        '</div>',
        unsafe_allow_html=True,
    )

    if st.button(
        "← Back to Research",
        key="back_to_research",
    ):
        st.session_state.show_evaluation = False
        st.rerun()

    render_evaluation_dashboard()

    st.markdown(
        '<div class="footer">'
        'ScholarAgent · Retrieval Evaluation'
        '<br>'
        'Recall@10 · Recall@5 · Mean Reciprocal Rank'
        '</div>',
        unsafe_allow_html=True,
    )

    st.stop()

# ============================================================
# HERO
# ============================================================

st.html(
    """
    <div class="hero">
        <div class="eyebrow">Research Intelligence</div>

        <div class="hero-title">
            Explore literature.<br>
            Synthesize evidence.
        </div>

        <div class="hero-subtitle">
            ScholarAgent searches across your research corpus,
            reranks the strongest evidence, and produces an
            evidence-grounded research synthesis using local inference.
        </div>
    </div>
    """
)


# ============================================================
# METRICS
# ============================================================

unique_papers = len(
    {item["paper_id"] for item in data}
)

total_chunks = len(data)

avg_chunk_length = int(
    sum(len(item["text"]) for item in data)
    / max(len(data), 1)
)

m1, m2, m3, m4 = st.columns(4)

with m1:
    st.html(
        f"""
        <div class="metric-card">
            <div class="metric-label">Research Papers</div>
            <div class="metric-value">{unique_papers}</div>
            <div class="metric-detail">Indexed literature sources</div>
        </div>
        """
    )

with m2:
    st.html(
        f"""
        <div class="metric-card">
            <div class="metric-label">Evidence Chunks</div>
            <div class="metric-value">{total_chunks}</div>
            <div class="metric-detail">Searchable knowledge units</div>
        </div>
        """
    )

with m3:
    st.html(
        """
        <div class="metric-card">
            <div class="metric-label">Retrieval</div>
            <div class="metric-value">Dense</div>
            <div class="metric-detail">
                MiniLM semantic embeddings
            </div>
        </div>
        """
    )

with m4:
    st.html(
        """
        <div class="metric-card">
            <div class="metric-label">Reranking</div>
            <div class="metric-value">Cross</div>
            <div class="metric-detail">
                MS-MARCO Cross-Encoder
            </div>
        </div>
        """
    )


# ============================================================
# PIPELINE VISUALIZATION
# ============================================================

st.markdown(
    '<div class="section-label">Inference Pipeline</div>',
    unsafe_allow_html=True,
)

st.html(
    """
    <div class="pipeline">
        <div class="pipeline-step active">
            <div class="pipeline-number">01</div>
            <div class="pipeline-name">Semantic Retrieval</div>
            <div class="pipeline-status">● READY</div>
        </div>

        <div class="pipeline-arrow">→</div>

        <div class="pipeline-step active">
            <div class="pipeline-number">02</div>
            <div class="pipeline-name">Evidence Reranking</div>
            <div class="pipeline-status">● READY</div>
        </div>

        <div class="pipeline-arrow">→</div>

        <div class="pipeline-step active">
            <div class="pipeline-number">03</div>
            <div class="pipeline-name">Local LLM Synthesis</div>
            <div class="pipeline-status">● READY</div>
        </div>
    </div>
    """
)


# ============================================================
# QUERY WORKSPACE
# ============================================================

st.markdown(
    '<div class="query-heading">What would you like to investigate?</div>',
    unsafe_allow_html=True,
)

st.markdown(
    '<div class="query-caption">'
    'Ask a question about the research literature in your corpus.'
    '</div>',
    unsafe_allow_html=True,
)

question = st.text_area(
    "Research question",
    value=st.session_state.last_question,
    height=130,
    placeholder=(
        "Example: What are the main approaches used "
        "to detect hallucinations in large language models?"
    ),
    label_visibility="collapsed",
)


# ============================================================
# EXAMPLE QUESTIONS
# ============================================================

st.markdown(
    '<div class="section-label">Research Prompts</div>',
    unsafe_allow_html=True,
)

examples = [
    "What are the main approaches used to detect hallucinations in large language models?",
    "What are the major challenges in hallucination detection?",
    "How can multiple signals be combined for token-level detection?",
    "What role does uncertainty play in hallucination detection?",
]

example_columns = st.columns(4)

for index, example in enumerate(examples):
    with example_columns[index]:
        if st.button(
            f"Example {index + 1}",
            use_container_width=True,
        ):
            st.session_state.last_question = example
            st.rerun()


# ============================================================
# ANALYZE BUTTON
# ============================================================

st.write("")

analyze_col, info_col = st.columns([1, 3])

with analyze_col:
    analyze = st.button(
        "✦  Analyze Research",
        type="primary",
        use_container_width=True,
    )

with info_col:
    st.caption(
        "Dense retrieval → Cross-Encoder reranking → local synthesis"
    )


# ============================================================
# ANALYSIS
# ============================================================

if analyze:
    question = question.strip()

    if not question:
        st.warning(
            "Enter a research question before starting the analysis."
        )
        st.stop()

    st.session_state.last_question = question

    if question not in st.session_state.history:
        st.session_state.history.append(question)

    status = st.status(
        "Running ScholarAgent pipeline...",
        expanded=True,
    )

    try:
        # --------------------------------------------------------
        # RETRIEVAL
        # --------------------------------------------------------

        status.write(
            "🔎 Searching the multi-paper evidence corpus..."
        )

        candidates = retrieve_candidates_cached(
            question,
            top_k=10,
        )

        status.write(
            f"✓ Retrieved {len(candidates)} candidate chunks."
        )

        # --------------------------------------------------------
        # RERANKING
        # --------------------------------------------------------

        status.write(
            "↗ Reranking evidence with the Cross-Encoder..."
        )

        results = rerank_candidates_cached(
            question,
            candidates,
            top_k=5,
        )

        status.write(
            "✓ Evidence reranking complete."
        )

        # --------------------------------------------------------
        # LOCAL LLM
        # --------------------------------------------------------

        status.write(
            "✦ Synthesizing an evidence-grounded answer..."
        )

        answer = generate_local_answer(question, results)

        status.write(
            "✓ Evidence-grounded synthesis complete."
        )

        status.update(
            label="Research analysis complete",
            state="complete",
            expanded=False,
        )

        st.session_state.last_results = results
        st.session_state.last_answer = answer

    except Exception as error:
        status.update(
            label="Analysis failed",
            state="error",
            expanded=True,
        )
        st.exception(error)
        st.stop()


# ============================================================
# RESULTS — SYNTHESIS
# ============================================================

if st.session_state.last_answer:
    st.markdown(
        '<div class="section-label">Research Synthesis</div>',
        unsafe_allow_html=True,
    )

    st.html(
        """
        <div class="answer-card">
            <div class="answer-badge">
                Local LLM • Evidence Grounded
            </div>
            <div class="answer-title">
                Synthesis
            </div>
        </div>
        """
    )

    st.markdown(st.session_state.last_answer)

    render_citation_map(
        st.session_state.last_answer,
        st.session_state.last_results,
    )


# ============================================================
# RESULTS — EVIDENCE TRACE
# ============================================================

if st.session_state.last_results:
    st.markdown(
        '<div class="section-label">Evidence Trace</div>',
        unsafe_allow_html=True,
    )

    st.caption(
        "Top-ranked evidence retrieved from the research corpus."
    )

    results = st.session_state.last_results

    for start in range(0, len(results), 2):
        row = results[start:start + 2]
        columns = st.columns(len(row))

        for index, result in enumerate(row):
            with columns[index]:
                title = result.get("title", "Untitled paper")
                paper_id = result.get("paper_id", "unknown")
                chunk_id = result.get("chunk_id", "unknown")
                retrieval_score = result.get("retrieval_score", 0)
                rerank_score = result.get("rerank_score", 0)
                arxiv_url = result.get("arxiv_url", "")

                st.html(
                    f"""
                    <div class="evidence-card">
                        <div class="evidence-rank">
                            EVIDENCE {start + index + 1}
                        </div>

                        <div class="evidence-title">
                            {title}
                        </div>

                        <div class="evidence-meta">
                            {paper_id}
                            &nbsp;•&nbsp;
                            Chunk {chunk_id}
                        </div>

                        <div class="score-pill">
                            Rerank: {rerank_score:.4f}
                        </div>
                    </div>
                    """
                )

                with st.expander("View evidence"):
                    st.write(result.get("text", ""))

                    st.caption(
                        f"Initial retrieval score: "
                        f"{retrieval_score:.4f}"
                    )

                    st.caption(
                        f"Cross-Encoder score: "
                        f"{rerank_score:.4f}"
                    )

                    if arxiv_url:
                        st.link_button(
                            "Open arXiv source ↗",
                            arxiv_url,
                            use_container_width=True,
                        )


# ============================================================
# TECHNICAL DETAILS
# ============================================================

if st.session_state.last_results:
    st.markdown(
        '<div class="section-label">System Architecture</div>',
        unsafe_allow_html=True,
    )

    tab1, tab2, tab3 = st.tabs(
        ["Architecture", "Retrieval", "Evidence"]
    )

    with tab1:
        st.markdown(
            """
            **ScholarAgent pipeline**

            `Research Question`
            → `Dense Retrieval`
            → `Cross-Encoder Reranking`
            → `Evidence Selection`
            → `Local LLM`
            → `Grounded Synthesis`

            The application uses a multi-stage retrieval architecture
            rather than sending the complete research corpus directly
            to the language model.
            """
        )

    with tab2:
        st.markdown(
            """
            **First stage — semantic retrieval**

            The question is embedded using:

            `sentence-transformers/all-MiniLM-L6-v2`

            The system retrieves the top **10 candidates** before
            Cross-Encoder reranking.
            """
        )

    with tab3:
        st.markdown(
            """
            **Second stage — evidence reranking**

            Retrieved candidates are scored using:

            `cross-encoder/ms-marco-MiniLM-L6-v2`

            The strongest **5 evidence chunks** are retained for
            downstream research synthesis.
            """
        )


# ============================================================
# FOOTER
# ============================================================

st.html(
    """
    <div class="footer">
        ScholarAgent · Multi-Paper Research Intelligence
        <br>
        Dense Retrieval · Cross-Encoder Reranking · Local LLM
    </div>
    """
)
