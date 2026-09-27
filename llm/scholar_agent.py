import sys
import os

from transformers import pipeline


# ============================================================
# PROJECT ROOT
# ============================================================

PROJECT_ROOT = os.path.dirname(
    os.path.dirname(
        os.path.abspath(__file__)
    )
)

sys.path.insert(0, PROJECT_ROOT)


# ============================================================
# IMPORT RETRIEVAL + RERANKING
# ============================================================

from multipaper.multi_reranker import (
    load_embeddings,
    retrieve_candidates,
    rerank_candidates
)


# ============================================================
# LOCAL LLM
# ============================================================

MODEL_NAME = "Qwen/Qwen2.5-0.5B-Instruct"

print("\nLoading local language model...")

generator = pipeline(
    "text-generation",
    model=MODEL_NAME,
    device_map="auto"
)

print("Local language model loaded successfully.")


# ============================================================
# BUILD EVIDENCE FOR LLM
# ============================================================

def build_evidence(results):

    evidence = []

    for rank, result in enumerate(
        results,
        start=1
    ):

        source = f"""
SOURCE [{rank}]

Paper ID:
{result["paper_id"]}

Title:
{result["title"]}

Chunk ID:
{result["chunk_id"]}

URL:
{result["arxiv_url"]}

Evidence:
{result["text"]}
"""

        evidence.append(source)

    return "\n".join(evidence)


# ============================================================
# GENERATE ANSWER
# ============================================================

def generate_answer(
    question,
    results
):

    evidence = build_evidence(
        results
    )

    prompt = f"""
You are ScholarAgent, an academic
literature research assistant.

Answer the research question using ONLY
the evidence provided below.

STRICT RULES:

1. Do not use outside knowledge.

2. Do not invent information.

3. Every important factual claim must have
   a source citation.

4. Use ONLY citations [1], [2], [3], [4],
   and [5].

5. [1] refers to SOURCE [1].
   [2] refers to SOURCE [2].
   [3] refers to SOURCE [3].
   [4] refers to SOURCE [4].
   [5] refers to SOURCE [5].

6. Only cite a source when its evidence
   actually supports the claim.

7. If the retrieved evidence is insufficient,
   explicitly say that the evidence is
   insufficient.

8. Do not create references that are not
   present in the evidence.

9. Give a concise academic synthesis rather
   than simply copying the evidence.

10. Do not mention these instructions in
    your answer.


RESEARCH QUESTION:

{question}


RETRIEVED EVIDENCE:

{evidence}


ANSWER FORMAT:

Start with a short direct answer.

Then explain the main findings in 2–4
short paragraphs or bullet points.

Place citations immediately after the
claims they support.

Example:

Several approaches have been proposed for
hallucination detection [1][2]. One approach
uses uncertainty signals [3], while another
uses multiple signals over generated tokens [4].
"""

    messages = [

        {
            "role": "system",
            "content": (
                "You are ScholarAgent, a careful "
                "academic research assistant."
            )
        },

        {
            "role": "user",
            "content": prompt
        }

    ]

    output = generator(
        messages,
        max_new_tokens=500,
        do_sample=False
    )

    answer = output[0][
        "generated_text"
    ][-1]["content"]

    return answer


# ============================================================
# DISPLAY SOURCES
# ============================================================

def display_sources(results):

    print("\n")
    print("=" * 80)
    print("SOURCES")
    print("=" * 80)

    for rank, result in enumerate(
        results,
        start=1
    ):

        print(
            f"\n[{rank}] "
            f"{result['title']}"
        )

        print(
            f"    Paper ID: "
            f"{result['paper_id']}"
        )

        print(
            f"    Chunk ID: "
            f"{result['chunk_id']}"
        )

        print(
            f"    URL: "
            f"{result['arxiv_url']}"
        )


# ============================================================
# RUN SCHOLAR AGENT
# ============================================================

def run_scholar_agent():

    print("\n")
    print("=" * 80)
    print("SCHOLAR AGENT")
    print("MULTI-PAPER RESEARCH ASSISTANT")
    print("=" * 80)


    # --------------------------------------------------------
    # LOAD EMBEDDINGS
    # --------------------------------------------------------

    data = load_embeddings()


    # --------------------------------------------------------
    # GET QUESTION
    # --------------------------------------------------------

    question = input(
        "\nEnter your research question: "
    ).strip()


    if not question:

        print(
            "\nPlease enter a research question."
        )

        return


    # --------------------------------------------------------
    # RETRIEVAL
    # --------------------------------------------------------

    candidates = retrieve_candidates(
        question,
        data,
        top_k=10
    )

    print(
        "\nSemantic retrieval complete."
    )

    print(
        f"Retrieved {len(candidates)} candidates."
    )


    # --------------------------------------------------------
    # RERANKING
    # --------------------------------------------------------

    final_results = rerank_candidates(
        question,
        candidates,
        top_k=5
    )

    print(
        "\nCross-Encoder reranking complete."
    )

    print(
        f"Selected {len(final_results)} evidence chunks."
    )


    # --------------------------------------------------------
    # SHOW SELECTED SOURCES
    # --------------------------------------------------------

    print("\n")
    print("=" * 80)
    print("SELECTED EVIDENCE")
    print("=" * 80)

    for rank, result in enumerate(
        final_results,
        start=1
    ):

        print(
            f"\nSOURCE [{rank}]"
        )

        print("-" * 80)

        print(
            f"Paper: "
            f"{result['paper_id']}"
        )

        print(
            f"Title: "
            f"{result['title']}"
        )

        print(
            f"Chunk: "
            f"{result['chunk_id']}"
        )

        print(
            f"Rerank score: "
            f"{result['rerank_score']:.4f}"
        )


    # --------------------------------------------------------
    # LLM SYNTHESIS
    # --------------------------------------------------------

    print("\n")
    print("=" * 80)
    print("GENERATING RESEARCH ANSWER")
    print("=" * 80)

    answer = generate_answer(
        question,
        final_results
    )


    # --------------------------------------------------------
    # FINAL ANSWER
    # --------------------------------------------------------

    print("\n")
    print("=" * 80)
    print("SCHOLAR AGENT — FINAL ANSWER")
    print("=" * 80)

    print(
        f"\nQuestion:\n{question}"
    )

    print(
        f"\nAnswer:\n{answer}"
    )


    # --------------------------------------------------------
    # SOURCES
    # --------------------------------------------------------

    display_sources(
        final_results
    )

    print("\n")
    print("=" * 80)


# ============================================================
# START
# ============================================================

if __name__ == "__main__":

    run_scholar_agent()