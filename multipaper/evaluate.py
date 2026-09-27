import json

from multi_reranker import (
    load_embeddings,
    retrieve_candidates,
    rerank_candidates
)


# ============================================================
# FILES
# ============================================================

QUESTIONS_FILE = (
    "multipaper/evaluation_questions.json"
)

RESULTS_FILE = (
    "multipaper/evaluation_results.json"
)


# ============================================================
# LOAD QUESTIONS
# ============================================================

def load_questions():

    with open(
        QUESTIONS_FILE,
        "r",
        encoding="utf-8"
    ) as file:

        questions = json.load(file)

    return questions


# ============================================================
# SAVE RESULTS
# ============================================================

def save_results(results):

    with open(
        RESULTS_FILE,
        "w",
        encoding="utf-8"
    ) as file:

        json.dump(
            results,
            file,
            indent=4,
            ensure_ascii=False
        )

    print(
        f"\nResults saved to: {RESULTS_FILE}"
    )


# ============================================================
# EVALUATE
# ============================================================

def evaluate():

    print("\n")
    print("=" * 80)
    print("SCHOLAR AGENT — RETRIEVAL EVALUATION")
    print("=" * 80)

    questions = load_questions()

    data = load_embeddings()

    print(
        f"\nEvaluation questions: "
        f"{len(questions)}"
    )

    print(
        f"Corpus chunks: "
        f"{len(data)}"
    )

    # --------------------------------------------------------
    # Store ALL results here
    # --------------------------------------------------------

    evaluation_results = {}

    # --------------------------------------------------------
    # Evaluate every question
    # --------------------------------------------------------

    for question_data in questions:

        question_id = question_data["id"]

        question = question_data["question"]

        print("\n")
        print("=" * 80)

        print(
            f"{question_id.upper()}: "
            f"{question}"
        )

        print("=" * 80)

        # ----------------------------------------------------
        # FIRST-STAGE RETRIEVAL
        # ----------------------------------------------------

        candidates = retrieve_candidates(
            question,
            data,
            top_k=10
        )

        print(
            "\nTop 10 semantic retrieval results:"
        )

        for rank, candidate in enumerate(
            candidates,
            start=1
        ):

            print(
                f"{rank}. "
                f"{candidate['paper_id']} "
                f"| chunk {candidate['chunk_id']} "
                f"| score "
                f"{candidate['retrieval_score']:.4f}"
            )

        # ----------------------------------------------------
        # CROSS-ENCODER RERANKING
        # ----------------------------------------------------

        reranked = rerank_candidates(
            question,
            candidates,
            top_k=5
        )

        print(
            "\nTop 5 after Cross-Encoder reranking:"
        )

        for rank, result in enumerate(
            reranked,
            start=1
        ):

            print(
                f"{rank}. "
                f"{result['paper_id']} "
                f"| chunk {result['chunk_id']} "
                f"| score "
                f"{result['rerank_score']:.4f}"
            )

        # ----------------------------------------------------
        # SAVE RESULTS FOR THIS QUESTION
        # ----------------------------------------------------

        evaluation_results[question_id] = {

            "question": question,

            "retrieval_top10": [],

            "reranked_top5": []
        }

        # Save top 10 retrieval results

        for rank, candidate in enumerate(
            candidates,
            start=1
        ):

            evaluation_results[
                question_id
            ][
                "retrieval_top10"
            ].append({

                "rank": rank,

                "paper_id":
                    candidate["paper_id"],

                "chunk_id":
                    candidate["chunk_id"],

                "title":
                    candidate["title"],

                "arxiv_url":
                    candidate["arxiv_url"],

                "retrieval_score":
                    candidate["retrieval_score"]
            })

        # Save top 5 reranked results

        for rank, result in enumerate(
            reranked,
            start=1
        ):

            evaluation_results[
                question_id
            ][
                "reranked_top5"
            ].append({

                "rank": rank,

                "paper_id":
                    result["paper_id"],

                "chunk_id":
                    result["chunk_id"],

                "title":
                    result["title"],

                "arxiv_url":
                    result["arxiv_url"],

                "retrieval_score":
                    result["retrieval_score"],

                "rerank_score":
                    result["rerank_score"]
            })

    # --------------------------------------------------------
    # SAVE EVERYTHING
    # --------------------------------------------------------

    save_results(
        evaluation_results
    )

    print("\n")
    print("=" * 80)
    print("EVALUATION RUN COMPLETE")
    print("=" * 80)

    print(
        f"\nQuestions processed: "
        f"{len(evaluation_results)}"
    )


# ============================================================
# RUN
# ============================================================

if __name__ == "__main__":

    evaluate()