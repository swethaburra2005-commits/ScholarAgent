import json

from multi_reranker import (
    load_embeddings,
    retrieve_candidates,
    rerank_candidates
)


QUESTIONS_FILE = (
    "multipaper/evaluation_questions.json"
)


OUTPUT_FILE = (
    "multipaper/evaluation_results.json"
)


def load_questions():

    with open(
        QUESTIONS_FILE,
        "r",
        encoding="utf-8"
    ) as file:

        return json.load(file)


def main():

    print("\n" + "=" * 70)
    print("SCHOLAR AGENT — RETRIEVAL EVALUATION")
    print("=" * 70)

    # Load questions
    questions = load_questions()

    print(
        f"\nEvaluation questions: "
        f"{len(questions)}"
    )

    # Load embedding index
    data = load_embeddings()

    results = []

    for number, item in enumerate(
        questions,
        start=1
    ):

        question_id = item["id"]
        question = item["question"]

        print("\n" + "-" * 70)
        print(
            f"QUESTION {number}/{len(questions)}"
        )
        print(
            f"ID: {question_id}"
        )
        print(
            f"Question: {question}"
        )
        print("-" * 70)

        # Stage 1: retrieve
        candidates = retrieve_candidates(
            question,
            data,
            top_k=10
        )

        # Stage 2: rerank
        ranked = rerank_candidates(
            question,
            candidates,
            top_k=5
        )

        question_results = {
            "id": question_id,
            "question": question,
            "results": []
        }

        for rank, result in enumerate(
            ranked,
            start=1
        ):

            print(
                f"\nRank {rank}"
            )

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

            question_results["results"].append(
                {
                    "rank": rank,
                    "paper_id": result[
                        "paper_id"
                    ],
                    "chunk_id": result[
                        "chunk_id"
                    ],
                    "title": result[
                        "title"
                    ],
                    "arxiv_url": result[
                        "arxiv_url"
                    ],
                    "rerank_score": result[
                        "rerank_score"
                    ],
                    "text": result[
                        "text"
                    ]
                }
            )

        results.append(
            question_results
        )

    # Save results
    with open(
        OUTPUT_FILE,
        "w",
        encoding="utf-8"
    ) as file:

        json.dump(
            results,
            file,
            indent=2,
            ensure_ascii=False
        )

    print("\n" + "=" * 70)
    print("EVALUATION RUN COMPLETE")
    print("=" * 70)

    print(
        f"\nResults saved to:"
    )

    print(
        OUTPUT_FILE
    )


if __name__ == "__main__":
    main()