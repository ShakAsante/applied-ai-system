"""Runs the heuristic and agentic recommenders through the same eval cases
and prints a side-by-side accuracy/diversity/bias comparison.

Usage: python src/evaluate.py
"""

from dotenv import load_dotenv

load_dotenv()

from embeddings import SongEmbeddingIndex
from evaluation import EVAL_CASES, evaluate_recommender, print_comparison
from rag_recommender import recommend_from_query
from recommender import load_songs, recommend_songs

K = 5


def main() -> None:
    songs = load_songs("data/songs.csv")
    index = SongEmbeddingIndex(songs)

    def heuristic_recommend(case) -> list:
        scored = recommend_songs(case["profile"], songs, k=K)
        return [song for song, _, _ in scored]

    def agentic_recommend(case) -> list:
        results = recommend_from_query(case["query"], songs, index, final_k=K)
        return [song for song, _, _ in results]

    heuristic_result = evaluate_recommender(
        "Heuristic", heuristic_recommend, EVAL_CASES, songs
    )
    agentic_result = evaluate_recommender(
        "Agentic (RAG)", agentic_recommend, EVAL_CASES, songs
    )

    print_comparison(heuristic_result, agentic_result)


if __name__ == "__main__":
    main()
