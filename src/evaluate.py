"""Runs the system on a fixed set of predefined inputs and prints a summary:
PASS/FAIL + confidence per case for the heuristic recommender, the agentic
(RAG) recommender, and the "Ask WaveTune" chat -- plus the existing aggregate
accuracy/diversity/bias comparison between the two recommenders.

Exercises the same methods the Streamlit app actually calls in production:
recommend_songs (heuristic), recommend_from_profile (agentic -- the shared
structured profile, not the older free-text recommend_from_query), and
rag_chat (the guardrailed chat widget).

Usage: python src/evaluate.py
"""

from dotenv import load_dotenv

load_dotenv()

from embeddings import SongEmbeddingIndex
from evaluation import (
    EVAL_CASES,
    evaluate_chat_case,
    evaluate_recommender,
    print_chat_summary,
    print_comparison,
    print_summary,
)
from rag_recommender import rag_chat, recommend_from_profile
from recommender import load_songs, load_songs_csv, recommend_songs

K = 5

# One chat prompt per eval case, phrased by vibe only (no genre/mood/energy
# stated outright) so a passing case proves the reply is actually grounded
# in the shared profile + retrieval, not just repeating the prompt back.
CHAT_CASES = [
    {"name": "high_energy_pop", "chat_prompt": "I need something to pump me up before the gym"},
    {"name": "chill_lofi", "chat_prompt": "I need something to help me focus while studying tonight"},
    {"name": "intense_rock", "chat_prompt": "Give me something loud and aggressive"},
    {"name": "peaceful_folk", "chat_prompt": "I want something calm and gentle to unwind to"},
    {"name": "moody_electronic", "chat_prompt": "I want something atmospheric and moody"},
]


def _describe_profile(profile: dict) -> str:
    acoustic = "acoustic" if profile["likes_acoustic"] else "non-acoustic"
    return (
        f"{profile['favorite_genre']} music with a {profile['favorite_mood']} mood, "
        f"energy around {profile['target_energy']:.2f}, {acoustic}."
    )


def main() -> None:
    songs = load_songs("data/songs.csv")
    index = SongEmbeddingIndex(songs)

    def heuristic_recommend(case) -> list:
        scored = recommend_songs(case["profile"], songs, k=K)
        return [song for song, _, _ in scored]

    def agentic_recommend(case) -> list:
        results = recommend_from_profile(case["profile"], songs, index, final_k=K)
        return [song for song, _, _ in results]

    heuristic_result = evaluate_recommender(
        "Heuristic", heuristic_recommend, EVAL_CASES, songs
    )
    agentic_result = evaluate_recommender(
        "Agentic (RAG)", agentic_recommend, EVAL_CASES, songs
    )

    print_summary(heuristic_result, agentic_result)
    print_comparison(heuristic_result, agentic_result)

    detailed_songs = load_songs_csv("data/songs_detailed.csv")
    chat_index = SongEmbeddingIndex(detailed_songs)
    profiles_by_name = {case["name"]: case["profile"] for case in EVAL_CASES}

    chat_results = []
    for chat_case in CHAT_CASES:
        profile = profiles_by_name[chat_case["name"]]
        prompt = chat_case["chat_prompt"]

        retrieval_query = f"{prompt} ({_describe_profile(profile)})"
        candidates = chat_index.query(retrieval_query, k=5)

        history = [{"role": "user", "content": prompt}]
        reply = rag_chat(history, chat_index, profile)

        chat_results.append(
            evaluate_chat_case(chat_case["name"], prompt, reply, candidates)
        )

    print_chat_summary(chat_results)


if __name__ == "__main__":
    main()
