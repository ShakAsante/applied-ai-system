"""End-to-end RAG flow: free-text query -> retrieval -> scoring -> generation."""

from typing import Dict, List, Tuple

from embeddings import SongEmbeddingIndex
from generation import generate_explanation, parse_query_to_preferences
from recommender import score_song


def recommend_from_query(
    query: str,
    songs: List[Dict],
    index: SongEmbeddingIndex,
    retrieve_k: int = 15,
    final_k: int = 5,
) -> List[Tuple[Dict, float, str]]:
    """Recommends songs from a free-text query using retrieval + scoring + generation.

    1. Retrieve: narrow the catalog to songs whose vibe matches the query.
    2. Score: rank the candidates with the existing feature-similarity scorer,
       using preferences Claude extracts from the query.
    3. Generate: replace the templated explanation with a natural-language one.
    """
    candidates = index.query(query, k=retrieve_k)
    user_prefs = parse_query_to_preferences(query)

    scored = []
    for song in candidates:
        score, reasons = score_song(user_prefs, song)
        scored.append((song, score, reasons))

    scored.sort(key=lambda item: item[1], reverse=True)
    top = scored[:final_k]

    return [
        (song, score, generate_explanation(user_prefs, song, score, reasons))
        for song, score, reasons in top
    ]


def _describe_profile(user_prefs: Dict) -> str:
    acoustic = "acoustic" if user_prefs["likes_acoustic"] else "non-acoustic"
    return (
        f"{user_prefs['favorite_genre']} music with a {user_prefs['favorite_mood']} mood, "
        f"energy around {user_prefs['target_energy']:.2f}, {acoustic}."
    )


def recommend_from_profile(
    user_prefs: Dict,
    songs: List[Dict],
    index: SongEmbeddingIndex,
    retrieve_k: int = 15,
    final_k: int = 5,
) -> List[Tuple[Dict, float, str]]:
    """Recommends songs from a structured profile using retrieval + scoring + generation.

    Same pipeline as recommend_from_query, but the preferences are already
    structured (from the same form the heuristic recommender uses), so there's
    no free text to parse -- it's turned into a query only for retrieval.
    """
    query = _describe_profile(user_prefs)
    candidates = index.query(query, k=retrieve_k)

    scored = []
    for song in candidates:
        score, reasons = score_song(user_prefs, song)
        scored.append((song, score, reasons))

    scored.sort(key=lambda item: item[1], reverse=True)
    top = scored[:final_k]

    return [
        (song, score, generate_explanation(user_prefs, song, score, reasons))
        for song, score, reasons in top
    ]
