"""End-to-end RAG flow: free-text query -> retrieval -> scoring -> generation."""

from typing import Dict, List, Tuple

from anthropic.types import MessageParam

from embeddings import SongEmbeddingIndex
from generation import chat_reply, generate_explanation, parse_query_to_preferences
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


def _format_context(songs: List[Dict]) -> str:
    lines = []
    for s in songs:
        line = f"- {s['title']} by {s['artist']} ({s['genre']}, {s['mood']} mood)"
        if s.get("description"):
            line += f": {s['description']}"
        if s.get("themes"):
            line += f" Themes: {s['themes']}."
        if s.get("best_for"):
            line += f" Best for: {s['best_for']}."
        lines.append(line)
    return "\n".join(lines)


def rag_chat(
    history: List[MessageParam],
    index: SongEmbeddingIndex,
    user_prefs: Dict,
    retrieve_k: int = 5,
) -> str:
    """Answers a chat message grounded in retrieved song context (RAG).

    `history` must end with the latest user message. `user_prefs` is the
    shared profile from the recommendation form -- it's blended into the
    retrieval query (so results lean toward the user's taste) and passed
    to Claude as a guardrail, keeping the conversation anchored to what
    the user said they wanted instead of drifting.
    """
    latest_message = history[-1]["content"]
    retrieval_query = f"{latest_message} ({_describe_profile(user_prefs)})"
    candidates = index.query(retrieval_query, k=retrieve_k)
    context = _format_context(candidates)

    return chat_reply(history[-16:], context, user_prefs)
