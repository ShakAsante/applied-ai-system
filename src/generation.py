"""Claude-powered generation: free-text preference parsing and explanations."""

import json
import os
from typing import Dict, List, Tuple

from anthropic import Anthropic
from anthropic.types import MessageParam

_MODEL = "claude-haiku-4-5"


def _client() -> Anthropic:
    return Anthropic(api_key=os.environ["ANTHROPIC_API_KEY"])


def _response_text(response) -> str:
    for block in response.content:
        if block.type == "text":
            return block.text
    raise ValueError("Response contained no text block")


def parse_query_to_preferences(query: str) -> Dict:
    """Turns a free-text mood/vibe description into structured user_prefs.

    Returns a dict shaped like the `user_prefs` argument expected by
    recommender.recommend_songs: favorite_genre, favorite_mood,
    target_energy, likes_acoustic.
    """
    system = (
        "Extract music preferences from the user's description. "
        "Reply with ONLY a JSON object with keys: "
        "favorite_genre (string), favorite_mood (string), "
        "target_energy (float 0.0-1.0), likes_acoustic (boolean). "
        "Make reasonable inferences when the user is not explicit."
    )

    response = _client().messages.create(
        model=_MODEL,
        max_tokens=200,
        system=system,
        messages=[{"role": "user", "content": query}],
    )

    text = _response_text(response).strip()
    text = text.removeprefix("```json").removeprefix("```").removesuffix("```").strip()

    return json.loads(text)


def generate_explanation(
    user_prefs: Dict, song: Dict, score: float, reasons: List[str]
) -> str:
    """Turns the scored feature breakdown into a natural-language explanation."""
    prompt = (
        f"A user with preferences {user_prefs} was recommended the song "
        f"'{song['title']}' by {song['artist']} (score {score:.2f}). "
        f"The scoring breakdown was: {'; '.join(reasons)}. "
        "Write one short, friendly sentence explaining why this song fits them."
    )

    response = _client().messages.create(
        model=_MODEL,
        max_tokens=120,
        messages=[{"role": "user", "content": prompt}],
    )

    return _response_text(response).strip()


def chat_reply(history: List[MessageParam], context: str, user_prefs: Dict) -> str:
    """Generates a conversational reply grounded in retrieved song context
    and anchored to the user's profile, so the conversation stays on-topic
    and consistent with their stated taste turn after turn.

    `history` is a list of {"role": "user"|"assistant", "content": str}
    ending with the latest user message. `context` is a block of retrieved
    song info (from the embedding index) to ground the answer in.
    """
    profile_line = (
        f"favorite genre: {user_prefs['favorite_genre']}; "
        f"favorite mood: {user_prefs['favorite_mood']}; "
        f"target energy: {user_prefs['target_energy']:.2f}; "
        f"acoustic preference: {'likes acoustic' if user_prefs['likes_acoustic'] else 'prefers non-acoustic'}."
    )

    system = (
        "You are WaveTune's music assistant. Stay focused on helping the "
        "user find and understand music from the catalog -- don't wander "
        "into unrelated topics. Answer using the catalog excerpts below "
        "when relevant, recommending specific songs by title and artist "
        "where it fits. Keep answers consistent with the user's stated "
        "profile unless they explicitly ask to explore outside it. Be "
        "concise and conversational. If nothing in the excerpts fits, "
        "say so.\n\n"
        f"User's profile -- {profile_line}\n\n"
        f"Relevant songs:\n{context}"
    )

    response = _client().messages.create(
        model=_MODEL,
        max_tokens=300,
        system=system,
        messages=history,
    )

    return _response_text(response).strip()
