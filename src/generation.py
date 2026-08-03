"""Claude-powered generation: free-text preference parsing and explanations."""

import json
import os
from typing import Dict, List, Tuple

from anthropic import Anthropic

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
