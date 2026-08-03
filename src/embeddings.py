"""Local semantic retrieval over the song catalog.

Uses sentence-transformers to embed a natural-language description of each
song, so users can query by vibe ("something moody to code to at night")
instead of only exact genre/mood fields.
"""

from typing import List, Dict
import numpy as np
from sentence_transformers import SentenceTransformer

_MODEL_NAME = "all-MiniLM-L6-v2"

def describe_song(song: Dict) -> str:
    """Builds a natural-language description of a song from its fields.

    Uses the rich descriptive fields (description, themes, tags, best_for)
    when available -- as in data/songs_detailed.csv -- since they carry far
    more retrievable meaning than the numeric features alone. Falls back to
    a synthesized description for catalogs that only have the basic fields.
    """
    if "description" in song:
        parts = [song["description"]]

        moods = song["mood"]
        if song.get("secondary_moods"):
            moods += f", {song['secondary_moods']}"
        parts.append(f"Genre: {song['genre']}. Mood: {moods}.")

        for field, label in [
            ("themes", "Themes"),
            ("tags", "Tags"),
            ("best_for", "Best for"),
        ]:
            if song.get(field):
                parts.append(f"{label}: {song[field]}.")

        return " ".join(parts)

    return (
        f"{song['title']} by {song['artist']} is a {song['mood']} {song['genre']} "
        f"track with {song['energy']:.2f} energy, {song['tempo_bpm']:.0f} BPM, "
        f"{song['valence']:.2f} valence, {song['danceability']:.2f} danceability, "
        f"and {song['acousticness']:.2f} acousticness."
    )


class SongEmbeddingIndex:
    """Embeds song descriptions and retrieves the closest matches to a query."""

    def __init__(self, songs: List[Dict], model_name: str = _MODEL_NAME):
        self.songs = songs
        self.model = SentenceTransformer(model_name)
        descriptions = [describe_song(song) for song in songs]
        embeddings = self.model.encode(descriptions, normalize_embeddings=True)
        self.embeddings = np.asarray(embeddings)

    def query(self, text: str, k: int = 10) -> List[Dict]:
        """Returns the top k songs whose description is closest to `text`."""
        query_embedding = self.model.encode([text], normalize_embeddings=True)[0]
        similarities = self.embeddings @ query_embedding
        top_indices = np.argsort(similarities)[::-1][:k]
        return [self.songs[i] for i in top_indices]
