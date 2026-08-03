from pathlib import Path

import streamlit as st
from dotenv import load_dotenv

load_dotenv()

from embeddings import SongEmbeddingIndex
from rag_recommender import recommend_from_profile
from recommender import load_songs, recommend_songs

DATA_PATH = Path(__file__).resolve().parent.parent / "data" / "songs.csv"

st.set_page_config(page_title="WaveTune", page_icon="🎵")


@st.cache_data
def get_songs():
    return load_songs(str(DATA_PATH))


@st.cache_resource(show_spinner="Loading embedding model...")
def get_embedding_index():
    return SongEmbeddingIndex(get_songs())


def render_results(results, explanation_is_reason_list: bool):
    for i, (song, score, explanation) in enumerate(results, start=1):
        with st.container(border=True):
            st.subheader(f"#{i}: {song['title']}")
            st.write(f"**{song['artist']}** · {song['genre']} · {song['mood']} · score {score:.2f}")
            if explanation_is_reason_list:
                for reason in explanation.split(", "):
                    st.write(f"- {reason}")
            else:
                st.write(explanation)


def main():
    st.title("🎵 WaveTune")
    st.caption(
        "Set your song profile and get recommendations from either the "
        "rule-based scorer or the Claude-powered agent."
    )

    songs = get_songs()
    genres = sorted({s["genre"] for s in songs})
    moods = sorted({s["mood"] for s in songs})

    mode = st.radio(
        "Recommendation mode",
        ["Heuristic (rule-based)", "Agentic (RAG)"],
        horizontal=True,
        help=(
            "Heuristic scores songs by exact genre/mood/energy matches. "
            "Agentic retrieves by vibe from your profile and lets Claude "
            "rank candidates and explain the results."
        ),
    )

    k = st.slider("Number of recommendations", min_value=1, max_value=10, value=5)

    with st.form("profile_form"):
        favorite_genre = st.selectbox("Favorite genre", genres)
        favorite_mood = st.selectbox("Favorite mood", moods)
        target_energy = st.slider("Target energy", 0.0, 1.0, 0.7)
        likes_acoustic = st.checkbox("I like acoustic songs")

        submitted = st.form_submit_button("Get recommendations")

    if not submitted:
        return

    user_prefs = {
        "favorite_genre": favorite_genre,
        "favorite_mood": favorite_mood,
        "target_energy": target_energy,
        "likes_acoustic": likes_acoustic,
    }

    if mode == "Heuristic (rule-based)":
        results = recommend_songs(user_prefs, songs, k=k)
        render_results(results, explanation_is_reason_list=True)
        return

    try:
        with st.spinner("Retrieving candidates and asking Claude..."):
            index = get_embedding_index()
            results = recommend_from_profile(user_prefs, songs, index, final_k=k)
    except KeyError:
        st.error("ANTHROPIC_API_KEY is not set. Add it to your .env file.")
        return
    except Exception as e:
        st.error(f"Couldn't get agentic recommendations: {e}")
        return

    render_results(results, explanation_is_reason_list=False)


main()
