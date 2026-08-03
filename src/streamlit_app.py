from pathlib import Path

import streamlit as st
from dotenv import load_dotenv

load_dotenv()

from embeddings import SongEmbeddingIndex
from rag_recommender import rag_chat, recommend_from_profile
from recommender import load_songs, load_songs_csv, recommend_songs

DATA_PATH = Path(__file__).resolve().parent.parent / "data" / "songs.csv"
DETAILED_DATA_PATH = Path(__file__).resolve().parent.parent / "data" / "songs_detailed.csv"

st.set_page_config(page_title="WaveTune", page_icon="🎵")


@st.cache_data
def get_songs():
    return load_songs(str(DATA_PATH))


@st.cache_resource(show_spinner="Loading embedding model...")
def get_embedding_index():
    return SongEmbeddingIndex(get_songs())


@st.cache_data
def get_detailed_songs():
    return load_songs_csv(str(DETAILED_DATA_PATH))


@st.cache_resource(show_spinner="Loading chat knowledge base...")
def get_chat_index():
    return SongEmbeddingIndex(get_detailed_songs())


def get_user_prefs() -> dict:
    """Reads the shared profile out of session_state.

    This is the single source of truth for "what does this user want" --
    both recommendation modes and the chat widget read from here, so a
    profile set once stays consistent everywhere in the app.
    """
    return {
        "favorite_genre": st.session_state.favorite_genre,
        "favorite_mood": st.session_state.favorite_mood,
        "target_energy": st.session_state.target_energy,
        "likes_acoustic": st.session_state.likes_acoustic,
    }


def render_profile_sidebar(genres, moods):
    with st.sidebar:
        st.header("Your profile")
        st.selectbox("Favorite genre", genres, key="favorite_genre")
        st.selectbox("Favorite mood", moods, key="favorite_mood")
        st.slider("Target energy", 0.0, 1.0, 0.7, key="target_energy")
        st.checkbox("I like acoustic songs", key="likes_acoustic")
        st.divider()
        st.slider("Number of recommendations", 1, 10, 5, key="result_count")


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


def render_chat_widget():
    st.markdown(
        """
        <style>
        /* A transform on any ancestor turns descendant position:fixed into
           position:relative-to-that-ancestor instead of the viewport, which
           is what was pulling the button off screen -- neutralize it. */
        [data-testid="stAppViewContainer"],
        [data-testid="stMain"],
        [data-testid="stMainBlockContainer"] {
            transform: none !important;
        }

        div[data-testid="stPopover"] {
            position: fixed !important;
            bottom: 24px !important;
            right: 24px !important;
            z-index: 999999 !important;
            width: auto !important;
        }
        div[data-testid="stPopover"] button[data-testid="stPopoverButton"] {
            border-radius: 50% !important;
            width: 56px !important;
            height: 56px !important;
            padding: 0 !important;
            font-size: 1.4rem !important;
            box-shadow: 0 2px 10px rgba(0, 0, 0, 0.35) !important;
        }
        div[data-testid="stPopoverBody"] {
            z-index: 999999 !important;
            width: 340px !important;
            max-height: 480px !important;
            overflow-y: auto !important;
        }
        </style>
        """,
        unsafe_allow_html=True,
    )

    st.session_state.setdefault("chat_messages", [])

    with st.popover("💬"):
        st.markdown("**Ask WaveTune**")

        user_prefs = get_user_prefs()
        acoustic = "acoustic" if user_prefs["likes_acoustic"] else "non-acoustic"
        st.caption(
            f"Grounded in your profile: {user_prefs['favorite_genre']} · "
            f"{user_prefs['favorite_mood']} · energy {user_prefs['target_energy']:.2f} · "
            f"{acoustic}"
        )

        for msg in st.session_state.chat_messages:
            with st.chat_message(msg["role"]):
                st.write(msg["content"])

        prompt = st.chat_input("Ask about music...")

        if prompt:
            st.session_state.chat_messages.append({"role": "user", "content": prompt})

            try:
                with st.spinner("Thinking..."):
                    index = get_chat_index()
                    reply = rag_chat(st.session_state.chat_messages, index, user_prefs)
            except KeyError:
                reply = "ANTHROPIC_API_KEY is not set. Add it to your .env file."
            except Exception as e:
                reply = f"Something went wrong: {e}"

            st.session_state.chat_messages.append({"role": "assistant", "content": reply})
            st.rerun()


def main():
    st.title("🎵 WaveTune")
    st.caption(
        "Set your profile in the sidebar, pick a recommendation mode, and go."
    )

    songs = get_songs()
    genres = sorted({s["genre"] for s in songs})
    moods = sorted({s["mood"] for s in songs})

    render_profile_sidebar(genres, moods)

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

    get_recommendations = st.button("Get recommendations", type="primary")

    st.session_state.setdefault("results", None)
    st.session_state.setdefault("results_is_reason_list", True)

    if get_recommendations:
        user_prefs = get_user_prefs()
        k = st.session_state.result_count

        if mode == "Heuristic (rule-based)":
            st.session_state.results = recommend_songs(user_prefs, songs, k=k)
            st.session_state.results_is_reason_list = True
        else:
            try:
                with st.spinner("Retrieving candidates and asking Claude..."):
                    index = get_embedding_index()
                    st.session_state.results = recommend_from_profile(
                        user_prefs, songs, index, final_k=k
                    )
                st.session_state.results_is_reason_list = False
            except KeyError:
                st.error("ANTHROPIC_API_KEY is not set. Add it to your .env file.")
            except Exception as e:
                st.error(f"Couldn't get agentic recommendations: {e}")

    if st.session_state.results:
        render_results(st.session_state.results, st.session_state.results_is_reason_list)


main()
render_chat_widget()
