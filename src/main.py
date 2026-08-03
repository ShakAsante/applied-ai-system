from dotenv import load_dotenv

from embeddings import SongEmbeddingIndex
from rag_recommender import recommend_from_query
from recommender import load_songs, recommend_songs

load_dotenv()


def run_rag_demo(songs) -> None:
    index = SongEmbeddingIndex(songs)
    query = "something moody and atmospheric to code to late at night"

    results = recommend_from_query(query, songs, index)

    print("\n" + "=" * 40)
    print(f"       RAG RECOMMENDATIONS FOR: \"{query}\"")
    print("=" * 40)

    for index_num, (song, score, explanation) in enumerate(results, start=1):
        print(f"\n#{index_num}: {song['title']} (score {score:.2f})")
        print(explanation)

    print("\n" + "=" * 40)


def main() -> None:
    songs = load_songs("data/songs.csv")

    # High-Energy Pop User
    user_prefs = {
        "favorite_genre": "pop",
        "favorite_mood": "happy",
        "target_energy": 0.9,
        "likes_acoustic": False
    }

    # Other test profiles:
    # user_prefs = {
    #     "favorite_genre": "lofi",
    #     "favorite_mood": "chill",
    #     "target_energy": 0.35,
    #     "likes_acoustic": True
    # }

    # user_prefs = {
    #     "favorite_genre": "rock",
    #     "favorite_mood": "intense",
    #     "target_energy": 0.95,
    #     "likes_acoustic": False
    # }

    recommendations = recommend_songs(user_prefs, songs, k=5)

    print("\n" + "=" * 40)
    print("       TOP MUSIC RECOMMENDATIONS")
    print("=" * 40)

    for index, (song, score, explanation) in enumerate(recommendations, start=1):
        print(f"\n#{index}: {song['title']}")
        print(f"Artist: {song['artist']}")
        print(f"Genre: {song['genre']} | Mood: {song['mood']}")
        print(f"Score: {score:.2f}")
        print("Reasons:")

        for reason in explanation.split(", "):
            print(f"  ✓ {reason}")

    print("\n" + "=" * 40)

    run_rag_demo(songs)


if __name__ == "__main__":
    main()