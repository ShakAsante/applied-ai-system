from src.evaluation import (
    catalog_coverage,
    catalog_genre_distribution,
    energy_mae,
    evaluate_recommender,
    genre_diversity,
    genre_distribution_skew,
    genre_match_rate,
    mood_match_rate,
    repeat_concentration,
)

SONGS = [
    {"id": 1, "genre": "pop", "mood": "happy", "energy": 0.8},
    {"id": 2, "genre": "pop", "mood": "happy", "energy": 0.7},
    {"id": 3, "genre": "lofi", "mood": "chill", "energy": 0.3},
    {"id": 4, "genre": "rock", "mood": "intense", "energy": 0.9},
]


def test_genre_match_rate():
    assert genre_match_rate(SONGS[:2], "pop") == 1.0
    assert genre_match_rate(SONGS, "pop") == 0.5
    assert genre_match_rate([], "pop") == 0.0


def test_mood_match_rate():
    assert mood_match_rate(SONGS, "chill") == 0.25


def test_energy_mae():
    recs = [{"energy": 0.8}, {"energy": 0.6}]
    assert abs(energy_mae(recs, 0.7) - 0.1) < 1e-9
    assert energy_mae([], 0.7) == 0.0


def test_genre_diversity():
    assert genre_diversity(SONGS[:2]) == 0.5  # both pop -> 1 unique / 2
    assert genre_diversity(SONGS[1:]) == 1.0  # lofi, rock -> all unique


def test_catalog_genre_distribution():
    dist = catalog_genre_distribution(SONGS)
    assert dist["pop"] == 0.5
    assert dist["lofi"] == 0.25
    assert dist["rock"] == 0.25


def test_genre_distribution_skew_matches_catalog_is_zero():
    # Recommending exactly the catalog mix should show no skew.
    skew, deviation = genre_distribution_skew(SONGS, SONGS)
    assert skew == 0.0
    assert all(abs(d) < 1e-9 for d in deviation.values())


def test_genre_distribution_skew_detects_overrepresentation():
    all_pop = [SONGS[0], SONGS[0], SONGS[0]]
    skew, deviation = genre_distribution_skew(all_pop, SONGS)
    assert skew > 0
    assert deviation["pop"] > 0
    assert deviation["rock"] < 0  # never recommended despite being in catalog


def test_catalog_coverage():
    assert catalog_coverage(SONGS, SONGS) == 1.0
    assert catalog_coverage(SONGS[:1], SONGS) == 0.25
    assert catalog_coverage([], SONGS) == 0.0


def test_repeat_concentration():
    same_song_everywhere = [SONGS[0]] * 4
    assert repeat_concentration(same_song_everywhere) == 1.0

    one_each = SONGS
    assert repeat_concentration(one_each) == 0.25


def test_evaluate_recommender_end_to_end():
    cases = [
        {
            "name": "pop_case",
            "profile": {
                "favorite_genre": "pop",
                "favorite_mood": "happy",
                "target_energy": 0.75,
                "likes_acoustic": False,
            },
        }
    ]

    def always_recommend_pop(case):
        return SONGS[:2]

    result = evaluate_recommender("test-system", always_recommend_pop, cases, SONGS)

    assert result.avg_genre_match_rate == 1.0
    assert result.avg_mood_match_rate == 1.0
    assert 0 <= result.quality_score <= 1
    assert result.catalog_coverage == 0.5
