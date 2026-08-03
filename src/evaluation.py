"""Evaluation harness for comparing the heuristic and agentic (RAG) recommenders.

Runs both systems against the same set of user intents (a structured profile
plus an equivalent free-text query per case) and scores each on:
- how well results match the stated preferences (accuracy)
- how varied the results are (diversity)
- whether the system, in aggregate, skews toward certain genres or keeps
  recommending the same handful of songs (bias)
"""

from collections import Counter
from dataclasses import dataclass, field
from typing import Callable, Dict, List, Tuple

# Each case pairs a structured profile (for scoring against) with an
# equivalent free-text query (for the agentic system's retrieval + parsing),
# spanning genres/moods/energy levels that actually exist in the catalog.
EVAL_CASES: List[Dict] = [
    {
        "name": "high_energy_pop",
        "profile": {
            "favorite_genre": "pop",
            "favorite_mood": "happy",
            "target_energy": 0.9,
            "likes_acoustic": False,
        },
        "query": "upbeat happy pop music, high energy, not acoustic",
    },
    {
        "name": "chill_lofi",
        "profile": {
            "favorite_genre": "lofi",
            "favorite_mood": "chill",
            "target_energy": 0.35,
            "likes_acoustic": True,
        },
        "query": "chill lofi music for relaxing, low energy, acoustic feel",
    },
    {
        "name": "intense_rock",
        "profile": {
            "favorite_genre": "rock",
            "favorite_mood": "intense",
            "target_energy": 0.95,
            "likes_acoustic": False,
        },
        "query": "intense high-energy rock, definitely not acoustic",
    },
    {
        "name": "peaceful_folk",
        "profile": {
            "favorite_genre": "folk",
            "favorite_mood": "peaceful",
            "target_energy": 0.3,
            "likes_acoustic": True,
        },
        "query": "peaceful acoustic folk music, low energy",
    },
    {
        "name": "moody_electronic",
        "profile": {
            "favorite_genre": "electronic",
            "favorite_mood": "dramatic",
            "target_energy": 0.8,
            "likes_acoustic": False,
        },
        "query": "dramatic, moody electronic music, high energy, not acoustic",
    },
]


# --- Per-case accuracy/diversity metrics -----------------------------------


def genre_match_rate(recs: List[Dict], favorite_genre: str) -> float:
    if not recs:
        return 0.0
    return sum(1 for s in recs if s["genre"] == favorite_genre) / len(recs)


def mood_match_rate(recs: List[Dict], favorite_mood: str) -> float:
    if not recs:
        return 0.0
    return sum(1 for s in recs if s["mood"] == favorite_mood) / len(recs)


def energy_mae(recs: List[Dict], target_energy: float) -> float:
    if not recs:
        return 0.0
    return sum(abs(s["energy"] - target_energy) for s in recs) / len(recs)


def genre_diversity(recs: List[Dict]) -> float:
    """Fraction of recommended slots that are a distinct genre (1.0 = no repeats)."""
    if not recs:
        return 0.0
    return len({s["genre"] for s in recs}) / len(recs)


# --- Aggregate bias metrics --------------------------------------------------


def catalog_genre_distribution(songs: List[Dict]) -> Dict[str, float]:
    counts = Counter(s["genre"] for s in songs)
    total = len(songs)
    return {genre: count / total for genre, count in counts.items()}


def genre_distribution_skew(
    all_recs: List[Dict], songs: List[Dict]
) -> Tuple[float, Dict[str, float]]:
    """Total variation distance between the recommended-genre mix and the
    catalog's genre mix, plus a per-genre (recommended share - catalog share)
    breakdown. 0.0 = recommendations mirror the catalog; higher = the system
    is steering toward (positive) or away from (negative) certain genres
    regardless of catalog availability.
    """
    catalog_dist = catalog_genre_distribution(songs)

    if not all_recs:
        return 0.0, {}

    rec_counts = Counter(s["genre"] for s in all_recs)
    total = len(all_recs)
    rec_dist = {genre: count / total for genre, count in rec_counts.items()}

    all_genres = set(catalog_dist) | set(rec_dist)
    deviation = {
        genre: rec_dist.get(genre, 0.0) - catalog_dist.get(genre, 0.0)
        for genre in all_genres
    }
    tvd = sum(abs(d) for d in deviation.values()) / 2

    return tvd, deviation


def catalog_coverage(all_recs: List[Dict], songs: List[Dict]) -> float:
    """Fraction of the catalog that ever gets recommended across all cases.
    Low coverage means the system leans on a small favored subset (filter bubble).
    """
    if not songs:
        return 0.0
    unique_recommended = {s["id"] for s in all_recs}
    return len(unique_recommended) / len(songs)


def repeat_concentration(all_recs: List[Dict]) -> float:
    """Share of all recommendation slots taken by the single most-recommended
    song. Higher = that song keeps winning regardless of the user intent.
    """
    if not all_recs:
        return 0.0
    counts = Counter(s["id"] for s in all_recs)
    return counts.most_common(1)[0][1] / len(all_recs)


# --- Chat (RAG) grounding metrics --------------------------------------------


@dataclass
class ChatCaseResult:
    name: str
    prompt: str
    reply: str
    matched_titles: List[str]
    confidence: float
    passed: bool


def evaluate_chat_case(
    name: str, prompt: str, reply: str, candidates: List[Dict]
) -> ChatCaseResult:
    """Checks whether a chat reply stays grounded in what was actually
    retrieved, instead of hallucinating songs or answering generically.
    Confidence is the fraction of retrieved candidates the reply names --
    partial credit for mentioning at least one, full credit is not expected
    since a reply only calls out a couple of songs by design.
    """
    reply_lower = reply.lower()
    matched = [s["title"] for s in candidates if s["title"].lower() in reply_lower]
    confidence = min(len(matched) / 2, 1.0) if candidates else 0.0

    return ChatCaseResult(
        name=name,
        prompt=prompt,
        reply=reply,
        matched_titles=matched,
        confidence=confidence,
        passed=confidence >= PASS_THRESHOLD,
    )


# --- Orchestration -----------------------------------------------------------


PASS_THRESHOLD = 0.5


@dataclass
class CaseResult:
    name: str
    genre_match_rate: float
    mood_match_rate: float
    energy_mae: float
    genre_diversity: float

    @property
    def confidence(self) -> float:
        """Per-case confidence, 0-1. Same components as the aggregate
        quality_score, but for a single case instead of averaged across all
        of them -- how well this one recommendation call matched intent.
        """
        return _mean(
            [
                self.genre_match_rate,
                self.mood_match_rate,
                1 - min(self.energy_mae, 1.0),
                self.genre_diversity,
            ]
        )

    @property
    def passed(self) -> bool:
        return self.confidence >= PASS_THRESHOLD


@dataclass
class EvaluationResult:
    system: str
    case_results: List[CaseResult] = field(default_factory=list)
    genre_distribution_skew: float = 0.0
    genre_deviation: Dict[str, float] = field(default_factory=dict)
    catalog_coverage: float = 0.0
    repeat_concentration: float = 0.0

    @property
    def pass_rate(self) -> float:
        return _mean(1.0 if c.passed else 0.0 for c in self.case_results)

    @property
    def avg_genre_match_rate(self) -> float:
        return _mean(c.genre_match_rate for c in self.case_results)

    @property
    def avg_mood_match_rate(self) -> float:
        return _mean(c.mood_match_rate for c in self.case_results)

    @property
    def avg_energy_mae(self) -> float:
        return _mean(c.energy_mae for c in self.case_results)

    @property
    def avg_genre_diversity(self) -> float:
        return _mean(c.genre_diversity for c in self.case_results)

    @property
    def quality_score(self) -> float:
        """Composite 0-1 score balancing preference accuracy against
        result diversity. Not a substitute for the individual metrics --
        useful only as an at-a-glance summary.
        """
        return _mean(
            [
                self.avg_genre_match_rate,
                self.avg_mood_match_rate,
                1 - min(self.avg_energy_mae, 1.0),
                self.avg_genre_diversity,
            ]
        )

    @property
    def over_represented_genres(self) -> Dict[str, float]:
        return {g: d for g, d in self.genre_deviation.items() if d > 0.15}

    @property
    def under_represented_genres(self) -> Dict[str, float]:
        return {g: d for g, d in self.genre_deviation.items() if d < -0.15}


def _mean(values) -> float:
    values = list(values)
    return sum(values) / len(values) if values else 0.0


def evaluate_recommender(
    system_name: str,
    recommend_fn: Callable[[Dict], List[Dict]],
    eval_cases: List[Dict],
    songs: List[Dict],
) -> EvaluationResult:
    """Runs `recommend_fn(case) -> List[song dict]` over every eval case and
    scores the results for accuracy, diversity, and aggregate bias.
    """
    case_results = []
    all_recs: List[Dict] = []

    for case in eval_cases:
        recs = recommend_fn(case)
        all_recs.extend(recs)

        profile = case["profile"]
        case_results.append(
            CaseResult(
                name=case["name"],
                genre_match_rate=genre_match_rate(recs, profile["favorite_genre"]),
                mood_match_rate=mood_match_rate(recs, profile["favorite_mood"]),
                energy_mae=energy_mae(recs, profile["target_energy"]),
                genre_diversity=genre_diversity(recs),
            )
        )

    skew, deviation = genre_distribution_skew(all_recs, songs)

    return EvaluationResult(
        system=system_name,
        case_results=case_results,
        genre_distribution_skew=skew,
        genre_deviation=deviation,
        catalog_coverage=catalog_coverage(all_recs, songs),
        repeat_concentration=repeat_concentration(all_recs),
    )


def print_summary(*results: EvaluationResult) -> None:
    """Prints a per-case PASS/FAIL + confidence table for one or more
    recommender systems -- the "predefined inputs in, pass/fail + confidence
    out" view, separate from the aggregate accuracy/bias comparison below.
    """
    print("\n" + "=" * 60)
    print("       RECOMMENDATION CASES: PASS/FAIL SUMMARY")
    print(f"       (pass threshold: confidence >= {PASS_THRESHOLD:.2f})")
    print("=" * 60)

    for r in results:
        print(f"\n  {r.system}")
        print(f"  {'-' * len(r.system)}")
        for c in r.case_results:
            status = "PASS" if c.passed else "FAIL"
            print(f"    [{status}] {c.name:<20} confidence {c.confidence:.2f}")
        print(f"    -> pass rate: {r.pass_rate:.0%} ({sum(1 for c in r.case_results if c.passed)}/{len(r.case_results)})")


def print_chat_summary(results: List[ChatCaseResult]) -> None:
    """Prints a PASS/FAIL + confidence table for chat (RAG) grounding cases."""
    print("\n" + "=" * 60)
    print("       CHAT (RAG) GROUNDING: PASS/FAIL SUMMARY")
    print(f"       (pass threshold: confidence >= {PASS_THRESHOLD:.2f})")
    print("=" * 60 + "\n")

    for c in results:
        status = "PASS" if c.passed else "FAIL"
        print(f"  [{status}] {c.name:<20} confidence {c.confidence:.2f}  prompt: \"{c.prompt}\"")
        if c.matched_titles:
            print(f"           grounded in: {', '.join(c.matched_titles)}")
        else:
            print("           grounded in: (no retrieved song named in reply)")

    passed = sum(1 for c in results if c.passed)
    print(f"\n  -> pass rate: {passed / len(results):.0%} ({passed}/{len(results)})" if results else "")


def print_comparison(*results: EvaluationResult) -> None:
    print("\n" + "=" * 60)
    print("       RECOMMENDER EVALUATION: HEURISTIC vs AGENTIC")
    print("=" * 60)

    header = f"{'Metric':<28}" + "".join(f"{r.system:>16}" for r in results)
    print("\n" + header)
    print("-" * len(header))

    rows = [
        ("Genre match rate", lambda r: f"{r.avg_genre_match_rate:.2f}"),
        ("Mood match rate", lambda r: f"{r.avg_mood_match_rate:.2f}"),
        ("Energy MAE (lower=better)", lambda r: f"{r.avg_energy_mae:.2f}"),
        ("Genre diversity", lambda r: f"{r.avg_genre_diversity:.2f}"),
        ("Quality score", lambda r: f"{r.quality_score:.2f}"),
        ("Catalog coverage", lambda r: f"{r.catalog_coverage:.2f}"),
        ("Repeat concentration", lambda r: f"{r.repeat_concentration:.2f}"),
        ("Genre dist. skew (bias)", lambda r: f"{r.genre_distribution_skew:.2f}"),
    ]

    for label, fmt in rows:
        print(f"{label:<28}" + "".join(f"{fmt(r):>16}" for r in results))

    print("\nBias flags (genre share vs catalog share, |deviation| > 0.15):")
    for r in results:
        over = r.over_represented_genres
        under = r.under_represented_genres
        print(f"\n  {r.system}:")
        if not over and not under:
            print("    none")
        for genre, dev in over.items():
            print(f"    over-represented: {genre} (+{dev:.2f})")
        for genre, dev in under.items():
            print(f"    under-represented: {genre} ({dev:.2f})")

    print("\n" + "=" * 60)
