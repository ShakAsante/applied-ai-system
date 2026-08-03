# Applied AI System

## Previous Project Summary

The previous project, i am extending is music reccomender and  its a  system that analyzes song features and user preferences to generate personalized recommendations. The system represents songs using attributes such as genre, mood, energy, tempo, valence, danceability, and acousticness.


# New Application & Extension

Title: Wavetune

Summary: Wavetune is a music recommendation system that analyzes song features and user preferences to generate personalized recommendations. The system represents songs using attributes such as genre, mood, energy, tempo, valence, danceability, and acousticness. It also integrates Retrieval-Augmented Generation (RAG) to retrieve relevant song metadata, contextual information, and user preference history before generating recommendations. The use of RAG is essential for improving accuracy.

# Setup

**REQUIREMENT!!** 

For the extended features such as RAG, you need to specify your `ANTHROPIC_API_KEY` in a `.env`

---

### Create a virtual environment (optional but recommended):

   ```bash
   python -m venv .venv
   source .venv/bin/activate      # Mac or Linux
   .venv\Scripts\activate         # Windows
```
### Install dependencies

```bash
pip install -r requirements.txt
```

### If you want to run the Web App:

```bash
python -m streamlit src/streamlit_app.py
```

### If you want to run the CLI Version.



```bash
python -m src.main
```



## Sample Interactions: 

The examples below are real, unedited output captured by running the system as documented above (not hand-written).

### 1. Heuristic mode — structured profile

**Input** (`src/main.py`, High-Energy Pop user):
```python
user_prefs = {
    "favorite_genre": "pop",
    "favorite_mood": "happy",
    "target_energy": 0.9,
    "likes_acoustic": False,
}
```

**Output:**
```
#1: Sunrise City
Artist: Neon Echo
Genre: pop | Mood: happy
Score: 4.66
Reasons:
  ✓ Genre matches preference (+1.0)
  ✓ Mood matches preference (+1.0)
  ✓ Energy similarity score (+1.84)
  ✓ Non-acoustic preference score (+0.82)

#2: Gym Hero
Artist: Max Pulse
Genre: pop | Mood: intense
Score: 3.89
Reasons:
  ✓ Genre matches preference (+1.0)
  ✓ Energy similarity score (+1.94)
  ✓ Non-acoustic preference score (+0.95)
```

### 2. Agentic mode — structured profile

**Input:** `user_prefs = {
    "favorite_genre": "pop",
    "favorite_mood": "happy",
    "target_energy": 0.9,
    "likes_acoustic": False,
}`

**Output:**
```
#1: Spacewalk Thoughts (score 3.04)
This ethereal ambient track perfectly captures that moody, introspective
vibe you love while keeping the energy low and atmospheric—no acoustic
instruments needed!

#2: Night Drive Loop (score 2.88)
With its moody atmosphere, perfectly balanced energy level, and synthetic
production style, "Night Drive Loop" captures the exact ambient vibe
you're looking for.
```
The query never mentions a genre or exact energy value — retrieval picks candidates by vibe similarity (embeddings), then Claude explains why each fits.

### 3. "Ask WaveTune" (RAG) chat — grounded in the shared profile

**Profile set in the sidebar:** genre `lofi`, mood `chill`, energy `0.35`, acoustic `yes`

**Input (chat message):** `"I need something to help me focus while studying tonight"`

**Output:**
> Perfect! I've got a couple of great options for you:
>
> **Focus Flow by LoRoom** is ideal for what you need—it's minimal lo-fi beats specifically designed to reduce distractions and keep you in deep concentration mode. Really clean and purposeful.
>
> **Midnight Coding by LoRoom** is another solid pick if you want something with a bit more warmth. It's a mellow lo-fi beat with soft textures that's built for late-night focus sessions, whether you're coding, studying, or reading.
>
> Both hit that chill vibe you love and have the low energy intensity that works well for sustained studying. **Focus Flow** might edge out slightly if you want maximum concentration, while **Midnight Coding** gives you a touch more atmosphere if you like a little company while you work.
>
> Which sounds better for your session tonight?

The chat message never states a genre or mood — the reply is grounded in the profile set in the sidebar (retrieval is biased toward it, and it's injected into Claude's system prompt as a guardrail), not just the literal wording of the message.

## Design Decisions: 

**Kept the heuristic recommender alongside the new RAG one, instead of replacing it.**
The original rule-based `score_song` is deterministic, free, and fully unit-testable — good properties to keep. Rather than assume RAG is strictly better, both paths were kept live so they could be run through the same [evaluation harness](src/evaluation.py) on the same cases and compared on accuracy, diversity, and bias. Trade-off: two recommendation code paths to maintain instead of one, but it turns "RAG is an improvement" from an assumption into something measurable (`python src/evaluate.py`).

**Retrieval is local (sentence-transformers), generation is Claude.**
Anthropic doesn't expose an embeddings endpoint, so the options were a second hosted API (e.g. Voyage) or a local embedding model. Local was chosen to avoid a second API key/cost/network dependency for something that's a solved problem at this catalog size (20 songs). Trade-off: the embedding model has to be downloaded on first run (a few hundred MB) and quality is lower than a large hosted model — acceptable here because retrieval only needs to narrow candidates, not make the final call.

**Retrieve-then-rescore, not pure vector search.**
Both `recommend_from_query`/`recommend_from_profile` embed to fetch a candidate pool by vibe, then re-rank that pool with the exact same deterministic `score_song` used by the heuristic path, and only then ask Claude to explain the result. This was chosen over trusting embedding similarity as the final ranking, because embeddings are good at "does this feel similar" but bad at "is this energy value close enough" — the numeric feature scoring is more precise for that. Trade-off: a more complex pipeline (embed → rescore → explain) than either approach alone.

**The agentic mode uses the same structured profile as the heuristic mode, not free text.**
Earlier versions had the agentic recommender parse a free-text query into preferences via an extra Claude call. That was dropped in favor of reading the same sidebar profile both modes now share. This makes the two modes an apples-to-apples comparison (same input, different pipeline) and — more importantly — gives the chat widget a stable profile to ground itself in. Trade-off: less "type anything and it figures it out" flexibility for one-off recommendations; free-text parsing (`parse_query_to_preferences`) is still used by the CLI demo in `main.py` where that flexibility matters more than consistency.

**The chat widget is guardrailed to the shared profile, not a general-purpose chatbot.**
The user's profile is both blended into the chat's retrieval query and injected into Claude's system prompt with an explicit instruction to stay on-topic and consistent with it. This was a deliberate constraint: a fully open chatbot bolted onto a recommender tends to wander off-topic or contradict the profile the user just set. The trade-off is a less "free" conversational experience, in exchange for answers that stay anchored to what the user actually said they want.

**Two catalogs instead of one.**
`data/songs.csv` (the original numeric-feature schema) still backs the heuristic/agentic recommendation forms, while `data/songs_detailed.csv` (richer text: description, themes, tags, best_for) backs the chat widget's retrieval specifically. Chat benefits from natural-language-dense fields that the original catalog doesn't have; the recommendation forms didn't need to change since they already depended on the original schema (and tests/model card reference it). Trade-off: two catalogs that need to be kept roughly in sync if songs are added or changed.

## Reflection & Model Card

[**Model Card**](model_card.md)