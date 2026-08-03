# 🎧 Model Card: Music Recommender Simulation

## 1. Model Name  

WaveTune: A, One solution, Music Recommender

---

## 2. Intended Use  

 The system is intended for users who want to discover new songs 

---

## 3. How the Model Works  

WaveTune is designed to provide personalized music recommendations by comparing a user's preferences with the characteristics of available songs.

Genre matches will receive higher priority because they strongly influence a song's style, whilst numerical features such as energy are scored based on how closely they match the user's target values.

These weights help the system rank songs and select the recommendations that best   fit the user's overall musical preferences.

To improve recommendation quality, WaveTune has a mode to incorporate Retrieval-Augmented Generation (RAG). Before generating recommendations, the system retrieves relevant song information and user preference data, ensuring that recommendations are based on accurate and contextually relevant information rather than relying solely on the model's internal knowledge. This retrieval step increases the accuracy, consistency, and personalization of the recommendations.

The weighted scoring approach, combined with RAG, enables the system to rank songs effectively and recommend tracks that best match the user's overall musical preferences.

---

## 4. Data  

WaveTune uses a music catalog containing 20 songs with information about each song's characteristics. Including characteristics such as the songs title, artist, genre, mood, energy, tempo, valence, danceability, and acousticness.

<!-- Describe the dataset the model uses.  

Prompts:  

- How many songs are in the catalog  
- What genres or moods are represented  
- Did you add or remove data  
- Are there parts of musical taste missing in the dataset   -->

---

## 5. Strengths  

The system is effective at identifying songs with similar genres, moods, and numerical characteristics such as energy, tempo, valence, danceability, and acousticness. Its weighted scoring system enables it to evaluate multiple song attributes simultaneously rather than relying on a single feature, resulting in more balanced and relevant recommendations.

The integration of Retrieval-Augmented Generation (RAG) further improves the system's performance by retrieving relevant song metadata and user preference information before recommendations are generated.

In the possibility the AI could have been abused, or misused, the proper guardrails were implemented to assure this wouldn't happen.

---

## 6. Limitations and Bias 
The system may over-prioritize genre due to its higher weighting compared to other song characteristics, which can cause tracks with different but potentially appealing qualities to receive lower rankings. While this improves genre-based matching, it may reduce the importance of other features such as energy, mood, or acousticness.

The system can also create a filter bubble by repeatedly recommending songs that are similar to the user's existing preferences, limiting exposure to new genres, artists, and musical styles. Incorporating Retrieval-Augmented Generation (RAG) can help

---

## 7. Evaluation  

Several different user profiles were tested to observe how the recommendation system responded to different preferences, including High-Energy Pop, Chill Lofi, and Deep Intense Rock users.

One surprising result was that songs from the user's favorite genre often ranked higher even when another song from a different genre had a closer match in energy or other musical features. I also ran comparisons by adjusting feature weights, such as increasing the importance of energy and reducing the importance of genre, to see how changes affected the rankings.

<!-- 
How you checked whether the recommender behaved as expected. 

Prompts:  

- Which user profiles you tested  
- What you looked for in the recommendations  
- What surprised you  
- Any simple tests or comparisons you ran   -->


---

## 8. Future Work  

Ideas for how you would improve the model next.  

- Adjust the scoring weights so the system does not rely too heavily on genre and gives more importance to features like mood, energy, tempo, and valence.
- Expand the dataset with a larger variety of genres, moods, and musical styles to reduce bias.
- Access real data from a actual database.
- Add a diversity system to prevent the recommender from repeatedly suggesting similar songs.
- Combine content-based filtering with collaborative filtering to learn from the preferences of similar users.
- Improve the scoring algorithm by considering relationships between features, such as how energy and mood work together to define a song's overall vibe.

---

## 9. Personal Reflection  

Building WaveTune helped me understand how recommender systems use data and algorithms to predict what users may enjoy.

I learned that recommendation systems are using different features, weights, and scoring methods to rank options based on user preferences.

This project changed the way I think about music recommendation apps because I realized that platforms like Spotify and YouTube rely on systems that balance personalization, discovery, and user behavior.

Another thing that suprised me was the AI's reliability and how hard it is to make it hallicunate with the proper guardrails, additionaly there are some issues with this project that can be fixed in a future evolution.

Just like how ai was important in live product of this project, it was just as important in the development of this project too, sometimes ai would assume things if there was specified information, for example, when i asked it whats the best api to get started with if i want to implement rag, with the most versatility, it say Voyage API, which is something i did not need at all. All i needed was a API Key from a LLM.