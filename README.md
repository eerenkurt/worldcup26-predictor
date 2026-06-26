# FIFA World Cup 2026 — AI Predictor

A modern, minimalist, end-to-end machine learning-based prediction engine for the FIFA World Cup 2026. This project leverages historical tournament data and real-time team form trends to simulate individual match outcomes and the entire tournament bracket.

 **Live Application Link:** [worldcup2026-predictor-erenkurt.streamlit.app](https://worldcup2026-predictor-erenkurt.streamlit.app/)

---

## Project Features

* **Single Match Predictor:** Analyzes ELO ratings, FIFA rankings, last 5-match goal/form metrics, and historical World Cup data of any two selected teams to generate real-time win probabilities.
* **Tournament Simulator:** Simulates the actual FIFA 2026 format consisting of 48 teams and 12 groups.
    * **AI Mode:** Automatically simulates the entire tournament and knockout stages (Round of 32, Round of 16, Quarter-Finals...) using the CatBoost model.
    * **User Mode:** Allows you to manually pick the group stage winners and dynamically locks the knockout bracket based on your choices.
* **UI/UX Design:** High-contrast, industrial-brutalist, and user-centric custom CSS interface inspired by the Apple design ecosystem.

---

## Model & Performance

* **Algorithm:** CatBoost Classifier (Gradient Boosting)
* **Accuracy Score:** **77.30%**
* **Dataset:** Official and competitive international match results from 1930 to the present (`results.csv`), combined with historical World Cup all-time stats.
* **Feature Pool:** 12 key metrics including `home_form`, `away_form`, `home_avg_scored`, `away_avg_scored`, `home_wc_appearances`, `home_titles`, and `home_finals_reached`.

---

## Tech Stack

* **Backend & ML:** Python, CatBoost, Scikit-Learn, Pandas, NumPy, Joblib
* **Frontend / UI:** Streamlit, Custom HTML & CSS (Apple Design Style)
* **Deployment:** GitHub & Streamlit Community Cloud

---
