# XAI-Soccer ⚽📊  
Explainable, minute-by-minute win probability modeling for football matches (Premier League baseline).

This project replicates the live win-probability mechanism used by major analytics platforms. It predicts **per-team scoring probability at each minute**, then simulates the rest of the game to produce **Home Win / Draw / Away Win** probabilities. Each probability is explained using **SHAP** and **LIME**, and a **Streamlit dashboard** lets you explore matches and run what-if simulations.


## What this project does

1. **Process event-level soccer data** into a minute-level match state.
2. Train a model to estimate **team scoring probability per minute**.
3. Convert those scoring probabilities into **win/draw/win** using Monte-Carlo simulation.
4. Explain *why* the model predicts a certain probability with **SHAP** and **LIME**.
5. Provide a **dashboard** to:
   - browse simulated matches  
   - view win probabilities over time  
   - inspect SHAP/LIME drivers at any minute  
   - simulate feature changes and see updated win probabilities


## Core features

- **Minute-level match state** with game context:
  - score differential
  - players on pitch (red cards)
  - yellow cards
  - xT pressure / momentum
- **Scoring probability model** (LightGBM baseline)
- **Win probability simulation**
- **Explainability**
  - SHAP barplot for any (match, team, minute)
  - LIME barplot for any (match, team, minute)
- **Streamlit app**
  - Home page listing matches
  - Match View page (win prob progression + SHAP/LIME)
  - Simulation page (change features and see new win probs)


## Setup
1. Clone this repository
2. Create a virtual environment and activate it
3. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```
4. Download the Wyscout dataset from Figshare (https://figshare.com/collections/Soccer_match_event_dataset/4415000/5)  and place it in the `wyscout_figshare_data/` folder.
5. Run the notebooks in order:
   - `1-data_preprocessing.ipynb`
   - `2-building-features.ipynb`
   - `3-modelling_extraction.ipynb`
6. Launch the Streamlit dashboard:
   ```bash
   streamlit run dashboard/Home.py
   ```  
