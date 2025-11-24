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


## Detailed Documentation

### Modelling
Building minute-by-minute win predictions is hard because most detailed live match data is owned by private companies. Thankfully, Wyscout makes a rich event dataset available for research, and this project starts by processing that data. First, the raw events are split into individual matches and converted into simple per-minute signals such as goals so far and yellow cards so far. Next, using information from previous matches, we add broader context features like a team’s average xT (a proxy for how well a team progresses the ball into dangerous areas), along with live game context such as red cards, home/away identification, and other match-state indicators. Finally, all of these pieces are merged into a single processed dataset that is used for modeling.

The project implements a Win Probability Model that estimates the chance of a team winning, drawing, or losing at any moment in a match. Rather than predicting the final score directly, we train a regression model using LightGBM to estimate each team’s probability of scoring in the remaining minutes based on the current game state. These scoring probabilities are then passed into a simulation engine, which runs many iterations of the rest of the match to convert scoring chances into Home Win, Draw, and Away Win probabilities.

The model uses a mix of game-state, discipline, and advanced xT features. Game-state features capture the live situation, such as score_differential, goals_scored, minutes_remaining, time_interval, and whether the team is at home. Advanced xT features include avg_team_xt as a measure of team strength, avg_opp_xt as a measure of opponent strength, and running_xt_differential to reflect live momentum. Discipline features account for numerical advantage and cautions, including player_differential (red cards), own_yellow_cards, and opposition_yellow_cards.

### Explainability
Since the model is already trained, the next step is to understand what actually drives its decisions. For that reason, I applied SHAP and LIME to interpret the predictions. These methods gave me both global explanations, which show overall feature importance across matches, and local explanations, which justify a specific prediction at a specific minute. From the plots, it’s clear that team xT is one of the strongest drivers. This makes sense because xT reflects how effectively a team progresses the ball into dangerous areas, so it captures both current form and overall quality. Player differential also shows up as a major factor, which is intuitive: when a team goes down to 10 or 9 players, it becomes much harder to cover space and defend, and that disadvantage often translates into a higher chance of losing.

[img]

However, explanation alone isn’t enough, because football is highly dynamic. The state of a match can shift dramatically from minute to minute ,  a single mistake, tackle, or sudden goal can flip the outlook. That’s why I built an interactive simulation layer, so the model can be explored in real time and users can see how win probabilities and feature drivers change as the match evolves.

### Dashboard Integration
