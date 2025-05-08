# Basic xG Model

A basic expected goals (xG) model built using football event data.  
This project uses shot location and distance to train a logistic regression model to predict the probability of a goal.

---

## Project Structure

```
Basic-xG-Model/
├── data/
│   └── 7478.json              # Sample match data (shots)
├── results/
│   └── shot_pitch_map.png      # Shot map visualization
├── src/
│   └── Basic_xG_Model.py       # Main Python script to build model and plot
├── README.md                   # Project overview (this file)
```

---

## How to Run

1. Clone the repository:
   ```bash
   git clone https://github.com/aiden-poswell/Basic-xG-Model.git
   cd Basic-xG-Model
   ```

2. Install required libraries:
   ```bash
   pip install matplotlib seaborn scikit-learn pandas
   ```

3. Run the script:
   ```bash
   python src/Basic_xG_Model.py
   ```

This will:
- Train a simple xG model,
- Plot the shot map (with xG values),
- Save the map inside the `results/` folder.

---

## Future Improvements

- Add shot angle, assist type, body part used.
- Train model on larger datasets.
- Create player-level or team-level xG dashboards.
- Add interactive visualizations.

---

## Author

- **Aiden Poswell**
