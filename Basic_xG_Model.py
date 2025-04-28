# --- Import Libraries ---
import json
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import matplotlib.patches as mpatches
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, roc_auc_score

# --- Load the JSON Data ---
with open('data/7478.json') as f:
    data = json.load(f)

# --- Extract Shot Events ---
shots = [event for event in data if event['type']['name'] == 'Shot']

# --- Convert to DataFrame ---
shots_df = pd.json_normalize(shots)

# --- Feature Engineering ---
shots_df['location_x'] = shots_df['location'].apply(lambda x: x[0])
shots_df['location_y'] = shots_df['location'].apply(lambda x: x[1])
shots_df['distance'] = ((120 - shots_df['location_x'])**2 + (40 - shots_df['location_y'])**2)**0.5
shots_df['is_goal'] = shots_df['shot.outcome.name'].apply(lambda x: 1 if x == 'Goal' else 0)
shots_df['shot_id'] = range(1, len(shots_df) + 1)

# --- Assign Fake Teams ---
mid_point = len(shots_df) // 2
shots_df['team'] = ['Team A' if i < mid_point else 'Team B' for i in range(len(shots_df))]

# --- Train Model ---
X = shots_df[['location_x', 'location_y', 'distance']]
y = shots_df['is_goal']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

model = LogisticRegression()
model.fit(X_train, y_train)

# --- Predict and Evaluate ---
y_pred = model.predict(X_test)
print("Accuracy:", accuracy_score(y_test, y_pred))
print("ROC AUC:", roc_auc_score(y_test, model.predict_proba(X_test)[:, 1]))

# --- Predict xG ---
shots_df['xG'] = model.predict_proba(X)[:, 1]

# --- Prepare plotting coordinates (flip Team B) ---
shots_df['plot_x'] = shots_df.apply(
    lambda row: row['location_x'] if row['team'] == 'Team A' else 120 - row['location_x'],
    axis=1
)
shots_df['plot_y'] = shots_df.apply(
    lambda row: row['location_y'] if row['team'] == 'Team A' else 80 - row['location_y'],
    axis=1
)

# --- Calculate total xG and goals for each team ---
team_a_xg = shots_df[shots_df['team'] == 'Team A']['xG'].sum()
team_b_xg = shots_df[shots_df['team'] == 'Team B']['xG'].sum()

team_a_goals = shots_df[(shots_df['team'] == 'Team A') & (shots_df['is_goal'] == 1)].shape[0]
team_b_goals = shots_df[(shots_df['team'] == 'Team B') & (shots_df['is_goal'] == 1)].shape[0]

# --- Create Pitch ---
def create_pitch(length=120, width=80):
    fig, ax = plt.subplots(figsize=(12, 8))
    ax.plot([0,0],[0,width], color="black")
    ax.plot([0,length],[width,width], color="black")
    ax.plot([length,length],[width,0], color="black")
    ax.plot([length,0],[0,0], color="black")
    ax.plot([length/2,length/2],[0,width], color="black")

    # Penalty Areas
    ax.plot([18,18],[62,18],color="black")
    ax.plot([0,18],[62,62],color="black")
    ax.plot([18,0],[18,18],color="black")
    ax.plot([length,length-18],[62,62],color="black")
    ax.plot([length-18,length-18],[62,18],color="black")
    ax.plot([length-18,length],[18,18],color="black")

    # 6-yard Boxes
    ax.plot([0,6],[50,50],color="black")
    ax.plot([6,6],[50,30],color="black")
    ax.plot([6,0],[30,30],color="black")
    ax.plot([length,length-6],[50,50],color="black")
    ax.plot([length-6,length-6],[50,30],color="black")
    ax.plot([length-6,length],[30,30],color="black")

    # Circles
    centreCircle = plt.Circle((length/2,width/2),8.1,color="black",fill=False)
    centreSpot = plt.Circle((length/2,width/2),0.71,color="black")
    leftPenSpot = plt.Circle((12,40),0.71,color="black")
    rightPenSpot = plt.Circle((length-12,40),0.71,color="black")

    ax.add_patch(centreCircle)
    ax.add_patch(centreSpot)
    ax.add_patch(leftPenSpot)
    ax.add_patch(rightPenSpot)

    ax.set_xlim(0, length)
    ax.set_ylim(0, width)
    ax.axis('off')

    return fig, ax

# --- Plot ---
fig, ax = create_pitch()

sns.scatterplot(
    x='plot_x', 
    y='plot_y', 
    hue='is_goal', 
    data=shots_df, 
    palette={0: 'red', 1: 'green'},
    s=150,
    edgecolor='black',
    ax=ax,
    legend=False
)

# Add Shot ID numbers inside each dot
for idx, row in shots_df.iterrows():
    ax.text(
        row['plot_x'],
        row['plot_y'],
        str(row['shot_id']),
        color='white',
        fontsize=8,
        weight='bold',
        ha='center',
        va='center'
    )

# --- Main Title ---
plt.title('Shot Map with xG Values', fontsize=20)

# --- Team xG Subheading ---
plt.gcf().text(
    0.5, 0.94,
    f"Team A xG: {team_a_xg:.2f}  |  Team B xG: {team_b_xg:.2f}",
    fontsize=12,
    ha='center'
)

# --- Scoreline inside top right (shifted properly) ---
ax.text(
    110, 78,  # move a bit left inside the pitch
    f"Team A {team_a_goals} - {team_b_goals} Team B",
    fontsize=12,
    ha='left',
    va='top'
)

# Invert X axis
plt.gca().invert_xaxis()

# --- Shot xG List Heading ---
plt.gcf().text(
    0.99, 0.83,
    "Shot xG List",
    fontsize=10,
    ha='right',
    va='top',
    weight='bold'
)

# --- Shot xG List Content ---
start_y = 0.81
dy = 0.024

for idx, row in shots_df.iterrows():
    text = f"{int(row['shot_id']):2}: {row['xG']:.2f} xG"
    t = plt.gcf().text(
        0.99, 
        start_y - idx * dy,
        text,
        fontsize=8,
        ha='right',
        va='top',
        fontfamily='monospace'
    )
    if row['is_goal'] == 1:
        t.set_weight('bold')

# --- Custom Legend ---
legend_elements = [
    mpatches.Patch(facecolor='green', edgecolor='black', label='Goal'),
    mpatches.Patch(facecolor='red', edgecolor='black', label='Miss')
]
plt.legend(
    handles=legend_elements,
    loc='lower center',
    bbox_to_anchor=(0.5, -0.08),
    ncol=2,
    frameon=False,
    fontsize=10
)

# --- Save and Show ---
fig.savefig('results/shot_pitch_map.png', dpi=300, bbox_inches='tight')
plt.show()
