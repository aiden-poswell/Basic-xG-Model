# --- Import Libraries ---
import json
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, roc_auc_score

# --- Load the JSON Data ---
with open('data/7478.json') as f:   # Change 7478.json if you uploaded a different match
    data = json.load(f) 

# --- Extract Shot Events ---
shots = [event for event in data if event['type']['name'] == 'Shot']

# --- Convert to DataFrame ---
shots_df = pd.json_normalize(shots)

# --- Feature Engineering ---
# Create features like shot location
shots_df['location_x'] = shots_df['location'].apply(lambda x: x[0])
shots_df['location_y'] = shots_df['location'].apply(lambda x: x[1])

# Calculate simple distance to goal
shots_df['distance'] = ((120 - shots_df['location_x'])**2 + (40 - shots_df['location_y'])**2)**0.5

# Target variable: was it a goal?
shots_df['is_goal'] = shots_df['shot.outcome.name'].apply(lambda x: 1 if x == 'Goal' else 0)

# Select features and target
X = shots_df[['location_x', 'location_y', 'distance']]
y = shots_df['is_goal']

# --- Train Test Split ---
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# --- Build a Logistic Regression Model ---
model = LogisticRegression()
model.fit(X_train, y_train)

# --- Predict and Evaluate ---
y_pred = model.predict(X_test)
print("Accuracy:", accuracy_score(y_test, y_pred))
print("ROC AUC:", roc_auc_score(y_test, model.predict_proba(X_test)[:,1]))

# --- Plot Shot Locations ---
plt.figure(figsize=(10, 6))
sns.scatterplot(x='location_x', y='location_y', hue='is_goal', data=shots_df, palette='coolwarm')
plt.title('Shot Locations (Goals vs Non-Goals)')
plt.xlabel('X Location')
plt.ylabel('Y Location')
plt.xlim(0, 120)
plt.ylim(0, 80)
plt.gca().invert_xaxis()
plt.savefig('results/shot_locations.png')
plt.show()

# --- Create a football pitch plot ---

def create_pitch(length=120, width=80, unity=1):
    """
    Creates a football pitch plot using matplotlib
    """
    fig, ax = plt.subplots(figsize=(length/unity, width/unity))
    # Pitch Outline & Centre Line
    plt.plot([0,0],[0,width], color="black")
    plt.plot([0,length],[width,width], color="black")
    plt.plot([length,length],[width,0], color="black")
    plt.plot([length,0],[0,0], color="black")
    plt.plot([length/2,length/2],[0,width], color="black")
    
    # Left Penalty Area
    plt.plot([18,18],[62,18],color="black")
    plt.plot([0,18],[62,62],color="black")
    plt.plot([18,0],[18,18],color="black")
    
    # Right Penalty Area
    plt.plot([length,length-18],[62,62],color="black")
    plt.plot([length-18,length-18],[62,18],color="black")
    plt.plot([length-18,length],[18,18],color="black")
    
    # Left 6-yard Box
    plt.plot([0,6],[50,50],color="black")
    plt.plot([6,6],[50,30],color="black")
    plt.plot([6,0],[30,30],color="black")
    
    # Right 6-yard Box
    plt.plot([length,length-6],[50,50],color="black")
    plt.plot([length-6,length-6],[50,30],color="black")
    plt.plot([length-6,length],[30,30],color="black")
    
    # Prepare Circles
    centreCircle = plt.Circle((length/2,width/2),8.1,color="black",fill=False)
    centreSpot = plt.Circle((length/2,width/2),0.71,color="black")
    leftPenSpot = plt.Circle((12,40),0.71,color="black")
    rightPenSpot = plt.Circle((length-12,40),0.71,color="black")
    
    # Draw Circles
    ax.add_patch(centreCircle)
    ax.add_patch(centreSpot)
    ax.add_patch(leftPenSpot)
    ax.add_patch(rightPenSpot)
    
    # Limit axes
    plt.xlim(0, length)
    plt.ylim(0, width)
    
    return fig, ax

# --- Plot Shots on Football Pitch ---

fig, ax = create_pitch()

sns.scatterplot(
    x='location_x', 
    y='location_y', 
    hue='is_goal', 
    data=shots_df, 
    palette='coolwarm', 
    ax=ax
)

plt.title('Shot Map: Goals vs Misses')
plt.legend(title='Goal')
plt.gca().invert_xaxis()
plt.savefig('results/shot_pitch_map.png')
plt.show()
