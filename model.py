import pandas as pd
import pickle
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report

# 1. Load the dataset
df = pd.read_csv('flood.csv')  # Make sure this file exists in the same folder

# 2. Check and drop missing values
print("Missing values per column:\n", df.isnull().sum())
df.dropna(inplace=True)

# 3. Convert FloodProbability to categorical (3 classes) if needed
if df['FloodProbability'].dtype != 'int' and df['FloodProbability'].nunique() > 3:
    df['FloodProbability'] = pd.cut(
        df['FloodProbability'],
        bins=[0, 0.33, 0.66, 1.0],
        labels=[0, 1, 2]
    )
    df['FloodProbability'] = df['FloodProbability'].astype(int)

# 4. Define selected features
selected_features = [
    'MonsoonIntensity',
    'TopographyDrainage',
    'RiverManagement',
    'Deforestation',
    'Urbanization',
    'ClimateChange',
    'DrainageSystems',
    'CoastalVulnerability',
    'Landslides',
    'Watersheds',
    'DeterioratingInfrastructure',
    'PopulationScore'
]

# 5. Prepare input (X) and target (y)
X = df[selected_features]
y = df['FloodProbability']

# 6. Split the data
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# 7. Train the model
model = LogisticRegression(max_iter=1000)
model.fit(X_train, y_train)

# 8. Evaluate the model
y_pred = model.predict(X_test)
print("\nModel Evaluation:")
print("Accuracy:", accuracy_score(y_test, y_pred))
print("\nClassification Report:\n", classification_report(y_test, y_pred))

# 9. Save the trained model using pickle
with open('model.pkl', 'wb') as f:
    pickle.dump(model, f)

print("\n✅ Model has been trained and saved as model.pkl")
