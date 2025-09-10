from flask import Flask, render_template, request
import numpy as np
import pickle
import os

app = Flask(__name__)

# Load your trained model (make sure model.pkl is in the same folder or provide full path)
model_path = os.path.join(os.path.dirname(__file__), 'model.pkl')
model = pickle.load(open(model_path, 'rb'))

# List of features the model expects (in this exact order)
feature_names = [
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

@app.route('/', methods=['GET'])
def home():
    """Render the home/landing page"""
    return render_template('home.html')

@app.route('/flood-detection', methods=['GET'])
def flood_detection_form():
    """Render the flood detection form page"""
    return render_template('index.html', prediction='')

@app.route('/flood-detection', methods=['POST'])
def predict():
    """Handle form submission and make prediction"""
    try:
        # Extract feature values from form and convert to float
        input_features = [float(request.form.get(feat)) for feat in feature_names]

        # Prepare data for prediction
        input_array = np.array([input_features])

        # Make prediction
        pred = model.predict(input_array)[0]

        # Convert prediction to readable label (customize if you have classes 0/1/2)
        if pred == 1:
            label = "Flood Risk Detected"
        elif pred == 2:
            label = "High Flood Risk"
        else:
            label = "No Flood Risk"

        return render_template('index.html', prediction=label)

    except Exception as e:
        return render_template('index.html', prediction=f"Error: {str(e)}")


if __name__ == '__main__':
    app.run(debug=True)
