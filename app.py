from flask import Flask, render_template, request
import numpy as np
import joblib

app = Flask(__name__)

# Load your trained model
model = joblib.load('model_knn_1_ws.joblib')

# Define routes
@app.route('/', methods=['GET'])
def home():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    # Extract features from the form
    feature1 = request.form['time_of_incident']
    feature2 = request.form['longitude']
    feature3 = request.form['latitude']

    try:
        # Convert inputs to float and reshape for the model
        features = np.array([[float(feature1), float(feature2), float(feature3)]])
        prediction = model.predict(features)
        prediction_value = prediction[0] if len(prediction) > 0 else None
        
        # Return prediction result in a new HTML page
        return f'<h2>Prediction: {prediction_value}</h2>'
    except ValueError:
        # Return error message if conversion fails
        return '<h2>Error: Please enter valid numerical values.</h2>'


if __name__ == '__main__':
    app.run(debug=True)