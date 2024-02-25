from flask import Flask, request, jsonify, render_template_string
import joblib
import numpy as np

app = Flask(__name__)

# Load your trained model
model = joblib.load('model_knn_1.joblib')

# HTML form
html_form = '''
<!DOCTYPE html>
<html>
<head>
    <title>Model Prediction</title>
</head>
<body>
    <h2>Enter Features for Prediction</h2>
    <form action="/predict" method="post">
        <label for="feature1">Feature 1:</label>
        <input type="text" id="feature1" name="feature1"><br><br>
        <label for="feature2">Feature 2:</label>
        <input type="text" id="feature2" name="feature2"><br><br>
        <label for="feature3">Feature 3:</label>
        <input type="text" id="feature3" name="feature3"><br><br>
        <input type="submit" value="Predict">
    </form>
</body>
</html>
'''

@app.route('/', methods=['GET'])
def home():
    return render_template_string(html_form)

@app.route('/predict', methods=['POST'])
def predict():
    # Extract features from the form
    feature1 = request.form['feature1']
    feature2 = request.form['feature2']
    feature3 = request.form['feature3']
    
    # Convert inputs to float and reshape for the model
    features = np.array([[float(feature1), float(feature2), float(feature3)]])
    prediction = model.predict(features)
    
    # Return prediction result in a new HTML page
    return f'<h2>Prediction: {prediction[0]}</h2>'

if __name__ == '__main__':
    app.run(debug=True)
