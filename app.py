from flask import Flask, request, render_template
import numpy as np
import joblib

app = Flask(__name__)

# Load the model and scaler
model = joblib.load('model.pkl')  # Make sure your model is trained and saved correctly
scaler = joblib.load('scaler.pkl')  # Load the scaler used during training

# Home Page Route
@app.route('/')
def home():
    return render_template('index.html')  # Ensure the index.html is present

# Prediction Route
@app.route('/predict', methods=['POST'])
def predict():
    try:
        # Collect the input data from the form (ensure the correct order and input types)
        features = [float(request.form[key]) for key in request.form]

        if len(features) != 15:  # Ensure that 15 features are provided
            return render_template('result.html', prediction_text="Error: Please enter all 15 features correctly.")
        
        # Preprocess the features (scale the features using the scaler used during training)
        features = np.array(features).reshape(1, -1)  # Reshape to ensure the input is 2D
        features_scaled = scaler.transform(features)  # Apply the scaler

        # Predict the outcome using the model
        prediction = model.predict(features_scaled)[0]

        # Based on prediction, return the result
        if prediction == 1:
            result_text = "Heart Disease Detected"
        else:
            result_text = "No Heart Disease Detected"

        return render_template('result.html', prediction_text=result_text)
    
    except Exception as e:
        return render_template('result.html', prediction_text=f"Error: {str(e)}")
    return render_template('result.html', prediction_text=f"Error: {str(e)}")
    

if __name__ == '__main__':
    app.run(debug=True)
