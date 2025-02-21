from flask import Flask, render_template, request
import joblib

app = Flask(__name__)

# Load trained model and encoders
model = joblib.load('model/crop_recommendation_model.pkl')
label_encoder_crop = joblib.load('model/label_encoder_crop.pkl')
label_encoder_state = joblib.load('model/label_encoder_state.pkl')
label_encoder_crop_type = joblib.load('model/label_encoder_crop_type.pkl')

limits = {
    "N": (10, 180),
    "P": (10, 125),
    "K": (10, 200),
    "pH": (3.00, 7.00),
    "rainfall": (3.00, 4000.00),
    "temperature": (1.00, 36.00),
}

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/input')
def input_page():
    return render_template('input.html')

@app.route('/predict', methods=['POST'])
def predict():
    try:
        # Form data
        state_name = request.form['state_name']
        crop_type = request.form['crop_type']
        N = float(request.form['nitrogen'])
        P = float(request.form['phosphorus'])
        K = float(request.form['potassium'])
        pH = float(request.form['ph'])
        rainfall = float(request.form['rainfall'])
        temperature = float(request.form['temperature'])

        # Validation
        errors = []
        for key, value in [("N", N), ("P", P), ("K", K), ("pH", pH), ("rainfall", rainfall), ("temperature", temperature)]:
            if not (limits[key][0] <= value <= limits[key][1]):
                errors.append(f"{key} must be between {limits[key][0]} and {limits[key][1]}.")
        
        if state_name not in label_encoder_state.classes_:
            errors.append("Invalid state name.")
        if crop_type not in label_encoder_crop_type.classes_:
            errors.append("Invalid crop type.")

        if errors:
            return render_template('input.html', error="; ".join(errors))

        # Encoding
        encoded_state = label_encoder_state.transform([state_name])[0]
        encoded_crop_type = label_encoder_crop_type.transform([crop_type])[0]
        input_data = [[encoded_state, encoded_crop_type, N, P, K, pH, rainfall, temperature]]

        # Prediction
        predicted_crop_index = model.predict(input_data)[0]
        predicted_crop = label_encoder_crop.inverse_transform([predicted_crop_index])[0]
        return render_template('predict.html', prediction=f"Recommended Crop: {predicted_crop}")
    
    except Exception as e:
        return render_template('predict.html', error=f"An error occurred: {str(e)}")

if __name__ == '__main__':
    app.run(debug=True)
