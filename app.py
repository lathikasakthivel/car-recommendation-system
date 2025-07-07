from flask import Flask, render_template, request, jsonify
import pandas as pd
import joblib
from sklearn.preprocessing import LabelEncoder, StandardScaler

# Initialize the Flask app
app = Flask(__name__)

# Load models and preprocessors
kmeans = joblib.load("kmeans_model.pkl")
random_forest = joblib.load("random_forest_model.pkl")
fuel_encoder = joblib.load("fuel_encoder.pkl")  # Load fuel encoder
transmission_encoder = joblib.load("transmission_encoder.pkl")  # Load transmission encoder
assembly_encoder = joblib.load("assembly_encoder.pkl")  # Load assembly encoder
scaler = joblib.load("scaler.pkl")  # Load scaler

# Load and preprocess the dataset
data = pd.read_csv('OLX_cars_dataset00.csv')

# Preprocess categorical columns as per the original dataset
data['Fuel'] = fuel_encoder.fit_transform(data['Fuel'])
data['Transmission'] = transmission_encoder.fit_transform(data['Transmission'])
data['Assembly'] = assembly_encoder.fit_transform(data['Assembly'])  # Assuming assembly is categorical
data[['Year']] = scaler.fit_transform(data[['Year']])  # Scale the year feature

# Homepage with form
@app.route('/')
def home():
    return render_template('form.html')

# Route to get user input and return prediction
@app.route('/predict', methods=['POST'])
def predict():
    if request.method == 'POST':
        # Get data from the form
        fuel_type = request.form['Fuel']
        transmission_type = request.form['Transmission']
        assembly = request.form['Assembly']  # New input for assembly
        year = int(request.form['Year'])  # New input for year

        # Encode and scale input data as the model expects
        fuel_type_encoded = fuel_encoder.transform([fuel_type])[0]  # Use the loaded fuel encoder
        transmission_type_encoded = transmission_encoder.transform([transmission_type])[0]  # Use the loaded transmission encoder
        assembly_encoded = assembly_encoder.transform([assembly])[0]  # Use the loaded assembly encoder
        year_scaled = scaler.transform([[year]])[0][0]  # Scale the year

        # Combine into feature array
        features = [[fuel_type_encoded,  transmission_type_encoded, assembly_encoded, year_scaled]]

        # Predict the cluster
        cluster = kmeans.predict(features)[0]
        recommendation_cluster = random_forest.predict(features)[0]
        filtered_cars = data[
            (data['fuel_type'] == fuel_type_encoded) &
            (data['transmission_type'] == transmission_type_encoded) &
            (data['assembly'] == assembly_encoded) &
            (data['year'] == year)  # Adjust this condition if necessary (e.g., use a range)
        ]

        # Get details of the filtered cars
        car_details = filtered_cars[['car_name', 'year', 'fuel_type', 'transmission_type','assembly']].to_dict(orient='records')

        # If no cars match the filter, provide a message
        if len(car_details) == 0:
            return jsonify({
                'Cluster': int(cluster),  # Convert to int for JSON serialization
                'Message': "No cars found matching your criteria.",
                'Car Details': car_details
            })

        # If cars are found, return them
        return jsonify({
            'Cluster': int(cluster),  # Convert to int for JSON serialization
            'Car Details': car_details
        })

if __name__ == "__main__":
    app.run(debug=True)
