# fit_label_encoder.py (Create this or add to your model training script)
from sklearn.preprocessing import LabelEncoder
import pickle

# Define all possible fuel types
all_fuel_types = ['Petrol', 'Diesel', 'Electric', 'Hybrid']  # Add other fuel types if needed

# Initialize and fit the LabelEncoder
label_encoder = LabelEncoder()
label_encoder.fit(all_fuel_types)

# Save the fitted LabelEncoder to a file
with open('label_encoder.pkl', 'wb') as f:
    pickle.dump(label_encoder, f)

print("LabelEncoder has been fitted and saved.")
