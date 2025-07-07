import pandas as pd
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.cluster import KMeans
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
import joblib

# Load data
data = pd.read_csv(r"C:\Users\Lathika S\carrec\cars data (1).csv")

# Preprocess data
fuel_encoder = LabelEncoder()
transmission_encoder = LabelEncoder()

# Encode categorical variables
data['fuel_type'] = fuel_encoder.fit_transform(data['fuel_type'])
data['transmission_type'] = transmission_encoder.fit_transform(data['transmission_type'])

# Feature scaling for continuous features
scaler = StandardScaler()
data[['Price']] = scaler.fit_transform(data[['Price']])

# Select features for clustering
features = data[['fuel_type', 'seating_capacity', 'transmission_type', 'Price']]

# Clustering with KMeans
kmeans = KMeans(n_clusters=5, random_state=42)
data['cluster'] = kmeans.fit_predict(features)

# Classification with RandomForest
X = features  # Independent variables (features)
y = data['cluster']  # Target variable (cluster labels)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Train the RandomForest classifier
clf = RandomForestClassifier(n_estimators=100, random_state=42)
clf.fit(X_train, y_train)

# Evaluate the model
y_pred = clf.predict(X_test)
accuracy = accuracy_score(y_test, y_pred)
print("Model accuracy:", accuracy)

# Save models and encoders using joblib
joblib.dump(kmeans, "kmeans_model.pkl")
joblib.dump(clf, "random_forest_model.pkl")
joblib.dump(fuel_encoder, "fuel_encoder.pkl")  # Save fuel encoder
joblib.dump(transmission_encoder, "transmission_encoder.pkl")  # Save transmission encoder
joblib.dump(scaler, "scaler.pkl")  # Save scaler
