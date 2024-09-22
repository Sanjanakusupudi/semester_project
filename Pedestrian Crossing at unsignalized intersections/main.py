import pandas as pd
import numpy as np
from sklearn.preprocessing import MinMaxScaler
from tensorflow.keras.models import load_model
from sklearn.metrics import accuracy_score, classification_report
import joblib

# Step 1: Load the test data from CSV
test_data_file = '/Users/sanjanakusupudi/Downloads/Pedestrian Crossing at unsignalized intersections/Data.csv'
test_data_df = pd.read_csv(test_data_file)

# Step 2: Split features and labels
X_test = test_data_df.iloc[:, :-1].values
y_test = test_data_df.iloc[:, -1].values

# Step 3: Preprocess the test data
# Load the scaler
scaler_path = '/Users/sanjanakusupudi/Downloads/Pedestrian Crossing at unsignalized intersections/scaler.pkl'
scaler = joblib.load(scaler_path)

# Scale the test data
X_test_scaled = scaler.transform(X_test)

# Step 4: Load the saved model
model_path = 'my_trained_model.h5'
model = load_model(model_path)

# Step 5: Make predictions on the test set
predictions = model.predict(X_test_scaled)
predicted_labels = np.argmax(predictions, axis=1)

# Step 6: Evaluate the model
accuracy = accuracy_score(y_test, predicted_labels)
print(f'Test accuracy: {accuracy}')

report = classification_report(y_test, predicted_labels)
print('Classification Report:')
print(report)