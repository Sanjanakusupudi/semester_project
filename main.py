import numpy as np
from tensorflow.keras.models import load_model
from sklearn.metrics import accuracy_score, classification_report
from preprocessor import preprocess_data  # Import the preprocessor function

# File paths
test_data_file = '/Users/sanjanakusupudi/Downloads/Pedestrian Crossing at unsignalized intersections/Data.csv'
scaler_path = '/Users/sanjanakusupudi/Downloads/Pedestrian Crossing at unsignalized intersections/scaler.pkl'
model_path = 'my_trained_model.h5'
target_column = 'Safety perception (y)'  # Set to None if you don't need labels for prediction

# Step 1: Preprocess the test data
# Note: If `target_column` is None, `preprocess_data` should be modified to handle unlabeled test data
X_test, y_test = preprocess_data(test_data_file, target_column, scaler_path)

# Step 2: Load the saved model
model = load_model(model_path)

# Step 3: Make predictions on the test set
predictions = model.predict(X_test)
predicted_labels = np.argmax(predictions, axis=1)

# Step 4: Evaluate the model
if y_test is not None:
    accuracy = accuracy_score(y_test, predicted_labels)
    print(f'Test accuracy: {accuracy}')

    report = classification_report(y_test, predicted_labels)
    print('Classification Report:')
    print(report)
else:
    print("Test predictions generated but no labels available for evaluation.")
