import pandas as pd
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Input
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.optimizers import Adam
from sklearn.model_selection import train_test_split
import joblib
from preprocessor import preprocess_data  # Import the preprocessor function

# File paths
file_path = '/Users/sanjanakusupudi/Downloads/Pedestrian Crossing at unsignalized intersections/Data.csv'
scaler_path = '/Users/sanjanakusupudi/Downloads/Pedestrian Crossing at unsignalized intersections/scaler.pkl'
target_column = 'Safety perception (y)'

# Step 1: Preprocess data
X, y = preprocess_data(file_path, target_column, scaler_path)

# Ensure labels are zero-indexed (if needed)
y = y - 1

# Calculate the number of unique classes
num_classes = len(y.unique())

# Step 2: Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

# Step 3: Convert labels to categorical one-hot encoding
y_train = to_categorical(y_train, num_classes)
y_test = to_categorical(y_test, num_classes)

# Step 4: Define the model architecture
model = Sequential()
model.add(Input(shape=(X_train.shape[1],)))  # Input layer
model.add(Dense(128, activation='relu'))  # First hidden layer
model.add(Dense(64, activation='relu'))   # Second hidden layer
model.add(Dense(32, activation='relu'))   # Third hidden layer
model.add(Dense(num_classes, activation='softmax'))  # Output layer

# Step 5: Compile the model
model.compile(optimizer=Adam(),
              loss='categorical_crossentropy',
              metrics=['accuracy'])

# Step 6: Train the model
history = model.fit(X_train, y_train, epochs=100, batch_size=250, validation_split=0.3)

# Step 7: Save the trained model
model.save('my_trained_model.h5')

# Step 8: Evaluate the model on the test data
test_loss, test_acc = model.evaluate(X_test, y_test, verbose=2)
print(f'Test accuracy: {test_acc}')
