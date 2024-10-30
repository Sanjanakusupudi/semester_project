import pandas as pd
import numpy as np
from sklearn.preprocessing import MinMaxScaler
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Input
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.optimizers import Adam
from sklearn.model_selection import train_test_split
import joblib

# Load your dataset
file_path = '/Users/sanjanakusupudi/Downloads/Pedestrian Crossing at unsignalized intersections/Data.csv'
try:
    data = pd.read_csv(file_path, sep=',')
    print("File loaded successfully.")
except FileNotFoundError:
    raise FileNotFoundError(f"File not found: {file_path}")
except Exception as e:
    raise Exception(f"Error loading file: {e}")

print("Data preview:")
print(data.head())
print("Data shape:", data.shape)

# Inspect the column names
print("Column names:", data.columns)

# Define X and y
# Ensure the column name matches exactly with the one in the DataFrame
target_column = 'Safety perception (y)'
if target_column not in data.columns:
    raise KeyError(f"Column '{target_column}' not found in the DataFrame columns: {data.columns}")

X = data.drop(columns=[target_column])
y = data[target_column]

# Ensure labels are zero-indexed
y = y - 1

# Calculate the number of unique classes
num_classes = len(y.unique())

# Split the data
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

# Fit the scaler on the training data
scaler = MinMaxScaler()
X_train_scaled = scaler.fit_transform(X_train)

# Save the scaler
scaler_path = '/Users/sanjanakusupudi/Downloads/Pedestrian Crossing at unsignalized intersections/scaler.pkl'
joblib.dump(scaler, scaler_path)

# Convert labels to categorical one-hot encoding
y_train = to_categorical(y_train, num_classes)
y_test = to_categorical(y_test, num_classes)

# Define the model architecture
model = Sequential()
model.add(Input(shape=(X_train_scaled.shape[1],)))  # Input layer
model.add(Dense(128, activation='relu'))  # First hidden layer
model.add(Dense(64, activation='relu'))   # Second hidden layer
model.add(Dense(32, activation='relu'))   # Third hidden layer
model.add(Dense(num_classes, activation='softmax'))  # Output layer

# Compile the model
model.compile(optimizer=Adam(),
              loss='categorical_crossentropy',
              metrics=['accuracy'])

# Train the model for 100 epochs
history = model.fit(X_train_scaled, y_train, epochs=100, batch_size=250, validation_split=0.3)

# Save the model
model.save('my_trained_model.h5')

# Evaluate the model
test_loss, test_acc = model.evaluate(X_test, y_test, verbose=2)
print(f'Test accuracy: {test_acc}')