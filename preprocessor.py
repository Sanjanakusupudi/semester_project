import pandas as pd
from sklearn.preprocessing import MinMaxScaler
from sklearn.impute import SimpleImputer
import joblib

def preprocess_data(file_path, target_column, scaler_path):
    # Step 1: Load the data
    try:
        data = pd.read_csv(file_path)
        print("File loaded successfully.")
    except FileNotFoundError:
        raise FileNotFoundError(f"File not found: {file_path}")
    except Exception as e:
        raise Exception(f"Error loading file: {e}")

    # Step 2: Check for missing values
    if data.isnull().values.any():
        print("Missing values found. Imputing missing values.")
        # Impute missing values with the column mean
        imputer = SimpleImputer(strategy="mean")
        data = pd.DataFrame(imputer.fit_transform(data), columns=data.columns)
    else:
        print("No missing values found.")

    # Step 3: Separate features (X) and target (y)
    if target_column not in data.columns:
        raise KeyError(f"Column '{target_column}' not found in the DataFrame columns: {data.columns}")
    
    X = data.drop(columns=[target_column])
    y = data[target_column]

    # Step 4: Check if there are any non-numeric columns and encode them if necessary
    if X.select_dtypes(include=['object']).shape[1] > 0:
        print("Non-numeric columns found. Encoding categorical variables.")
        X = pd.get_dummies(X, drop_first=True)

    # Step 5: Fit and save the scaler on the features
    scaler = MinMaxScaler()
    X_scaled = scaler.fit_transform(X)
    joblib.dump(scaler, scaler_path)

    # Step 6: Return the processed features and target
    return X_scaled, y

# Run the preprocessor
if __name__ == "__main__":
    file_path = '/Users/sanjanakusupudi/Downloads/Pedestrian Crossing at unsignalized intersections/Data.csv'
    target_column = 'Safety perception (y)'
    scaler_path = '/Users/sanjanakusupudi/Downloads/Pedestrian Crossing at unsignalized intersections/scaler.pkl'

    X, y = preprocess_data(file_path, target_column, scaler_path)
    print("Data preprocessed and scaler saved.")
