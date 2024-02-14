import joblib
import pandas as pd
import numpy as np

def load_model_and_scaler(model_path='home_price_model.pkl', scaler_path='scaler.pkl'):
    """Load the model and scaler from disk."""
    model = joblib.load(model_path)
    scaler = joblib.load(scaler_path)
    return model, scaler

def prepare_data(data):
    """Prepare input data for prediction."""
    df = pd.DataFrame(data)
    df['ElementarySchoolName'] = df['ElementarySchoolName'].astype('category').cat.codes
    numeric_features = ['CDOM', 'LotSizeAreaSQFT', 'SqFtTotal']
    df[numeric_features] = scaler.transform(df[numeric_features])
    return df

def make_predictions(model, scaler, data):
    """Make predictions using the preprocessed data."""
    df_input = prepare_data(data)
    predictions = model.predict(df_input)
    return pd.Series(predictions).apply(np.exp)

# Load the model and scaler
home_price_model, scaler = load_model_and_scaler(model_path='home_price_model.pkl', scaler_path='scaler.pkl')

# Example input data
data_example = {
    "BathsTotal": [3.0, 2.1, 1.1],
    "BedsTotal": [4, 4, 1],
    "CDOM": [52, 58, 38],
    "LotSizeAreaSQFT": [7100.28, 8712.0, 1306.8],
    "SqFtTotal": [2484, 2631, 884],
    "ElementarySchoolName": ["Allen", "Fisher", "Bright"]
}

# Make predictions
predictions = make_predictions(home_price_model, scaler, data_example)

# Print the predictions
print(predictions)
