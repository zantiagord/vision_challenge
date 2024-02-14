# Home Price Prediction Model

This section explains how to import the `home_price_model` and use it to predict the price of a home based on various features.

## Requirements

Ensure you have `scikit-learn` installed in your environment, as the model uses `scikit-learn` for loading and prediction.

```bash
pip install scikit-learn
```

## Importing the Model

First, import the necessary modules and load the models:

```python
import joblib

# Load the model
model_path = 'path_to_your_saved_model/home_price_model.pkl'
home_price_model = joblib.load(model_path)
scaler = joblib.load('scaler.pkl')

```

## Making Predictions 

To make predictions, you need to provide input data as a dictionary or a DataFrame with the following features: 'ClosePrice', 'BathsTotal', 'BedsTotal', 'CDOM', 'LotSizeAreaSQFT', 'SqFtTotal', 'ElementarySchoolName'. Here's an example using a dictionary:

```python
import pandas as pd

input_data = {
    "BathsTotal":[
       3.0,
       2.1,
       1.1
    ],
    "BedsTotal":[
       4,
       4,
       1
    ],
    "CDOM":[
       52,
       58,
       38
    ],
    "LotSizeAreaSQFT":[
       7100.28,
       8712.0,
       1306.8
    ],
    "SqFtTotal":[
       2484,
       2631,
       884
    ],
    "ElementarySchoolName":[
       "Allen",
       "Fisher",
       "Bright"
    ]
 }


df_input = pd.DataFrame(data_example)
df_input['ElementarySchoolName'] = df_input['ElementarySchoolName'].astype('category').cat.codes
df_input[['CDOM', 'LotSizeAreaSQFT', 'SqFtTotal']] = scaler.fit_transform(df_input[['CDOM', 'LotSizeAreaSQFT', 'SqFtTotal']])
predictions = home_price_model.predict(df_input)
# do an inverse np.log transformation to get the real values
predictions = pd.Series(predictions).apply(lambda x: np.exp(x))
# show the predictions
print(predictions)
```

## Explanation


In this example, we first load the home_price_model using joblib.load(). We then create a DataFrame input_df with our input features total baths, total beds, cumulative days on market (CDOM), lot size area in square feet, total square footage of the home, and the name of the elementary school. Before making predictions, we scale the numerical features using the scaler loaded from scaler.pkl. This ensures that the input data is on the same scale as the data used to train the model. After scaling, we use the .predict() method on our model with input_df as the argument to predict the home price. The output is a prediction of the home price based on the given features.

Remember to replace 'path_to_your_saved_model/home_price_model.pkl' with the actual path to your saved model file and provide information about where the scaler model was saved.

## example

an example about how to use it is provided in predict_example.py.