# Home Price Prediction Model

This section explains how to import the `home_price_model` and use it to predict the price of a home based on various features.

## Requirements


```bash
pip install -r requirements.txt 
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

```
## Explanation

Overall, this code demonstrates how to load a trained machine learning model and scaler, preprocess input data for prediction, and make predictions on new data using the trained model.


## Example

an example about how to use it is provided in predict_example.py.