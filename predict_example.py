import joblib
import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
import json

model_path = 'home_price_model.pkl'
home_price_model = joblib.load(model_path)
scaler = joblib.load('scaler.pkl')
data_example = {
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