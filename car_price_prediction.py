import pandas as pd
import numpy as np
from sklearn import linear_model
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import r2_score
df = pd.read_csv('car_dataset.csv')
encoders = {}
features = ['name','fuel','seller_type','transmission','owner']
for feature in features:
    le = LabelEncoder()
    df[feature] = le.fit_transform(df[feature].str.lower())
    encoders[feature] = le
x = df[['name','year','fuel','seller_type','km_driven','transmission','owner']]
y = df[['selling_price']]
model = linear_model.LinearRegression()
x_train,x_test,y_train,y_test = train_test_split(x,y,test_size = 0.1)
model.fit(x_train,y_train)
y_pred = model.predict(x_test)
print(f"r2_score: {r2_score(y_test['selling_price'],y_pred)}")
x_pred = {}
for i in x:
    if i in features:
        value = input('Enter '+i+' :').lower()
        x_pred[i] = encoders[i].transform([value])[0]
    else:
        x_pred[i] = int(input('Enter '+i+' :'))
input_values = pd.DataFrame(np.array(list(x_pred.values())).reshape(1,-1),columns = x.columns)
print(model.predict(input_values))