import pandas as pd
import numpy as np
from sklearn import linear_model
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import r2_score
from fastapi import FastAPI
from pydantic import BaseModel
from fastapi.middleware.cors import CORSMiddleware

# Define request body
class CarData(BaseModel):
    name: str
    year: int
    fuel: str
    seller_type: str
    km_driven: int
    transmission: str
    owner: str
app = FastAPI()
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # allow live server
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


# Train model once at startup
df = pd.read_csv("car_dataset.csv")

encoders = {}
features = ["name", "fuel", "seller_type", "transmission", "owner"]
for feature in features:
    le = LabelEncoder()
    df[feature] = le.fit_transform(df[feature].str.lower())
    encoders[feature] = le

X = df[["name", "year", "fuel", "seller_type", "km_driven", "transmission", "owner"]]
y = df[["selling_price"]]

model = linear_model.LinearRegression()
x_train, x_test, y_train, y_test = train_test_split(X, y, test_size=0.1)
model.fit(x_train, y_train)

@app.post("/predict")
def predict(data: CarData):
    try:
        # Encode categorical features
        input_dict = {
            "name": encoders["name"].transform([data.name.lower()])[0],
            "year": data.year,
            "fuel": encoders["fuel"].transform([data.fuel.lower()])[0],
            "seller_type": encoders["seller_type"].transform([data.seller_type.lower()])[0],
            "km_driven": data.km_driven,
            "transmission": encoders["transmission"].transform([data.transmission.lower()])[0],
            "owner": encoders["owner"].transform([data.owner.lower()])[0],
        }

        input_df = pd.DataFrame([input_dict])
        prediction = model.predict(input_df)[0][0]  # scalar value
        return {"price": float(prediction)}

    except Exception as e:
        return {"error": str(e)}
