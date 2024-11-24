from fastapi import FastAPI, UploadFile, File 
from fastapi.responses import StreamingResponse
from pydantic import BaseModel, Field
from enum import Enum
from typing import List, Optional
import pickle 
import pandas as pd
import numpy as np
from io import StringIO
import os 

app = FastAPI()

class SellerType(str, Enum):
    Individual = 'Individual'
    Dealer = 'Dealer'
    TrustmarkDealer = 'Trustmark Dealer'

class Fuel(str, Enum):
    Diesel = 'Diesel' 
    Petrol = 'Petrol' 
    LPG = 'LPG' 
    CNG = 'CNG'

class Transmission(str, Enum):
    Manual = 'Manual' 
    Automatic = 'Automatic'

class Owner(str, Enum):
    FirstOwner = 'First Owner'
    SecondOwner = 'Second Owner'
    ThirdOwner = 'Third Owner'
    More = 'Fourth & Above Owner'
    Test = 'Test Drive Car'

class Item(BaseModel):
    name: str = Field(description= 'Название автомобиля', example= 'Maruti Swift Dzire VDI', pattern= r"[A-Za-z0-9\s]+$")
    year: int = Field(..., description= 'Год выпуска автомобиля', example='2020', gt=1900)
    selling_price: Optional[int] = None 
    km_driven: int = Field(..., description= 'Пробег')
    fuel: Fuel 
    seller_type: SellerType
    transmission: Transmission
    owner: Owner
    mileage: float = Field(..., description= 'Расход топлива(в километрах на литр)')
    engine: int = Field(..., description= 'Мощность двигателя')
    max_power: float = Field(..., description= 'Максимальная мощность')
    torque: str = Field(..., description= 'Момент силы')
    seats: int = Field(..., description = 'Число сидений')


class Items(BaseModel):
    objects: List[Item]

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))  # Папка проекта
MODELS_DIR = os.path.join(BASE_DIR, "models")
# загружаю несколько предобученных моделей, которые выдавали более менее нормальные предсказания
with open(os.path.join(MODELS_DIR, "car_price_predict_Ridge.pkl"), "rb") as f:
        ridge = pickle.load(f)
with open(os.path.join(MODELS_DIR,"scaler_categorical.pkl"), "rb") as f:
        scaler = pickle.load(f)





@app.get("/")
def root():
    return {'message': 'API для предсказания стоимости автомобилей'}


@app.post("/linear_regressor_1/predict_item")
def predict_item(item: Item) -> float:
    data = pd.DataFrame(item.model_dump(), index=[0])
    data_cat = pd.get_dummies(data, drop_first=True)
    expected_columns = scaler.feature_names_in_  
    missing_cols = set(expected_columns) - set(data_cat.columns)
    for col in missing_cols:
        data_cat[col] = 0
    data_cat = data_cat[expected_columns]
    data_cat = pd.DataFrame(scaler.transform(data_cat))
    pred = ridge.predict(data_cat)
    predicted_price = float(np.exp(pred)[0])
    return  predicted_price



@app.post("/predict_items")
def predict_items(items: List[Item]) -> List[float]:
    data = pd.DataFrame([item.model_dump() for item in items])
    data_cat = pd.get_dummies(data, drop_first=True)
    expected_columns = scaler.feature_names_in_  
    missing_cols = set(expected_columns) - set(data_cat.columns)
    for col in missing_cols:
        data_cat[col] = 0
    data_cat = data_cat[expected_columns]   
    data_cat = pd.DataFrame(scaler.transform(data_cat))
    predicted_prices = np.exp(ridge.predict(data_cat))
    return [float(price) for price in predicted_prices]

@app.post("/upload_csv/")
async def upload_csv(file: UploadFile = File(...)):
    content = await file.read()
    df = pd.read_csv(StringIO(content.decode('utf-8')))

    items = []
    for index, row in df.iterrows():
        try:
            item = Item(**row.to_dict())
            items.append(item)
        except Exception as e:
            return {"error": f"Row {index} is invalid: {str(e)}"}

    data = pd.DataFrame([item.model_dump() for item in items])
    data_cat = pd.get_dummies(data, drop_first=True)
    expected_columns = scaler.feature_names_in_
    missing_cols = set(expected_columns) - set(data_cat.columns)
    for col in missing_cols:
        data_cat[col] = 0
    data_cat = data_cat[expected_columns]
    data_cat = pd.DataFrame(scaler.transform(data_cat))
    predicted_prices = np.exp(ridge.predict(data_cat))

    df['predicted_price'] = predicted_prices

    output = StringIO()
    df.to_csv(output, index=False)
    output.seek(0)

    return StreamingResponse(output, media_type="text/csv", headers={"Content-Disposition": "attachment; filename=predicted_prices.csv"})
