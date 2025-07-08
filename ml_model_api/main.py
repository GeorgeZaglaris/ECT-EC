from fastapi import FastAPI
from pydantic import BaseModel
from pycaret.regression import load_model, predict_model
import pandas as pd

app = FastAPI()

model = load_model('my_model_et')

class PMVInput(BaseModel):
    air_temperature: float
    mean_radiant_temperature:float
    relative_humidity: float
    air_velocity: float
    clo: float
    met: float

@app.post("/predict")
def predict_pmv(data: PMVInput):
    try:
        # Μετατροπή εισόδου σε dataframe
        input_data = pd.DataFrame([{
            "Air temperature (C)": data.air_temperature,
            "Mean Radiant Temperature(C)":data.mean_radiant_temperature,
            "Relative humidity (%)": data.relative_humidity,
            "Air velocity (m/s)": data.air_velocity,
            "Clo": data.clo,
            "Met": data.met
        }])
        
        print("Input data:\n", input_data)  # debug print

        # Πρόβλεψη
        output = predict_model(model, data=input_data)
        print("Prediction result:\n", output)  # debug print

        return {"predicted_PMV": output['prediction_label'][0]}
    
    except Exception as e:
        return {"error": str(e)}
