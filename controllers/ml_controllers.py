from fastapi import APIRouter, HTTPException, status
import pickle
import numpy as np  
from ml_models import load_model_from_pickle
from models import FeaturesInput, PredictionModel
import warnings

warnings.filterwarnings("ignore", category=UserWarning, module="sklearn")


model = load_model_from_pickle('logreg_balanced.pkl')

router = APIRouter()
preds = []

@router.get("/ml")
def get_preds():
    return preds

@router.post("/ml", status_code = status.HTTP_201_CREATED, response_model = PredictionModel)
def predict(pred_input : FeaturesInput):
    input_data = np.array([
    [
        pred_input.Mean_Integrated,
        pred_input.SD,
        pred_input.EK,
        pred_input.Skewness,
        pred_input.Mean_DMSNR_Curve,
        pred_input.SD_DMSNR_Curve,
        pred_input.EK_DMSNR_Curve,
        pred_input.Skewness_DMSNR_Curve
    ]
])
    true_value = pred_input.Class
    prediction_f = model.predict(input_data)
    preds_proba = model.predict_proba(input_data)
    predict_dict = {
        "Class": true_value,
        "predict": int(prediction_f), 
        "predict_proba":float(preds_proba[0][1])}
    preds.append(predict_dict)

    return predict_dict

@router.delete("/ml")
def delete_preds():
    global preds
    preds = []
    return {"message": "Todas las predicciones han sido eliminadas"}


@router.put("/ml/{prediction_idx}", response_model=PredictionModel)
def update_prediction(prediction_idx: int, pred_input: FeaturesInput):
    if 0 <= prediction_idx < len(preds):
        input_data = np.array([
            [
                pred_input.Mean_Integrated,
                pred_input.SD,
                pred_input.EK,
                pred_input.Skewness,
                pred_input.Mean_DMSNR_Curve,
                pred_input.SD_DMSNR_Curve,
                pred_input.EK_DMSNR_Curve,
                pred_input.Skewness_DMSNR_Curve
            ]
        ])
        prediction_f = model.predict(input_data)
        preds_proba = model.predict_proba(input_data)
        
        updated_prediction = {
            "predict": float(prediction_f),
            "predict_proba": float(preds_proba[0][1])
        }
        
        preds[prediction_idx] = updated_prediction
        return updated_prediction
    else:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="Predicción no encontrada")

