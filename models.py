from pydantic import BaseModel


class FeaturesInput(BaseModel):
    Mean_Integrated:float
    SD: float
    EK: float
    Skewness: float
    Mean_DMSNR_Curve: float
    SD_DMSNR_Curve:float
    EK_DMSNR_Curve:float
    Skewness_DMSNR_Curve:float
    Class : int
    

class PredictionModel(BaseModel):
    Class: int
    predict: int
    predict_proba: float