from pydantic import BaseModel

class PatientData(BaseModel):
    age: float
    sexe: int
    temperature: float
    douleur: float
    vomissement: int
    leucocytes: float
    neutrophile: float
    crp: float
    score_alvarado: float
    echographie: float