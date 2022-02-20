from fastapi import FastAPI
from app.model import Device
from app.utils import data_to_csv
from app.ia_process import get_prediction

app = FastAPI()
print("Démarrage du serveur \n")


@app.get("/")
async def root():
    return {"message": "Salut à tous\n"}


@app.post("/send_data", tags=["Send_Data"])
async def send_data(device: Device):
    print("Requete pour prédiction\n")
    # data_to_csv(device)
    return get_prediction()







