from fastapi import FastAPI
from app.model import Device
# import utils
import ia_process as ia

app = FastAPI()
print("Démarrage du serveur \n")


@app.get("/")
async def root():
    return {"message": "Salut à tous"}


@app.post("/send_data", tags=["Send_Data"])
async def send_data(device: Device):
    print("Requete sendData\n")
    ia.data_to_csv(device)
    return ia.get_prediction()







