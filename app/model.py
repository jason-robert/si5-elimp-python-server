from typing import List
from pydantic import BaseModel


class Device(BaseModel):
    labels:str
    values:List[str]