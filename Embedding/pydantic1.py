from pydantic import BaseModel

class Embedding(BaseModel):

    name:str
student= {'name':"Aayush"}

Student=Embedding(**student)