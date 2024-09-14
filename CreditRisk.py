from pydantic import BaseModel

class CreditRiskData(BaseModel):
    age:float
    income:float
    home:float
    emp_length:float
    intent:float
    amount:float
    rate:float
    status:float
    percent_income:float
    cred_length:float

