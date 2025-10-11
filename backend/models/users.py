from typing import Generic, Optional, TypeVar,  List
from pydantic.generics import GenericModel
from pydantic import BaseModel, Field, EmailStr, constr
from enum import Enum

T = TypeVar('T')

#Login
class Login(BaseModel):
    username: str
    password: str

#Register
# ---------- Gender Enum ----------
class GenderEnum(str, Enum):
    male = "Male"
    female = "Female"
    other = "Other"

# ---------- CareRecipient Input Model ----------
class CareRecipientCreate(BaseModel):
    full_name: str = Field(..., example="Alice Smith")
    email: EmailStr = Field(..., example="alice@example.com")
    phone_number: constr(min_length=10, max_length=10) = Field(..., example="9876543210")
    age: int = Field(..., example=70)
    gender: GenderEnum = Field(..., example="Female")
    respiratory_condition_status: bool = Field(default=False)

# ---------- CareTaker Registration Model ----------
class Register(BaseModel):
    id:str
    email: EmailStr = Field(..., example="john@example.com")
    username: str = Field(..., example="john_doe")
    phone_number: constr(min_length=10, max_length=10) = Field(..., example="9999999999")
    password: str = Field(..., example="strongpassword123")
    full_name: str = Field(..., example="John Doe")
    # Must provide at least 1 care recipient
    care_recipients: List[CareRecipientCreate] = Field(..., min_items=1)

#response model
class ResponseSchema(BaseModel):
    code: str
    status: str
    message: str
    result: Optional[T]= None

#token
class Token(BaseModel):
    access_token: str
    token_type: str