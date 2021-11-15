from datetime import datetime
from typing import Optional, Any
from pydantic import BaseModel

class Request(BaseModel):
    pass

class SetDeviceRequest(Request):
    device: str

class LoadModelRequest(Request):
    path: str

class Response(BaseModel):
    status: bool = False
    output: Any = None
    execution_begin: datetime = ""
    execution_end: datetime = ""
    execution_time: float = 0.0
    error_message: Optional[str] = None
