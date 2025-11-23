from pydantic import BaseModel


class get_user(BaseModel):
    access_token: str
    user_id: str