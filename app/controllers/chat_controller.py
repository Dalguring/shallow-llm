from fastapi import APIRouter, HTTPException
from pydantic import BaseModel
from app.models.llm_model import LLMModel

router = APIRouter()


class ChatRequest(BaseModel):
    user_input: str


llm_model = LLMModel()


@router.post("/chat")
async def chat_endpoint(request: ChatRequest):
    user_input = request.user_input.strip()

    if not user_input:
        raise HTTPException(status_code=400, detail="User input is empty.")

    try:
        response, time_taken = llm_model.generate_response(user_input)
        return {"response": response.split('#answer:')[-1].strip(),
                "time taken": f"{time_taken:.2f} seconds"}
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"An error occurred: {str(e)}")
