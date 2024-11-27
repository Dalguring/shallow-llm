from fastapi import FastAPI
from app.controllers import chat_controller
from fastapi.middleware.cors import CORSMiddleware

app = FastAPI()

app.include_router(chat_controller.router)
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# 앱 실행
if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
