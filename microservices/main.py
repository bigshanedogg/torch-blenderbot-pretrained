import os
import sys
import random
import uvicorn
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
sys.path.append(os.getcwd())
from microservices.setting.config import load_config
from microservices.services import empathetic_bot, persona_bot
from microservices.utils.common import read_json

def create_app():

    # initialize app
    app = FastAPI(title="AIBUD")

    # define router
    app.include_router(empathetic_bot.router, prefix='/empathetic-bot', tags=["EmpatheticBotServices"])
    app.include_router(persona_bot.router, prefix='/persona-bot', tags=["PersonaBotServices"])
    # app.include_router(blender_bot.router, tags=["BlenderBotServices"])

    # cors setting
    origins = [
        "*"
    ]

    app.add_middleware(
        CORSMiddleware,
        allow_origins=origins,
        allow_credentials=True,
        allow_methods=["*"],
        allow_headers=["*"],
    )

    return app

app = create_app()

if __name__ == "__main__":
    json_path = "./microservices/server_config.json"
    server_config = read_json(json_path)
    env_config = load_config(server_config["env"])

    random.seed(12345)
    host = server_config["host"]
    port = server_config["port"]
    if isinstance(port, str): port = int(port)
    uvicorn.run("main:app", host=host, port=port, reload=["PROJ_RELOAD"])