import openai

from langchain_openai import ChatOpenAI

from agents_in_langgraph.config import open_ai as open_ai_config

def get_base_url() -> str:
    return open_ai_config.BASE_URL

def get_api_key() -> str:
    return open_ai_config.API_KEY

def new_open_ai():
    return openai.OpenAI(
        base_url=open_ai_config.BASE_URL,
        api_key=open_ai_config.API_KEY,  # this is also the default, it can be omitted
    )

def new_chat_open_ai(model: str, temperature: float = 0.0):
    return ChatOpenAI(
        base_url=get_base_url(),
        api_key=get_api_key(),
        temperature=temperature, model=model)