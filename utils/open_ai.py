import openai

from lang_graph_project.config import open_ai as open_ai_config

def get_base_url() -> str:
    return open_ai_config.BASE_URL

def get_api_key() -> str:
    return open_ai_config.API_KEY

def new_open_ai_client():
    return openai.OpenAI(
        base_url=open_ai_config.BASE_URL,
        api_key=open_ai_config.API_KEY,  # this is also the default, it can be omitted
    )
