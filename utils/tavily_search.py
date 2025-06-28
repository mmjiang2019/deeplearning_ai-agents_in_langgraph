# from langchain_community.tools.tavily_search import TavilySearchResults # deprecated
from tavily import TavilyClient
from langchain_tavily import TavilySearch

from agents_in_langgraph.config import tavily_search as tavily_config

def get_tavily_key() -> str:
    return tavily_config.TAVILY_API_KEY

def new_trvily_search(
        max_results: int=4, 
        tavily_api_key: str = get_tavily_key()
        ):
    return TavilySearch(
        max_results=max_results, 
        tavily_api_key = tavily_api_key)

def new_trvily_client(
        tavily_api_key: str = get_tavily_key()
        ):
    return TavilyClient(
        api_key=os.environ.get("TAVILY_API_KEY"),
        )