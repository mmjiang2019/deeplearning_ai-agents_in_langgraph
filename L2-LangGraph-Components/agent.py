import operator

from typing import TypedDict, Annotated
from langchain_core.messages import AnyMessage, SystemMessage, HumanMessage, ToolMessage
from langgraph.graph import StateGraph, END
# from langchain_community.tools.tavily_search import TavilySearchResults
from langchain_tavily import TavilySearch
from IPython.display import Image

from agents_in_langgraph.utils.open_ai import new_chat_open_ai, new_trvily_search

class AgentState(TypedDict):
    messages: Annotated[list[AnyMessage], operator.add]


class Agent():
    def __init__(self, model, tools, system:str=""):
        self.system = system
        graph = StateGraph(AgentState)
        graph.add_node("llm", self.call_openai)
        graph.add_node("action", self.take_action)
        graph.add_conditional_edges(
            "llm", 
            self.exists_action, 
            {True: "action", False: END},
        )
        graph.add_edge("action", "llm")
        graph.set_entry_point("llm")

        self.graph = graph.compile()
        self.tools = {t.name: t for t in tools}
        self.model = model.bind_tools(tools)

    def exists_action(self, state: AgentState):
        result = state['messages'][-1]
        return len(result.tool_calls) > 0
    
    def call_openai(self, state: AgentState):
        messages = state['messages']
        if self.system:
            messages = [SystemMessage(content=self.system)] + messages
        message = self.model.invoke(messages)
        return {'messages': [message]}
    
    def take_action(self, state: AgentState):
        tool_calls = state['messages'][-1].tool_calls
        results = []
        for t in tool_calls:
            print(f"Calling: {t}")
            if not t['name'] in self.tools:      # check for bad tool name from LLM
                print("\n ....bad tool name....")
                result = "bad tool name, retry"  # instruct LLM to retry if bad
            else:
                result = self.tools[t['name']].invoke(t['args'])
            results.append(ToolMessage(tool_call_id=t['id'], name=t['name'], content=str(result)))
        print("Back to the model!")
        return {'messages': results}


if __name__ == '__main__':
    
    # TODO: define a custom tool implemented by ourselves
    tool =  new_trvily_search()
    print(type(tool))
    print(tool.name)

    prompt = """You are a smart research assistant. Use the search engine to look up information. \
    You are allowed to make multiple calls (either together or in sequence). \
    Only look up information when you are sure of what you want. \
    If you need to look up some information before asking a follow up question, you are allowed to do that!
    """

    model_name = "qwen2.5-it:3b"
    
    #reduce inference cost
    print(f"inital conversation:")
    model = new_chat_open_ai(model=model_name)
    abot = Agent(model, [tool], system=prompt)
    Image(abot.graph.get_graph().draw_png())

    messages = [HumanMessage(content="What is the weather in sf?")]
    result = abot.graph.invoke({"messages": messages})

    print(f"result content: \n{result['messages'][-1].content}")

    messages = [HumanMessage(content="What is the weather in SF and LA?")]
    result = abot.graph.invoke({"messages": messages})
    print(f"result content: \n{result['messages'][-1].content}")
    print(f"=================================================================================================")

    # Note, the query was modified to produce more consistent results. 
    # Results may vary per run and over time as search information and models change.
    print(f"new conservative:")
    query = "Who won the super bowl in 2024? In what state is the winning team headquarters located? \
    What is the GDP of that state? Answer each question." 
    messages = [HumanMessage(content=query)]

    model = new_chat_open_ai(model=model_name)  # requires more advanced model
    abot = Agent(model, [tool], system=prompt)
    result = abot.graph.invoke({"messages": messages})
    print(f"result content: \n{result['messages'][-1].content}")
    print(f"=================================================================================================")
