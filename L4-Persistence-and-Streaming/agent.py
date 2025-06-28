import operator
import asyncio

from typing import TypedDict, Annotated
from langchain_core.messages import AnyMessage, SystemMessage, HumanMessage, ToolMessage
from langgraph.graph import StateGraph, END
from IPython.display import Image
from langgraph.checkpoint.sqlite import SqliteSaver
from langgraph.checkpoint.aiosqlite import AsyncSqliteSaver

from agents_in_langgraph.utils.open_ai import new_chat_open_ai
from agents_in_langgraph.utils.tavily_search import new_tavily_search

class AgentState(TypedDict):
    messages: Annotated[list[AnyMessage], operator.add]


class Agent():
    def __init__(self, model, tools, checkpointer, system:str=""):
        self.system = system
        graph = StateGraph(AgentState)
        graph.add_node("llm", self.call_openai)
        graph.add_node("action", self.take_action)
        graph.add_conditional_edges(
            "llm", # where the conditional edge starts
            self.exists_action, # function where to go after LLM returns
            {True: "action", False: END}, # dict of possible destinations nodes
        )
        graph.add_edge("action", "llm") # add an edge from action to llm
        graph.set_entry_point("llm") # set the entry point to llm

        # compile the graph, returning a langchain runnable
        # here we use a sqlite checkpointer
        self.graph = graph.compile(checkpointer=checkpointer)
        self.tools = {t.name: t for t in tools}
        self.model = model.bind_tools(tools) # bind the tools to the model

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
    
    def stream(self, input, config):
        for event in self.graph.stream(input, config):
            for v in event.values():
                print(v['messages'])

async def stream_token():
    with AsyncSqliteSaver.from_conn_string(":memory:") as memory:
                abot = Agent(model, [tool], system=prompt, checkpointer=memory)

                messages = [HumanMessage(content="What is the weather in SF?")]
                thread = {"configurable": {"thread_id": "4"}}
                async for event in abot.graph.astream_events({"messages": messages}, thread, version="v1"):
                    kind = event["event"]
                    if kind == "on_chat_model_stream":
                        content = event["data"]["chunk"].content
                        if content:
                            # Empty content in the context of OpenAI means
                            # that the model is asking for a tool to be invoked.
                            # So we only print non-empty content
                            print(content, end="|")

if __name__ == '__main__':
    
    # TODO: define a custom tool implemented by ourselves
    tool = new_tavily_search(max_results=2)
    print(f"tavily search tool:")
    print(f"type: {type(tool)}, name: {tool.name}")

    prompt = """You are a smart research assistant. Use the search engine to look up information. \
    You are allowed to make multiple calls (either together or in sequence). \
    Only look up information when you are sure of what you want. \
    If you need to look up some information before asking a follow up question, you are allowed to do that!
    """

    model_name = "qwen2.5-it:3b"
    
    #reduce inference cost
    print(f"inital conversation:")
    with SqliteSaver.from_conn_string(":memory:") as memory:
        model = new_chat_open_ai(model=model_name)
        abot = Agent(model, [tool], checkpointer=memory, system=prompt)
        # Comment temporarily
        # Image(abot.graph.get_graph().draw_png())

        messages = [HumanMessage(content="What is the weather in sf?")]
        thread = {"configurable": {"thread_id": "1"}}
        # Different from the code block in L1, we use stream instead of invoke
        # result = abot.graph.invoke({"messages": messages})
        # print(f"result content: \n{result['messages'][-1].content}")
        for event in abot.graph.stream({"messages": messages}, thread):
            for v in event.values():
                print(v['messages'])


        messages = [HumanMessage(content="What is the weather in SF and LA?")]
        # result = abot.graph.invoke({"messages": messages})
        # print(f"result content: \n{result['messages'][-1].content}")
        
        abot.stream(input={"messages": messages}, config=thread)
        print(f"=================================================================================================")

        # Note, the query was modified to produce more consistent results. 
        # Results may vary per run and over time as search information and models change.
        print(f"new conservative:")
        query = "Who won the super bowl in 2024? In what state is the winning team headquarters located? \
        What is the GDP of that state? Answer each question." 
        messages = [HumanMessage(content=query)]

        model = new_chat_open_ai(model=model_name)  # requires more advanced model
        abot = Agent(model, [tool], checkpointer=memory, system=prompt)
        # result = abot.graph.invoke({"messages": messages})
        # print(f"result content: \n{result['messages'][-1].content}")
        abot.stream(input={"messages": messages}, config=thread)
        print(f"=================================================================================================")

    # Streaming tokens
    # TODO: need validations in the future
    asyncio.to_thread(
        stream_token()
    )