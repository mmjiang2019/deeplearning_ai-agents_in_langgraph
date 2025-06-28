import operator
import asyncio

from uuid import uuid4

from typing import TypedDict, Annotated
from langchain_core.messages import AnyMessage, SystemMessage, HumanMessage, ToolMessage
from langgraph.graph import StateGraph, END
from IPython.display import Image
from langgraph.checkpoint.sqlite import SqliteSaver
from langgraph.checkpoint.sqlite.aio import AsyncSqliteSaver

from agents_in_langgraph.utils.open_ai import new_chat_open_ai
from agents_in_langgraph.utils.tavily_search import new_tavily_search


def reduce_messages(left: list[AnyMessage], right: list[AnyMessage]) -> list[AnyMessage]:
    """
    In previous examples we've annotated the `messages` state key
    with the default `operator.add` or `+` reducer, which always
    appends new messages to the end of the existing messages array.

    Now, to support replacing existing messages, we annotate the
    `messages` key with a customer reducer function, which replaces
    messages with the same `id`, and appends them otherwise.
    """
    # assign ids to messages that don't have them
    for message in right:
        if not message.id:
            message.id = str(uuid4())
    # merge the new messages with the existing messages
    merged = left.copy()
    for message in right:
        for i, existing in enumerate(merged):
            # replace any existing messages with the same id
            if existing.id == message.id:
                merged[i] = message
                break
        else:
            # append any new messages to the end
            merged.append(message)
    return merged

class AgentState(TypedDict):
    messages: Annotated[list[AnyMessage], reduce_messages]


class Agent():
    def __init__(self, model, tools, checkpointer=None, system:str=""):
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
        self.graph = graph.compile(
            checkpointer=checkpointer,
            # add human intterrupt before "action" node
            interrupt_before=["action"],
            )
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

        # Manual human approval
        abot.stream(input={"messages": messages}, config=thread)
        state = abot.graph.get_state(thread)
        print(f"state: \n{state}, next state: {state.next}")
        print(f"=================================================================================================")

        # Continue after interrupt
        print(f"Continue after interrupt")
        abot.stream(input=None, config=thread)
        state = abot.graph.get_state(thread)
        print(f"state: \n{state}, next state: {state.next}")

        messages = [HumanMessage("Whats the weather in LA?")]
        thread = {"configurable": {"thread_id": "2"}}

        abot.stream(input={"messages": messages}, config=thread)

        # Wile loop for human to decide whether to proceed or quit
        while abot.graph.get_state(thread).next:
            print("\n", abot.graph.get_state(thread),"\n")
            _input = input("proceed?")
            if _input != "y":
                print("aborting")
                break
            for event in abot.graph.stream(None, thread):
                for v in event.values():
                    print(v)