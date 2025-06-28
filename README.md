# deeplearning_ai-agents_in_langgraph
## L1 Simple ReAct(React and Action) Agent from Scratch
based on https://til.simonwillison.net/llms/python-react-pattern

## L2 LangGraph Components
### tavily api key 问题
可以通过以下网站，获取 api key（默认会给有一个免费的，默认1,000 credit/month）
Getting Started with Tavily Search | Tavily AI[https://app.tavily.com/home]

### tavily search API 问题
TavilySearchResult 已经被废弃，换成: TavilySearch

### libgraphviz 问题
需要安装 graphviz 及相关依赖包

#### Ubuntu
sudo apt-get install -y pygraphviz libgraphviz-dev

#### Windows
避坑指南：Windows下pygraphviz安装全攻略[https://blog.csdn.net/daqianai/article/details/148660117]
##### windows的 **powershell** 里执行 (换行连接符为 `)
uv add -C="--global-option=build_ext" `
  -C="--global-option=-IC:\Program Files\Graphviz\include" `
  -C="--global-option=-LC:\Program Files\Graphviz\lib" pygraphviz==1.14
 
##### windows的 **cmd** 里执行 (换行连接符为 ^)
uv add -C="--global-option=build_ext" ^
  -C="--global-option=-IC:\Program Files\Graphviz\include" ^
  -C="--global-option=-LC:\Program Files\Graphviz\lib" pygraphviz==1.14

## L3 Persistence and Streaming
关于 SqliteSaver 的使用问题
### 1. langgraph.checkpoint.sqlite 并不在 panggraph 包里，而是在 langgraph-checkpoint-sqlite 里，需要单独安装：
uv add langgraph-checkpoint-sqlite
uv pip install langgraph-checkpoint-sqlite
以上方式二选一皆可

### 2. SqliteSaver.from_conn_string(":memory:") 使用
这里的 SqliteSaver.from_conn_string(":memory:") 是一个 context 上下文 manager， 并不是一个 checkpointer。两种方法解决以上问题
#### 方法一： 直接使用 SqliteSaver 作为 checkpointer
with SqliteSaver.from_conn_string(":memory:") as memory:
    model = new_chat_open_ai(model=model_name)
    abot = Agent(model, [tool], checkpointer=memory, system=prompt)

#### 方法二： 使用 SqliteSaver.from_conn_string 作为 checkpointer
from contextlib import ExitStack

# Use contextlib to manually enter the context manager and keep the object
stack = ExitStack()

# Create the memory context manager and enter the context
memory= stack.enter_context(SqliteSaver.from_conn_string(":memory:"))

# Now the checkpointer can be reused across cells
abot = Agent(model, [tool], system=prompt, checkpointer=memory)

thread = {"configurable": {"thread_id": "1"}}

for event in abot.graph.stream({"messages": messages}, thread):
    for v in event.values():
        print(v['messages'])
...
# Close the context manually when done
stack.close()

参考链接：
AttributeError: '_GeneratorContextManager' object has no attribute 'get_next_version'[https://github.com/langchain-ai/langgraph/discussions/1696]

# Requirements
## python 版本
python >= 3.13.5