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

# Requirements
## python 版本
python >= 3.13.5