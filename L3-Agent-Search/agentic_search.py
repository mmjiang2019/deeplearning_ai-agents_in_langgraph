import json
from pygments import highlight, lexers, formatters

from agents_in_langgraph.utils.tavily_search import new_trvily_client

class tavily_search:
    def __init__(self):
        self.client = new_trvily_client()

    def search(self, query):
        # run search
        result = self.client.search(
            query,
            include_answer=True)
        
        return result
    
    def parse_result(self, date):
        # parse JSON
        parsed_json = json.loads(data.replace("'", '"'))

        # pretty print JSON with syntax highlighting
        formatted_json = json.dumps(parsed_json, indent=4)
        colorful_json = highlight(
            formatted_json,
            lexers.JsonLexer(),
            formatters.TerminalFormatter())
        

if __name__ == "__main__":
    client = tavily_search()
    result = client.search("what is the current weather in San Francisco?")
    
    # print first result
    data = result["results"][0]["content"]
    print(data)

    colorful_json = client.parse_result(data)
    print(colorful_json)