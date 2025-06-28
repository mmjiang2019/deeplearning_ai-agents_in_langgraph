import requests
from bs4 import BeautifulSoup
from duckduckgo_search import DDGS
import re

class regular_search:
    def __init__(self, max_results: int = 6):
        self.max_results = max_results
        self.ddg = DDGS()

    def search(self, query, max_results=6):
        if not max_results:
            max_results = self.max_results
        
        try:
            results = self.ddg.text(query, max_results=max_results)
            return [i["href"] for i in results]
        except Exception as e:
            print(f"returning previous results due to exception reaching ddg.")
            results = [ # cover case where DDG rate limits due to high deeplearning.ai volume
                "https://weather.com/weather/today/l/USCA0987:1:US",
                "https://weather.com/weather/hourbyhour/l/54f9d8baac32496f6b5497b4bf7a277c3e2e6cc5625de69680e6169e7e38e9a8",
            ]
            return results  

    def scrape_weather_info(self, url):
        """Scrape content from the given URL"""
        if not url:
            return "Weather information could not be found."
        
        # fetch data
        headers = {'User-Agent': 'Mozilla/5.0'}
        response = requests.get(url, headers=headers)
        if response.status_code != 200:
            return "Failed to retrieve the webpage."

        # parse result
        soup = BeautifulSoup(response.text, 'html.parser')
        print(f"Website: {url}\n\n")
        print(str(soup.body)[:50000]) # limit long outputs
        return soup
    
    def extract_text(self, soup):
        # extract text
        weather_data = []
        for tag in soup.find_all(['h1', 'h2', 'h3', 'p']):
            text = tag.get_text(" ", strip=True)
            weather_data.append(text)

        # combine all elements into a single string
        weather_data = "\n".join(weather_data)

        # remove all spaces from the combined text
        weather_data = re.sub(r'\s+', ' ', weather_data)

        print(f"Website: {url}\n\n")
        print(weather_data)
        
        return weather_data


if __name__ == '__main__':
    # choose location (try to change to your own city!)

    city = "San Francisco"

    query = f"""
        what is the current weather in {city}?
        Should I travel there today?
        "weather.com"
    """

    reg_search = regular_search()
    # use DuckDuckGo to find websites and take the first result
    result = reg_search.search(query)
    for i in result:
        print(i)

    # scrape first wesbsite
    url = result[0]
    soup = reg_search.scrape_weather_info(url)

    # extract text
    weather_data = reg_search.extract_text(soup)
    