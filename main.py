import os
import certifi
from dotenv import load_dotenv
from langchain_openai import ChatOpenAI
from langchain_community.tools.tavily_search import TavilySearchResults
from langchain import hub
from langchain.tools import tool
import requests

from langchain.agents import create_react_agent, AgentExecutor  



os.environ["SSL_CERT_FILE"] = certifi.where()
load_dotenv()

API_BASE_URL = os.getenv("API_BASE_URL")
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
TAVILY_API_KEY = os.getenv("TAVILY_API_KEY")
WEATHERSTACK_API_KEY = os.getenv("WEATHERSTACK_API_KEY")

search_tool = TavilySearchResults(max_results=2)

@tool
def get_weather_data(city:str) -> str:
    """Fetch current weather information for a city."""
    url =( f"https://api.weatherstack.com/current?"
          f"access_key={WEATHERSTACK_API_KEY}&query={city}"
          )
    response = requests.get(url)
    data = response.json()

    if "current" not in data:
        return F"Could not retrieve weather data for {city}. Please check the city name and try again."
    return (
        f"city: {city}\n"
        f"temperature: {data['current']['temperature']}°C\n"
        f"weather: {data['current']['weather_descriptions'][0]}\n"
        f"Humidity: {data['current']['humidity']}%\n"
    ) 

result = search_tool.invoke("Give me the lastest news about AI")
result

llm = ChatOpenAI(
    model="gpt-4o-mini",
    temperature=0,
    api_key=OPENAI_API_KEY,
    base_url=API_BASE_URL
)

response = llm.invoke("tell me a joke about AI")
response

prompt = hub.pull("hwchase17/react")

tools = [search_tool, get_weather_data]

agent = create_react_agent(
    llm=llm,
    tools=tools,
    prompt=prompt
)

agent_executor = AgentExecutor(
    agent= agent,
    tools=tools,
    verbose=True
)

response = agent_executor.invoke({
    "input": "tell me lastest hunan news and the weather in changsha"
})

