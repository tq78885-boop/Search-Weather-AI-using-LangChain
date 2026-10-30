import os
import certifi
from dotenv import load_dotenv
from langchain_openai import ChatOpenAI
from langchain_community.tools.tavily_search import TavilySearchResults
from langchain import hub
from langchain.tools import tool
import requests
from langchain.agents import create_react_agent, AgentExecutor
import streamlit as st

# ========== 基础配置与环境加载 ==========
# 页面全局配置
st.set_page_config(
    page_title="智能资讯天气助手",
    page_icon="🌤️",
    layout="centered"
)

# SSL 证书配置 + 环境变量加载
os.environ["SSL_CERT_FILE"] = certifi.where()
# 基于脚本目录定位 .env，避免切换工作目录启动时加载不到配置而误报缺少 API Key
load_dotenv(os.path.join(os.path.dirname(os.path.abspath(__file__)), ".env"))

API_BASE_URL = os.getenv("API_BASE_URL")
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
TAVILY_API_KEY = os.getenv("TAVILY_API_KEY")
WEATHERSTACK_API_KEY = os.getenv("WEATHERSTACK_API_KEY")

# ========== 工具定义（与原代码完全一致） ==========
@tool
def get_weather_data(city: str) -> str:
    """Fetch current weather information for a city."""
    # 对城市名做 URL 编码，避免中文/空格直接拼接导致请求异常
    url = (f"https://api.weatherstack.com/current?"
           f"access_key={WEATHERSTACK_API_KEY}&query={requests.utils.quote(city)}")
    try:
        # 必须设置超时：否则网络挂起时请求会一直阻塞，界面无限转圈
        response = requests.get(url, timeout=10)
        response.raise_for_status()
        data = response.json()
    except requests.RequestException as e:
        # 网络层错误以字符串返回，让 Agent 能感知并继续，而不是直接让界面崩溃
        return f"Request weather data failed for {city}: {e}. Please check network and try again."
    except ValueError:
        return f"Received invalid weather response for {city}. Please try again later."

    if "current" not in data:
        return f"Could not retrieve weather data for {city}. Please check the city name and try again."

    # 防御性取值：字段缺失或 weather_descriptions 为空时不再抛 IndexError/KeyError
    current = data["current"]
    descriptions = current.get("weather_descriptions") or []
    weather_desc = descriptions[0] if descriptions else "N/A"
    return (
        f"city: {city}\n"
        f"temperature: {current.get('temperature', 'N/A')}°C\n"
        f"weather: {weather_desc}\n"
        f"Humidity: {current.get('humidity', 'N/A')}%\n"
    )

# ========== Agent 初始化（缓存复用，避免重复创建） ==========
@st.cache_resource(show_spinner="正在初始化智能助手...")
def init_agent_executor():
    """初始化并缓存 Agent 执行器，页面刷新不会重复创建"""
    # 初始化搜索工具
    search_tool = TavilySearchResults(max_results=2)
    tools = [search_tool, get_weather_data]

    # 初始化大模型
    llm = ChatOpenAI(
        model="gpt-4o-mini",
        temperature=0,
        api_key=OPENAI_API_KEY,
        base_url=API_BASE_URL
    )

    # 拉取 ReAct 提示词
    prompt = hub.pull("hwchase17/react")

    # 创建 ReAct Agent
    agent = create_react_agent(
        llm=llm,
        tools=tools,
        prompt=prompt
    )

    # 创建 Agent 执行器
    agent_executor = AgentExecutor(
        agent=agent,
        tools=tools,
        verbose=True,
        handle_parsing_errors=True,  # 增强鲁棒性，自动处理解析错误
        max_iterations=8,  # 限制推理步数，避免工具反复失败时界面长时间转圈
    )
    return agent_executor

# ========== 页面交互逻辑 ==========
def main():
    st.title("🌤️ 智能资讯与天气助手")
    st.caption("基于 LangChain ReAct Agent | 支持全网新闻搜索 + 全球城市天气查询")

    # 前置校验：检查必要 API Key 是否配置
    required_env = ["OPENAI_API_KEY", "TAVILY_API_KEY", "WEATHERSTACK_API_KEY"]
    missing_env = [key for key in required_env if not os.getenv(key)]
    if missing_env:
        st.error(f"⚠️ 缺少必要环境变量：{', '.join(missing_env)}")
        st.info("请在项目根目录的 .env 文件中配置所有 API Key 后重启应用")
        st.stop()

    # 初始化 Agent
    agent_executor = init_agent_executor()

    # 用户输入表单
    with st.form("query_form", clear_on_submit=False):
        user_query = st.text_input(
            "请输入你的问题",
            placeholder="例如：告诉我湖南最新的新闻和长沙的天气"
        )
        submit_btn = st.form_submit_button("🔍 提交查询", use_container_width=True)

    # 提交后执行查询
    if submit_btn and user_query:
        with st.spinner("Agent 正在思考与执行中..."):
            try:
                # 调用 Agent 执行
                result = agent_executor.invoke({"input": user_query})

                # 展示最终回答
                st.subheader("📌 回答结果")
                st.success(result["output"])

                # 折叠展示原始返回数据（可选，方便调试）
                with st.expander("查看完整返回数据"):
                    st.json(result)

            except Exception as e:
                st.error(f"执行失败：{str(e)}")
                st.info("请检查网络连接与 API Key 是否有效")

if __name__ == "__main__":
    main()
