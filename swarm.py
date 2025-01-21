import asyncio
from autogen_agentchat.agents import AssistantAgent
from autogen_ext.models.openai import OpenAIChatCompletionClient
from autogen_agentchat.teams import MagenticOneGroupChat
from autogen_agentchat.ui import Console
from autogen_agentchat.conditions import ExternalTermination, TextMentionTermination
from autogen_agentchat.teams import Swarm

from serpapi import GoogleSearch
from typing import Any, Dict, List
import os
from datetime import datetime

import yfinance as yf

from key import YOUR_API_KEY


def get_model_client_deepseek() -> OpenAIChatCompletionClient:  # type: ignore
    return OpenAIChatCompletionClient(
        model="deepseek-chat",
        api_key=YOUR_API_KEY,
        base_url="https://api.deepseek.com/v1",
        model_capabilities={
            "json_output": True,
            "vision": False,
            "function_calling": True,
        },
    )


async def get_stock_data(symbol: str) -> Dict[str, Any]:
    """
    Get real stock market data for a given symbol with improved error handling

    Args:
        symbol: Stock ticker symbol (e.g. 'TSLA')

    Returns:
        Dict containing price, volume, PE ratio and market cap
    """
    try:
        # 创建股票对象
        stock = yf.Ticker(symbol)

        # 获取实时价格数据
        price_info = stock.history(period="1d")
        if not price_info.empty:
            current_price = price_info["Close"].iloc[-1]
        else:
            current_price = None

        # 获取其他信息
        info = stock.info

        return {
            "price": current_price,
            "volume": info.get("regularMarketVolume"),
            "pe_ratio": info.get("forwardPE"),
            "market_cap": info.get("marketCap"),
            "last_updated": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        }

    except Exception as e:
        print(f"Error fetching stock data for {symbol}: {str(e)}")
        return {
            "price": None,
            "volume": None,
            "pe_ratio": None,
            "market_cap": None,
            "error": str(e),
        }


async def get_news(query: str) -> List[Dict[str, str]]:
    """Get recent news articles about a company"""
    params = {
        "engine": "google_news",
        "q": query,
        "gl": "us",
        "hl": "en",
        "api_key": os.getenv("SERPAPI_KEY"),
        "num": 3,  # 限制结果数量
    }

    try:
        search = GoogleSearch(params)
        results = search.get_dict()

        news_items = []
        for article in results.get("news_results", []):
            # 获取更多文章细节
            title = article.get("title", "").strip()
            source = article.get("source", {})
            source_name = source.get("name", "")
            authors = source.get("authors", [])
            author_text = f"By {', '.join(authors)}" if authors else ""

            # 提取或构建摘要
            snippet = article.get("snippet", "")
            description = article.get("description", "")
            link_text = article.get("link_text", "")

            # 选择最长的非空内容作为摘要
            summary_candidates = [s for s in [snippet, description, link_text] if s]
            summary = max(summary_candidates, key=len) if summary_candidates else title

            # 格式化日期
            date_str = article.get("date", "")
            try:
                if date_str:
                    date_obj = datetime.strptime(date_str.split(",")[0], "%m/%d/%Y")
                    formatted_date = date_obj.strftime("%Y-%m-%d")
                else:
                    formatted_date = datetime.now().strftime("%Y-%m-%d")
            except:
                formatted_date = date_str

            news_items.append(
                {
                    "title": title,
                    "date": formatted_date,
                    "summary": f"{summary} {author_text}".strip(),
                    "source": source_name,
                }
            )

        return news_items

    except Exception as e:
        print(f"Error fetching news: {str(e)}")
        return []


async def main() -> None:
    model_client = get_model_client_deepseek()

    planner = AssistantAgent(
        "planner",
        model_client=model_client,
        handoffs=["financial_analyst", "news_analyst", "writer"],
        system_message="""你是一名研究规划协调员。
        通过委派给专业智能体来协调市场研究：
        - 金融分析师：负责股票数据分析
        - 新闻分析师：负责新闻收集和分析
        - 撰写员：负责编写最终报告
        始终先发送你的计划，然后再移交给适当的智能体。
        每次只能移交给一个智能体。
        当研究完成时使用 TERMINATE 结束。""",
    )

    financial_analyst = AssistantAgent(
        "financial_analyst",
        model_client=model_client,
        handoffs=["planner"],
        tools=[get_stock_data],
        system_message="""你是一名金融分析师。
        使用 get_stock_data 工具分析股市数据。
        提供金融指标的深入见解。
        分析完成后务必移交回规划协调员。""",
    )

    news_analyst = AssistantAgent(
        "news_analyst",
        model_client=model_client,
        handoffs=["planner"],
        tools=[get_news],
        system_message="""你是一名新闻分析师。
        使用 get_news 工具收集和分析相关新闻。
        总结新闻中的关键市场见解。
        分析完成后务必移交回规划协调员。""",
    )

    writer = AssistantAgent(
        "writer",
        model_client=model_client,
        handoffs=["planner"],
        system_message="""你是一名财经报告撰写员。
        将研究发现编译成清晰简洁的报告。
        撰写完成后务必移交回规划协调员。""",
    )

    # Define termination condition
    text_termination = TextMentionTermination("TERMINATE")
    termination = text_termination

    research_team = Swarm(
        participants=[planner, financial_analyst, news_analyst, writer],
        termination_condition=termination,
    )

    task = "为特斯拉(TSLA)股票进行市场研究，并用中文回答"
    await Console(research_team.run_stream(task=task))


asyncio.run(main())
