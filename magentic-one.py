import asyncio
from autogen_agentchat.agents import AssistantAgent, UserProxyAgent, CodeExecutorAgent
from autogen_ext.models.openai import OpenAIChatCompletionClient
from autogen_agentchat.teams import MagenticOneGroupChat
from autogen_agentchat.ui import Console
from autogen_ext.code_executors.docker import DockerCommandLineCodeExecutor
from autogen_core import CancellationToken
from autogen_agentchat.messages import TextMessage

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


async def main() -> None:
    model_client = get_model_client_deepseek()

    assistant = AssistantAgent(
        "assistant",
        model_client=model_client,
    )

    user_proxy = UserProxyAgent(
        "user_proxy",
    )

    team = MagenticOneGroupChat([assistant, user_proxy], model_client=model_client)
    await Console(team.run_stream(task="用Python寫一個Hello World程式"))


asyncio.run(main())
