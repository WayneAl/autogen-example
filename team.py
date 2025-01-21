import asyncio
from autogen_agentchat.agents import AssistantAgent
from autogen_agentchat.conditions import (
    MaxMessageTermination,
    ExternalTermination,
    TextMentionTermination,
)
from autogen_agentchat.teams import RoundRobinGroupChat
from autogen_ext.models.openai import OpenAIChatCompletionClient

from key import YOUR_API_KEY


def get_model_client_deepseek() -> OpenAIChatCompletionClient:  # type: ignore
    return OpenAIChatCompletionClient(
        model="deepseek-chat",
        api_key=YOUR_API_KEY,
        base_url="https://api.deepseek.com/v1",
        model_capabilities={
            "json_output": False,
            "vision": False,
            "function_calling": True,
        },
    )


def print_formatted_result(task_result):
    print("\n" + "=" * 60)
    print("代码评审过程".center(60))
    print("=" * 60 + "\n")

    for msg in task_result.messages:
        if msg.source == "user":
            print("📋 需求描述：")
        elif msg.source == "primary":
            print("👨‍💻 开发工程师提交：")
        elif msg.source == "critic":
            print("🔍 代码审查反馈：")

        print("-" * 40)
        print(f"{msg.content}\n")

        if msg.models_usage:
            print(f"Token统计：")
            print(f"· 提示tokens: {msg.models_usage.prompt_tokens}")
            print(f"· 生成tokens: {msg.models_usage.completion_tokens}")
            print(
                f"· 总计tokens: {msg.models_usage.prompt_tokens + msg.models_usage.completion_tokens}\n"
            )

    print("=" * 60)
    print("评审结果：".center(60))
    print("=" * 60)
    print(f"\n{task_result.stop_reason}\n")


async def main() -> None:
    # 创建 OpenAI 模型客户端
    model_client = get_model_client_deepseek()

    # 创建Python开发工程师
    Programmer_Agent = AssistantAgent(
        "programmer",
        model_client=model_client,
        system_message="""你是一个专业的Python开发工程师。
    请基于需求编写清晰、可维护且符合PEP8规范的Python代码。
    代码要包含:
    - 清晰的注释和文档字符串
    - 适当的错误处理
    - 代码性能优化
    - 单元测试
    """,
    )

    # 创建代码审计专家
    CodeReviewer_Agent = AssistantAgent(
        "code_reviewer",
        model_client=model_client,
        system_message="""你是一位资深的代码审查专家。请对代码进行全面的评审,包括:
    - 代码规范性和可读性
    - 设计模式的使用
    - 性能和效率
    - 安全性考虑
    - 测试覆盖率
    - 潜在问题
    当代码符合要求时,回复'同意通过'。""",
    )

    # 定义终止条件:当评论员同意时停止任务
    text_termination = TextMentionTermination("同意通过")

    # 创建一个包含主要智能助手和评论员的团队
    team = RoundRobinGroupChat(
        [Programmer_Agent, CodeReviewer_Agent], termination_condition=text_termination
    )

    # 示例任务:实现一个文件处理类
    task = """
    请实现一个文件处理类 FileProcessor,要求:
    1. 支持读取、写入和追加文本文件
    2. 包含基本的文件统计功能(行数、字符数、单词数)
    3. 支持文件加密/解密功能
    4. 实现异常处理
    5. 编写完整的单元测试
    """

    # 在脚本中运行时使用 `asyncio.run(...)`
    result = await team.run(task=task)
    print_formatted_result(result)


asyncio.run(main())
