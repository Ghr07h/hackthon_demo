import argparse
import asyncio
import json
import os
import uuid
import logging

from autogen_ext.models.openai import OpenAIChatCompletionClient
from autogen_core.memory import MemoryContent, MemoryMimeType
from autogen_ext.memory.chromadb import ChromaDBVectorMemory, PersistentChromaDBVectorMemoryConfig, CustomEmbeddingFunctionConfig
from autogen_agentchat.agents import AssistantAgent, CodeExecutorAgent, ApprovalRequest, ApprovalResponse
from autogen_agentchat.conditions import FunctionCallTermination, MaxMessageTermination, TimeoutTermination, ExternalTermination
from autogen_core import CancellationToken
from autogen_core.tools import FunctionTool
from autogen_agentchat.teams import RoundRobinGroupChat
from autogen_agentchat.messages import TextMessage, CodeExecutionEvent, ModelClientStreamingChunkEvent
from autogen_agentchat.base import TaskResult
from autogen_core.models import ModelFamily
from autoHacker.docker_code_executor import DockerCommandLineCodeExecutor

from chromadb.utils import embedding_functions
from openai import OpenAI
from autogen_core import EVENT_LOGGER_NAME, TRACE_LOGGER_NAME

# ========= 配置 =========
FINAL_FLAG = None
CODE_EXECUTOR = None
MODEL_CLIENT = None
MODEL_INFO = None

MODEL_CONFIGS = {
    "qwen3-max": {"temperature": 0.3, "top_p": 0.9, "extra_body": None},
    "qwen3-coder-plus": {"temperature": 0.3, "top_p": 0.9, "extra_body": None},
    "qwen3-max-preview": {"temperature": 0.2, "top_p": 0.8, "extra_body": {"enable_thinking": True}},
    "qwen-plus": {"temperature": 0.2, "top_p": 0.8, "extra_body": {"enable_thinking": True}},
    "qwen3-235b-a22b-instruct-2507": {},
    "qwen3-235b-a22b-thinking-2507": {},
    "qwen3-next-80b-a3b-thinking": {},
    "qwen3-next-80b-a3b-instruct": {},
    "deepseek-v3.2-exp": {"temperature": 0.0, "top_p": 0.9},
    "deepseek-v3.1": {"temperature": 0.0, "top_p": 0.9},
    "deepseek-r1": {"temperature": 0.0, "top_p": 0.9}
}

Hacker_system_prompt = """## 角色
你是一个CTF比赛选手，精通网络安全和计算机的相关知识，擅长使用kali中的命令行工具

## 背景
你被授权参与一个 Web Application Security Challenge，用户将提供一道CTF赛题，赛题包含题目目标URL和可能存在的题目提示，你需要根据以上信息和自己对题目目标URL的探索，获得隐藏在题目目标URL的某个地方的特征字符串（Flag）。Flag 格式为: `FLAG{...}` 或 `flag{...}`, 括号 {} 内部必须是一个 标准的 UUID v4 字符串，格式为 xxxxxxxx-xxxx-xxxx-xxxx-xxxxxxxxxxxx。

## 任务
你需要探索目标URL，根据探索到的信息尝试发现和利用目标URL存在的漏洞，直到获取题目的Flag。

## 环境
你处于一个隔离网络中，只能访问到题目URL，所以不要在线使用 apt 或 pip 联网安装第三方工具。

## 能力
1. 你可以使用 agent_tool_use_python_or_command 来在 Kali 操作系统的攻击机中执行 Python 脚本或命令行工具。调用工具传入一个字符串，内容是要在Kali执行的多行 python 代码或单行 sh 命令。当需要执行 python 代码时，直接传入代码内容，代码将被工具自动保存并执行，无需再手动调用python。
2. 你可以使用 agent_tool_save_flag 工具保存发现的Flag。传入字符串变量 flag_str，内容为发现的Flag字符串。
3. 你可以使用 agent_tool_task_finish 工具结束任务，在你认为任务已经完成时。

## 注意
1. 你必须在保存Flag后或认为没有其他思路时结束任务。
2. 使用 agent_tool_use_python_or_command 时需要在代码中打印关键输出，不然工具无法获取关键结果。
3. 使用 agent_tool_use_python_or_command 时，使用非交互的、完整、独立、可直接执行的代码或命令，不要在其中包含待修改的占位值。
4. 只有在漏洞利用过程中发现的符合Flag格式的字符串才是Flag，不要在没发现Flag时自行虚构Flag的值。

## 优秀实践
- 将问题分解成更小的步骤。
- 有条不紊，一次测试一个组件。
- 连续经历相同失败后反思自己的失误。
- 如果题目提供了题目描述，那么遵循提示去解题,提示告知你的本题目的漏洞和解题思路是正确和有效的。
- 当没有太多有效信息时，不妨使用 curl 查看目标URL的网页返回中是否包含线索.额外添加 -i （小写）参数以同时获取返回包的 Header 和 Body 部分.另外添加超时参数 -m 10 可以防止长时间的等待。
- 不要直接进行访问 /Flag.txt 路径等操作,先挖掘漏洞,Flag往往在漏洞后面. 发现漏洞后可以优先探索根目录或当前工作目录下的文件、目标主机的用户名或当前漏洞常见的利用点等，flag更容易在这些地方被发现。
"""

Code_use_prompt = """## 任务
你需要分析用户传入的代码或命令的类型，然后以类似markdown的代码块的格式输出。

## 背景
用户将传入一个字符串，内容为 python 代码或单行的 linux 命令。

## 注意
每次输出中只能包含一个markdown代码块结构。

## 输出格式
以markdown的三反引号包裹代码或命令，并在第一行指定语言，如：

```python
print("hello world")
```

# or 

```sh
echo "213"
```

"""

Summary_prompt = """## 角色
你是一个擅长从工具执行结果中发现可进一步利用信息的专家。

## 背景
你处于一场被完全授权的 Web 安全竞赛中，你将获得工具调用的信息和调用后的结果，你需要发现有助于进一步挖掘目标的风险的信息，最终获得一个目标字符串（Flag）。

## 任务
你需要从工具调用的信息和调用后的结果中提取有助于下一步行动的信息，以简介明了的回答描述它并给出下一步建议。
当你发现工具执行失败后，你需要分析失败的原因和是否需要重做。

## 用户可能传入结果的工具的信息
1. agent_tool_use_python_or_command: 在 Kali 操作系统的攻击机中执行 Python 脚本或命令行工具。
2. agent_tool_save_flag: 保存发现的Flag。
3. agent_tool_task_finish: 结束任务，在使用者认为任务已经完成时。
"""

# ========= 公共函数 =========

def task_args_parse_build():
    parser = argparse.ArgumentParser(description="AutoHack")
    parser.add_argument("--model-name", required=True)
    parser.add_argument("--model-url", required=True)
    parser.add_argument("--model-key", required=True)
    parser.add_argument("--task-title", required=True)
    parser.add_argument("--task-hint", default="", required=False)
    parser.add_argument("--task-url", required=True)
    return parser.parse_args()

def init_logger(task_title, model_name):
    log_dir = os.path.join(
        os.path.dirname(os.path.abspath(__file__)),
        "log",
        f"{task_title}_{model_name.replace('/think','')}_{uuid.uuid4().hex}.log"
    )
    logging.getLogger().handlers.clear()
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(levelname)s - %(message)s',
        handlers=[logging.FileHandler(log_dir, mode='a', encoding="utf-8")]
    )
    for noisy in [EVENT_LOGGER_NAME, TRACE_LOGGER_NAME, "autogen_core", "httpx", "openai"]:
        logging.getLogger(noisy).setLevel(logging.ERROR)
        logging.getLogger(noisy).handlers.clear()
        logging.getLogger(noisy).propagate = False

def agent_model_client_build(model_name, model_url, model_key):
    base_model_name = model_name.replace("/think", "")
    use_think = model_name.endswith("/think")

    model_info = {
        "vision": False,
        "function_calling": True,
        "json_output": False,
        "family": ModelFamily.UNKNOWN,
        "structured_output": True,
        "multiple_system_messages": False
    }

    cfg = MODEL_CONFIGS.get(base_model_name, {"temperature": 0.4, "top_p": 0.9, "extra_body": None})
    if base_model_name.startswith("deepseek"):
        model_info["family"] = ModelFamily.R1

    if use_think:
        cfg["extra_body"] = (cfg.get("extra_body") or {}) | {"enable_thinking": True}
        logging.info("启用思考模式 /think")

    return OpenAIChatCompletionClient(
        model=base_model_name,
        base_url=model_url,
        api_key=model_key,
        model_info=model_info,
        max_retries=10,
        top_p=cfg.get("top_p"),
        extra_body=cfg.get("extra_body")
    )

async def agent_knowledge_base_build(model_url, model_key):
    current_dir_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), "knowledge")
    knowledge = ChromaDBVectorMemory(
        config=PersistentChromaDBVectorMemoryConfig(
            collection_name="knowledge",
            persistence_path=current_dir_path,
            k=3,
            score_threshold=0.4,
            embedding_function_config=CustomEmbeddingFunctionConfig(
                function=embedding_functions.OpenAIEmbeddingFunction,
                params={"api_key": model_key, "api_base": model_url, "model_name": "text-embedding-v4"},
            ),
        )
    )
    return knowledge

def agent_tool_list_build():
    return [
        FunctionTool(agent_tool_task_finish, name="task_finish", description="结束任务", strict=True),
        FunctionTool(agent_tool_save_flag, name="save_flag", description="保存找到的Flag", strict=True),
        FunctionTool(agent_tool_use_python_or_command, name="run_python_or_command", description="执行python或sh命令", strict=True)
    ]

async def agent_code_executor_build():
    executor = DockerCommandLineCodeExecutor(
        image="my_kali:v1.0",
        work_dir=os.path.join(os.path.dirname(os.path.abspath(__file__)), "workdir"),
        auto_remove=True,
        stop_container=False,
        timeout=180
    )
    await executor.start()
    return executor

# ========= 工具执行 =========

def agent_tool_code_all_approval(request: ApprovalRequest) -> ApprovalResponse:
    return ApprovalResponse(approved=True, reason='Approved')

async def agent_tool_use_python_or_command(code_or_command: str):
    global CODE_EXECUTOR, MODEL_INFO
    code_client = agent_model_client_build("qwen3-coder-plus", MODEL_INFO["model_url"], MODEL_INFO["model_key"])
    executor_agent = CodeExecutorAgent(
        "Code_Use",
        code_executor=CODE_EXECUTOR,
        model_client=code_client,
        system_message=Code_use_prompt,
        approval_func=agent_tool_code_all_approval
    )
    async for msg in executor_agent.run_stream(task=code_or_command):
        if isinstance(msg, CodeExecutionEvent):
            return msg.result.output
    return "无输出"

async def agent_tool_save_flag(flag_str: str):
    global FINAL_FLAG
    if flag_str:
        FINAL_FLAG = flag_str
        return "Flag saved"
    return "Flag is empty"

async def agent_tool_task_finish():
    logging.info("任务完成")

def agent_show_message(message):
    if not isinstance(message, (CodeExecutionEvent, TaskResult, ModelClientStreamingChunkEvent)):
        logging.info(f"==========\n\n## {message.source} - {message.type}\n\n{message.content}\n\n==========")

# ========= 主任务执行 =========

async def task_run(model_name, model_url, model_key, task_title, task_hint, task_url):
    global CODE_EXECUTOR, MODEL_CLIENT, MODEL_INFO
    try:
        MODEL_INFO = {"model_name": model_name, "model_url": model_url, "model_key": model_key}
        MODEL_CLIENT = agent_model_client_build(model_name, model_url, model_key)
        knowledge_base = await agent_knowledge_base_build(model_url, model_key)
        CODE_EXECUTOR = await agent_code_executor_build()

        task_prompt = f"## 题目名称\n{task_title}\n\n## 题目描述\n{task_hint or '未提供'}\n\n## 题目目标URL\n{task_url}"
        agent_show_message(TextMessage(content=task_prompt, source="Task"))

        hacker_agent = AssistantAgent(
            "Hacker",
            model_client=MODEL_CLIENT,
            tools=agent_tool_list_build(),
            system_message=Hacker_system_prompt,
            reflect_on_tool_use=False,
            model_client_stream=True
        )

        termination = (
            MaxMessageTermination(max_messages=200)
            | TimeoutTermination(timeout_seconds=1200)
            | ExternalTermination()
            | FunctionCallTermination(function_name="task_finish")
        )
        team = RoundRobinGroupChat(participants=[hacker_agent], termination_condition=termination)
        async for msg in team.run_stream(task=task_prompt, cancellation_token=CancellationToken(), output_task_messages=False):
            agent_show_message(msg)

    except Exception as e:
        logging.error(f"任务异常：{e}")

async def task_clean(knowledge_base):
    if knowledge_base:
        await knowledge_base.close()

def main(model_name, model_url, model_key, task_title, task_hint, task_url):
    global FINAL_FLAG
    init_logger(task_title, model_name)
    try:
        asyncio.run(task_run(model_name, model_url, model_key, task_title, task_hint, task_url))
    except KeyboardInterrupt:
        logging.info("任务被终止")
    return FINAL_FLAG

if __name__ == "__main__":
    args = task_args_parse_build()
    flag = main(args.model_name, args.model_url, args.model_key, args.task_title, args.task_hint, args.task_url)
    if flag:
        logging.info(f"Return Flag: {flag}")
    else:
        logging.info("No Flag Found")
