from typing import Dict, Optional
from langchain.chains.llm import LLMChain
from langchain.memory import ConversationBufferMemory
from langchain_core.prompts import PromptTemplate
from langchain_ollama import OllamaLLM


# 全局存储：管理所有会话的记忆
SESSION_MEMORIES: Dict[str, ConversationBufferMemory] = {}

# 实验必需函数1：chat_with_langchain_memory（对话主函数）
def chat_with_langchain_memory(message: str, session_id: str) -> dict:

    # 引用全局的会话记忆字典
    global SESSION_MEMORIES

    # 1. 为当前会话创建/获取专属记忆
    # 逻辑：如果session_id不在全局字典中，说明是新会话，创建新记忆；否则复用旧记忆
    if session_id not in SESSION_MEMORIES:
        SESSION_MEMORIES[session_id] = ConversationBufferMemory(
            memory_key="history",
            return_messages=False
        )
    # 获取当前会话的记忆实例
    current_memory = SESSION_MEMORIES[session_id]

    # 2. 创建大模型实例（Ollama）
    # 配置说明：
    # - model: 测试环境可用的轻量模型（启动快，不超时）
    # - base_url: Ollama默认服务地址（测试环境固定）
    # - temperature: 0.0（确定性回复，测试结果稳定，便于验证）
    llm = OllamaLLM(
        model = "qwen3:8b",
        base_url = "http://localhost:11434",
        temperature = 0.0
    )

    # 3. 创建提示词模板（定义对话格式）
    # 输入变量：
    # - history: 历史对话（来自Memory）
    # - input: 当前用户消息（来自参数message）
    prompt_template = PromptTemplate(
        input_variables = ["history", "input"],
        template="""以下是历史对话：
{history}

用户：{input}
助手："""
    )

    # 4. 创建对话链：串联模型、提示词、记忆
    chain = LLMChain(
        llm = llm,
        prompt = prompt_template,
        memory = current_memory
    )

    # 5. 执行对话，获取AI回复
    # 调用predict时，传入"input"参数（对应用户当前消息）
    ai_response = chain.predict(input=message)

    # 6. 获取当前记忆状态（用于测试验证）
    memory_variables = current_memory.load_memory_variables({})

    # 7. 返回结果（严格按实验要求的格式）
    return {
        "response": ai_response,
        "memory_variables": memory_variables
    }

# 实验必需函数2：get_memory_summary（获取会话历史摘要）
def get_memory_summary(session_id: str) -> str :

    global SESSION_MEMORIES

    # 1. 会话不存在：直接返回空字符串
    if session_id not in SESSION_MEMORIES:
        return ""

    # 2. 获取该会话的记忆实例
    memory = SESSION_MEMORIES[session_id]
    return memory.load_memory_variables({}).get("history", "")

# 辅助函数：clear_memory（测试用，非必需但推荐保留）
def clear_memory(session_id: Optional[str] = None):

    global SESSION_MEMORIES
    if session_id is None:
        SESSION_MEMORIES.clear()
    elif session_id in SESSION_MEMORIES:
        del SESSION_MEMORIES[session_id]

# 本地测试代码（帮助你理解运行流程，可直接运行）
if __name__ == "__main__":
    # 1. 清空所有旧记忆（避免干扰测试）
    clear_memory()

    # 2. 模拟第一个会话（session_id = "user_001"）
    print("=== 会话1：用户001的对话 ===")
    session1_id = "user_001"
    # 第1轮对话：用户发送"我叫小明"
    res1 = chat_with_langchain_memory("我叫小明", session1_id)
    print(f"用户：我叫小明")
    print(f"AI：{res1['response']}")
    print(f"当前记忆：{res1['memory_variables']['history']}\n")

    # 第2轮对话：用户问"我叫什么"（验证记忆是否生效）
    res2 = chat_with_langchain_memory("我叫什么名字？", session1_id)
    print(f"用户：我叫什么名字？")
    print(f"AI：{res2['response']}")
    print(f"当前记忆：{res2['memory_variables']['history']}\n")

    # 3. 模拟第二个会话（session_id = "user_002"，验证会话隔离）
    print("=== 会话2：用户002的对话（验证隔离） ===")
    session2_id = "user_002"
    res3 = chat_with_langchain_memory("你好呀", session2_id)
    print(f"用户：你好呀")
    print(f"AI：{res3['response']}")
    print(f"会话2的记忆：{res3['memory_variables']['history']}")
    print(f"会话1的记忆（未被影响）：{get_memory_summary(session1_id)}\n")

    # 4. 测试历史摘要功能
    print("=== 会话1的完整历史摘要 ===")
    summary = get_memory_summary(session1_id)
    print(summary)
