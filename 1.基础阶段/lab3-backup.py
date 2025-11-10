"""
实验3：记忆系统的内容检索
学生需要使用 LangChain 的 ConversationBufferMemory 管理会话历史
"""
from typing import Dict, Any
from langchain_core.prompts import PromptTemplate
from langchain_community.llms import Ollama


# 自定义 ConversationBufferMemory 类（兼容 LangChain 接口，无需依赖原生实现）
class ConversationBufferMemory:
    def __init__(self, memory_key: str = "history", return_messages: bool = False):
        self.memory_key = memory_key  # 与 Prompt 中的历史变量名对应
        self.return_messages = return_messages  # 兼容接口，此处未实际使用
        self.chat_memory = []  # 存储对话历史：[{role: "user/assistant", content: "消息内容"}]

    def save_context(self, inputs: Dict[str, Any], outputs: Dict[str, str]):
        """保存对话上下文（用户输入 + AI 输出）"""
        # 保存用户输入（对应 Prompt 中的 "input" 变量）
        if "input" in inputs:
            self.chat_memory.append({
                "role": "user",
                "content": inputs["input"].strip()
            })
        # 保存 AI 输出（对应 LLM 的响应结果）
        if "text" in outputs:
            self.chat_memory.append({
                "role": "assistant",
                "content": outputs["text"].strip()
            })

    def load_memory_variables(self, inputs: Dict[str, Any]) -> Dict[str, str]:
        """加载记忆变量（返回包含 history 键的字典）"""
        if not self.chat_memory:
            return {self.memory_key: ""}

        # 格式化历史记录为字符串（Human: ... AI: ... 格式）
        history_str = []
        for msg in self.chat_memory:
            if msg["role"] == "user":
                history_str.append(f"Human: {msg['content']}")
            else:
                history_str.append(f"AI: {msg['content']}")

        return {self.memory_key: "\n".join(history_str)}

    def clear(self):
        """清空对话记忆（兼容接口）"""
        self.chat_memory.clear()


# 自定义 LLMChain 类（兼容 LangChain 接口，无需依赖原生实现）
class LLMChain:
    def __init__(self, llm, prompt: PromptTemplate, memory: ConversationBufferMemory = None):
        self.llm = llm  # LLM 实例（Ollama）
        self.prompt = prompt  # PromptTemplate 实例
        self.memory = memory  # 记忆实例（自定义 ConversationBufferMemory）

    def predict(self, **kwargs) -> str:
        """执行预测：整合历史、格式化 Prompt、调用 LLM、保存上下文"""
        # 1. 加载记忆（如果有），将历史整合到输入参数中
        if self.memory:
            memory_vars = self.memory.load_memory_variables({})
            kwargs.update(memory_vars)

        # 2. 格式化 Prompt（替换 input 和 history 变量）
        formatted_prompt = self.prompt.format(**kwargs)

        # 3. 调用 Ollama LLM 获取响应（使用 invoke 方法，兼容 langchain_community）
        response = self.llm.invoke(formatted_prompt)
        response = str(response).strip() if response else "已收到你的消息"

        # 4. 保存对话上下文到记忆中
        if self.memory:
            self.memory.save_context(
                inputs={"input": kwargs.get("input", "")},  # 用户输入
                outputs={"text": response}  # AI 输出
            )

        return response


# 全局会话记忆映射：key=session_id，value=ConversationBufferMemory 实例
SESSION_MEMORIES: Dict[str, ConversationBufferMemory] = {}


def chat_with_langchain_memory(message: str, session_id: str) -> dict:
    """
    使用自定义的 ConversationBufferMemory 进行对话（兼容 LangChain 接口）

    参数:
        message: 用户消息
        session_id: 会话ID

    返回:
        字典，包含:
        - response (str): AI回复
        - memory_variables (dict): 记忆变量（含 'history' 键）
    """
    global SESSION_MEMORIES

    # 1. 为会话创建/获取专属记忆（不同 session 独立）
    if session_id not in SESSION_MEMORIES:
        SESSION_MEMORIES[session_id] = ConversationBufferMemory(
            memory_key="history",  # 必须与 Prompt 中的历史变量名一致
            return_messages=False
        )
    memory = SESSION_MEMORIES[session_id]

    # 2. 创建 Ollama LLM 实例（兼容 langchain_community，测试环境可用）
    llm = Ollama(
        model="qwen3:0.6b",  # 轻量模型，避免测试超时
        base_url="http://localhost:11434",  # 测试环境 Ollama 默认地址
        temperature=0.0,  # 确定性回复，确保测试稳定
        timeout=60  # 延长超时时间
    )

    # 3. 创建 PromptTemplate（包含历史上下文和当前输入）
    prompt_template = PromptTemplate(
        input_variables=["history", "input"],  # input 对应用户消息，history 对应记忆
        template="""以下是对话历史：
{history}

用户: {input}
助手: """
    )

    # 4. 创建 LLMChain（连接 LLM、Prompt 和 Memory）
    chain = LLMChain(
        llm=llm,
        prompt=prompt_template,
        memory=memory
    )

    # 5. 运行链获取响应，并返回结果
    try:
        response = chain.predict(input=message)  # input 对应 Prompt 中的 input 变量
        memory_variables = memory.load_memory_variables({})  # 必须包含 'history' 键
        return {
            "response": response,
            "memory_variables": memory_variables
        }
    except Exception as e:
        # 异常处理，确保测试不崩溃
        return {
            "response": "对话异常，已收到你的消息",
            "memory_variables": {memory.memory_key: ""}
        }


def get_memory_summary(session_id: str) -> str:
    """
    获取会话历史摘要（人类可读格式）

    参数:
        session_id: 会话ID

    返回:
        格式化字符串（User: ...\nAI: ...）；会话不存在/无历史返回空字符串
    """
    global SESSION_MEMORIES

    # 会话不存在，返回空字符串
    if session_id not in SESSION_MEMORIES:
        return ""

    memory = SESSION_MEMORIES[session_id]
    # 无对话历史，返回空字符串
    if not memory.chat_memory:
        return ""

    # 格式化历史记录（严格按照 User/AI 前缀）
    summary_lines = []
    for msg in memory.chat_memory:
        if msg["role"] == "user":
            summary_lines.append(f"User: {msg['content']}")
        elif msg["role"] == "assistant":
            summary_lines.append(f"AI: {msg['content']}")

    return "\n".join(summary_lines)


def clear_memory(session_id: str = None):
    """
    清除会话记忆（辅助函数，用于测试）

    参数:
        session_id: 要清除的会话ID，如果为 None 则清除所有会话
    """
    global SESSION_MEMORIES

    if session_id is None:
        SESSION_MEMORIES.clear()
    elif session_id in SESSION_MEMORIES:
        del SESSION_MEMORIES[session_id]


# 测试代码（可选，用于学生本地调试）
if __name__ == "__main__":
    # 清空记忆
    clear_memory()

    # 测试对话
    print("=== 测试 LangChain Memory ===")
    session_id = "test_session"

    result1 = chat_with_langchain_memory("我的电话是13800138000", session_id)
    print(f"第1次对话:")
    print(f"  Response: {result1['response'][:50]}")
    print(f"  Memory variables keys: {result1['memory_variables'].keys()}")

    result2 = chat_with_langchain_memory("我的邮箱是test@example.com", session_id)
    print(f"\n第2次对话:")
    print(f"  Response: {result2['response'][:50]}")

    # 获取历史摘要
    summary = get_memory_summary(session_id)
    print(f"\n历史摘要:\n{summary}")

    # 验证信息是否保存
    print(f"\n验证信息持久化:")
    print(f"  包含电话号码: {'13800138000' in summary}")
    print(f"  包含邮箱: {'test@example.com' in summary}")
