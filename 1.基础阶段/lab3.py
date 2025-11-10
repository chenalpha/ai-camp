"""
实验3核心目标：
1. 理解会话记忆的本质（跨轮对话状态保存）
2. 掌握 LangChain 核心组件接口设计（Memory、LLMChain、PromptTemplate）
3. 实现会话隔离（不同用户/会话的记忆独立）
"""
from typing import Dict, Any
# 仅保留实验必需的导入（基于测试环境兼容版本）
from langchain_core.prompts import PromptTemplate
from langchain_community.llms import Ollama


# ------------------------------------------------------------------------------
# 核心类1：自定义 ConversationBufferMemory（会话记忆核心）
# 作用：保存单一会话的历史对话，提供"保存上下文"和"加载记忆"的接口
# ------------------------------------------------------------------------------
class ConversationBufferMemory:
    """
    会话记忆类：专门存储单个会话的历史对话（用户输入+AI回复）
    核心能力：保存对话、加载历史、清空记忆
    """

    def __init__(self, memory_key: str = "history", return_messages: bool = False):
        # memory_key：记忆变量名（必须与PromptTemplate中的历史变量名一致）
        self.memory_key = memory_key
        # return_messages：兼容LangChain接口（本实验用不到，仅保留兼容性）
        self.return_messages = return_messages
        # 核心存储：用列表保存对话历史，每个元素是{角色: 内容}的字典
        self.chat_memory = []

    def save_context(self, inputs: Dict[str, Any], outputs: Dict[str, str]):
        """
        保存一轮对话上下文（用户输入 + AI输出）
        inputs: 用户输入字典（key为"input"，value为用户消息）
        outputs: AI输出字典（key为"text"，value为AI回复）
        """
        # 保存用户输入到历史
        if "input" in inputs:
            self.chat_memory.append({
                "role": "user",  # 标记角色为用户
                "content": inputs["input"].strip()  # 去除首尾空格，规范格式
            })
        # 保存AI输出到历史
        if "text" in outputs:
            self.chat_memory.append({
                "role": "assistant",  # 标记角色为助手
                "content": outputs["text"].strip()
            })

    def load_memory_variables(self, inputs: Dict[str, Any]) -> Dict[str, str]:
        """
        加载记忆变量（给Prompt提供历史对话字符串）
        返回：字典，key为self.memory_key（即"history"），value为格式化的历史对话
        """
        # 如果没有历史对话，返回空字符串
        if not self.chat_memory:
            return {self.memory_key: ""}

        # 格式化历史对话（人类可读格式：User: ... AI: ...）
        history_lines = []
        for msg in self.chat_memory:
            if msg["role"] == "user":
                history_lines.append(f"Human: {msg['content']}")
            else:
                history_lines.append(f"AI: {msg['content']}")

        # 用换行符连接所有对话，形成完整历史
        return {self.memory_key: "\n".join(history_lines)}

    def clear(self):
        """清空记忆（辅助功能，测试时用）"""
        self.chat_memory.clear()


# ------------------------------------------------------------------------------
# 核心类2：自定义 LLMChain（逻辑串联核心）
# 作用：串联 PromptTemplate（提示词）、LLM（大模型）、Memory（记忆）
# ------------------------------------------------------------------------------
class LLMChain:
    """
    对话链类：整合"记忆→提示词→大模型"的完整流程
    核心逻辑：加载历史→生成提示词→调用模型→保存历史
    """

    def __init__(self, llm, prompt: PromptTemplate, memory: ConversationBufferMemory = None):
        self.llm = llm  # 大模型实例（这里是Ollama）
        self.prompt = prompt  # 提示词模板（定义对话格式）
        self.memory = memory  # 记忆实例（关联的会话记忆）

    def predict(self, **kwargs) -> str:
        """
        执行一轮对话：
        kwargs: 输入参数（必须包含"input"，即当前用户消息）
        返回：AI的最终回复
        """
        # 1. 加载记忆：如果有记忆，就把历史对话加入输入参数
        if self.memory:
            memory_vars = self.memory.load_memory_variables({})  # 加载历史
            kwargs.update(memory_vars)  # 把历史对话（key=history）加入参数

        # 2. 生成完整提示词：用用户消息+历史对话替换Prompt中的变量
        # 例如：kwargs = {"input": "你好", "history": "Human: 之前的消息..."}
        formatted_prompt = self.prompt.format(**kwargs)

        # 3. 调用大模型：把完整提示词传给Ollama，获取回复
        response = self.llm.invoke(formatted_prompt)
        # 处理空响应（避免报错）
        response = str(response).strip() if response else "已收到你的消息"

        # 4. 保存历史：把当前轮的"用户输入+AI回复"存入记忆
        if self.memory:
            self.memory.save_context(
                inputs={"input": kwargs.get("input", "")},  # 当前用户消息
                outputs={"text": response}  # 当前AI回复
            )

        return response


# ------------------------------------------------------------------------------
# 全局存储：管理所有会话的记忆
# ------------------------------------------------------------------------------
# 字典结构：key = 会话ID（session_id），value = 该会话的ConversationBufferMemory实例
# 作用：实现不同会话的记忆隔离（A用户的对话不会影响B用户）
SESSION_MEMORIES: Dict[str, ConversationBufferMemory] = {}


# ------------------------------------------------------------------------------
# 实验必需函数1：chat_with_langchain_memory（对话主函数）
# ------------------------------------------------------------------------------
def chat_with_langchain_memory(message: str, session_id: str) -> dict:
    """
    对外提供的对话接口：
    参数：
        message: 当前用户发送的消息（字符串）
        session_id: 会话唯一标识（区分不同用户/对话）
    返回：
        dict: 包含"response"（AI回复）和"memory_variables"（当前记忆状态）
    """
    global SESSION_MEMORIES  # 引用全局的会话记忆字典

    # 1. 为当前会话创建/获取专属记忆
    # 逻辑：如果session_id不在全局字典中，说明是新会话，创建新记忆；否则复用旧记忆
    if session_id not in SESSION_MEMORIES:
        SESSION_MEMORIES[session_id] = ConversationBufferMemory(
            memory_key="history",  # 记忆变量名必须和Prompt中的一致
            return_messages=False
        )
    # 获取当前会话的记忆实例
    current_memory = SESSION_MEMORIES[session_id]

    # 2. 创建大模型实例（Ollama）
    # 配置说明：
    # - model: 测试环境可用的轻量模型（启动快，不超时）
    # - base_url: Ollama默认服务地址（测试环境固定）
    # - temperature: 0.0（确定性回复，测试结果稳定，便于验证）
    llm = Ollama(
        model="qwen3:0.6b",
        base_url="http://localhost:11434",
        temperature=0.0,
        timeout=60
    )

    # 3. 创建提示词模板（定义对话格式）
    # 输入变量：
    # - history: 历史对话（来自Memory）
    # - input: 当前用户消息（来自参数message）
    prompt_template = PromptTemplate(
        input_variables=["history", "input"],  # 必须包含这两个变量
        template="""以下是对话历史：
{history}

用户: {input}
助手: """  # 固定格式，引导AI按"助手"角色回复
    )

    # 4. 创建对话链：串联模型、提示词、记忆
    chain = LLMChain(
        llm=llm,
        prompt=prompt_template,
        memory=current_memory  # 绑定当前会话的记忆
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


# ------------------------------------------------------------------------------
# 实验必需函数2：get_memory_summary（获取会话历史摘要）
# ------------------------------------------------------------------------------
def get_memory_summary(session_id: str) -> str:
    """
    获取某个会话的完整历史摘要（人类可读格式）
    参数：session_id 会话ID
    返回：
        格式化字符串："User: 消息1\nAI: 回复1\nUser: 消息2..."
        会话不存在/无历史时，返回空字符串
    """
    global SESSION_MEMORIES

    # 1. 会话不存在：直接返回空字符串
    if session_id not in SESSION_MEMORIES:
        return ""

    # 2. 获取该会话的记忆实例
    memory = SESSION_MEMORIES[session_id]

    # 3. 无对话历史：返回空字符串
    if not memory.chat_memory:
        return ""

    # 4. 格式化历史记录（按"User: ... AI: ..."格式整理）
    summary_lines = []
    for msg in memory.chat_memory:
        if msg["role"] == "user":
            summary_lines.append(f"User: {msg['content']}")
        elif msg["role"] == "assistant":
            summary_lines.append(f"AI: {msg['content']}")

    # 5. 用换行符连接所有对话，返回最终摘要
    return "\n".join(summary_lines)


# ------------------------------------------------------------------------------
# 辅助函数：clear_memory（测试用，非必需但推荐保留）
# ------------------------------------------------------------------------------
def clear_memory(session_id: str = None):
    """
    清空记忆：
    - session_id为None：清空所有会话的记忆
    - 传入具体session_id：仅清空该会话的记忆
    """
    global SESSION_MEMORIES
    if session_id is None:
        SESSION_MEMORIES.clear()
    elif session_id in SESSION_MEMORIES:
        del SESSION_MEMORIES[session_id]


# ------------------------------------------------------------------------------
# 本地测试代码（帮助你理解运行流程，可直接运行）
# ------------------------------------------------------------------------------
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