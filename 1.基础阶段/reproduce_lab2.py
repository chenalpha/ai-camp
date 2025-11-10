from typing import Dict, List
import httpx
import json

# 全局会话存储：以session_id为键，值为消息列表（每条消息包含role和content）
# 消息格式示例：[{"role": "user", "content": "你好"}, {"role": "assistant", "content": "你好！"}]
SESSION_HISTORY: Dict[str, List[Dict[str, str]]] = {}


def chat_with_memory(message: str, session_id: str) -> dict:
    """
    带记忆功能的多会话隔离对话函数
    :param message: 用户当前发送的消息内容
    :param session_id: 会话唯一标识符（用于隔离不同会话的历史）
    :return: 包含AI回复、历史长度、会话ID的字典
    """
    # 全局变量声明：操作全局的会话存储字典
    global SESSION_HISTORY

    # 步骤1：初始化会话（若session_id不存在，创建空列表存储其历史消息）
    if session_id not in SESSION_HISTORY:
        SESSION_HISTORY[session_id] = []

    # 步骤2：计算本次对话前的历史消息数量（历史消息是user和assistant成对存储的，总条数即为历史长度）
    history_length = len(SESSION_HISTORY[session_id])

    # 步骤3：构建包含历史上下文的消息列表（用于传递给Ollama模型）
    context_messages = SESSION_HISTORY[session_id].copy()
    # 先复制历史消息，再追加当前用户消息，形成完整上下文
    context_messages.append(
        {"role": "user", "content": message}
    )

    # 步骤4：调用Ollama API获取AI回复（使用测试环境适配的qwen3小模型，确保响应速度）
    url = "http://localhost:11434/api/chat"
    payload = {
        "model": "qwen3:8b",
        "messages": context_messages,
        "stream": False,
        "temperature": 0.0,
        "max_tokens": 64
    }

    # 发送HTTP请求并处理响应
    with httpx.Client(timeout=30) as client:
        response = client.post(
            url,
            content = json.dumps(payload)
        )
        ai_response = response.json()["message"]["content"].strip()

    # 步骤5：保存本次对话到历史记录（用户消息和AI回复成对存储）
    SESSION_HISTORY[session_id].append({"role": "user", "content": message})
    SESSION_HISTORY[session_id].append({"role": "assistant", "content": ai_response})

    # 步骤6：返回结构化结果（包含AI回复、历史长度、会话ID）
    return {
        "response": ai_response,
        "history_length": history_length,
        "session_id": session_id
    }


def clear_session(session_id: str = None):
    """
    清除会话历史（用于测试时重置环境）
    :param session_id: 要清除的会话ID，为None时清除所有会话
    """
    global SESSION_HISTORY
    if session_id is None:
        SESSION_HISTORY.clear()  # 清除所有会话
    elif session_id in SESSION_HISTORY:
        del SESSION_HISTORY[session_id]  # 清除指定会话


# 测试代码（本地调试用，需确保Ollama服务已启动并加载qwen3:0.6b模型）
if __name__ == "__main__":
    clear_session()  # 测试前清空所有历史

    # 测试单会话历史累积
    print("=== 单会话历史累积测试 ===")
    test_session = "test_single"
    res1 = chat_with_memory("你好", test_session)
    print(f"第1次对话 - 历史长度: {res1['history_length']}, 回复: {res1['response'][:50]}")

    res2 = chat_with_memory("我叫张三", test_session)
    print(f"第2次对话 - 历史长度: {res2['history_length']}, 回复: {res2['response'][:50]}")

    res3 = chat_with_memory("我刚才说了什么？", test_session)
    print(f"第3次对话 - 历史长度: {res3['history_length']}, 回复: {res3['response'][:50]}")

    # 测试多会话隔离
    print("\n=== 多会话隔离测试 ===")
    clear_session()  # 重置环境
    res_a = chat_with_memory("我喜欢苹果", "session_A")
    res_b = chat_with_memory("我喜欢香蕉", "session_B")
    print(f"Session A 回复: {res_a['response'][:30]}")
    print(f"Session B 回复: {res_b['response'][:30]}")