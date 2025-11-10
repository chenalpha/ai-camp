from typing import Dict
from langchain.memory import ConversationBufferMemory

# 全局字典：key=session_id，value=Memory实例
SESSION_MEMORY: Dict[str, ConversationBufferMemory] = {}

def get_memory(session_id: str) -> ConversationBufferMemory:
    """获取会话专属的 Memory（不存在则创建）"""
    if session_id not in SESSION_MEMORY:
        SESSION_MEMORY[session_id] = ConversationBufferMemory(
            memory_key="history",
            return_messages=False,
            output_key="text"
        )
    return SESSION_MEMORY[session_id]

# 小明的会话ID
session1 = "user_001"
# 小红的会话ID
session2 = "user_002"

# 小明的对话
memory1 = get_memory(session1)
memory1.save_context({"input": "我叫小明"}, {"text": "你好小明！"})

# 小红的对话
memory2 = get_memory(session2)
memory2.save_context({"input": "我叫小红"}, {"text": "你好小红！"})

# 验证隔离：小明的记忆里没有小红的对话
print(f"小明的历史：{memory1.load_memory_variables({})['history']}")
print(f"小红的历史：{memory2.load_memory_variables({})['history']}" )

