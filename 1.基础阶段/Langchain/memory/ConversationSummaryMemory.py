from langchain.memory import ConversationSummaryMemory
from langchain_ollama import OllamaLLM

llm = OllamaLLM(
    model = "qwen3:8b"
)

# 初始化：需要传入 LLM（用于总结历史）
memory = ConversationSummaryMemory(
    memory_key = "history",
    llm = llm,
    return_messages=False
)

# 模拟多轮长对话
memory.save_context({"input": "我叫小明，今年25岁"}, {"text": "你好小明！25岁很年轻呀～"})
memory.save_context({"input": "我在一家互联网公司做开发，每天要写代码、改bug"}, {"text": "开发工作很有挑战性～"})
memory.save_context({"input": "我周末喜欢爬山、看书，偶尔和朋友聚餐"}, {"text": "生活很丰富呀～"})

# 加载总结后的历史
summary = memory.load_memory_variables(inputs = {})
print(f"总结后的历史：{summary['history']}")
