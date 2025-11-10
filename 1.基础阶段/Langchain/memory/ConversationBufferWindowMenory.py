from langchain.memory import ConversationBufferWindowMemory

# 初始化：只保留最近2轮对话
memory = ConversationBufferWindowMemory(
    memory_key = "history",
    k = 2
)

# 模拟3轮对话
memory.save_context({"input": "我叫小明"}, {"text": "你好小明！"})
memory.save_context({"input": "我喜欢爬山"}, {"text": "爬山很健康～"})
memory.save_context({"input": "我还喜欢看书"}, {"text": "看书能增长知识～"})

# 加载历史：只保留最后2轮（第2、3轮）
history = memory.load_memory_variables(inputs={})
print(history["history"])