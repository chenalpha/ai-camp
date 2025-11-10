from langchain.memory import ConversationBufferMemory

memory = ConversationBufferMemory(memory_key = "history")

memory.save_context(
    inputs = {"input": "我叫小明"},
    outputs = {"output": "你好小明！很高兴认识你~"}
)

history = memory.load_memory_variables(inputs = {})
print(f"加载的历史对话：{history}")

memory.save_context(
    inputs={"input": "我叫什么名字？"},
    outputs={"text": "你叫小明呀～"}
)

history2 = memory.load_memory_variables(inputs = {})
print(f"加载完整的历史对话：{history2}")
