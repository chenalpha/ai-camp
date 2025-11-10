
from langchain.memory import ConversationBufferMemory
from langchain_core.prompts import PromptTemplate
from langchain.chains import LLMChain
from langchain_ollama import OllamaLLM

# 1. 初始化 Memory（笔记本）
memory = ConversationBufferMemory(
    memory_key="history",  # 记忆变量名，后续 Prompt 要用到
    return_messages=False  # 返回字符串格式的历史（方便 Prompt 拼接）
)

# 2. 初始化 LLM（大模型）
llm = OllamaLLM(
    model="qwen3:8b",
    temperature=0.0
)

# 3. 初始化 PromptTemplate（带历史变量的提示词）
prompt = PromptTemplate(
    input_variables=["history", "input"],  # 必须包含 memory_key（history）和当前输入（input）
    template="""以下是之前的对话：
{history}

当前用户输入：{input}
AI回复："""
)

# 4. 用 LLMChain 串联 Prompt + LLM + Memory（实验3的核心逻辑）
chain = LLMChain(
    llm=llm,
    prompt=prompt,
    memory=memory,  # 绑定 Memory，自动 save/load
    verbose=True,  # 开启调试模式，能看到完整的 Prompt（方便学习）
)

# 5. 开始对话（多轮测试）
print("=== 第1轮对话 ===")
response1 = chain.predict(input="我叫小明")  # input 是当前用户消息
print(f"AI：{response1}\n")

print("=== 第2轮对话 ===")
response2 = chain.predict(input="我叫什么名字？")
print(f"AI：{response2}\n")

print("=== 第3轮对话 ===")
response3 = chain.predict(input="我想了解LangChain，能简单介绍下吗？")
print(f"AI：{response3}\n")

print("=== 第4轮对话 ===")
response4 = chain.predict(input="刚才我问的是什么问题？")
print(f"AI：{response4}\n")