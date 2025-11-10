from langchain_ollama import OllamaLLM
from langchain_core.prompts import PromptTemplate
from langchain.chains import LLMChain

# 1. 初始化组件
llm = OllamaLLM(model="qwen3:8b", temperature=0.3)
prompt = PromptTemplate(
    input_variables=["question"],
    template="请简单回答以下问题：{question}"
)

# 2. 创建链式调用（正确顺序：prompt 格式化 → llm 生成）
chain = prompt | llm  # 核心：先处理提示词，再传给LLM

# 3. 执行 Chain（仅用最新版支持的 `invoke` 方法，统一输入格式）
# 方式1：字典输入（推荐，标准化用法，适配所有场景）
response1 = chain.invoke({"question": "什么是LangChain？"})
print(f"回答1：{response1}\n")

# 方式2：显式指定 input 参数（等价于方式1，更清晰）
response2 = chain.invoke(input={"question": "LangChain 能做什么？"})
print(f"回答2：{response2}\n")

# 方式3：流式输出（可选，逐字打印，适合长文本）
print("回答3：", end="", flush=True)
for chunk in chain.stream({"question": "LangChain 适合哪些开发者使用？"}):
    print(chunk, end="", flush=True)
