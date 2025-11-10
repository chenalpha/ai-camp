from langchain_ollama import OllamaLLM
from langchain_core.prompts import PromptTemplate
from langchain_core.runnables import RunnablePassthrough, RunnableLambda, RunnableSequence

# 1. 初始化 Ollama LLM
llm = OllamaLLM(model="qwen3:8b", temperature=0.3)

# 2. 定义 PromptTemplate（保持原逻辑）
prompt1 = PromptTemplate(
    input_variables=["user_question"],
    template="用户问题：{user_question}\n请提取1-2个核心关键词（用中文）："
)

prompt2 = PromptTemplate(
    input_variables=["user_question", "keywords"],
    template="用户问题：{user_question}\n核心关键词：{keywords}\n请基于关键词写一段简洁的回答（50字以内）："
)

# 3. 构建链式调用（新版 RunnableSequence 语法）
# 链1：用户问题 → 提取关键词
chain1 = prompt1 | llm

# 链2：用户问题 + 关键词 → 生成回答
# 需用 RunnablePassthrough 传递原始用户问题，再结合 chain1 的输出（keywords）
chain2 = {
    "user_question": RunnablePassthrough(),
    "keywords": chain1
} | prompt2 | llm

# 4. 组合为 SequentialChain（新版 RunnableSequence 语法）
overall_chain = RunnableSequence(
    {"user_question": RunnablePassthrough()},  # 输入用户问题
    chain2,  # 按顺序执行 chain1 → chain2
)

# 5. 运行并输出结果
result = overall_chain.invoke("LangChain 的 Chain 组件有什么用？")
print("\n最终结果：")
print(f"回答：{result}")

# （若需同时输出关键词，可拆分 chain1 单独调用）
keywords = chain1.invoke("LangChain 的 Chain 组件有什么用？")
print(f"关键词：{keywords}")
