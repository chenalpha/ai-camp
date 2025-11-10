# from langchain_community.llms.ollama import Ollama
from langchain_core.prompts import PromptTemplate
from langchain_ollama import OllamaLLM

# 1. 创建大模型实例（本地Ollama）
llm = OllamaLLM(
    model="qwen3:8b",
    temperature=0.3
)

# 2. 创建提示词模板
prompt = PromptTemplate(
    input_variables = ["question"],
    template = "请简单回答：{question}"
)

# 3. 格式化提示词 + 调用模型
formatted_prompt = prompt.format(question="什么是LangChain？")
response = llm.invoke(formatted_prompt)

print(response)
