from langchain_ollama import OllamaLLM
from langchain_core.prompts import PromptTemplate
from langchain.chains import LLMChain
from langchain.memory import ConversationBufferMemory

# 1. 初始化组件
llm = OllamaLLM(
    model="qwen3:8b",
    temperature=0.0
)
memory = ConversationBufferMemory(memory_key="history")
prompt = PromptTemplate(
    input_variables=["history", "input"],
    template="""以下是之前的对话：
{history}
    
当前用户输入：{input}
AI回复："""
)

# 2. 创建 LLMChain（串联 Prompt + LLM + Memory）
chain = LLMChain(
    llm=llm,
    memory=memory,
    prompt=prompt,
    verbose=True
)

# 3. 多轮对话（只需调用 predict，流程自动完成）
print("第1轮：")
print(chain.predict(input="我叫小明"), "\n")

print("第2轮：")
print(chain.predict(input="我叫什么名字？"), "\n")

print("第3轮：")
print(chain.predict(input="我想学习 Chain，能给我点建议吗？"), "\n")

print("第4轮：")
print(chain.predict(input="我刚才问的是什么问题？"), "\n")
