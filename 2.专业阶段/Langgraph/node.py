from langgraph.graph import StateGraph, END
from typing import TypedDict
from langchain_ollama import OllamaLLM

llm = OllamaLLM(model="qwen3:8b")

class State(TypedDict):
    topic: str
    draft: str  # 允许初始为空，由节点函数填充

# 定义“写报告”节点（LLM调用）
def write_draft(state: State) -> State:
    # 1. 读取State中的主题
    topic = state["topic"]

    # 2. 调用LLM生成草稿（核心逻辑）
    prompt = f"围绕主题{topic}写一份100字左右的报告草稿"
    draft = llm.invoke(prompt)

    # 3. 返回新State
    return {"topic": topic, "draft": draft}

# 构建 LangGraph 工作流
graph_builder = StateGraph(State)
graph_builder.add_node("write", write_draft)
graph_builder.set_entry_point("write")
graph_builder.add_edge("write", END)
graph = graph_builder.compile()

# 调用图：初始状态只需传入 topic（draft 可缺省，节点会自动填充）
result = graph.invoke({"topic": "LangGraph核心概念"})
print(f"报告草稿：{result['draft']}")
