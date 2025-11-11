from langgraph.graph import StateGraph, END
from typing import TypedDict
from langchain_ollama import OllamaLLM

llm = OllamaLLM(model="qwen3:8b")

class State(TypedDict):
    topic: str
    draft: str  # 允许初始为空，由节点函数填充
    final: str

# 定义两个节点：写草稿→优化草稿
def write_draft(state: State) -> State:
    # 1. 读取State中的主题
    topic = state["topic"]

    # 2. 调用LLM生成草稿（核心逻辑）
    prompt = f"围绕主题{topic}写一份100字左右的报告草稿"
    draft = llm.invoke(prompt)

    # 3. 返回新State
    return {"topic": topic, "draft": draft, "final": ""}

def optimize_draft(state: State) -> State:
    prompt = f"优化草稿：{state['draft']}"
    final = llm.invoke(prompt)
    return {"topic": state["topic"],"draft": state["draft"] , "final": final}


# 构建 LangGraph 工作流
graph_builder = StateGraph(State)
graph_builder.add_node("write", write_draft)
graph_builder.add_node("optimize", optimize_draft)
graph_builder.set_entry_point("write")
graph_builder.add_edge("write", "optimize")
graph_builder.add_edge("optimize", END)
graph = graph_builder.compile()

# 调用工作流（初始状态只需传topic，draft和final由节点自动填充）
result = graph.invoke({"topic": "LangGraph核心概念"})
# 输出结果（同时展示草稿和优化后版本，方便对比）
print(f"=== 报告草稿 ===\n{result['draft']}\n")
print(f"=== 优化后报告 ===\n{result['final']}")
