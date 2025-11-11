from langgraph.graph import StateGraph, END
from typing import TypedDict
from langchain_ollama import OllamaLLM

llm = OllamaLLM(model="qwen3:8b")

class State(TypedDict):
    topic: str
    draft: str
    is_satisfied: bool # 是否满意草稿（用于判断）

def write_draft(state: State) -> State:
    topic = state["topic"]
    prompt = f"围绕主题{topic}写一份100字左右的报告草稿"
    draft = llm.invoke(prompt)
    return {"topic": topic, "draft": draft, "is_satisfied": False}

# 路由函数：根据is_satisfied决定流转方向
def route_draft(state: State) -> str:
    if state["is_satisfied"]:
        return END
    else:
        return "optimize"

# 优化节点：优化后标记为“满意”（简化逻辑）
def optimize_draft(state: State) -> State:
    prompt = f"优化草稿：{state['draft']}"
    final = llm.invoke(prompt)
    return {**state, "draft": final, "is_satisfied": True}

# 构建图：写草稿→判断→优化/终点
graph_builder = StateGraph(State)
graph_builder.add_node("write", write_draft)
graph_builder.add_node("optimize", optimize_draft)
graph_builder.set_entry_point("write")
# 条件边：写草稿→路由函数（动态决定下一个节点）
graph_builder.add_conditional_edges("write", route_draft)
graph_builder.add_edge("optimize", END)
graph = graph_builder.compile()

result = graph.invoke({"topic": "LangGraph核心概念"})
print(f"最终报告：{result['draft']}")
