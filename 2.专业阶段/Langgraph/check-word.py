from langgraph.graph import StateGraph, END
from typing import TypedDict
from langchain_ollama import OllamaLLM

llm = OllamaLLM(model="qwen3:8b")

class State(TypedDict):
    topic: str
    draft: str
    word_count: int

def write_draft(state: State) -> State:
    topic = state["topic"]
    prompt = f"围绕主题{topic}写一份100字左右的报告草稿"
    draft = llm.invoke(prompt)
    return {"topic": topic, "draft": draft,"word_count": 0}

def check_word_count(state: State) -> State:
    word_count = len(state["draft"])
    return {**state,"word_count": word_count}

def optimize_draft(state: State) -> State:
    draft = state["draft"]
    prompt = f"优化草稿：{state['draft']}"
    final = llm.invoke(prompt)
    return {**state, "draft": final}

graph_builder = StateGraph(State)
graph_builder.add_node("write", write_draft)
graph_builder.add_node("check", check_word_count)
graph_builder.add_node("optimize", optimize_draft)

graph_builder.set_entry_point("write")
graph_builder.add_edge("write", "check")
graph_builder.add_edge("check", "optimize")
graph_builder.add_edge("optimize", END)
graph = graph_builder.compile()

if __name__ == "__main__":
    # 输入原料：只需要提供topic（其他字段会在节点中生成）
    input_state = {"topic": "LangGraph三大核心概念（State/Node/Edge）"}
    # 执行工作流
    result = graph.invoke(input_state)
    # 打印结果
    print("="*50)
    print(f"报告主题：{result['topic']}")
    print(f"原始草稿字数：{result['word_count']}字")
    print("="*50)
    print("优化后的报告内容：")
    print(result["draft"])
