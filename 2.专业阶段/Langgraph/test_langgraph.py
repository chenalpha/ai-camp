from langgraph.graph import StateGraph, END
from typing import TypedDict

# 1. 定义状态（共享数据）
class State(TypedDict):
    message: str

# 2. 定义节点（处理函数）
def hello_node(state: State) -> State:
    return {"message": f"Hello!你输入的是：{state['message']}"}

# 3. 构建图
graph_builder = StateGraph(State)
graph_builder.add_node("hello", hello_node)
graph_builder.set_entry_point("hello")
graph_builder.add_edge("hello", END)
graph = graph_builder.compile()

# 4. 运行图
result = graph.invoke({"message": "LangGraph入门"})
print(result["message"])
