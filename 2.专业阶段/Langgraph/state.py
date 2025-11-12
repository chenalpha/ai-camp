from langgraph.graph import StateGraph, END
from typing import TypedDict

class State(TypedDict):
    message: str
    count: int

def hello_node(state: State) -> State:
    # 读取State中的message和count，修改后返回新State
    new_count = state["count"] + 1
    return {
        "message": f"Hello! 你的输入是{state['message']}",
        "count": new_count
    }

graph_builder = StateGraph(State)
graph_builder.add_node("hello", hello_node)
graph_builder.set_entry_point("hello")
graph_builder.add_edge("hello", END)
graph = graph_builder.compile()

# 投入原料时，必须包含所有必填字段（message和count）
result = graph.invoke({"message": "LangGraph入门", "count": 1})
print(result)