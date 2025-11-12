"""
实验5：LangGraph工作流路径验证
学生需要使用 LangGraph 构建状态图工作流，实现基于输入意图的条件路由
"""
from typing import TypedDict, Literal, List
from langgraph.graph import StateGraph, END
from typing_extensions import TypedDict



class WorkflowState(TypedDict):
    """
    工作流状态定义
    包含工作流执行过程中需要传递和更新的所有数据
    """
    input: dict  # 原始输入数据
    intent: str  # 识别出的意图类型
    route_taken: str  # 实际执行的路由路径
    intermediate_results: List[str]  # 中间节点的处理结果
    final_output: str  # 最终输出内容
    execution_path: List[str]  # 节点执行顺序列表（新增，用于追踪执行路径）

def start_node(state: WorkflowState) -> WorkflowState:
    intent = state["input"].get("intent", "")

    return {
        **state,
        "intent": intent,
        "route_taken": "",
        "intermediate_results": [],
        "final_output": "",
        "execution_path": ["start"]  # 记录执行路径
    }

def order_handler_node(state: WorkflowState) -> WorkflowState:

    query = state["input"].get("query", "无具体查询内容")
    processing_result = f"订单处理：收到下单请求 - {query}"
    return {
        **state,
        "route_taken": "order_handler",
        "intermediate_results": state["intermediate_results"] + [processing_result],
        "execution_path": state["execution_path"] + ["order_handler"]  # 追加执行路径
    }

def info_handler_node(state: WorkflowState) -> WorkflowState:

    query = state["input"].get("query", "无具体查询内容")
    processing_result = f"信息查询处理：收到查询请求 - {query}"
    return {
        **state,
        "route_taken": "info_handler",
        "intermediate_results": state["intermediate_results"] + [processing_result],
        "execution_path": state["execution_path"] + ["info_handler"]  # 追加执行路径
    }

def error_handler_node(state: WorkflowState) -> WorkflowState:

    intent = state["intent"]
    # 模拟错误处理逻辑
    error_result = f"错误处理：未识别的意图 - '{intent}'，请使用支持的意图（order/info）"
    return {
        **state,
        "route_taken": "error_handler",
        "intermediate_results": state["intermediate_results"] + [error_result],
        "execution_path": state["execution_path"] + ["error_handler"]  # 追加执行路径
    }

def end_node(state: WorkflowState) -> WorkflowState:

    intermediate_results = state["intermediate_results"]
    # 整合中间结果为最终输出
    final_output = "\n".join(intermediate_results) + "\n处理完成"

    return {
        **state,
        "final_output": final_output,
        "execution_path": state["execution_path"] + ["end"]
    }

def router_intent(state: WorkflowState) -> Literal["order_handler", "info_handler", "error_handler"]:
    intent = state["intent"]
    if intent == "order":
        return "order_handler"
    elif intent == "info":
        return "info_handler"
    else:
        return "error_handler"

def run_workflow(input_data: dict) -> dict:
    """
    执行 LangGraph 工作流，根据意图进行路由

    参数:
        input_data: 输入字典，必须包含以下键:
            - intent (str): 用户意图类型（"order" 或 "info"）
            - query (str): 用户查询内容

    返回:
        字典，包含以下键:
        - route_taken (str): 实际执行的路由名称（如 "order_handler"）
        - final_output (str): 工作流的最终输出内容
        - execution_path (list): 节点执行顺序列表

    实现要求:
        1. 创建 StateGraph 并定义工作流状态
        2. 添加以下节点：
           - start: 起始节点，解析用户意图
           - order_handler: 订单处理节点（当 intent=="order" 时）
           - info_handler: 信息查询节点（当 intent=="info" 时）
           - end: 结束节点，整合输出
        3. 实现条件路由逻辑：
           - 当 intent == "order" 时，路由到 order_handler
           - 当 intent == "info" 时，路由到 info_handler
           - 其他情况路由到 error_handler
        4. 追踪执行路径（execution_path）

    提示:
        - 使用 StateGraph 创建工作流图
        - 使用 add_node() 添加节点
        - 使用 add_conditional_edges() 实现条件路由
        - 在每个节点函数中更新 state 字典
    """
    # TODO: 实现 LangGraph 工作流
    # 提示:
    # 1. 导入 from langgraph.graph import StateGraph, END
    # 2. 定义节点函数（start_node, order_handler_node, info_handler_node, end_node）
    # 3. 定义路由决策函数 route_intent
    # 4. 创建 StateGraph 并添加节点和边
    # 5. 编译并执行工作流

    graph = StateGraph(WorkflowState)

    graph.add_node("start", start_node)
    graph.add_node("order_handler", order_handler_node)
    graph.add_node("info_handler", info_handler_node)
    graph.add_node("error_handler", error_handler_node)
    graph.add_node("end", end_node)

    graph.add_conditional_edges(
        "start",
        router_intent,
        {
            "order_handler": "order_handler",
            "info_handler": "info_handler",
            "error_handler": "error_handler"
        }
    )

    graph.add_edge("order_handler", "end")
    graph.add_edge("info_handler", "end")
    graph.add_edge("error_handler", "end")
    graph.add_edge("end", END)

    graph.set_entry_point("start")
    app = graph.compile()

    initial_state: WorkflowState = {
        "input": input_data,
        "intent": "",
        "route_taken": "",
        "intermediate_results": [],
        "final_output": "",
        "execution_path": []
    }
    final_state = app.invoke(initial_state)

    return {
        "route_taken": final_state["route_taken"],
        "final_output": final_state["final_output"],
        "execution_path": final_state["execution_path"]
    }

# 辅助函数（已在 run_workflow 中实现）


# 测试代码
if __name__ == "__main__":
    # 测试订单处理路由
    print("=== 测试订单处理路由 ===")
    result1 = run_workflow({"intent": "order", "query": "我要下单购买商品"})
    print(f"路由路径: {result1['route_taken']}")
    print(f"执行顺序: {result1['execution_path']}")
    print(f"最终输出: {result1['final_output']}")

    # 测试信息查询路由
    print("\n=== 测试信息查询路由 ===")
    result2 = run_workflow({"intent": "info", "query": "查询订单状态"})
    print(f"路由路径: {result2['route_taken']}")
    print(f"执行顺序: {result2['execution_path']}")
    print(f"最终输出: {result2['final_output']}")

    # 测试无效意图
    print("\n=== 测试无效意图 ===")
    result3 = run_workflow({"intent": "unknown", "query": "随机查询"})
    print(f"路由路径: {result3['route_taken']}")
    print(f"执行顺序: {result3['execution_path']}")
    print(f"最终输出: {result3['final_output']}")
