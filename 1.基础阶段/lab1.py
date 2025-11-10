"""
实验1：结构化提示词与输出
学生需要实现 classify_text 函数，使用 Pydantic 模型返回结构化的文本分类结果
"""
from typing import List  # 用于定义列表类型注解（如List[str]）
from pydantic import BaseModel, Field, validator, field_validator  # Pydantic核心组件，用于数据模型定义和验证
import httpx  # 用于发送HTTP请求（替代requests，支持异步）
import json  # 用于JSON字符串和字典的相互转换


class TextClassification(BaseModel):  # 继承BaseModel，使类具备数据验证能力
    """
    文本分类结果的数据模型
    """
    # 定义category字段：分类类别
    category: str = Field(
        ...,  # ... 表示该字段为必填项
        description="文本分类类别，必须是以下之一：'新闻', '技术', '体育', '娱乐', '财经'"
    )
    # 定义confidence_score字段：置信度
    confidence_score: float = Field(
        ...,
        ge=0.0,  # 约束：大于等于0.0
        le=1.0,  # 约束：小于等于1.0
        description="分类置信度，范围0.0-1.0"
    )
    # 定义keywords字段：关键词列表
    keywords: List[str] = Field(
        ...,
        min_length=1,  # 约束：列表长度至少1
        max_length=5,  # 约束：列表长度最多5
        description="从文本中提取的1-5个关键词"
    )

    @field_validator('category')
    def check_category(cls, v):
        value_category = ['新闻', '技术', '体育', '娱乐', '财经']
        if v not in value_category:
            raise ValueError(f'分类必须是以下之一: {value_category}')
        return v


def classify_text(text: str) -> TextClassification:  # 输入文本，返回TextClassification实例
    """
    对输入文本进行分类，返回结构化的分类结果
    """
    # 1. 构建结构化Prompt
    prompt = f"""
    任务：对文本进行分类并提取关键词，返回严格符合格式的JSON。

    待分类文本：{text}

    类别定义（必须严格遵守）：
    - 技术：与科技产品、技术研发、人工智能、软件、硬件、编程相关的内容（如产品发布、技术突破、算法更新）
    - 新闻：对近期发生的社会事件、时事动态的报道（不包含技术、体育、娱乐、财经领域的专业内容）
    - 体育：与体育赛事、运动员、体育规则、体育组织相关的内容
    - 娱乐：与影视、明星、综艺、音乐、游戏等娱乐产业相关的内容
    - 财经：与金融、股市、企业财报、经济政策、商业活动相关的内容

    要求：
    1. 分类类别（category）：从上述5个类别中选择，优先匹配最相关的专业领域
    2. 置信度（confidence_score）：0.0-1.0之间的浮点数，表示分类确定性
    3. 关键词（keywords）：从文本中提取1-5个核心词（如技术类提取产品名、技术名词）

    输出格式：
    仅返回JSON字符串，不包含任何解释。JSON必须包含以下字段：
    - category: 分类结果
    - confidence_score: 置信度
    - keywords: 关键词列表

    示例：
    输入文本："苹果发布最新款iPhone手机"
    输出：{{"category": "技术", "confidence_score": 0.95, "keywords": ["苹果", "iPhone", "手机"]}}

    输入文本："世界杯决赛巴西队3-2战胜德国队"
    输出：{{"category": "体育", "confidence_score": 0.98, "keywords": ["世界杯", "巴西队", "德国队"]}}

    输入文本："OpenAI发布GPT-5大模型，支持多模态生成"
    输出：{{"category": "技术", "confidence_score": 0.99, "keywords": ["OpenAI", "GPT-5", "大模型"]}}
    """.strip()  # 去除首尾空白，避免格式混乱

    # 2. 调用Ollama API
    url = "http://localhost:11434/api/generate"  # Ollama生成接口的地址
    payload = {  # 请求参数
        "model": "qwen3:8b",  # 指定使用的模型
        "prompt": prompt,  # 传入构建好的提示词
        "format": "json",  # 要求模型输出JSON格式
        "stream": False,  # 非流式响应（一次性返回结果）
        "timeout": 60,  # 模型处理超时时间（秒）
        "options": {
            "num_thread": 4,  # 模型推理使用的线程数（加速处理）
            "temperature": 0.0  # 随机性参数（0表示确定性输出，减少错误）
        }
    }

    # 创建HTTP客户端，设置超时时间60秒
    with httpx.Client(timeout=httpx.Timeout(60.0)) as client:
        # 发送POST请求，传入参数
        response = client.post(url, json=payload)

    # 解析响应为字典
    response_data = response.json()
    # 提取模型返回的内容（JSON字符串）
    json_str = response_data.get("response", "")
    if not json_str:  # 若返回内容为空，抛出错误
        raise ValueError("模型返回内容为空")

    # 将JSON字符串解析为字典
    result_dict = json.loads(json_str)
    # 用Pydantic验证字典并返回TextClassification实例
    return TextClassification.model_validate(result_dict)


# 测试代码（可选，用于学生本地调试）
if __name__ == "__main__":  # 当脚本直接运行时执行（导入时不执行）
    test_texts = [  # 测试用例
        "OpenAI发布GPT-5，性能提升10倍",
        "中国队在巴黎奥运会夺得金牌",
        "A股市场今日大涨，沪指突破3000点"
    ]

    for text in test_texts:  # 遍历测试用例
        try:
            result = classify_text(text)  # 调用分类函数
            # 打印结果
            print(f"\n文本: {text}")
            print(f"分类: {result.category}")
            print(f"置信度: {result.confidence_score}")
            print(f"关键词: {result.keywords}")
        except Exception as e:  # 捕获错误并打印
            print(f"错误: {e}")