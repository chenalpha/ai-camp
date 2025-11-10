"""
实验4：LangChain 链的确定性输出
学生需要使用 LangChain 的 PromptTemplate 和 LLMChain 生成广告文案
"""
# 提示：需要导入 LangChain 相关模块
from langchain.chains.llm import LLMChain
from langchain_core.prompts import PromptTemplate
from langchain_ollama import OllamaLLM

def generate_ad(input_dict: dict) -> dict:

    # 1. 创建 PromptTemplate，包含产品和特性变量
    template = """你是一个专业的广告文案创作者。请为以下产品创作一段吸引人的广告文案：

    产品名称：{product}
    核心特性：{feature}

    要求：
    1. 文案中必须包含产品名称"{product}"
    2. 文案中必须包含特性词汇"{feature}"（可以是完整词汇或核心关键词）
    3. 长度控制在30-80字之间
    4. 语言简洁有力，突出产品优势
    5. 直接输出文案内容，不要包含其他说明文字

    请确保在文案中明确提及"{feature}"这一特性。

    广告文案："""

    prompt = PromptTemplate(
        input_variables = ["product", "feature"],
        template = template
    )

    # 2. 创建 Ollama LLM 实例
    llm = OllamaLLM(
        model = "qwen3:8b",
        base_url="http://localhost:11434"
    )

    # 3. 创建 LLMChain，连接 Prompt 和 LLM
    chain = LLMChain(
        llm = llm,
        prompt = prompt
    )

    # 4. 运行链，传入产品和特性参数
    product = input_dict["product"]
    feature = input_dict["feature"]

    ad_copy = chain.run(product=product, feature=feature)

    # 5. 提取文案并清理格式
    ad_copy = ad_copy.strip()

    # 6. 计算字数
    word_count = len(ad_copy)

    # 7. 返回结构化结果
    return {
        "ad_copy": ad_copy,
        "word_count": word_count,
        "template_used": "广告文案生成模板"
    }

# 测试代码（可选，用于学生本地调试）
if __name__ == "__main__":
    # 测试基础版本
    print("=== 测试基础广告生成 ===")
    test_inputs = [
        {"product": "智能手表", "feature": "心率监测"},
        {"product": "无线耳机", "feature": "降噪功能"},
        {"product": "扫地机器人", "feature": "自动避障"}
    ]

    for input_data in test_inputs:
        try:
            result = generate_ad(input_data)
            print(f"\n产品: {input_data['product']}")
            print(f"文案: {result['ad_copy']}")
            print(f"字数: {result['word_count']}")
            print(f"模板: {result['template_used']}")
        except Exception as e:
            print(f"错误: {e}")