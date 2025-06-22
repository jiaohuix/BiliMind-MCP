import os
import requests

from dotenv import load_dotenv

# 加载环境变量
load_dotenv()

# 定义常量
API_KEY = os.getenv("OPENAI_API_KEY", "sk-")
API_BASE = os.getenv("API_BASE", "https://api.siliconflow.cn/v1")


def call_qwen_api(transcript_text: str) -> str:
    """调用Qwen API生成总结"""
    api_base = API_BASE
    api_key = API_KEY
    model = "Qwen/Qwen3-8B"
    
    # 构建提示词
    prompt = f"""
你是一个专业的笔记助手，擅长将视频转录内容整理成清晰、有条理且信息丰富的笔记。

语言要求：
- 笔记必须使用 **中文** 撰写。
- 专有名词、技术术语、品牌名称和人名应适当保留 **英文**。

输出说明：
- 仅返回最终的 **Markdown 内容**。
- **不要**将输出包裹在代码块中。
- 如果要加粗并保留编号，应使用 `1\\. **内容**`（加反斜杠），防止被误解析为有序列表。
- 或者使用 `## 1. 内容` 的形式作为标题。

视频转录内容：

---
{transcript_text}
---

你的任务：
根据上面的转录内容，生成结构化的笔记，遵循以下原则：

1. **完整信息**：记录尽可能多的相关细节，确保内容全面。
2. **去除无关内容**：省略广告、填充词、问候语和不相关的言论。
3. **保留关键细节**：保留重要事实、示例、结论和建议。
4. **可读布局**：必要时使用项目符号，并保持段落简短，增强可读性。
5. 视频中提及的数学公式必须保留，并以 LaTeX 语法形式呈现，适合 Markdown 渲染。

额外任务：
1. 为每个主要标题（`##`）添加时间标记，格式为 `*Content-[mm:ss]`。
2. 如果某个部分涉及视觉演示、代码演示或UI交互，在该部分末尾插入截图提示，格式为 `*Screenshot-[mm:ss]`。
3. 在笔记末尾添加一个专业的AI总结，简要概括整个视频的内容。

请提供完整的笔记内容。
"""

    headers = {
        "Content-Type": "application/json",
        "Authorization": f"Bearer {api_key}"
    }
    
    data = {
        "model": model,
        "messages": [
            {
                "role": "system",
                "content": "你是一个专业的笔记助手，擅长将视频转录内容整理成笔记。"
            },
            {
                "role": "user",
                "content": prompt
            }
        ],
        "temperature": 0.7
    }
    
    try:
        response = requests.post(f"{api_base}/chat/completions", headers=headers, json=data)
        response.raise_for_status()
        result = response.json()
        return result["choices"][0]["message"]["content"]
    except Exception as e:
        print(f"调用API失败: {e}")
        if 'response' in locals() and response:
            print(f"响应状态码: {response.status_code}")
            print(f"响应内容: {response.text}")
        return ""

def main():
    # 输入和输出文件路径
    input_file = 'tests/faster_whisper_result.txt'
    output_file = 'tests/video_summary.md'
    
    # 检查输入文件是否存在
    if not os.path.exists(input_file):
        print(f"错误: 输入文件 {input_file} 不存在")
        return
    
    print(f"读取转录文件: {input_file}")
    try:
        # 直接读取整个文件内容
        with open(input_file, 'r', encoding='utf-8') as f:
            transcript_text = f.read()
        
        if not transcript_text:
            print("错误: 文件为空")
            return
        
        print(f"成功读取转录文件，内容长度: {len(transcript_text)} 字符")
    except Exception as e:
        print(f"读取文件失败: {e}")
        return
    
    print("调用Qwen API生成笔记...")
    summary = call_qwen_api(transcript_text)
    
    if summary:
        # 保存笔记
        with open(output_file, 'w', encoding='utf-8') as f:
            f.write(summary)
        print(f"笔记已保存到: {output_file}")
    else:
        print("生成笔记失败")

if __name__ == "__main__":
    main()
