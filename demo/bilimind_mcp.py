import os
import sys
import json
import time
import requests
from typing import Dict

import yt_dlp
from faster_whisper import WhisperModel
from dotenv import load_dotenv
from datetime import datetime
from mcp.server.fastmcp import FastMCP

# 加载环境变量
load_dotenv()



# 定义常量
API_BASE = "https://api.siliconflow.cn/v1"
API_KEY = os.getenv("OPENAI_API_KEY", "sk-")
MCP_PORT =  int(os.getenv("MCP_PORT", 8001))
MODEL_NAME = "Qwen/Qwen3-8B"
DEFAULT_OUTPUT_DIR = "downloads"
DEFAULT_MODEL_DIR = "models"
WHISPER_MODEL_SIZE = "tiny"  # 固定使用tiny模型

# 初始化FastMCP服务器
mcp = FastMCP("bili_note_generator", port=MCP_PORT)

class BilibiliDownloader:
    """哔哩哔哩视频下载器"""
    
    def __init__(self, output_dir: str = DEFAULT_OUTPUT_DIR):
        self.output_dir = output_dir
        os.makedirs(self.output_dir, exist_ok=True)

    def download_audio(self, video_url: str) -> dict:
        """下载B站视频的音频"""
        print(f"开始下载视频音频: {video_url}")
        output_path = os.path.join(self.output_dir, "%(id)s.%(ext)s")
        
        ydl_opts = {
            'format': 'bestaudio[ext=m4a]/bestaudio/best',
            'outtmpl': output_path,
            'postprocessors': [{
                'key': 'FFmpegExtractAudio',
                'preferredcodec': 'mp3',
            }],
            'quiet': True,
        }

        with yt_dlp.YoutubeDL(ydl_opts) as ydl:
            info = ydl.extract_info(video_url, download=True)
            video_id = info.get("id")
            audio_path = os.path.join(self.output_dir, f"{video_id}.mp3")
            
        print(f"音频下载完成: {audio_path}")
        return {
            'file_path': audio_path,
            'title': info.get("title"),
            'duration': info.get("duration", 0),
            'cover_url': info.get("thumbnail"),
            'video_id': video_id,
        }

class WhisperTranscriber:
    """使用Faster-Whisper转录音频"""
    
    def __init__(self, model_dir: str = DEFAULT_MODEL_DIR):
        self.model_dir = model_dir
        os.makedirs(model_dir, exist_ok=True)
    
    def check_model_files(self) -> bool:
        """检查模型文件是否已完整存在"""
        model_size = WHISPER_MODEL_SIZE
        model_path = os.path.join(self.model_dir, model_size)
        
        # 需要检查的文件列表
        required_files = [
            "model.bin",
            "config.json",
            "tokenizer.json",
            "vocabulary.txt"
        ]
        
        # 检查每个文件是否存在
        all_files_exist = True
        for filename in required_files:
            file_path = os.path.join(model_path, filename)
            if not os.path.exists(file_path):
                print(f"缺少模型文件: {filename}")
                all_files_exist = False
                break
        
        return all_files_exist
        
    def download_model(self, use_mirror: bool = True) -> bool:
        """下载模型文件，使用镜像站点"""
        model_size = WHISPER_MODEL_SIZE
        print(f"开始从镜像站点下载 {model_size} 模型...")
        
        # 使用镜像站点
        base_url = "https://hf-mirror.com" if use_mirror else "https://huggingface.co"
        repo_id = f"guillaumekln/faster-whisper-{model_size}"
        
        # 创建模型目录
        model_dir = os.path.join(self.model_dir, model_size)
        os.makedirs(model_dir, exist_ok=True)
        
        # 需要下载的文件列表
        files_to_download = [
            "model.bin",
            "config.json",
            "tokenizer.json",
            "vocabulary.txt"
        ]
        
        # 下载文件
        for filename in files_to_download:
            file_path = os.path.join(model_dir, filename)
            
            if os.path.exists(file_path):
                print(f"文件 {filename} 已存在，跳过下载")
                continue
            
            url = f"{base_url}/{repo_id}/resolve/main/{filename}"
            print(f"下载 {filename} 从 {url}...")
            
            try:
                response = requests.get(url, stream=True)
                response.raise_for_status()
                
                total_size = int(response.headers.get('content-length', 0))
                block_size = 1024  # 1 KB
                downloaded = 0
                
                with open(file_path, 'wb') as f:
                    for data in response.iter_content(block_size):
                        downloaded += len(data)
                        f.write(data)
                        
                        # 显示下载进度
                        done = int(50 * downloaded / total_size) if total_size > 0 else 0
                        sys.stdout.write(f"\r[{'=' * done}{' ' * (50-done)}] {downloaded/1024/1024:.2f}/{total_size/1024/1024:.2f} MB")
                        sys.stdout.flush()
                
                print(f"\n{filename} 下载完成")
                
            except Exception as e:
                print(f"下载 {filename} 失败: {e}")
                if os.path.exists(file_path):
                    os.remove(file_path)  # 删除可能部分下载的文件
                return False
        
        return True
        
    def transcribe(self, audio_path: str) -> Dict:
        """转录音频文件"""
        model_size = WHISPER_MODEL_SIZE
        model_path = os.path.join(self.model_dir, model_size)
        
        # 检查是否存在全局模型目录
        global_model_path = os.path.join(DEFAULT_MODEL_DIR, model_size)
        global_model_exists = os.path.exists(os.path.join(global_model_path, "model.bin"))
        
        # 优先使用全局模型目录
        if global_model_exists:
            print(f"发现全局模型目录 {global_model_path}，将使用该目录")
            model_path = global_model_path
        
        # 检查模型文件是否完整存在
        if os.path.exists(os.path.join(model_path, "model.bin")):
            print(f"发现本地模型 {model_size}，直接加载...")
            try:
                model = WhisperModel(
                    model_path,
                    device="cpu", 
                    compute_type="int8",
                    local_files_only=True
                )
            except Exception as e:
                print(f"加载本地模型失败: {e}")
                print("尝试重新下载...")
                if not self.download_model():
                    raise Exception("无法下载模型")
                model = WhisperModel(
                    model_path,
                    device="cpu", 
                    compute_type="int8",
                    local_files_only=True
                )
        else:
            print(f"未发现本地模型 {model_size}，开始下载...")
            if self.download_model():
                model = WhisperModel(
                    model_path,
                    device="cpu", 
                    compute_type="int8",
                    local_files_only=True
                )
            else:
                raise Exception("无法下载模型")
        
        # 执行转录
        print(f"开始转录: {audio_path}")
        segments, info = model.transcribe(audio_path, language="zh", beam_size=5)
        
        # 打印检测到的语言和概率
        print(f"检测到语言: '{info.language}' (概率: {info.language_probability:.2f})")
        
        # 收集所有文本片段
        full_text = ""
        segments_list = list(segments)  # 将生成器转换为列表
        
        for segment in segments_list:
            full_text += segment.text + " "
        
        return {
            "full_text": full_text.strip(),
            "segments": segments_list,
            "language": info.language
        }

class NotesGenerator:
    """使用LLM生成笔记"""
    
    def __init__(self, api_base: str = API_BASE, 
                 api_key: str = API_KEY,
                 model: str = MODEL_NAME):
        self.api_base = api_base
        self.api_key = api_key
        self.model = model
        
    def generate_notes(self, transcript_text: str, video_title: str = "", tags: str = "") -> str:
        """根据转录文本生成笔记"""
        print("开始生成笔记...")
        
        # 构建提示词
        prompt = f"""
你是一个专业的笔记助手，擅长将视频转录内容整理成清晰、有条理且信息丰富的笔记。

语言要求：
- 笔记必须使用 **中文** 撰写。
- 专有名词、技术术语、品牌名称和人名应适当保留 **英文**。

视频标题：
{video_title}

视频标签：
{tags}

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
            "Authorization": f"Bearer {self.api_key}"
        }
        
        data = {
            "model": self.model,
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
            response = requests.post(f"{self.api_base}/chat/completions", headers=headers, json=data)
            response.raise_for_status()
            result = response.json()
            return result["choices"][0]["message"]["content"]
        except Exception as e:
            print(f"调用API失败: {e}")
            if 'response' in locals() and response:
                print(f"响应状态码: {response.status_code}")
                print(f"响应内容: {response.text}")
            return ""

# 实现MCP工具
@mcp.tool()
async def generate_bilibili_notes(video_url: str) -> str:
    """
    从B站视频生成笔记。该工具会下载视频音频，转录为文本，然后生成结构化笔记。
    
    Args:
        video_url: B站视频链接，例如 https://www.bilibili.com/video/BV1z65TzuE94
    
    Returns:
        str: 生成的笔记内容（Markdown格式）
    """
    # 记录开始时间
    start_time = time.time()
    
    # 创建临时目录
    timestamp = datetime.now().strftime("%Y%m%d%H%M%S")
    output_dir = f"downloads_{timestamp}"
    
    # 检查全局模型目录是否存在
    global_model_dir = DEFAULT_MODEL_DIR
    model_path = os.path.join(global_model_dir, WHISPER_MODEL_SIZE)
    
    # 如果全局目录不存在或不完整，则使用临时目录
    model_dir = global_model_dir
    if not (os.path.exists(model_path) and 
            os.path.exists(os.path.join(model_path, "model.bin"))):
        model_dir = f"models_{timestamp}"
    
    os.makedirs(output_dir, exist_ok=True)
    
    try:
        # 步骤1: 下载视频音频
        downloader = BilibiliDownloader(output_dir=output_dir)
        audio_info = downloader.download_audio(video_url)
        
        # 步骤2: 转录音频
        transcriber = WhisperTranscriber(model_dir=model_dir)
        transcript = transcriber.transcribe(audio_info['file_path'])
        
        # 步骤3: 生成笔记
        notes_generator = NotesGenerator()
        notes = notes_generator.generate_notes(
            transcript["full_text"], 
            video_title=audio_info['title'],
            tags=""
        )
        
        # 删除音频文件
        if os.path.exists(audio_info['file_path']):
            os.remove(audio_info['file_path'])
            print(f"已删除音频文件: {audio_info['file_path']}")
            
        # 记录结束时间
        end_time = time.time()
        processing_time = end_time - start_time
        
        # 添加处理信息
        processing_info = f"""
---

**处理信息**:
- 视频标题: {audio_info['title']}
- 视频ID: {audio_info['video_id']}
- 处理时间: {processing_time:.2f} 秒
- 使用模型: faster-whisper-{WHISPER_MODEL_SIZE}
- 生成时间: {datetime.now().strftime("%Y-%m-%d %H:%M:%S")}

---
"""
        
        return notes + processing_info
        
    except Exception as e:
        error_message = f"生成笔记失败: {str(e)}"
        print(error_message)
        return error_message
    finally:
        # 清理临时文件
        if os.path.exists(output_dir):
            import shutil
            shutil.rmtree(output_dir, ignore_errors=True)
        
        # 如果使用的是临时模型目录，也清理它
        if model_dir != global_model_dir and os.path.exists(model_dir):
            import shutil
            shutil.rmtree(model_dir, ignore_errors=True)

@mcp.tool()
async def get_current_time() -> str:
    """
    获取当前的日期和时间，并返回格式化的时间字符串，包括星期几和其他详细信息。
    
    Returns:
        str: 当前时间的详细信息，格式为 "YYYY年MM月DD日 星期X HH:MM:SS"
    """
    # 获取当前时间
    current_time = datetime.now()
    
    # 格式化日期和时间
    formatted_time = current_time.strftime("%Y年%m月%d日 %H:%M:%S")
    
    # 获取星期几（英文）
    weekday_english = current_time.strftime("%A")
    
    # 将英文星期几转换为中文
    weekday_chinese = {
        "Monday": "星期一",
        "Tuesday": "星期二",
        "Wednesday": "星期三",
        "Thursday": "星期四",
        "Friday": "星期五",
        "Saturday": "星期六",
        "Sunday": "星期日"
    }.get(weekday_english, "未知")
    
    # 组合最终的返回字符串
    result = f"{formatted_time} {weekday_chinese}"
    
    return result

if __name__ == "__main__":
    # 初始化并运行服务器
    mcp.run(transport='sse')  # 服务器发送事件
