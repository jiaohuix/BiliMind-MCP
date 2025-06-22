import os
import sys
import json
import time
import requests
import argparse
from typing import Dict, List, Any
import yt_dlp
from faster_whisper import WhisperModel

from dotenv import load_dotenv

# 加载环境变量
load_dotenv()

# 定义常量
API_KEY = os.getenv("OPENAI_API_KEY", "sk-")
API_BASE = os.getenv("API_BASE", "https://api.siliconflow.cn/v1")


class BilibiliDownloader:
    """哔哩哔哩视频下载器"""
    
    def __init__(self, output_dir: str = "downloads"):
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
    
    def __init__(self, model_dir: str = "./models"):
        self.model_dir = model_dir
        os.makedirs(model_dir, exist_ok=True)
        
    def download_model(self, model_size: str = "tiny", use_mirror: bool = True) -> bool:
        """下载模型文件，使用镜像站点"""
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
        
    def transcribe(self, audio_path: str, model_size: str = "tiny") -> Dict:
        """转录音频文件"""
        model_path = os.path.join(self.model_dir, model_size)
        
        # 检查是否已经下载了模型
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
                if not self.download_model(model_size=model_size):
                    raise Exception("无法下载模型")
                model = WhisperModel(
                    model_path,
                    device="cpu", 
                    compute_type="int8",
                    local_files_only=True
                )
        else:
            print(f"未发现本地模型 {model_size}，开始下载...")
            if self.download_model(model_size=model_size):
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

# 定义常量
API_KEY = os.getenv("OPENAI_API_KEY", "sk-")
API_BASE = os.getenv("API_BASE", "https://api.siliconflow.cn/v1")

class NotesGenerator:
    """使用LLM生成笔记"""
    
    def __init__(self, api_base: str = API_BASE, 
                 api_key: str = API_KEY,
                 model: str = "Qwen/Qwen3-8B"):
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

def main():
    DEFAULT_VIDEO_URL = "https://www.bilibili.com/video/BV1z65TzuE94"
    
    parser = argparse.ArgumentParser(description='从B站视频生成笔记')
    parser.add_argument('--url', '-u', default=DEFAULT_VIDEO_URL, 
                        help=f'B站视频链接 (默认: {DEFAULT_VIDEO_URL})')
    parser.add_argument('--output', '-o', default='video_notes.md', help='输出笔记文件路径')
    parser.add_argument('--model-size', '-m', default='tiny', choices=['tiny', 'base', 'small', 'medium', 'large-v3'], 
                        help='Whisper模型大小')
    parser.add_argument('--keep-audio', '-k', action='store_true', help='保留下载的音频文件')
    
    args = parser.parse_args()
    
    print(f"处理视频: {args.url}")
    print(f"使用模型: {args.model_size}")
    print(f"输出文件: {args.output}")
    
    # 步骤1: 下载视频音频
    downloader = BilibiliDownloader()
    try:
        audio_info = downloader.download_audio(args.url)
    except Exception as e:
        print(f"下载音频失败: {e}")
        return
    
    # 步骤2: 转录音频
    transcriber = WhisperTranscriber()
    try:
        transcript = transcriber.transcribe(audio_info['file_path'], model_size=args.model_size)
    except Exception as e:
        print(f"转录音频失败: {e}")
        return
    
    # 步骤3: 生成笔记
    notes_generator = NotesGenerator()
    try:
        notes = notes_generator.generate_notes(
            transcript["full_text"], 
            video_title=audio_info['title'],
            tags=""
        )
    except Exception as e:
        print(f"生成笔记失败: {e}")
        return
    
    # 保存笔记
    with open(args.output, 'w', encoding='utf-8') as f:
        f.write(notes)
    print(f"笔记已保存到: {args.output}")
    
    # 保存转录文本
    transcript_file = f"{os.path.splitext(args.output)[0]}_transcript.txt"
    with open(transcript_file, 'w', encoding='utf-8') as f:
        f.write(transcript["full_text"])
        f.write("\n\n分段详情:\n")
        for segment in transcript["segments"]:
            f.write(f"[{segment.start:.2f}s -> {segment.end:.2f}s] {segment.text}\n")
    print(f"转录文本已保存到: {transcript_file}")
    
    # 如果不保留音频文件，则删除
    if not args.keep_audio and os.path.exists(audio_info['file_path']):
        os.remove(audio_info['file_path'])
        print(f"已删除音频文件: {audio_info['file_path']}")
    else:
        print(f"音频文件保留在: {audio_info['file_path']}")

if __name__ == "__main__":
    main() 