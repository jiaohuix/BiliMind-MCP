# tests/test_faster_whisper.py
from faster_whisper import WhisperModel
import os
import sys
import requests
import json

def download_model_manually(model_size="tiny", output_dir="./models", use_mirror=True):
    """手动下载模型文件，使用镜像站点"""
    print(f"开始从镜像站点下载 {model_size} 模型...")
    
    # 确保输出目录存在
    os.makedirs(output_dir, exist_ok=True)
    
    # 使用镜像站点
    base_url = "https://hf-mirror.com" if use_mirror else "https://huggingface.co"
    repo_id = f"guillaumekln/faster-whisper-{model_size}"
    
    # 创建模型目录
    model_dir = os.path.join(output_dir, model_size)
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
    
    # 创建一个模型配置文件，指示这是从镜像站点下载的
    with open(os.path.join(model_dir, "download_info.json"), "w", encoding="utf-8") as f:
        json.dump({
            "source": "hf-mirror.com" if use_mirror else "huggingface.co",
            "model_size": model_size,
            "download_date": str(os.path.getctime(os.path.join(model_dir, "model.bin")))
        }, f, indent=2)
    
    return True

def transcribe_audio(file_path, model_dir="./models"):
    """使用 faster-whisper 转录音频文件"""
    
    model_size = "tiny"
    model_path = os.path.join(model_dir, model_size)
    
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
            if not download_model_manually(model_size=model_size, output_dir=model_dir, use_mirror=True):
                raise Exception("无法下载模型")
            model = WhisperModel(
                model_path,
                device="cpu", 
                compute_type="int8",
                local_files_only=True
            )
    else:
        print(f"未发现本地模型 {model_size}，开始下载...")
        if download_model_manually(model_size=model_size, output_dir=model_dir, use_mirror=True):
            model = WhisperModel(
                model_path,
                device="cpu", 
                compute_type="int8",
                local_files_only=True
            )
        else:
            raise Exception("无法下载模型")
    
    # 执行转录
    print(f"开始转录: {file_path}")
    segments, info = model.transcribe(file_path, language="zh", beam_size=5)
    
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

def main():
    # 测试音频文件路径
    test_audio_path = 'downloads/BV1z65TzuE94.mp3'
    model_dir = "./models"
    
    # 确保模型下载目录存在
    os.makedirs(model_dir, exist_ok=True)
    
    print(f"开始处理音频文件: {test_audio_path}")
    
    try:
        # 执行转录
        result = transcribe_audio(test_audio_path, model_dir)
        
        if result:
            print("\n转录成功!")
            print(f"完整文本: {result['full_text']}")
            
            print("\n分段详情:")
            for segment in result['segments']:
                print(f"[{segment.start:.2f}s -> {segment.end:.2f}s] {segment.text}")
            
            # 保存结果到文件
            with open('tests/faster_whisper_result.txt', 'w', encoding='utf-8') as f:
                f.write(result['full_text'])
                f.write("\n\n分段详情:\n")
                for segment in result['segments']:
                    f.write(f"[{segment.start:.2f}s -> {segment.end:.2f}s] {segment.text}\n")
            
            print("\n转录结果已保存到 tests/faster_whisper_result.txt")
    
    except Exception as e:
        print(f"转录过程中出现错误: {e}")

if __name__ == "__main__":
    main()