# BiliMind-MCP

## 功能特点
- 从B站视频生成结构化笔记
- 支持音频自动下载和转录
- 使用 Whisper 进行语音识别
- 使用 LLM 生成结构化笔记
- 提供 MCP 服务接口

## 环境准备

### 系统要求
- Python >= 3.10
- FFmpeg（用于音频处理）

### 1. 安装系统依赖
```bash
# Ubuntu/Debian
sudo apt update && sudo apt install ffmpeg -y

# CentOS/RHEL
sudo yum install ffmpeg -y
```

### 2. 安装 uv
```bash
# 使用 pip 安装 uv
pip install uv

# 或使用 curl 安装
curl -LsSf https://astral.sh/uv/install.sh | sh
```

### 3. 创建并激活虚拟环境
```bash
# 创建虚拟环境
uv venv

# 激活虚拟环境
source .venv/bin/activate  # Linux/macOS
# 或
.venv\Scripts\activate     # Windows
```

### 4. 安装项目依赖
```bash
uv pip install -r requirements.txt
```

### 5. 配置环境变量
```bash
# 复制环境变量示例文件
cp .env.example .env

# 编辑 .env 文件，填入必要的配置
nano .env
```

## 使用方法

### 启动 MCP 服务器
```bash
python demo/bilimind_mcp.py
```

### API 调用示例
```python
from mcp.client import Client

# 连接到 MCP 服务器
client = Client("bili_note_generator")

# 生成笔记
video_url = "https://www.bilibili.com/video/BVxxxxxx"
notes = await client.generate_bilibili_notes(video_url)
```

## 环境变量说明
- `OPENAI_API_KEY`: LLM API密钥
- `API_BASE`: API基础URL
- `MCP_PORT`: MCP服务器端口（默认8001）
- `DEFAULT_OUTPUT_DIR`: 下载文件保存目录
- `DEFAULT_MODEL_DIR`: 模型文件保存目录
- `WHISPER_MODEL_SIZE`: Whisper模型大小（默认tiny）

## 注意事项
- 首次运行会自动下载 Whisper 模型文件
- 音频文件会在处理完成后自动删除
- 需要确保有足够的磁盘空间存储临时文件和模型文件
- API调用需要有效的 API 密钥

## 目录结构

```
BiliMind-MCP/
├── demo/ # 示例代码
├── tests/ # 测试文件
├── .env.example # 环境变量示例
├── requirements.txt # 项目依赖
└── README.md # 项目文档
```

## 致谢
本项目受到 [BiliNote](https://github.com/JefferyHcool/BiliNote) 项目的启发。BiliNote 是一个优秀的开源 AI 视频笔记助手，支持多平台视频内容的自动笔记生成。


## License

MIT License