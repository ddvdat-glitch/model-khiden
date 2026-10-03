# 🚀 WukongTTS 快速开始指南

## 📋 前置要求

- Python 3.8+
- CUDA 11.8+ (推荐，CPU模式也可用)
- 16GB+ 内存 (GPU模式需要8GB+显存)

## ⚡ 一键启动

### 1. 环境设置
```bash
# 克隆项目（如果还没有）
git clone <your-repo-url>
cd "WukongChat TTS"

# 运行设置脚本
chmod +x setup.sh
./setup.sh
```

### 2. 获取F5-TTS
```bash
git clone https://github.com/SWivid/F5-TTS.git
cd F5-TTS
pip install -e .
cd ..
```

### 3. 准备参考音频
- 下载或录制一段5-12秒的孙悟空配音
- 保存为 `audio/wukong_ref.wav`
- 格式：WAV，16kHz或22kHz

### 4. 训练模型
```bash
python train.py
```

### 5. 启动系统
```bash
# Web界面（推荐）
python wukong_gradio.py

# 命令行模式
python wukong_tts.py --interactive

# 演示模式
python demo.py
```

## 🎯 核心功能

### 💬 文本对话
```python
from wukong_tts import WukongTTS

wukong = WukongTTS()
response = wukong.generate_text("你是谁？")
print(response)  # 俺老孙是齐天大圣孙悟空！
```

### 🔊 语音合成
```python
wukong.text_to_speech("俺老孙来也！", "output.wav")
```

### 🎙️ 完整对话
```python
response, audio_path = wukong.chat_with_voice("师父被抓了怎么办？")
```

## 📁 项目结构

```
WukongChat TTS/
├── 🐒 wukong_tts.py       # 核心TTS系统
├── 🌐 wukong_gradio.py    # Web界面
├── 🧪 test_wukong.py      # 测试脚本
├── 🎭 demo.py             # 演示脚本
├── 🏋️ train.py            # 模型训练
├── ⚙️ setup.sh            # 环境设置
├── 📋 requirements.txt    # 依赖列表
├── 🎵 audio/              # 参考音频
├── 📊 dataset/            # 数据集和模型
├── 🔊 output/             # 生成音频
└── 🤖 F5-TTS/             # TTS模型
```

## 🔧 常见问题

### Q: 模型下载很慢？
A: 使用国内镜像或手动下载Qwen2.5模型

### Q: 显存不够？
A: 调整batch_size或使用CPU模式

### Q: 语音效果不好？
A: 检查参考音频质量和格式

### Q: 训练失败？
A: 确保数据集格式正确，检查依赖安装

## 📞 获取帮助

- 查看详细文档：`README.md`
- 运行测试：`python test_wukong.py`
- 查看配置：`config.yaml`
- 系统演示：`python demo.py`

## 🎉 开始体验

访问 http://localhost:7860 开始与孙悟空对话！

> *"俺老孙有七十二变，筋斗云一个跟头十万八千里！有什么本事尽管来试试！"*
