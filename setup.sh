#!/bin/bash
# WukongTTS 环境设置脚本

echo "🐒 WukongTTS 环境设置开始..."

# 检查Python版本
python_version=$(python3 --version 2>&1 | awk '{print $2}')
echo "Python版本: $python_version"

# 升级pip
echo "📦 升级pip..."
python3 -m pip install --upgrade pip

# 设置pip源（可选）
echo "🔧 配置pip源..."
pip config set global.index-url https://pypi.tuna.tsinghua.edu.cn/simple

# 安装基础依赖
echo "📚 安装基础依赖..."
pip install -r requirements.txt

# 检查CUDA
if command -v nvidia-smi &> /dev/null; then
    echo "🎮 检测到NVIDIA GPU"
    nvidia-smi --query-gpu=name --format=csv,noheader
else
    echo "⚠️  未检测到NVIDIA GPU，将使用CPU模式"
fi

# 设置F5-TTS
echo "🎵 设置F5-TTS..."
if [ -d "F5-TTS" ]; then
    echo "F5-TTS目录已存在"
    cd F5-TTS
    pip install -e .
    cd ..
else
    echo "❌ F5-TTS目录不存在，请先克隆F5-TTS仓库"
    echo "运行: git clone https://github.com/SWivid/F5-TTS.git"
fi

# 创建必要目录
echo "📁 创建目录结构..."
mkdir -p audio output logs

# 检查数据集
echo "📊 检查数据集..."
if [ -f "dataset/train/lora/西游记白话文.json" ]; then
    echo "✅ 数据集文件存在"
    echo "数据条数: $(jq length dataset/train/lora/西游记白话文.json)"
else
    echo "❌ 数据集文件不存在"
fi

echo "✨ 环境设置完成！"
echo ""
echo "📋 下一步："
echo "1. 准备参考音频文件到 audio/wukong_ref.wav"
echo "2. 运行训练: python train.py"
echo "3. 测试系统: python test_wukong.py"
echo "4. 启动Web界面: python wukong_gradio.py"
echo ""
echo "🚀 快速开始："
echo "python wukong_tts.py --interactive  # 命令行交互模式"
echo "python wukong_gradio.py            # Web界面模式"
