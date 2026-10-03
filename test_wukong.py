#!/usr/bin/env python3
"""
测试WukongTTS系统
"""

import os
import sys
from pathlib import Path

# 添加当前目录到路径
sys.path.append(str(Path(__file__).parent))

from wukong_tts import WukongTTS


def test_text_generation():
    """测试文本生成功能"""
    print("=== 测试文本生成 ===")
    
    try:
        # 初始化系统（仅LLM部分）
        wukong = WukongTTS(
            base_model_path="qwen/Qwen2.5-7B-Instruct",
            lora_path="./dataset/output"
        )
        
        # 测试问题列表
        test_prompts = [
            "你是谁？",
            "你有什么本领？",
            "师父被妖怪抓走了怎么办？",
            "天庭的玉皇大帝怎么样？",
            "你和二师弟猪八戒关系如何？"
        ]
        
        for i, prompt in enumerate(test_prompts, 1):
            print(f"\n测试 {i}: {prompt}")
            response = wukong.generate_text(prompt)
            print(f"悟空: {response}")
            print("-" * 50)
            
    except Exception as e:
        print(f"文本生成测试失败: {e}")


def test_full_system():
    """测试完整系统（需要参考音频）"""
    print("\n=== 测试完整系统 ===")
    
    # 检查参考音频是否存在
    ref_audio_path = "./audio/wukong_ref.wav"
    if not os.path.exists(ref_audio_path):
        print(f"参考音频不存在: {ref_audio_path}")
        print("请准备孙悟空的参考音频文件")
        return
    
    try:
        wukong = WukongTTS(
            base_model_path="qwen/Qwen2.5-7B-Instruct",
            lora_path="./dataset/output",
            ref_audio_path=ref_audio_path
        )
        
        # 测试完整流程
        prompt = "你好，孙悟空！"
        response, audio_path = wukong.chat_with_voice(prompt, "./output")
        
        if audio_path and os.path.exists(audio_path):
            print(f"✅ 测试成功！音频已保存到: {audio_path}")
        else:
            print("⚠️ 音频生成失败，但文本生成正常")
            
    except Exception as e:
        print(f"完整系统测试失败: {e}")


def create_demo_ref_audio():
    """创建演示用的参考音频文件说明"""
    print("\n=== 参考音频准备指南 ===")
    print("为了获得最佳效果，请准备以下格式的参考音频：")
    print("1. 格式：WAV文件，16kHz或22kHz采样率")
    print("2. 时长：5-12秒")
    print("3. 内容：清晰的孙悟空配音（动画、游戏等）")
    print("4. 质量：无背景音乐，语音清晰")
    print("5. 保存路径：./audio/wukong_ref.wav")
    print()
    print("参考文本应该与音频内容匹配，例如：")
    print("'俺老孙来也！妖怪哪里逃！'")
    print()
    print("可以从以下来源获取孙悟空音频：")
    print("- 西游记动画片配音")
    print("- 游戏《黑神话：悟空》配音")
    print("- 其他孙悟空相关影视作品")


def main():
    """主测试函数"""
    print("WukongTTS 系统测试")
    print("=" * 50)
    
    # 检查基本环境
    print("检查环境...")
    
    # 检查数据集
    dataset_path = "./dataset/train/lora/西游记白话文.json"
    if os.path.exists(dataset_path):
        print(f"✅ 数据集存在: {dataset_path}")
    else:
        print(f"❌ 数据集不存在: {dataset_path}")
    
    # 检查LoRA权重
    lora_path = "./dataset/output"
    if os.path.exists(lora_path):
        print(f"✅ LoRA输出目录存在: {lora_path}")
    else:
        print(f"⚠️ LoRA输出目录不存在: {lora_path}")
        print("   请先运行 python train.py 进行模型训练")
    
    # 检查F5-TTS
    f5tts_path = "./F5-TTS"
    if os.path.exists(f5tts_path):
        print(f"✅ F5-TTS目录存在: {f5tts_path}")
    else:
        print(f"❌ F5-TTS目录不存在: {f5tts_path}")
    
    print("-" * 50)
    
    # 运行测试
    try:
        # 测试1: 仅文本生成
        test_text_generation()
        
        # 测试2: 完整系统（如果有参考音频）
        test_full_system()
        
        # 创建参考音频指南
        create_demo_ref_audio()
        
    except Exception as e:
        print(f"测试过程中出现错误: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    main()
