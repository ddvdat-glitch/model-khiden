#!/usr/bin/env python3
"""
WukongTTS 演示脚本
展示系统的主要功能
"""

import os
import sys
import time
from pathlib import Path

# 添加当前目录到路径
sys.path.append(str(Path(__file__).parent))

from wukong_tts import WukongTTS


def print_banner():
    """打印欢迎横幅"""
    banner = """
    ╔══════════════════════════════════════════════════════════════╗
    ║                    🐒 孙悟空TTS系统演示 🐒                     ║
    ║                                                              ║
    ║              基于 Qwen2.5 + F5-TTS 技术栈                    ║
    ║                                                              ║
    ║    "俺老孙来也！七十二变，筋斗云，样样精通！"                  ║
    ╚══════════════════════════════════════════════════════════════╝
    """
    print(banner)


def demo_text_generation():
    """演示文本生成功能"""
    print("\n" + "="*60)
    print("📝 文本生成演示")
    print("="*60)
    
    try:
        print("正在初始化孙悟空AI...")
        wukong = WukongTTS(
            base_model_path="qwen/Qwen2.5-7B-Instruct",
            lora_path="./dataset/output"
        )
        
        demo_questions = [
            "你是谁？",
            "你有什么本领？", 
            "师父唐僧怎么样？",
            "你对如来佛祖有什么看法？",
            "你最讨厌什么？"
        ]
        
        print("\n开始对话演示...")
        for i, question in enumerate(demo_questions, 1):
            print(f"\n[问题 {i}] {question}")
            print("🤔 孙悟空思考中...")
            
            start_time = time.time()
            response = wukong.generate_text(question, temperature=0.8)
            end_time = time.time()
            
            print(f"🐒 孙悟空: {response}")
            print(f"⏱️  生成时间: {end_time - start_time:.2f}秒")
            print("-" * 50)
            
            # 短暂暂停
            time.sleep(1)
            
    except Exception as e:
        print(f"❌ 文本生成演示失败: {e}")
        print("请确保已正确训练LoRA模型")


def demo_audio_synthesis():
    """演示音频合成功能"""
    print("\n" + "="*60)
    print("🔊 音频合成演示")
    print("="*60)
    
    ref_audio_path = "./audio/wukong_ref.wav"
    
    if not os.path.exists(ref_audio_path):
        print(f"❌ 参考音频文件不存在: {ref_audio_path}")
        print("请准备孙悟空的参考音频文件")
        print("\n📋 参考音频要求:")
        print("- 格式: WAV")
        print("- 时长: 5-12秒")
        print("- 内容: 清晰的孙悟空配音")
        print("- 路径: ./audio/wukong_ref.wav")
        return
    
    try:
        print("正在初始化完整TTS系统...")
        wukong = WukongTTS(
            base_model_path="qwen/Qwen2.5-7B-Instruct",
            lora_path="./dataset/output",
            ref_audio_path=ref_audio_path
        )
        
        demo_texts = [
            "俺老孙来也！妖怪哪里逃！",
            "师父不用怕，有俺老孙在！",
            "吃俺老孙一棒！",
            "俺老孙有七十二变，筋斗云一个跟头十万八千里！"
        ]
        
        print("\n开始音频合成演示...")
        os.makedirs("./output", exist_ok=True)
        
        for i, text in enumerate(demo_texts, 1):
            print(f"\n[音频 {i}] 合成文本: {text}")
            
            output_path = f"./output/demo_audio_{i}.wav"
            start_time = time.time()
            
            success = wukong.text_to_speech(text, output_path)
            end_time = time.time()
            
            if success:
                print(f"✅ 合成成功: {output_path}")
                print(f"⏱️  合成时间: {end_time - start_time:.2f}秒")
            else:
                print("❌ 合成失败")
            
            print("-" * 50)
            time.sleep(1)
            
    except Exception as e:
        print(f"❌ 音频合成演示失败: {e}")
        print("请检查F5-TTS安装和参考音频配置")


def demo_interactive_chat():
    """演示交互式对话"""
    print("\n" + "="*60)
    print("💬 交互式对话演示")
    print("="*60)
    
    try:
        print("正在初始化交互系统...")
        wukong = WukongTTS(
            base_model_path="qwen/Qwen2.5-7B-Instruct",
            lora_path="./dataset/output",
            ref_audio_path="./audio/wukong_ref.wav"
        )
        
        print("\n🎯 开始互动演示（输入'quit'退出）")
        print("💡 试试问孙悟空一些问题吧！")
        print("-" * 50)
        
        conversation_count = 0
        while conversation_count < 3:  # 限制演示轮数
            try:
                user_input = input(f"\n[轮次 {conversation_count + 1}] 你: ").strip()
                
                if user_input.lower() in ['quit', 'exit', '退出']:
                    break
                
                if not user_input:
                    print("请输入一些内容...")
                    continue
                
                print("🤔 孙悟空思考中...")
                response, audio_path = wukong.chat_with_voice(user_input, "./output")
                
                print(f"🐒 孙悟空: {response}")
                if audio_path:
                    print(f"🔊 音频已保存: {audio_path}")
                
                conversation_count += 1
                
            except KeyboardInterrupt:
                break
        
        print("\n👋 演示结束，感谢体验！")
        
    except Exception as e:
        print(f"❌ 交互演示失败: {e}")


def show_system_info():
    """显示系统信息"""
    print("\n" + "="*60)
    print("ℹ️  系统信息")
    print("="*60)
    
    import torch
    
    # Python环境
    print(f"Python版本: {sys.version.split()[0]}")
    
    # PyTorch信息
    print(f"PyTorch版本: {torch.__version__}")
    print(f"CUDA可用: {'是' if torch.cuda.is_available() else '否'}")
    if torch.cuda.is_available():
        print(f"GPU设备: {torch.cuda.get_device_name()}")
        print(f"显存: {torch.cuda.get_device_properties(0).total_memory / 1024**3:.1f}GB")
    
    # 文件检查
    files_to_check = [
        ("数据集", "./dataset/train/lora/西游记白话文.json"),
        ("LoRA权重", "./dataset/output"),
        ("F5-TTS", "./F5-TTS"),
        ("参考音频", "./audio/wukong_ref.wav"),
    ]
    
    print("\n文件检查:")
    for name, path in files_to_check:
        status = "✅" if os.path.exists(path) else "❌"
        print(f"{status} {name}: {path}")


def main():
    """主演示函数"""
    print_banner()
    
    print("欢迎使用孙悟空TTS系统演示！")
    print("本演示将展示系统的主要功能...")
    
    # 显示系统信息
    show_system_info()
    
    # 演示选项
    demos = [
        ("文本生成", demo_text_generation),
        ("音频合成", demo_audio_synthesis),
        ("交互对话", demo_interactive_chat),
    ]
    
    print("\n" + "="*60)
    print("📋 可用演示:")
    for i, (name, _) in enumerate(demos, 1):
        print(f"{i}. {name}")
    print("0. 全部演示")
    print("q. 退出")
    
    while True:
        try:
            choice = input("\n请选择演示项目 (0-3, q): ").strip().lower()
            
            if choice == 'q':
                print("再见！俺老孙去也！")
                break
            
            elif choice == '0':
                # 全部演示
                for name, demo_func in demos:
                    print(f"\n🚀 开始 {name} 演示...")
                    demo_func()
                    input("\n按Enter继续下一个演示...")
                break
            
            elif choice.isdigit() and 1 <= int(choice) <= len(demos):
                # 单个演示
                idx = int(choice) - 1
                name, demo_func = demos[idx]
                print(f"\n🚀 开始 {name} 演示...")
                demo_func()
                break
            
            else:
                print("请输入有效选项")
                
        except KeyboardInterrupt:
            print("\n再见！俺老孙去也！")
            break
        except Exception as e:
            print(f"出现错误: {e}")
    
    print("\n🎉 演示结束！")
    print("如需了解更多功能，请查看 README.md 或运行:")
    print("- python wukong_tts.py --help")
    print("- python wukong_gradio.py")


if __name__ == "__main__":
    main()
