#!/usr/bin/env python3
"""
WukongTTS Gradio Web界面
"""

import os
import sys
import gradio as gr
import torch
from pathlib import Path

# 添加当前目录到路径
sys.path.append(str(Path(__file__).parent))

from wukong_tts import WukongTTS

# 全局变量
wukong_system = None


def initialize_system():
    """初始化WukongTTS系统"""
    global wukong_system
    
    if wukong_system is None:
        try:
            print("正在初始化WukongTTS系统...")
            wukong_system = WukongTTS(
                base_model_path="qwen/Qwen2.5-7B-Instruct",
                lora_path="./dataset/output",
                ref_audio_path="./audio/wukong_ref.wav"
            )
            print("系统初始化完成！")
            return True
        except Exception as e:
            print(f"系统初始化失败: {e}")
            return False
    return True


def chat_with_wukong(message, history):
    """
    与孙悟空对话
    
    Args:
        message: 用户输入
        history: 对话历史
        
    Returns:
        更新后的历史记录
    """
    if not message.strip():
        return history
    
    # 初始化系统
    if not initialize_system():
        history.append([message, "系统初始化失败，请检查模型配置"])
        return history
    
    try:
        # 生成回复
        response = wukong_system.generate_text(message)
        history.append([message, response])
        
    except Exception as e:
        history.append([message, f"出错了: {str(e)}"])
    
    return history


def generate_audio(text):
    """
    生成语音
    
    Args:
        text: 要合成的文本
        
    Returns:
        音频文件路径或错误信息
    """
    if not text.strip():
        return None, "请输入要合成的文本"
    
    if not initialize_system():
        return None, "系统未初始化"
    
    try:
        # 确保输出目录存在
        output_dir = "./output"
        os.makedirs(output_dir, exist_ok=True)
        
        # 生成音频文件名
        import time
        audio_filename = f"wukong_audio_{int(time.time())}.wav"
        audio_path = os.path.join(output_dir, audio_filename)
        
        # 合成语音
        success = wukong_system.text_to_speech(text, audio_path)
        
        if success and os.path.exists(audio_path):
            return audio_path, "语音合成成功！"
        else:
            return None, "语音合成失败，请检查TTS模型配置"
            
    except Exception as e:
        return None, f"语音合成出错: {str(e)}"


def chat_with_voice(message, history):
    """
    对话+语音合成
    
    Args:
        message: 用户输入
        history: 对话历史
        
    Returns:
        (更新的历史, 音频文件, 状态信息)
    """
    if not message.strip():
        return history, None, "请输入消息"
    
    if not initialize_system():
        return history + [[message, "系统初始化失败"]], None, "系统初始化失败"
    
    try:
        # 生成文本回复
        response = wukong_system.generate_text(message)
        history.append([message, response])
        
        # 生成语音
        output_dir = "./output"
        os.makedirs(output_dir, exist_ok=True)
        
        import time
        audio_filename = f"wukong_voice_{int(time.time())}.wav"
        audio_path = os.path.join(output_dir, audio_filename)
        
        success = wukong_system.text_to_speech(response, audio_path)
        
        if success and os.path.exists(audio_path):
            return history, audio_path, "对话和语音生成成功！"
        else:
            return history, None, "文本生成成功，但语音合成失败"
            
    except Exception as e:
        history.append([message, f"出错了: {str(e)}"])
        return history, None, f"出错: {str(e)}"


def get_system_status():
    """获取系统状态"""
    status = []
    
    # 检查数据集
    if os.path.exists("./dataset/train/lora/西游记白话文.json"):
        status.append("✅ 数据集已准备")
    else:
        status.append("❌ 数据集缺失")
    
    # 检查LoRA权重
    if os.path.exists("./dataset/output"):
        status.append("✅ LoRA权重目录存在")
    else:
        status.append("❌ LoRA权重未训练")
    
    # 检查F5-TTS
    if os.path.exists("./F5-TTS"):
        status.append("✅ F5-TTS已安装")
    else:
        status.append("❌ F5-TTS未安装")
    
    # 检查参考音频
    if os.path.exists("./audio/wukong_ref.wav"):
        status.append("✅ 参考音频已准备")
    else:
        status.append("❌ 参考音频缺失")
    
    # 检查GPU
    if torch.cuda.is_available():
        status.append(f"✅ GPU可用: {torch.cuda.get_device_name()}")
    else:
        status.append("⚠️ 仅CPU模式")
    
    return "\n".join(status)


def create_interface():
    """创建Gradio界面"""
    
    with gr.Blocks(title="孙悟空TTS系统", theme=gr.themes.Soft()) as demo:
        gr.Markdown("""
        # 🐒 孙悟空TTS系统
        
        基于Qwen2.5 LoRA微调 + F5-TTS的孙悟空语音合成系统
        
        > *俺老孙来也！有什么事尽管说，俺老孙都能帮你解决！*
        """)
        
        with gr.Tab("💬 文本对话"):
            with gr.Row():
                with gr.Column(scale=3):
                    chatbot = gr.Chatbot(
                        label="与孙悟空对话",
                        height=400,
                        placeholder="点击下方输入框开始对话..."
                    )
                    msg = gr.Textbox(
                        label="输入消息",
                        placeholder="输入你想对孙悟空说的话...",
                        lines=2
                    )
                    with gr.Row():
                        send_btn = gr.Button("发送", variant="primary")
                        clear_btn = gr.Button("清空对话")
                
                with gr.Column(scale=1):
                    gr.Markdown("### 💡 使用提示")
                    gr.Markdown("""
                    - 可以问孙悟空关于西游记的问题
                    - 尝试问他的本领和经历
                    - 模型会以孙悟空的语气回复
                    """)
            
            # 绑定事件
            send_btn.click(
                chat_with_wukong,
                inputs=[msg, chatbot],
                outputs=[chatbot]
            ).then(lambda: "", outputs=msg)
            
            msg.submit(
                chat_with_wukong,
                inputs=[msg, chatbot],
                outputs=[chatbot]
            ).then(lambda: "", outputs=msg)
            
            clear_btn.click(lambda: [], outputs=[chatbot])
        
        with gr.Tab("🔊 语音合成"):
            with gr.Row():
                with gr.Column():
                    tts_input = gr.Textbox(
                        label="输入文本",
                        placeholder="输入要合成的文本...",
                        lines=3,
                        value="俺老孙来也！妖怪哪里逃！"
                    )
                    tts_btn = gr.Button("合成语音", variant="primary")
                    
                with gr.Column():
                    audio_output = gr.Audio(
                        label="合成的语音",
                        type="filepath"
                    )
                    tts_status = gr.Textbox(
                        label="状态",
                        interactive=False
                    )
            
            tts_btn.click(
                generate_audio,
                inputs=[tts_input],
                outputs=[audio_output, tts_status]
            )
        
        with gr.Tab("🎙️ 语音对话"):
            with gr.Row():
                with gr.Column(scale=2):
                    voice_chatbot = gr.Chatbot(
                        label="语音对话",
                        height=300
                    )
                    voice_msg = gr.Textbox(
                        label="输入消息",
                        placeholder="输入消息，将生成文本回复和语音...",
                        lines=2
                    )
                    voice_btn = gr.Button("发送并生成语音", variant="primary")
                    
                with gr.Column(scale=1):
                    voice_audio = gr.Audio(
                        label="孙悟空的语音回复",
                        type="filepath"
                    )
                    voice_status = gr.Textbox(
                        label="状态",
                        interactive=False
                    )
            
            voice_btn.click(
                chat_with_voice,
                inputs=[voice_msg, voice_chatbot],
                outputs=[voice_chatbot, voice_audio, voice_status]
            ).then(lambda: "", outputs=voice_msg)
        
        with gr.Tab("⚙️ 系统状态"):
            status_text = gr.Textbox(
                label="系统状态",
                value=get_system_status(),
                lines=10,
                interactive=False
            )
            refresh_btn = gr.Button("刷新状态")
            
            refresh_btn.click(
                get_system_status,
                outputs=[status_text]
            )
            
            gr.Markdown("""
            ### 📋 使用说明
            
            1. **准备工作**：
               - 确保已训练LoRA模型（运行 `python train.py`）
               - 准备孙悟空参考音频文件到 `./audio/wukong_ref.wav`
               - 安装F5-TTS依赖
            
            2. **文本对话**：直接与孙悟空进行文本对话
            
            3. **语音合成**：输入任意文本，生成孙悟空语音
            
            4. **语音对话**：完整的对话+语音合成流程
            
            ### 🔧 故障排除
            
            - 如果文本生成失败，检查LoRA模型是否正确训练
            - 如果语音合成失败，检查参考音频文件和F5-TTS配置
            - 建议使用GPU以获得更好的性能
            """)
    
    return demo


def main():
    """启动Gradio应用"""
    print("启动孙悟空TTS系统Web界面...")
    
    # 检查基本环境
    print("检查系统环境...")
    print(get_system_status())
    
    # 创建界面
    demo = create_interface()
    
    # 启动应用
    demo.launch(
        server_name="0.0.0.0",
        server_port=7860,
        share=False,
        inbrowser=True
    )


if __name__ == "__main__":
    main()
