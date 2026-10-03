#!/usr/bin/env python3
"""
WukongTTS - 孙悟空语音合成系统
集成Qwen2.5 LoRA微调模型和F5-TTS语音合成
"""

import os
import sys
import argparse
import torch
import soundfile as sf
from pathlib import Path
from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import PeftModel
import warnings
warnings.filterwarnings("ignore")

# 添加F5-TTS路径
sys.path.append(str(Path(__file__).parent / "F5-TTS" / "src"))

try:
    from f5_tts.api import F5TTS
except ImportError:
    print("警告: 无法导入F5-TTS，请确保已正确安装F5-TTS依赖")
    F5TTS = None


class WukongTTS:
    """孙悟空TTS系统主类"""
    
    def __init__(self, 
                 base_model_path="qwen/Qwen2.5-7B-Instruct",
                 lora_path="./dataset/output",
                 ref_audio_path="./audio/wukong_ref.wav",
                 device=None):
        """
        初始化WukongTTS系统
        
        Args:
            base_model_path: Qwen2.5基础模型路径
            lora_path: LoRA微调权重路径
            ref_audio_path: 孙悟空参考音频路径
            device: 计算设备
        """
        self.base_model_path = base_model_path
        self.lora_path = lora_path
        self.ref_audio_path = ref_audio_path
        
        if device is None:
            self.device = "cuda" if torch.cuda.is_available() else "cpu"
        else:
            self.device = device
            
        print(f"使用设备: {self.device}")
        
        # 初始化组件
        self.tokenizer = None
        self.model = None
        self.tts_model = None
        self.ref_text = "俺老孙来也！妖怪哪里逃！"  # 默认参考文本
        
        self._load_llm_model()
        self._load_tts_model()
    
    def _load_llm_model(self):
        """加载语言模型和LoRA权重"""
        print("正在加载Qwen2.5模型...")
        
        try:
            # 加载tokenizer
            self.tokenizer = AutoTokenizer.from_pretrained(
                self.base_model_path,
                trust_remote_code=True,
                use_fast=False
            )
            
            # 加载基础模型
            self.model = AutoModelForCausalLM.from_pretrained(
                self.base_model_path,
                device_map="auto" if self.device == "cuda" else None,
                torch_dtype=torch.bfloat16 if self.device == "cuda" else torch.float32,
                trust_remote_code=True
            ).eval()
            
            # 加载LoRA权重
            if os.path.exists(self.lora_path):
                print(f"正在加载LoRA权重: {self.lora_path}")
                self.model = PeftModel.from_pretrained(self.model, self.lora_path)
                print("LoRA权重加载成功！")
            else:
                print(f"警告: LoRA权重路径不存在 {self.lora_path}，使用原始模型")
                
        except Exception as e:
            print(f"加载语言模型时出错: {e}")
            raise
    
    def _load_tts_model(self):
        """加载F5-TTS模型"""
        if F5TTS is None:
            print("跳过TTS模型加载，F5-TTS未安装")
            return
            
        try:
            print("正在加载F5-TTS模型...")
            self.tts_model = F5TTS(device=self.device)
            print("F5-TTS模型加载成功！")
        except Exception as e:
            print(f"加载TTS模型时出错: {e}")
            self.tts_model = None
    
    def generate_text(self, prompt, max_length=512, temperature=0.8, top_p=0.9):
        """
        生成孙悟空风格的文本回复
        
        Args:
            prompt: 用户输入
            max_length: 最大生成长度
            temperature: 温度参数
            top_p: top_p采样参数
            
        Returns:
            生成的文本
        """
        if self.model is None or self.tokenizer is None:
            return "模型未正确加载"
        
        # 构建对话消息
        messages = [
            {"role": "system", "content": "你是大名鼎鼎的齐天大圣，花果山水帘洞美猴王孙悟空。说话要嚣张一点，自称俺老孙。"},
            {"role": "user", "content": prompt}
        ]
        
        try:
            # 使用chat template
            text = self.tokenizer.apply_chat_template(
                messages,
                tokenize=False,
                add_generation_prompt=True
            )
            
            # 编码输入
            inputs = self.tokenizer([text], return_tensors="pt")
            if self.device == "cuda":
                inputs = {k: v.cuda() for k, v in inputs.items()}
            
            # 生成回复
            with torch.no_grad():
                outputs = self.model.generate(
                    **inputs,
                    max_new_tokens=max_length,
                    temperature=temperature,
                    top_p=top_p,
                    do_sample=True,
                    pad_token_id=self.tokenizer.eos_token_id
                )
            
            # 解码输出
            response = self.tokenizer.decode(
                outputs[0][inputs['input_ids'].shape[1]:], 
                skip_special_tokens=True
            )
            
            return response.strip()
            
        except Exception as e:
            print(f"文本生成时出错: {e}")
            return "俺老孙今天有些不在状态，改日再聊！"
    
    def text_to_speech(self, text, output_path="output.wav"):
        """
        将文本转换为语音
        
        Args:
            text: 要合成的文本
            output_path: 输出音频文件路径
            
        Returns:
            是否成功
        """
        if self.tts_model is None:
            print("TTS模型未加载，无法进行语音合成")
            return False
            
        if not os.path.exists(self.ref_audio_path):
            print(f"参考音频文件不存在: {self.ref_audio_path}")
            return False
        
        try:
            print(f"正在合成语音: {text}")
            
            # 调用F5-TTS进行语音合成
            wav, sr, spec = self.tts_model.infer(
                ref_file=self.ref_audio_path,
                ref_text=self.ref_text,
                gen_text=text,
                file_wave=output_path,
                seed=None
            )
            
            print(f"语音合成完成，保存到: {output_path}")
            return True
            
        except Exception as e:
            print(f"语音合成时出错: {e}")
            return False
    
    def chat_with_voice(self, prompt, output_dir="./output"):
        """
        完整的对话+语音合成流程
        
        Args:
            prompt: 用户输入
            output_dir: 输出目录
            
        Returns:
            (文本回复, 音频文件路径)
        """
        # 确保输出目录存在
        os.makedirs(output_dir, exist_ok=True)
        
        # 生成文本回复
        print(f"用户: {prompt}")
        response = self.generate_text(prompt)
        print(f"悟空: {response}")
        
        # 生成音频
        audio_path = os.path.join(output_dir, f"wukong_response_{len(prompt)%1000}.wav")
        success = self.text_to_speech(response, audio_path)
        
        if success:
            return response, audio_path
        else:
            return response, None


def main():
    """命令行入口"""
    parser = argparse.ArgumentParser(description="孙悟空TTS系统")
    parser.add_argument("--base_model", default="qwen/Qwen2.5-7B-Instruct", help="基础模型路径")
    parser.add_argument("--lora_path", default="./dataset/output", help="LoRA权重路径")
    parser.add_argument("--ref_audio", default="./audio/wukong_ref.wav", help="参考音频路径")
    parser.add_argument("--output_dir", default="./output", help="输出目录")
    parser.add_argument("--prompt", help="单次对话输入")
    parser.add_argument("--interactive", action="store_true", help="交互式模式")
    
    args = parser.parse_args()
    
    # 初始化系统
    print("正在初始化WukongTTS系统...")
    wukong = WukongTTS(
        base_model_path=args.base_model,
        lora_path=args.lora_path,
        ref_audio_path=args.ref_audio
    )
    
    if args.prompt:
        # 单次对话模式
        response, audio_path = wukong.chat_with_voice(args.prompt, args.output_dir)
        if audio_path:
            print(f"音频已保存到: {audio_path}")
    
    elif args.interactive:
        # 交互式对话模式
        print("\n=== 孙悟空对话系统 ===")
        print("输入'quit'或'exit'退出")
        print("-" * 50)
        
        while True:
            try:
                user_input = input("\n你: ").strip()
                if user_input.lower() in ['quit', 'exit', '退出']:
                    print("再见！俺老孙去也！")
                    break
                
                if not user_input:
                    continue
                
                response, audio_path = wukong.chat_with_voice(user_input, args.output_dir)
                if audio_path:
                    print(f"🔊 音频: {audio_path}")
                    
            except KeyboardInterrupt:
                print("\n再见！俺老孙去也！")
                break
            except Exception as e:
                print(f"出错了: {e}")
    
    else:
        print("请指定 --prompt 或 --interactive 模式")
        print("使用 --help 查看帮助")


if __name__ == "__main__":
    main()
