# Chat-Wukong

**Chat-Wukong** 是利用《西游记》白话文中所有关于孙悟空的台词和语句，基于 **Qwen2.5** 大语言模型进行 **LoRA微调** 得到的模仿孙悟空语气的聊天语言模型。

> 孙悟空，又名美猴王、齐天大圣、斗战胜佛。由开天辟地产生的仙石孕育而生，拜菩提祖师为师学得七十二变、筋斗云等神通，后大闹天宫，被如来佛祖压在五行山下。经观音菩萨点化，保护唐僧西天取经，一路降妖除魔，历经九九八十一难，最终取得真经，修成正果。

Chat-Wukong 项目旨在提供一个完整的流程，让用户可以基于任意小说或剧本中的特定角色，利用大语言模型微调和先进的文本到语音（TTS）技术，创建一个高度个性化、智能且声音逼真的 AI 角色。

> *本项目灵感来源于 Chat-嬛嬛 (Chat-HuanHuan) 项目。*

---

## Step 1: 环境准备

本文基础环境如下：
----------------
ubuntu 22.04
python 3.12
cuda 12.6
pytorch 2.3.0
transformers >= 4.45.0
accelerate == 0.32.1
peft == 0.11.1
datasets == 2.20.0
sentencepiece (版本根据模型需要)
----------------
```
> 本文默认学习者已安装好以上 Pytorch(cuda) 环境，如未安装请自行安装。

首先 `pip` 换源加速下载并安装依赖包

```shell
# 升级pip
python -m pip install --upgrade pip
# 更换 pypi 源加速库的安装
pip config set global.index-url https://pypi.tuna.tsinghua.edu.cn/simple

# 安装基础依赖
pip install modelscope==1.16.1
pip install transformers>=4.45.0
pip install accelerate==0.32.1
pip install peft==0.11.1
pip install datasets==2.20.0

# 安装 Qwen 模型所需依赖 (根据实际模型调整)
pip install tiktoken
pip install einops

# 安装 F5-TTS 相关依赖 (根据实际 F5-TTS 实现调整)
# pip install torchaudio # 通常 PyTorch 已包含
# pip install phonemizer # 如果需要音素化
# pip install vocos # 如果使用 Vocos 作为声码器
# pip install datasets # 已安装
# pip install accelerate # 已安装
# pip install tqdm # 通常已包含
# pip install tensorboard # 如果需要日志
```

---

## Step 2: 数据准备

首先，我们需要准备《西游记》中孙悟空的对话数据。

1.  **原始数据**:《西游记》的白话文版本文本。例如：
    ```
    原来孙悟空走了以后，有一个混世魔王独占了水帘洞，并且抢走了许多猴子猴孙。孙悟空听到这些以后，气得咬牙跺脚。他问清了混世魔王的住处，决定找混世魔王报仇，便驾着筋斗云，朝北方飞去。

    不一会儿，孙悟空就来到混世魔王的水脏洞前，对门前的小妖喊到∶“你家那个狗屁魔王，多次欺负我们猴子。我今天来，要和那魔王比比高低！

    ”小妖跑进洞里，报告魔王。魔王急忙穿上铁甲，提着大刀，在小妖们的簇拥下走出洞门。

    孙悟空赤手空拳，夺过了混世魔王的大刀，把他劈成了两半。然后，拔下一把毫毛咬碎喷了出去，毫毛变成许多小猴子，直杀进洞里，把所有的妖精全杀死，然后救出被抢走的小猴子，放了一把火烧了水脏洞。
    ```

2.  **提取对话**: 由于原始文本是叙述性文本，需要从中提取出孙悟空的对话。可以使用正则表达式匹配 `"孙悟空.*[：“]([^"”]+)["”]` 或类似模式，或者借助大模型（如 Qwen 自身）来提取。这里推荐使用一个小工具：[extract-dialogue](https://github.com/KMnO4-zx/extract-dialogue.git) 来辅助提取。

    提取后的中间格式可能如下：
    ```
    [
        {"role": "孙悟空", "content": "你家那个狗屁魔王，多次欺负我们猴子。我今天来，要和那魔王比比高低！"},
        {"role": "孙悟空", "content": "俺老孙自有道理！"},
        {"role": "孙悟空", "content": "妖怪！哪里逃！"},
        ...
    ]
    ```

3.  **构建 QA 对**: 将提取出的孙悟空对话整理成指令跟随（Instruction Following）的格式。由于原始数据可能缺乏明确的问答结构，可以采用以下策略：
    *   **自问自答**: 将一句孙悟空的话作为 `output`，为其构造一个合理的 `instruction`（例如，描述场景或提出问题）。
    *   **上下文构造**: 如果有连续的孙悟空发言，可以将前面的发言作为 `input`，当前发言作为 `output`。
    *   **借助大模型**: 使用少量示例作为 prompt，让大模型（如另一个 Qwen 模型）帮助生成符合风格的 `instruction` 和 `input`。

    最终整理成 `json` 格式的数据，如下：

    ```json
    [
        {
            "instruction": "有妖怪霸占了你的水帘洞，还抓走了你的猴子猴孙，你该如何应对？",
            "input": "",
            "output": "你家那个狗屁魔王，多次欺负我们猴子。我今天来，要和那魔王比比高低！"
        },
        {
            "instruction": "别人问你有什么本领时，你会如何回答？",
            "input": "",
            "output": "俺老孙有七十二变，筋斗云一个跟头十万八千里！"
        },
        {
            "instruction": "遇到妖怪时，你通常会说什么？",
            "input": "",
            "output": "妖怪！哪里逃！吃俺老孙一棒！"
        }
    ]
    ```

> 西游记白话文原始数据：[*西游记*](https://github.com/KMnO4-zx/huanhuan-chat/blob/master/dataset/input/wukong/%E8%A5%BF%E6%B8%B8%E8%AE%B0%E7%99%BD%E8%AF%9D%E6%96%87.txt)
> （请将提取并整理好的孙悟空 QA 数据集保存为类似 `wukong_qa.json` 的文件）

所以，在这一步处理数据的大致思路就是：

***1. 从原始文本中提取出目标角色（孙悟空）的对话 &emsp;2. 将对话转换成 QA 指令跟随格式***

> *这一步可以增加数据增强的环节，比如利用生成式模型根据已有数据生成更多风格一致的对话。*

---

## Step 3: 模型训练 (使用 Qwen2.5)

这一步我们将使用 **Qwen2.5** 系列模型进行 LoRA 微调。

1.  **下载模型**:
    创建一个 `model_download.py` 文件，输入以下内容来下载 Qwen2.5 模型（请根据需要选择具体版本，如 `Qwen2.5-7B-Instruct`）：

    ```python
    import torch
    from modelscope import snapshot_download, AutoModel, AutoTokenizer
    import os

    # 替换为你想要的 Qwen2.5 模型 ID 和缓存路径
    model_id = 'qwen/Qwen2.5-7B-Instruct'
    cache_dir = '/root/autodl-tmp' # 请修改为你的模型下载路径

    model_dir = snapshot_download(model_id, cache_dir=cache_dir, revision='master')
    print(f"Model downloaded to: {model_dir}")
    ```

    在命令行运行：
    ```shell
    python model_download.py
    ```

2.  **准备训练代码**:
    参考 `self-llm` 项目中针对 Qwen 或 Qwen2 的 LoRA 微调教程。你需要准备一个 `train.py` 脚本，主要配置包括：
    *   加载你下载的 Qwen2.5 模型。
    *   加载你准备好的 `wukong_qa.json` 数据集。
    *   配置 LoRA 参数（如 `r`, `alpha`, `dropout` 等）。
    *   设置训练参数（如 `epochs`, `learning_rate`, `per_device_train_batch_size` 等）。
    *   设置输出路径（如 `output_dir`）。

    > *可以参考 `self-llm` 中 Qwen2 或类似模型的 Lora 微调教程进行修改。*

    在命令行运行以下指令进行训练：
    ```shell
    python train.py
    ```

    > *注意：记得在 `train.py` 中正确设置模型路径、数据集路径和输出路径。*

    训练时间根据数据量和硬件性能而定，训练完成后会在指定的 `output_dir` 下生成 LoRA 权重。

3.  **测试模型**:
    训练完成后，可以编写一个简单的脚本 `test_wukong.py` 来测试微调后的模型：

    ```python
    from transformers import AutoModelForCausalLM, AutoTokenizer
    import torch
    from peft import PeftModel
    import os

    # 修改为你的实际路径
    base_model_path = '/root/autodl-tmp/qwen/Qwen2.5-7B-Instruct' # Qwen2.5 基础模型路径
    lora_path = './output/qwen2.5_wukong_lora/checkpoint-xxx' # 修改为你的 LoRA 输出路径和 checkpoint

    # 加载 tokenizer
    tokenizer = AutoTokenizer.from_pretrained(base_model_path, trust_remote_code=True)

    # 加载基础模型
    model = AutoModelForCausalLM.from_pretrained(
        base_model_path,
        device_map="auto",
        torch_dtype=torch.bfloat16, # 或 float16, 根据显存选择
        trust_remote_code=True
    ).eval()

    # 加载 LoRA 权重
    model = PeftModel.from_pretrained(model, model_id=lora_path)

    # 构造测试 prompt
    prompt = "你是谁？"

    messages = [
        {"role": "system", "content": "你是大名鼎鼎的齐天大圣，花果山水帘洞美猴王孙悟空。说话要嚣张一点，自称俺老孙。"}, # 设定角色
        {"role": "user", "content": prompt}
    ]

    # 使用 Qwen2.5 的 chat template
    text = tokenizer.apply_chat_template(
        messages,
        tokenize=False,
        add_generation_prompt=True
    )

    model_inputs = tokenizer([text], return_tensors="pt").to(model.device)

    generated_ids = model.generate(
        model_inputs.input_ids,
        max_new_tokens=512,
        temperature=0.7, # 可调整
        top_p=0.9,       # 可调整
        do_sample=True   # 启用采样
    )

    generated_ids = [
        output_ids[len(input_ids):] for input_ids, output_ids in zip(model_inputs.input_ids, generated_ids)
    ]

    response = tokenizer.batch_decode(generated_ids, skip_special_tokens=True)[0]

    print('User: ', prompt)
    print('Wukong: ', response)

    ```
    运行测试脚本：
    ```shell
    python test_wukong.py
    ```

---

## Step 4: 集成 F5-TTS 语音合成

为了让 Chat-Wukong 能"说话"，我们需要将文本回复通过 TTS 模型转换为语音。这里我们选择 **F5-TTS**。

1.  **获取 F5-TTS**: 克隆 F5-TTS 仓库：
    ```shell
    git clone https://github.com/SWivid/F5-TTS.git
    cd F5-TTS
    pip install -e .
    cd ..
    ```

2.  **准备参考音频**: 你需要一段高质量的孙悟空声音参考音频。
    - 格式：WAV文件，16kHz或22kHz采样率
    - 时长：5-12秒
    - 内容：清晰的孙悟空配音（动画、游戏等）
    - 质量：无背景音乐，语音清晰
    - 保存路径：`./audio/wukong_ref.wav`

3.  **使用完整的WukongTTS系统**: 我们已经创建了完整的集成系统。

---

## Step 5: 快速开始

### 🚀 一键设置

```shell
# 1. 克隆并设置环境
chmod +x setup.sh
./setup.sh

# 2. 准备参考音频（手动）
# 将孙悟空音频文件复制到 audio/wukong_ref.wav

# 3. 训练模型
python train.py

# 4. 测试系统
python test_wukong.py
```

### 💬 使用方式

**命令行交互模式：**
```shell
python wukong_tts.py --interactive
```

**Web界面模式：**
```shell
python wukong_gradio.py
```

**单次对话：**
```shell
python wukong_tts.py --prompt "你是谁？"
```

### 🎯 功能特性

- ✅ **智能对话**: 基于Qwen2.5的孙悟空角色扮演
- ✅ **语音合成**: F5-TTS生成孙悟空语音
- ✅ **Web界面**: 友好的Gradio交互界面
- ✅ **命令行工具**: 灵活的CLI接口
- ✅ **批量处理**: 支持多轮对话和音频生成

### 📁 项目结构

```
WukongChat TTS/
├── wukong_tts.py          # 主要TTS系统
├── wukong_gradio.py       # Web界面
├── test_wukong.py         # 测试脚本
├── train.py               # LoRA训练脚本
├── setup.sh               # 环境设置脚本
├── requirements.txt       # 依赖列表
├── dataset/               # 数据集目录
│   ├── input/wukong/      # 原始文本数据
│   ├── train/lora/        # 训练数据
│   └── output/            # 模型输出
├── F5-TTS/               # F5-TTS模型
├── audio/                # 参考音频
│   └── wukong_ref.wav    # 孙悟空参考音频
└── output/               # 生成的音频文件
```

---

## Step 6: 高级用法

### 🎛️ 参数调优

**文本生成参数：**
```python
response = wukong.generate_text(
    prompt="你好",
    max_length=512,      # 最大生成长度
    temperature=0.8,     # 创造性（0.1-1.0）
    top_p=0.9           # 多样性控制
)
```

**语音合成参数：**
- 参考音频质量直接影响合成效果
- 建议使用清晰、无噪音的孙悟空配音
- 参考文本应与音频内容匹配

### 🔧 故障排除

**常见问题：**

1. **模型加载失败**
   - 检查网络连接（首次会下载Qwen2.5模型）
   - 确保有足够的磁盘空间和显存

2. **LoRA权重未找到**
   - 先运行 `python train.py` 训练模型
   - 检查 `./dataset/output/` 目录

3. **语音合成失败**
   - 确保参考音频文件存在
   - 检查F5-TTS安装是否正确
   - 验证音频格式（推荐WAV格式）

4. **显存不足**
   - 调整训练参数中的batch_size
   - 使用梯度检查点
   - 考虑使用CPU模式

### 📊 性能优化

**GPU推荐配置：**
- 显存 >= 8GB（推荐16GB+）
- CUDA 11.8+
- PyTorch 2.0+

**CPU模式：**
- 内存 >= 16GB
- 生成速度较慢但可用

---

## 🎉 完成！

现在你拥有了一个完整的孙悟空TTS系统！

**快速体验：**
```shell
# 启动Web界面
python wukong_gradio.py

# 或使用命令行
python wukong_tts.py --interactive
```

访问 `http://localhost:7860` 开始与孙悟空对话！