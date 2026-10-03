from datasets import Dataset
import pandas as pd
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM, DataCollatorForSeq2Seq, TrainingArguments, Trainer, GenerationConfig
from peft import LoraConfig, TaskType, get_peft_model


def process_func(example):
    MAX_LENGTH = 512    # Qwen分词器对中文支持较好，适当增加长度
    input_ids, attention_mask, labels = [], [], []
    
    # 使用Qwen2.5的chat template格式
    messages = [
        {"role": "system", "content": "你是大名鼎鼎的齐天大圣，花果山水帘洞美猴王孙悟空。说话要嚣张一点，自称俺老孙。"},
        {"role": "user", "content": example['instruction'] + example['input']},
        {"role": "assistant", "content": example['output']}
    ]
    
    # 构建完整的对话文本
    text = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=False)
    
    # 分别编码instruction和response部分
    instruction_text = tokenizer.apply_chat_template(
        messages[:-1], tokenize=False, add_generation_prompt=True
    )
    
    instruction = tokenizer(instruction_text, add_special_tokens=False)
    response = tokenizer(example['output'] + tokenizer.eos_token, add_special_tokens=False)
    
    input_ids = instruction["input_ids"] + response["input_ids"]
    attention_mask = instruction["attention_mask"] + response["attention_mask"]
    labels = [-100] * len(instruction["input_ids"]) + response["input_ids"]
    
    if len(input_ids) > MAX_LENGTH:  # 做截断
        input_ids = input_ids[:MAX_LENGTH]
        attention_mask = attention_mask[:MAX_LENGTH]
        labels = labels[:MAX_LENGTH]
    return {
        "input_ids": input_ids,
        "attention_mask": attention_mask,
        "labels": labels
    }

if __name__ == "__main__":
    # 使用Qwen2.5-7B-Instruct模型
    model_path = 'qwen/Qwen2.5-7B-Instruct'  # 可以修改为本地下载的模型路径
    
    model = AutoModelForCausalLM.from_pretrained(model_path, device_map="auto", torch_dtype=torch.bfloat16, trust_remote_code=True)
    model.enable_input_require_grads() # 开启梯度检查点时，要执行该方法
    tokenizer = AutoTokenizer.from_pretrained(model_path, use_fast=False, trust_remote_code=True)
    
    # Qwen2.5通常已经有pad_token，但以防万一
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    # 加载孙悟空数据集
    df = pd.read_json('./dataset/train/lora/西游记白话文.json')
    ds = Dataset.from_pandas(df)
    tokenized_id = ds.map(process_func, remove_columns=ds.column_names)

    # 适用于Qwen2.5的LoRA配置
    config = LoraConfig(
        task_type=TaskType.CAUSAL_LM, 
        target_modules=["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"],
        inference_mode=False, # 训练模式
        r=16, # 对于Qwen2.5，可以使用稍大的秩
        lora_alpha=32, # Lora alpha参数
        lora_dropout=0.1# Dropout 比例
    )
    model = get_peft_model(model, config)
    model.print_trainable_parameters() # 打印总训练参数

    args = TrainingArguments(
        output_dir="./dataset/output",  # 输出到dataset目录下
        per_device_train_batch_size=2,  # 减小batch size避免显存不足
        gradient_accumulation_steps=8,  # 增加梯度累积步数保持有效batch size
        logging_steps=10,
        num_train_epochs=3,
        save_steps=100,
        learning_rate=5e-5,  # Qwen2.5适合稍小的学习率
        save_on_each_node=True,
        gradient_checkpointing=True,
        warmup_ratio=0.1,  # 添加warmup
        logging_dir="./dataset/output/logs",
        save_total_limit=3,  # 只保留最近3个checkpoint
        load_best_model_at_end=True,
        metric_for_best_model="loss",
        greater_is_better=False,
        report_to=None,  # 不使用wandb等
    )
    trainer = Trainer(
        model=model,
        args=args,
        train_dataset=tokenized_id,
        data_collator=DataCollatorForSeq2Seq(tokenizer=tokenizer, padding=True),
    )
    trainer.train() # 开始训练 
    # 在训练参数中设置了自动保存策略此处并不需要手动保存。

















