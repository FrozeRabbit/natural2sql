from datasets import load_dataset               # 载入数据集，支持 json / csv / parquet 等格式
from pathlib import Path                        # 处理文件路径
from typing import List
from transformers import (
    GPT2LMHeadModel,                             # 基于 GPT-2 的语言模型，用于自回归生成
    GPT2TokenizerFast,                          # 快速版 GPT-2 分词器，进行分词和编码
    TrainingArguments,                          # Trainer 训练参数配置项
    Trainer                                     # HuggingFace 高级训练类，管理断点/结果/日志
)
from peft import LoraConfig, get_peft_model     # 载入 LoRA 设置和将 LoRA 添加到原始模型中#！！！
import torch                                    # 基础 PyTorch 模块

# 基本参数
MODEL_NAME = "gpt2-large" #模型名：可换成 gpt2-medium 或 distilgpt2
DATA_PATH = "data/test.jsonl" #格式为 JSONL，含 prompt 和 sql 两个字段
MAX_LEN_SRC = 640 #最大输入长度，根据 GPU 调整
BATCH_SIZE = 2 #单卡 batch 大小
ACC_STEPS = 4 #梯度累积步数，混合 batch 缓解 GPU 压力，等效 batch=8
LR = 1e-5 #学习率
NUMT = 3 #训练轮数
SST = 500 #保存频率

# 载入数据集并分段
raw_ds = load_dataset(
    "json",                                       # 指定格式：JSON 或 JSONL
    data_files={"train": DATA_PATH},             # 输入文件
    split="train"                                # 只载入 train 分割
)

# 数据格式已经是 {"prompt": ..., "output": ...} 的对
# 下面会把 prompt + output 分辨地分词，并对 prompt 部分打 -100 mask

# =初始化分词器
tok = GPT2TokenizerFast.from_pretrained(MODEL_NAME)
tok.pad_token = tok.eos_token                      # GPT-2 本身无 pad，用 <eos> 代替

# 分词 + 揭挠并打印标签区
def tok_and_mask(batch):
    input_texts = [p + s for p, s in zip(batch["prompt"], batch["output"])]
    enc = tok(input_texts, padding="longest", truncation=True, max_length=MAX_LEN_SRC)

    input_ids = enc["input_ids"]
    attention_mask = enc["attention_mask"]

    labels = []
    for i, (p, ids) in enumerate(zip(batch["prompt"], input_ids)):
        prompt_ids = tok(p, add_special_tokens=False)["input_ids"]
        plen = len(prompt_ids)

        # 防止 plen > len(ids)（因为 truncation 会把 prompt 砍掉）
        if plen > len(ids):
            plen = len(ids)

        lbl = ids.copy()
        lbl[:plen] = [-100] * plen  # -100 不计算 loss
        labels.append(lbl)

    enc["labels"] = labels

    # 再次确保所有 input_ids 都合法
    for i, ids in enumerate(enc["input_ids"]):
        for tokid in ids:
            if tokid >= len(tok) or tokid < 0:
                print(f"❌ 非法 token ID: {tokid}，样本 index: {i}")
                raise ValueError("存在非法 token id，终止训练。")

    return enc

# 进行批量处理
# remove_columns 用来删除 prompt/output 原始文本，节省显存
ds_tok = raw_ds.map(tok_and_mask, batched=True, remove_columns=raw_ds.column_names)

# 加载 GPT-2 模型 + LoRA 配置
base_model = GPT2LMHeadModel.from_pretrained(MODEL_NAME)#！！！

# LoRA 设置：只对注意力层插入 LoRA#！！！
lora_cfg = LoraConfig(
    r=8,
    lora_alpha=16,
    target_modules=["c_attn"],     # GPT2 的注意力层（Wqkv联合在 c_attn 里）
    lora_dropout=0.05,
    bias="none",
    task_type="CAUSAL_LM"
)
model = get_peft_model(base_model, lora_cfg)       # 返回包含 LoRA 的 model#！！！
model.resize_token_embeddings(len(tok))

# 训练参数
args = TrainingArguments(
    output_dir="out", # 训练时保存的目录
    overwrite_output_dir=True,
    num_train_epochs=NUMT,
    per_device_train_batch_size=BATCH_SIZE,
    gradient_accumulation_steps=ACC_STEPS,
    learning_rate=LR,
    fp16=torch.cuda.is_available(), # 如果支持 GPU，则使用 fp16 混合精度
    logging_steps=50, #日志步长
    save_steps=SST,
    save_total_limit=3, #只保留最近的3个模型
    evaluation_strategy="no" # 未分配 eval 数据时，设置为 "no"
)

# Trainer 构造器：自动创建数据加载器
trainer = Trainer(
    model=model,
    args=args,
    train_dataset=ds_tok,
    tokenizer=tok,
)

# 进行训练
trainer.train()

# 保存模型和分词器
Path("model").mkdir(exist_ok=True)
model.save_pretrained("model")
tok.save_pretrained("model")
print("模型和分词器已保存到 ./model")
