import argparse, torch, pathlib, os
from datasets import load_dataset #HuggingFace 的数据加载工具，用来快速加载各种 标准或本地 NLP 数据集，同时支持缓存、切分、预处理等功能。
from transformers import (AutoTokenizer, #自动加载与模型匹配的分词器（Tokenizer），自动加载相应的词表、tokenizer config（如 SentencePiece、BPE 等）
                          AutoModelForSeq2SeqLM, #加载 Seq2Seq 架构的模型（如 T5、BART、FLAN-T5）
                          Seq2SeqTrainer,  #数据结构类，用于定义训练参数，如 batch size、lr、epoch 数、输出路径等。
                          Seq2SeqTrainingArguments,  #用于训练 Seq2Seq 模型的“自动训练器”。能：封装了整个训练流程，包括：数据分发（支持多 GPU），自动 loss 计算和反向传播，模型保存与评估，支持使用 generate() 进行预测（如自动翻译或 Text2SQL）
                          DataCollatorForSeq2Seq) 
from peft import LoraConfig, get_peft_model #用Lora进行低成本微调

# ////////////////////参数/////////////////
BASE_DIR = os.path.abspath("model/original/flan-t5-large") # 本地模型路径
JSON_DIR = "data/normalized.jsonl" # 本地数据集路径
MODEL_DIR = "model_dir/output1" # 输出目录
MAX_IN, MAX_OUT = 512, 256         # token 长度

# /////////////////////tokenization//////////////////
def tokenize_fn(batch, tok):
    model_in  = tok(batch["input_text"],
                    truncation=True, max_length=MAX_IN, 
                    padding="max_length")
    model_out = tok(batch["label_text"],
                    truncation=True, max_length=MAX_OUT, 
                    padding="max_length")
    return {
        "input_ids"      : model_in["input_ids"],
        "attention_mask" : model_in["attention_mask"],
        "labels"         : model_out["input_ids"]      
    }

# ///////////////主程序///////////////
def main(model,data,output_dir="model_dir"):
    # tokenizer & dataset
    tok = AutoTokenizer.from_pretrained(model, local_files_only=True)
    ds  = load_dataset("json", data_files=data, split="train")
    ds  = ds.map(tokenize_fn, fn_kwargs={"tok": tok},
                 batched=True, remove_columns=ds.column_names)

    model = AutoModelForSeq2SeqLM.from_pretrained(model, local_files_only=True)

    # 训练参数
    training_args = Seq2SeqTrainingArguments(
        output_dir="out",
        per_device_train_batch_size=4,
        gradient_accumulation_steps=2,
        num_train_epochs=5,
        learning_rate=2e-5,
        predict_with_generate=True,
        fp16=torch.cuda.is_available(),
        logging_steps=50,
        save_steps=500,
        save_total_limit=2
    )

    # Trainer
    trainer = Seq2SeqTrainer(
        model=model,
        args=training_args,
        train_dataset=ds,
        tokenizer=tok,
    )
    trainer.train()

    # 保存 LoRA adapter + tokenizer
    model.save_pretrained(output_dir)
    tok.save_pretrained(output_dir)
    print(f"LoRA adapter & tokenizer saved to {output_dir}")

if __name__ == "__main__":
    model = BASE_DIR
    data  = JSON_DIR
    output_dir= MODEL_DIR
    main(model, data, output_dir)
