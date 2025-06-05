#!/usr/bin/env python
# coding: utf-8

import os
import json
import sqlite3
from typing import List, Dict

import torch
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM


def evaluate_queries(ddl_file,json_path,db_path,model_path,log_path):

    # 将DDL文件内容加载为字符串
    if not os.path.isfile(ddl_file):
        raise FileNotFoundError(f"DDL file not found: {ddl_file}")

    with open(ddl_file, "r", encoding="utf-8") as f:
        ddl_content = f.read().strip()
        if not ddl_content.endswith(";"):
            ddl_content += ";"
    context_ddl = ddl_content

    # 加载JSON文件中的问题和SQL对
    try:
        with open(json_path, "r", encoding="utf-8") as jf:
            q_list: List[Dict[str, str]] = json.load(jf)
    except json.JSONDecodeError as e:
        raise ValueError(f"Failed to parse JSON at {json_path}: {e}")

    # 加载模型和分词器
    if device is None:
        device = "cuda" if torch.cuda.is_available() else "cpu"

    tokenizer = AutoTokenizer.from_pretrained(model_path, local_files_only=True)
    model = AutoModelForSeq2SeqLM.from_pretrained(model_path, local_files_only=True)
    model.to(device)
    model.eval()

    # 连接到SQLite数据库
    conn = sqlite3.connect(db_path)
    cursor = conn.cursor()

    logs: List[Dict] = []

    # 遍历问题列表，生成SQL并执行
    for entry in q_list:
        question = entry.get("question", "").strip()
        ground_sql = entry.get("sql", "").strip()

        # 模型输入格式化
        input_text = f"Question: {question} Context: {context_ddl}"
        inputs = tokenizer(
            input_text,
            return_tensors="pt",
            padding=True,
            truncation=True,
            max_length=1024,
        ).to(device)

        # 用模型生成SQL语句
        with torch.no_grad():
            generated_ids = model.generate(
                **inputs,
                max_new_tokens=256,
                early_stopping=True,
                num_beams=4
            )
        pred_sql = tokenizer.batch_decode(generated_ids, skip_special_tokens=True)[0].strip()

        # 将正确的sql和预测的sql转换为SQLite可执行的格式
        def execute_sql_safe(sql: str):
            try:
                cursor.execute(sql)
                return cursor.fetchall()
            except Exception:
                return []

        ground_result = execute_sql_safe(ground_sql)
        pred_result = execute_sql_safe(pred_sql)

        # 计算结果相似度
        ground_set = set(ground_result)
        pred_set = set(pred_result)
        union_size = len(ground_set.union(pred_set))
        inter_size = len(ground_set.intersection(pred_set))
        if union_size > 0:
            similarity = inter_size / union_size
        else:
            # 如果两个集合都为空，则认为相似度为1.0
            similarity = 1.0 if not ground_set and not pred_set else 0.0

        # 记录日志
        logs.append({
            "question": question, # 问题文本
            "ground_sql": ground_sql, # 真实SQL
            "pred_sql": pred_sql, # 预测SQL
            "ground_result": ground_result, # 真实SQL执行结果
            "pred_result": pred_result, # 预测SQL执行结果
            "similarity": round(similarity, 4) # 结果相似度
        })

    # 保存日志到json文件
    with open(log_path, "w", encoding="utf-8") as lf:
        json.dump(logs, lf, ensure_ascii=False, indent=2)

    conn.close()
    print(f"Evaluation complete. Logs saved to '{log_path}'.")


# 数据库DDL和JSON文件路径
if __name__ == "__main__":
    ddl_file    = "test/DDL/ddl"            # DDL文件路径
    json_file   = "test/json/test.json"     # JSON文件路径，包含问题和SQL对
    sqlite_db   = "test/database/bioinfo.db"  # SQLite数据库路径
    model_dir   = "model_dir/output1"       # 模型目录，包含分词器和模型权重
    log_file    = "query_logs.json" # 日志文件路径

    evaluate_queries(
        ddl_file=ddl_file,
        json_path=json_file,
        db_path=sqlite_db,
        model_path=model_dir,
        log_path=log_file
    )
