import argparse, json, pathlib, sys
from datasets import load_dataset, disable_caching

disable_caching()                          # 避免写 .cache

def main(origin_js,out_js):
    out_path = pathlib.Path(out_js)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    ds = load_dataset(                     # 自动识别 json / jsonl
        "json", data_files=origin_js, split="train")

    def normalize(sample):
        return {
            "input_text" : sample["prompt"].replace("\n", " "),
            "label_text" : sample["output"]
        }

    ds = ds.map(normalize, remove_columns=ds.column_names)
    ds.to_json(out_path, orient="records", lines=True,
               force_ascii=False)
    print(f"Saved {len(ds):,} samples → {out_path}")

if __name__ == "__main__":
    origin_js = "data/test.jsonl"  # 默认输入
    out_js = "data/train.jsonl"  # 默认输出
    main(origin_js, out_js)
"""
把原始 JSON/JSONL（字段 prompt / output） 规范化：
1. 去掉 prompt 中的换行放入 input_text
2. output 原样放入 label_text
结果每行 JSON 存两列：input_text, label_text
"""