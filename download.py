from transformers import AutoTokenizer, AutoModelForSeq2SeqLM

AutoTokenizer.from_pretrained("google/flan-t5-large").save_pretrained("model/original/flan-t5-large")
AutoModelForSeq2SeqLM.from_pretrained("google/flan-t5-large").save_pretrained("model/original/flan-t5-large")
