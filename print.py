from transformers import pipeline
pipe = pipeline(
    "text2text-generation",
    model="model_dir/T5-LM-Large-text2sql" 
)
prompt = ("question: check the name begins with capital 'A' "
          "context: CREATE TABLE Companies(id int, name varchar);")
print(pipe(prompt, max_new_tokens=64)[0]["generated_text"])
