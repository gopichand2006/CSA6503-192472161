from transformers import pipeline

generator = pipeline("text-generation", model="gpt2")

prompt = "Once upon a time, there was a young student who"

result = generator(
    prompt,
    max_length=80,
    num_return_sequences=1
)

print("Generated Text:")
print(result[0]["generated_text"])