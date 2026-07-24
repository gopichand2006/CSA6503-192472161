import torch
from transformers import BertTokenizer, BertModel

tokenizer = BertTokenizer.from_pretrained("bert-base-uncased")
model = BertModel.from_pretrained("bert-base-uncased")

sentences = [
    "I like learning AI.",
    "I enjoy studying Artificial Intelligence."
]

embeddings = []

for sentence in sentences:
    inputs = tokenizer(sentence, return_tensors="pt")

    with torch.no_grad():
        output = model(**inputs)

    # Mean pooling
    embedding = output.last_hidden_state.mean(dim=1)
    embeddings.append(embedding)

similarity = torch.nn.functional.cosine_similarity(
    embeddings[0],
    embeddings[1]
)

print("Sentence 1:", sentences[0])
print("Sentence 2:", sentences[1])
print("Cosine Similarity:", similarity.item())