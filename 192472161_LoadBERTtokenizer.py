from transformers import BertTokenizer

tokenizer = BertTokenizer.from_pretrained("bert-base-uncased")

sentence = "I am studying Artificial Intelligence."

tokens = tokenizer.tokenize(sentence)

print("Sentence:", sentence)
print("Tokens:", tokens)