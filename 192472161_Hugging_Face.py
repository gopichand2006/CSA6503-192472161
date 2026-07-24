from transformers import pipeline

classifier = pipeline("sentiment-analysis")

text = "This is a very good product."

result = classifier(text)

print("Sentiment:", result)