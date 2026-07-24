from transformers import pipeline

qa = pipeline("question-answering")

context = """
Artificial Intelligence is a branch of computer science.
It enables machines to perform tasks that normally require human intelligence.
"""

question = "What is Artificial Intelligence?"

answer = qa(
    question=question,
    context=context
)

print("Answer:", answer["answer"])