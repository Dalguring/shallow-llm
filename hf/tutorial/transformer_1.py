from transformers import pipeline

classifier = pipeline("sentiment-analysis")

results = classifier(["I'm fine really!", "Go away!"])
for result in results:
    print(f"label: {result['label']}, with score: {round(result['score'], 4)}")

