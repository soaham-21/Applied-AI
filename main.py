def write_to_file(title,content):
    with open("outputs.txt", "a", encoding = "utf-8") as f:
        f.write(f"\n{'='*50}\n")
        f.write(f"{title}\n")
        f.write(f"{'-'*50}\n")
        f.write(str(content) + "\n")

from transformers import pipeline
print("TEXT CLASSIFICATION")
print("Sentiment Analysis")
analyser = pipeline("sentiment-analysis", model="distilbert-base-uncased-finetuned-sst-2-english")
text = "This is the applied AI track and I am happy to analyse the sentiment in this text"
result = analyser(text)

print("Input: ", text)
print("Output: ", result)
print()

result = analyser(text)
write_to_file("Sentiment Analysis", result)

print("Topic Classification")
classifier = pipeline("zero-shot-classification", model="facebook/bart-large-mnli")
text = "Dhurandar part 2 was released recently and it has been getting great reviews from the audience."
labels = ["politics","sports", "entertainment", "education" , "technology"]
print(classifier(text, labels))


zs_result = classifier(text, candidate_labels=labels)
write_to_file("Topic Classification", zs_result)

print("SUMMARIZATION")
summarizer = pipeline("summarization", model="facebook/bart-large-cnn")
long_text = """Artificial Intelligence is transforming the modern world. It allows machines to learn from data,
identify patterns, and make decisions. AI is widely used in healthcare to diagnose diseases,
in finance to detect fraud, in education to personalize learning, and in many other industries
to automate tasks and improve efficiency."""
summary = summarizer(long_text, max_length =30, min_length= 10, do_sample=False)

print("original Text: ", long_text)
print("Summary: ", summary[0]["summary_text"])
print()

write_to_file("Summarization", summary[0]["summary_text"])

print("QUESTION ANSWERING")
qna = pipeline("question-answering", model="distilbert-base-cased-distilled-squad")
context = """Hugging Face is a company that develops tools for natural language processing.
Their Transformers library provides thousands of pre-trained models that developers can use easily."""

question = "What does hugging face develop?"
answer = qna(question=question, context=context)

print("Context: ", context)
print("Question: ", question)
print("Answer: ", answer)
print()

write_to_file("Question Answering", answer["answer"])

