# Applied AI Track – Assignment 1

## Hugging Face Transformers

## Objective

The objective of this assignment is to get hands-on experience with modern NLP tools using the **Hugging Face Transformers** library and perform multiple NLP tasks using pre-trained models.

The tasks implemented in this project are:

* Sentiment Analysis (Text Classification)
* Zero-Shot Topic Classification
* Text Summarization
* Question Answering

---

## Approach

This project uses the `pipeline` API from the Transformers library, which allows easy access to powerful pre-trained NLP models without training them from scratch.

Each NLP task is implemented using a dedicated pipeline and model:

| Task                 | Pipeline                   | Model Used                                        | Purpose                               |
| -------------------- | -------------------------- | ------------------------------------------------- | ------------------------------------- |
| Sentiment Analysis   | `sentiment-analysis`       | `distilbert-base-uncased-finetuned-sst-2-english` | Classify text as positive/negative    |
| Topic Classification | `zero-shot-classification` | `facebook/bart-large-mnli`                        | Classify text into custom labels      |
| Summarization        | `summarization`            | `facebook/bart-large-cnn`                         | Generate a short summary of long text |
| Question Answering   | `question-answering`       | `distilbert-base-cased-distilled-squad`           | Answer questions from context         |

A helper function is used to store outputs of each task into an `outputs.txt` file for record-keeping and submission.

---

## How to Set Up and Run

### 1️. Create Virtual Environment (Python 3.11 recommended)

```bash
py -3.11 -m venv venv
venv\Scripts\activate
```

### 2️. Install Dependencies

```bash
pip install -r requirements.txt
```

### 3️. Run the program (use py or python keyword according to your system)

```bash
py main.py
```

After running, an `outputs.txt` file will be generated containing results of all tasks.

---

## Difficulties Faced

1. **Pipeline task errors** due to incompatibility with the latest Transformers version.
2. Confusion between different output formats of pipelines (e.g., `summary_text`, `answer`, `generated_text`).
3. Handling structured outputs and storing them properly into a file.

---

## Resolutions

* Used a compatible Python version (3.11) with a virtual environment.
* Pinned Transformers to a stable version (`4.40.2`) for compatibility with standard pipeline tasks.
* Carefully examined the output structure of each pipeline to extract the correct fields.
* Created a reusable function to store outputs cleanly into a text file.

---

## Learnings

* How to use Hugging Face pipelines for real NLP tasks.
* Understanding differences between various NLP models and their purposes.
* Importance of Python and library version compatibility in ML projects.
* Working with model outputs (lists, dictionaries) and handling them correctly.

---

## Project Structure

```
├── main.py
├── outputs.txt
└── README.md
```

---

## Conclusion

This assignment demonstrates how powerful pre-trained NLP models from Hugging Face can be used to perform complex tasks like classification, summarization, and question answering with minimal code, while also highlighting practical challenges related to environment setup and compatibility.