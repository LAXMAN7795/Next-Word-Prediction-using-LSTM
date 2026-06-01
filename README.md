# Next Word Prediction using LSTM

## 📌 Project Overview

The **Next Word Prediction using LSTM** project is a Natural Language Processing (NLP) application that predicts the most probable next word based on a sequence of input words provided by the user.

The project utilizes a **Long Short-Term Memory (LSTM)** neural network, a specialized Recurrent Neural Network (RNN) architecture designed to learn long-term dependencies in sequential data. The trained model is deployed through a user-friendly **Streamlit** web application, enabling real-time next-word prediction.

This project demonstrates the practical implementation of deep learning techniques for language modeling and text generation tasks.

---

## 🚀 Features

* Predicts the next word from a given text sequence.
* Interactive web interface using Streamlit.
* Deep Learning model built using TensorFlow and Keras.
* Uses tokenization and sequence padding for text preprocessing.
* Fast and real-time prediction.
* Easy-to-use and lightweight application.

---

## 🛠️ Technologies Used

### Programming Language

* Python

### Libraries & Frameworks

* TensorFlow / Keras
* NumPy
* Streamlit
* Pickle
* NLTK

### Machine Learning

* Long Short-Term Memory (LSTM)
* Natural Language Processing (NLP)

---

## 🧠 How the Model Works

### Step 1: Data Collection

A text corpus is collected and used as the training dataset.

Example:

```text
Artificial Intelligence is transforming the world.
Machine Learning is a subset of Artificial Intelligence.
```

---

### Step 2: Text Preprocessing

The text data undergoes several preprocessing steps:

* Convert text into lowercase
* Remove unnecessary characters
* Tokenization
* Create input sequences
* Sequence padding

Example:

```text
Input Sentence:
Artificial Intelligence is transforming the world

Generated Sequences:
Artificial
Artificial Intelligence
Artificial Intelligence is
Artificial Intelligence is transforming
...
```

---

### Step 3: Tokenization

The tokenizer converts words into numerical representations.

Example:

```python
{
    "artificial": 1,
    "intelligence": 2,
    "is": 3,
    "transforming": 4
}
```

---

### Step 4: Sequence Padding

Sequences are padded to ensure equal length.

Example:

```python
[0, 0, 1, 2, 3]
[0, 1, 2, 3, 4]
```

---

### Step 5: Model Training

The model is trained using an LSTM architecture.

Typical Architecture:

```python
Embedding Layer
↓
LSTM Layer
↓
Dense Layer
↓
Softmax Output Layer
```

The model learns contextual relationships between words and predicts the next most probable word.

---

## 🔍 Prediction Workflow

### User Input

```text
Machine Learning is
```

### Tokenization

```python
[12, 34, 56]
```

### Padding

```python
[0, 0, 12, 34, 56]
```

### LSTM Prediction

```python
Predicted Word Index = 78
```

### Word Mapping

```python
78 → powerful
```

### Final Output

```text
Predicted Next Word: powerful
```

---

## 💻 Streamlit Application Workflow

The application performs the following steps:

1. Load trained LSTM model.
2. Load saved tokenizer.
3. Accept user input.
4. Convert text into sequences.
5. Pad the sequence.
6. Generate prediction.
7. Display predicted next word.

---

## 📜 Code Explanation

### Loading the Trained Model

```python
model = load_model('lstm_model.h5')
```

Loads the pre-trained LSTM model.

---

### Loading the Tokenizer

```python
with open('tokenizer.pickle', 'rb') as handle:
    tokenizer = pickle.load(handle)
```

Loads the tokenizer used during training.

---

### Prediction Function

```python
def predict_next_word(model, tokenizer, text, max_sequence_len):
```

This function:

* Converts text to numerical sequence.
* Applies padding.
* Uses the trained model for prediction.
* Finds the word corresponding to the predicted index.
* Returns the predicted word.

---

### Streamlit User Interface

```python
st.title("Next Word Prediction using LSTM")
```

Creates the application title.

```python
input_text = st.text_input("Input Text")
```

Accepts user input.

```python
if st.button("Predict"):
```

Triggers prediction when clicked.

---

## 📊 Model Architecture

```text
Input Layer
     │
Embedding Layer
     │
LSTM Layer
     │
Dense Layer
     │
Softmax Layer
     │
Predicted Word
```

---

## ⚙️ Installation

### Clone Repository

```bash
git clone https://github.com/yourusername/next-word-prediction.git
```

```bash
cd next-word-prediction
```

---

### Create Virtual Environment

```bash
python -m venv venv
```

Activate:

Windows

```bash
venv\Scripts\activate
```

Linux / Mac

```bash
source venv/bin/activate
```

---

### Install Dependencies

```bash
pip install -r requirements.txt
```

---

## ▶️ Run the Application

```bash
streamlit run app.py
```

Application will open in your browser:

```text
http://localhost:8501
```

---

## 📋 Requirements

```text
streamlit
tensorflow
numpy
nltk
pickle-mixin
```

Install manually:

```bash
pip install streamlit tensorflow numpy nltk pickle-mixin
```

---

## 🎯 Applications

* Text Generation
* Smart Keyboard Suggestions
* Chatbots
* Search Query Completion
* Content Writing Assistance
* Auto-Completion Systems
* AI Writing Tools

---

## 🔮 Future Enhancements

* Top-K predictions
* Beam Search decoding
* Transformer-based architecture
* Multi-language support
* GPT-style text generation
* Better UI/UX design
* Cloud deployment using AWS/Azure/GCP
* REST API integration

---

## 📈 Learning Outcomes

Through this project, the following concepts are explored:

* Natural Language Processing (NLP)
* Tokenization
* Sequence Modeling
* Deep Learning
* LSTM Networks
* TensorFlow/Keras
* Streamlit Deployment
* Model Serialization
* Real-time Prediction Systems

---

## 👨‍💻 Author

**Laxman Sannu Gouda**

Artificial Intelligence & Data Science Graduate

---

## 📄 License

This project is developed for educational and learning purposes. Feel free to use, modify, and enhance it for personal or academic projects.

---

⭐ If you found this project useful, consider giving it a star on GitHub!
