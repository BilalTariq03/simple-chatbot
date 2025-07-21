# AI Chatbot using TensorFlow & NLTK

A simple AI chatbot built with **Python**, **TensorFlow**, and **Natural Language Toolkit (NLTK)**. It uses a **feed-forward neural network** to classify user input into predefined intents, and replies with contextually appropriate responses.

---

## Features

- Tokenization and lemmatization of input
- Bag-of-Words (BoW) vectorization
- Feedforward neural network using Keras
- Custom intent classification with confidence filtering
- CLI-based conversation interface
- Easily extendable with new intents

---

## Requirements

- Python 3.10+
- TensorFlow 2.x
- NLTK
- NumPy
- Pickle (standard library)

Install dependencies:

```bash
pip install tensorflow nltk numpy
```
Download necessary NLTK resources:

import nltk

nltk.download('punkt')

nltk.download('wordnet')

## Training the model
Run training script
```bash
python training.py
```

This will
- Read and preprocess the intents
- Convert patterns into training data using BoW
- Train a model using a simple dense neural network
- Save the trained model and vocabulary files

## Running the chatbot
```bash
python chat.py
```

Then you can type your message in the terminal

## Sample Intents(intents.json)
```json
{
  "tag": "greetings",
  "patterns": ["hello", "hi", "how are you"],
  "responses": ["Hi there!", "Hello!", "How can I help you?"]
}
```
you can add more intents similarly with your own patterns and responses

## Model Architecture
- Dense(128) → ReLU
- Dropout(0.5)
- Dense(64) → ReLU
- Dense(len(classes)) → Softmax

optimized using
```python
SGD(learning_rate=0.01, momentum=0.9, nesterov=True)
```

## License
This project is for learning and educational purposes.
