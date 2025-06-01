# ✅ Imports
import gradio as gr
from tensorflow.keras.models import load_model
import pickle
import numpy as np
from tensorflow.keras.preprocessing.sequence import pad_sequences

# ✅ Load saved model and tokenizer
model = load_model("lstm_model.h5")

with open("tokenizer.pkl", "rb") as f:
    tokenizer = pickle.load(f)

# ✅ Text cleaning function (same as training time)
def clean_text(text):
    import re
    from nltk.corpus import stopwords
    from nltk.stem import PorterStemmer
    import nltk
    nltk.download('stopwords', quiet=True)

    ps = PorterStemmer()
    stop_words = set(stopwords.words('english'))
    text = text.lower()
    text = re.sub(r'[^a-z\s]', '', text)
    words = text.split()
    words = [ps.stem(word) for word in words if word not in stop_words]
    return " ".join(words)

# ✅ Sentiment prediction function
def predict_sentiment(text):
    cleaned = clean_text(text)
    sequence = tokenizer.texts_to_sequences([cleaned])
    padded = pad_sequences(sequence, maxlen=100)
    pred = model.predict(padded)[0][0]
    label = "Positive 😊" if pred > 0.5 else "Negative 😞"
    return f"Prediction: {label}
Confidence Score: {pred:.2f}"

# ✅ Gradio UI
gr.Interface(
    fn=predict_sentiment,
    inputs=gr.Textbox(lines=4, placeholder="Enter an Amazon review here..."),
    outputs="text",
    title="Amazon Review Sentiment Analyzer",
    description="Enter a product review to detect if it's Positive or Negative."
).launch(share=True)  # ✅ This gives you a public link
