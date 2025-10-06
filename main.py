import gradio as gr
import json
import random
import pickle
import nltk
import string
from nltk.tokenize import TreebankWordTokenizer
from nltk.stem import WordNetLemmatizer

nltk.download('wordnet')
nltk.download('omw-1.4')

# Load model, vectorizer, and intents
with open("chat_model.pkl", "rb") as f:
    model = pickle.load(f)

with open("vectorizer.pkl", "rb") as f:
    vectorizer = pickle.load(f)

with open("intents.json") as f:
    intents = json.load(f)

lemmatizer = WordNetLemmatizer()
tokenizer = TreebankWordTokenizer()

def preprocess(text):
    tokens = tokenizer.tokenize(text)
    tokens = [lemmatizer.lemmatize(t.lower()) for t in tokens if t not in string.punctuation]
    return " ".join(tokens)

def chat(user_input):
    if not user_input:
        return "Please ask me something about Aakansha!"
    processed = preprocess(user_input)
    X = vectorizer.transform([processed])
    intent = model.predict(X)[0]
    for i in intents["intents"]:
        if i["tag"] == intent:
            return random.choice(i["responses"])
    return "Hmm, I’m not sure about that. Try asking about my projects, skills, or education!"

# Build Gradio App
with gr.Blocks(theme=gr.themes.Soft()) as app:
    gr.Markdown(
        "<h1 style='text-align:center;'>🌸 Aakansha's AI Portfolio Assistant</h1>",
        elem_id="header"
    )

    with gr.Row():
        # Left column: profile + summary
        with gr.Column(scale=1):
            gr.Image("me.jpg", label="Aakansha Arora", show_label=False, width=200)
            gr.Markdown("""
### 👋 Hi, I'm Aakansha Arora  
**Software Engineer | MS CS @ Purdue | Ex-IBM**  

💡 I love building human-centered products — from AI assistants to dashboards.  
📸 I capture stories on [Instagram](https://www.instagram.com/one_last.click)  
✍️ I write on [Medium](https://medium.com/@aakansha.a03)  
🎨 I paint when I’m not coding.  
📧 Reach me at: **aakansha.a03@gmail.com**
            """)

        # Right column: chatbot
        with gr.Column(scale=2):
            gr.Markdown("### 🤖 Chat with my AI Assistant")
            chatbot = gr.ChatInterface(
                fn=chat,
                title="Aakansha's AI Assistant",
                description="Ask me about my work, skills, education, or projects!"
            )

if __name__ == "__main__":
    app.launch()
