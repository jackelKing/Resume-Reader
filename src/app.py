import os
import gradio as gr
import joblib
from utils import clean_text

vectorizer = joblib.load("models/vectorizer.joblib")
clf = joblib.load("models/role_clf.joblib")

def predict_resume(text):
    cleaned = clean_text(text)
    X_vect = vectorizer.transform([cleaned])
    pred = clf.predict(X_vect)[0]
    return pred

demo = gr.Interface(
    fn=predict_resume,
    inputs=gr.Textbox(lines=15, placeholder="Paste resume text here..."),
    outputs="text",
    title="Resume Classifier",
    description="Upload/paste a resume and get predicted job role."
)

if __name__ == "__main__":
    port = int(os.environ.get("PORT", 7860))  # Render assigns PORT
    demo.launch(server_name="0.0.0.0", server_port=port)
