import gradio as gr
import joblib
from utils import clean_text

# Load models
vectorizer = joblib.load("models/vectorizer.joblib")
clf = joblib.load("models/role_clf.joblib")

def predict_resume(text):
    cleaned = clean_text(text)
    X_vect = vectorizer.transform([cleaned])
    pred = clf.predict(X_vect)[0]
    return pred

# Gradio Interface
demo = gr.Interface(
    fn=predict_resume,
    inputs=gr.Textbox(lines=15, placeholder="Paste resume text here..."),
    outputs="text",
    title="Resume Classifier",
    description="Upload/paste a resume and get predicted job role."
)

# For Render deployment
app = gr.mount_gradio_app(app=None, blocks=demo, path="/")
