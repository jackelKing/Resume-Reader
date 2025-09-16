import re
import nltk
nltk.download('stopwords')
from nltk.corpus import stopwords

stop_words = set(stopwords.words('english'))

def clean_text(text):
    """
    Lowercase, remove special characters, numbers, and stopwords.
    """
    text = text.lower()
    text = re.sub(r'\d+', '', text)
    text = re.sub(r'\W+', ' ', text)
    words = [w for w in text.split() if w not in stop_words]
    return " ".join(words)

def extract_skills(text, skills_list):
    """
    Extract skills present in the resume from a skills dictionary.
    """
    text = text.lower()
    extracted = [skill for skill in skills_list if skill.lower() in text]
    return extracted
