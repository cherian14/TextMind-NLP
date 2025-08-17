\
import os
import string
from typing import List, Tuple, Union

import pdfplumber
import spacy
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize

# Load spaCy model once
try:
    nlp = spacy.load("en_core_web_sm")
except Exception as e:
    nlp = None
    print("[WARN] spaCy model en_core_web_sm not loaded. Some features will be limited.")
    
STOP_WORDS = set(stopwords.words("english"))

def read_text(path: str) -> str:
    """
    Reads a .txt file into a string.
    """
    with open(path, "r", encoding="utf-8", errors="ignore") as f:
        return f.read()

def parse_resume(path: str) -> str:
    """
    Extract text from a PDF or TXT resume.
    """
    ext = os.path.splitext(path)[1].lower()
    if ext == ".pdf":
        text = []
        with pdfplumber.open(path) as pdf:
            for page in pdf.pages:
                page_text = page.extract_text() or ""
                text.append(page_text)
        return "\n".join(text).strip()
    elif ext == ".txt":
        return read_text(path)
    else:
        raise ValueError("Unsupported resume format. Use .pdf or .txt")

def preprocess_text(text: str) -> str:
    """
    Lowercase, tokenize, remove stopwords and punctuation.
    """
    tokens = word_tokenize(text.lower())
    tokens = [t for t in tokens if t not in STOP_WORDS and t not in string.punctuation]
    return " ".join(tokens)

def extract_keywords(text: str) -> List[str]:
    """
    Simple keyword extractor using spaCy noun chunks (if available).
    Falls back to frequent tokens.
    """
    keywords = []
    if nlp:
        doc = nlp(text)
        for chunk in doc.noun_chunks:
            k = chunk.text.strip().lower()
            if len(k) > 2:
                keywords.append(k)
    else:
        # Fallback: return the preprocessed tokens (not ideal, but works)
        pre = preprocess_text(text)
        keywords = [w for w in pre.split() if len(w) > 2]
    # De-duplicate while preserving order
    seen = set()
    out = []
    for k in keywords:
        if k not in seen:
            seen.add(k)
            out.append(k)
    return out[:100]

def match_resume_to_jd(resume_text: str, jd_text: str) -> float:
    """
    Compute cosine similarity (TF-IDF) between resume and JD.
    Returns a percentage 0..100.
    """
    resume_proc = preprocess_text(resume_text)
    jd_proc = preprocess_text(jd_text)
    vectorizer = TfidfVectorizer()
    vectors = vectorizer.fit_transform([resume_proc, jd_proc])
    sim = cosine_similarity(vectors[0:1], vectors[1:2])[0][0]
    return float(sim * 100.0)

def rank_resume_against_many(resume_text: str, items: List[dict], top_k: int = 5) -> List[Tuple[int, dict, float]]:
    """
    Rank a resume against many JDs.
    Returns list of (index, item, score) sorted by score desc.
    """
    scores = []
    for i, item in enumerate(items):
        jd_text = item.get("description") or item.get("text") or ""
        score = match_resume_to_jd(resume_text, jd_text)
        scores.append((i, item, score))
    scores.sort(key=lambda x: x[2], reverse=True)
    return scores[:top_k]
