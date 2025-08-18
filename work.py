import streamlit as st
import pandas as pd
import pdfplumber
import docx
import pytesseract
from PIL import Image
import spacy
import nltk
from nltk.corpus import stopwords
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import re
import io
import time
import os
import requests
from bs4 import BeautifulSoup
from concurrent.futures import ThreadPoolExecutor

# --- PAGE CONFIGURATION ---
st.set_page_config(
    page_title="Advanced Resume Parser with Job Scraping",
    layout="wide",
    initial_sidebar_state="expanded",
    page_icon="favicon.ico"  # Assumes favicon.ico is in the same directory
)

# --- AESTHETIC STYLING ---
st.markdown("""
<style>
    .stApp {
        background-color: #0D1117;
        color: #C9D1D9;
        font-family: 'Arial', sans-serif;
    }
    h1, h2, h3 {
        color: #58A6FF;
        text-shadow: 1px 1px 4px #000000;
    }
    [data-testid="stSidebar"] {
        background-color: #161B22;
        border-right: 1px solid #30363D;
    }
    .stButton > button {
        border: 2px solid #58A6FF;
        border-radius: 25px;
        background-color: transparent;
        color: #58A6FF;
        padding: 12px 30px;
        font-weight: bold;
        transition: all 0.3s ease;
        font-size: 16px;
    }
    .stButton > button:hover {
        background-color: #58A6FF;
        color: #0D1117;
        box-shadow: 0 0 15px #58A6FF;
    }
    .stFileUploader, .stTextInput, .stSelectbox {
        border-radius: 8px;
        border: 1px solid #30363D;
        background-color: #161B22;
        padding: 10px;
    }
    .stProgress > div > div > div > div {
        background-color: #58A6FF;
    }
    .section-container {
        background-color: #161B22;
        border-radius: 12px;
        padding: 20px;
        margin-bottom: 20px;
        border: 1px solid #30363D;
        box-shadow: 0 4px 20px rgba(0, 0, 0, 0.3);
    }
    .stDataFrame {
        border-radius: 10px;
        overflow: hidden;
    }
</style>
""", unsafe_allow_html=True)

# --- INITIAL SETUP ---
try:
    nltk.download('stopwords', quiet=True)
    stop_words = set(stopwords.words('english'))
except Exception as e:
    st.error(f"Failed to download NLTK data: {e}")
    st.stop()

@st.cache_resource
def load_spacy_model():
    try:
        return spacy.load("en_core_web_sm")
    except Exception as e:
        st.error(f"Failed to load spaCy model. Ensure 'en_core_web_sm' is installed: `python -m spacy download en_core_web_sm`. Error: {e}")
        st.stop()

nlp = load_spacy_model()

# --- LOAD 500+ JOB DATASET FROM ONLINE CSV ---
@st.cache_data
def load_online_job_dataset():
    try:
        url = "https://raw.githubusercontent.com/binoydutt/Resume-Job-Description-Matching/master/data.csv"
        df = pd.read_csv(url)
        # Standardize columns to match app's expected format
        df = df.rename(columns={
            "position": "Title",
            "company": "Company",
            "location": "Location",
            "Job Description": "Description"
        })
        # Add a placeholder Skills column if missing
        if "Skills" not in df.columns:
            df["Skills"] = [[] for _ in range(len(df))]
        st.info(f"Loaded {len(df)} job postings from online dataset.")
        return df
    except Exception as e:
        st.error(f"Error loading online dataset: {e}")
        return pd.DataFrame()

# --- JOB SCRAPING FUNCTION (Optional, as fallback) ---
def scrape_job_descriptions(query="software engineer", num_pages=10, max_jobs=500):
    jobs = []
    base_url = "https://www.indeed.com/jobs"
    headers = {"User-Agent": "Mozilla/5.0"}

    def scrape_page(page):
        params = {"q": query, "start": page * 10}
        response = requests.get(base_url, params=params, headers=headers)
        soup = BeautifulSoup(response.text, "html.parser")
        job_cards = soup.find_all("div", class_="job_seen_beacon")
        page_jobs = []
        for card in job_cards:
            title = card.find("h2").text.strip()
            company = card.find("span", class_="companyName").text.strip() if card.find("span", class_="companyName") else "N/A"
            location = card.find("div", class_="companyLocation").text.strip() if card.find("div", class_="companyLocation") else "N/A"
            description = card.find("div", class_="job-snippet").text.strip() if card.find("div", class_="job-snippet") else "N/A"
            page_jobs.append({
                "Title": title,
                "Company": company,
                "Location": location,
                "Description": description,
                "Skills": extract_skills_from_description(description)
            })
        return page_jobs

    with ThreadPoolExecutor(max_workers=5) as executor:
        results = list(executor.map(scrape_page, range(num_pages)))
    
    for page_jobs in results:
        jobs.extend(page_jobs)
        if len(jobs) >= max_jobs:
            break
    
    return pd.DataFrame(jobs[:max_jobs])

def extract_skills_from_description(desc):
    doc = nlp(desc.lower())
    skills = [ent.text for ent in doc.ents if ent.label_ in ["SKILL", "ORG", "PRODUCT"]]  # Custom skill extraction
    return list(set(skills))

# --- CORE FUNCTIONS ---
def extract_text_from_pdf(file):
    try:
        with pdfplumber.open(file) as pdf:
            return "".join(page.extract_text() or "" for page in pdf.pages)
    except Exception as e:
        st.error(f"Error reading PDF: {e}")
        return ""

def extract_text_from_docx(file):
    try:
        doc = docx.Document(file)
        return " ".join([p.text for p in doc.paragraphs if p.text])
    except Exception as e:
        st.error(f"Error reading DOCX: {e}")
        return ""

def extract_text_from_image(file):
    try:
        img = Image.open(file)
        text = pytesseract.image_to_string(img)
        return text
    except Exception as e:
        st.error(f"Error reading image: {e}")
        return ""

def extract_resume_text(uploaded_file):
    file_extension = uploaded_file.name.split('.')[-1].lower()
    try:
        if file_extension == 'pdf':
            return extract_text_from_pdf(io.BytesIO(uploaded_file.read()))
        elif file_extension == 'docx':
            return extract_text_from_docx(io.BytesIO(uploaded_file.read()))
        elif file_extension in ['png', 'jpg', 'jpeg']:
            return extract_text_from_image(io.BytesIO(uploaded_file.read()))
        else:
            st.error(f"Unsupported file type: .{file_extension}")
            return ""
    except Exception as e:
        st.error(f"Error processing file {uploaded_file.name}: {e}")
        return ""

def preprocess_text(text):
    try:
        text = re.sub(r'\s+', ' ', text.lower()).strip()
        doc = nlp(text)
        tokens = [token.text for token in doc if token.text not in stop_words and not token.is_punct]
        return " ".join(tokens)
    except Exception as e:
        st.error(f"Error preprocessing text: {e}")
        return ""

def extract_all_entities(text):
    try:
        doc = nlp(text.lower())
        entities = {
            "Name": "",
            "Email": "",
            "Phone": "",
            "Skills": [],
            "Education": [],
            "Experience": [],
            "Designations": []
        }
        
        SKILLS_DB = [
            'python', 'java', 'c++', 'c#', 'sql', 'javascript', 'typescript', 'react', 'angular', 'vue',
            'node.js', 'django', 'flask', 'spring', 'machine learning', 'deep learning', 'nlp', 'generative ai', 'llms',
            'data science', 'data analysis', 'pandas', 'numpy', 'scikit-learn', 'tensorflow', 'pytorch',
            'aws', 'azure', 'gcp', 'docker', 'kubernetes', 'git', 'ci/cd', 'agile', 'scrum', 'api', 'rest', 'graphql'
        ]
        
        # Name
        entities["Name"] = next((ent.text.title() for ent in doc.ents if ent.label_ == 'PERSON'), "Not Found")
        
        # Contact Info
        email = re.search(r'\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Z|a-z]{2,}\b', text)
        phone = re.search(r'(\(?\d{3}\)?[-.\s]?)?(\d{3}[-.\s]?\d{4})', text)
        entities["Email"] = email.group(0) if email else "Not Found"
        entities["Phone"] = phone.group(0) if phone else "Not Found"
        
        # Skills
        entities["Skills"] = list(set([skill for skill in SKILLS_DB if re.search(r'\b' + re.escape(skill) + r'\b', text, re.IGNORECASE)]))
        
        # Education (Fixed: Handle empty or incomplete tuples)
        education_patterns = re.findall(r'\b(bachelor|master|phd|degree)\b.*?\b(computer science|data science|engineering|information technology|business administration)\b', text, re.IGNORECASE)
        entities["Education"] = []
        for e in education_patterns:
            if len(e) == 2:  # Ensure tuple has exactly 2 elements
                entities["Education"].append(f"{e[0]} in {e[1]}")
            else:
                continue  # Skip invalid patterns
        
        # Experience
        entities["Experience"] = re.findall(r'\b(\d+\s*(?:year|yr)s?\s*(?:experience)?)\b', text, re.IGNORECASE)
        
        # Designations
        entities["Designations"] = re.findall(r'\b(software engineer|data scientist|machine learning engineer|developer|analyst|manager|consultant|architect)\b', text, re.IGNORECASE)
        
        for key in entities:
            if isinstance(entities[key], list):
                entities[key] = list(set(entities[key]))
        
        return entities
    except Exception as e:
        st.error(f"Error extracting entities: {e}")
        return entities

def compute_match_score(resume_text, job_description):
    try:
        documents = [resume_text, job_description]
        vectorizer = TfidfVectorizer()
        tfidf_matrix = vectorizer.fit_transform(documents)
        similarity = cosine_similarity(tfidf_matrix[0:1], tfidf_matrix[1:2])[0][0]
        return round(float(similarity) * 100, 2)
    except Exception as e:
        st.error(f"Error computing match score: {e}")
        return 0.0

def skill_gap_analysis(resume_skills, job_skills):
    try:
        if any(isinstance(i, list) for i in job_skills):
            flat_job_skills = [item.lower() for sublist in job_skills for item in sublist]
        else:
            flat_job_skills = [skill.lower() for skill in job_skills]
        flat_resume_skills = [skill.lower() for skill in resume_skills]
        return list(set(flat_job_skills) - set(flat_resume_skills))
    except Exception as e:
        st.error(f"Error in skill gap analysis: {e}")
        return []

# Train custom SpaCy model (simple example; expand for production)
@st.cache_resource
def train_custom_spacy(scraped_jobs):
    # For demo: Use scraped skills to update SKILLS_DB or fine-tune
    all_skills = set()
    for _, row in scraped_jobs.iterrows():
        all_skills.update(row["Skills"])
    # In production, fine-tune SpaCy with new entities
    return list(all_skills)  # Return dynamic skills

# --- SIDEBAR ---
with st.sidebar:
    st.markdown("<h2 style='text-align: center;'>Advanced Resume Parser</h2>", unsafe_allow_html=True)
    st.markdown("### With Job Scraping Integration")
    st.markdown("---")
    st.markdown("""
    Automate resume parsing and match with scraped job data. 
    Extracts details using NLP and provides skill-gap analysis.
    """)
    st.markdown("---")
    st.info("Scrape jobs, upload resumes, and analyze.")

# --- MAIN APPLICATION ---
st.markdown("<h1 style='text-align: center;'>Advanced Resume Parser with Job Scraping</h1>", unsafe_allow_html=True)
st.markdown("Parse resumes and match against scraped job descriptions.", unsafe_allow_html=True)

# Step 0: Load 500+ Online Job Dataset or Scrape
st.markdown('<div class="section-container">', unsafe_allow_html=True)
st.header("Step 0: Load or Scrape Job Descriptions")

load_option = st.radio("Choose Data Source", ("Load 500+ Jobs from Online CSV (GitHub)", "Scrape Custom Jobs"))

if load_option == "Load 500+ Jobs from Online CSV (GitHub)":
    if st.button("Load Online Dataset"):
        with st.spinner("Loading job dataset..."):
            job_df = load_online_job_dataset()
            if not job_df.empty:
                st.session_state['job_df'] = job_df
                st.success(f"Loaded {len(job_df)} job postings!")
                st.dataframe(job_df[['Title', 'Company', 'Location']], use_container_width=True)
                # Train custom model on loaded data
                dynamic_skills = train_custom_spacy(job_df)
                st.info(f"Extracted {len(dynamic_skills)} unique skills for custom analysis.")
            else:
                st.error("Failed to load dataset.")

else:
    job_query = st.text_input("Enter Job Query (e.g., software engineer)", "software engineer")
    num_pages = st.slider("Number of Pages to Scrape", 1, 50, 10)
    max_jobs = st.slider("Max Jobs to Scrape (up to 500)", 10, 500, 500)

    if st.button("Scrape Jobs"):
        with st.spinner("Scraping job descriptions..."):
            try:
                scraped_df = scrape_job_descriptions(job_query, num_pages, max_jobs)
                if scraped_df.empty:
                    st.error("No jobs scraped.")
                else:
                    st.session_state['job_df'] = scraped_df
                    st.success(f"Scraped {len(scraped_df)} job postings!")
                    st.dataframe(scraped_df[['Title', 'Company', 'Location']], use_container_width=True)
                    # Train custom model on scraped data
                    dynamic_skills = train_custom_spacy(scraped_df)
                    st.info(f"Extracted {len(dynamic_skills)} unique skills for custom analysis.")
            except Exception as e:
                st.error(f"Error scraping jobs: {e}")
st.markdown('</div>', unsafe_allow_html=True)

# Step 1: Upload Resumes
st.markdown('<div class="section-container">', unsafe_allow_html=True)
st.header("Step 1: Upload Candidate Resumes")
st.markdown("Upload resumes in PDF, DOCX, or image formats (PNG, JPG, JPEG) for analysis.")

uploaded_files = st.file_uploader("Choose resume files", type=["pdf", "docx", "png", "jpg", "jpeg"], accept_multiple_files=True)

# Step 2: Analyze and Rank Candidates
if 'job_df' in st.session_state and uploaded_files:
    st.markdown('<div class="section-container">', unsafe_allow_html=True)
    st.header("Step 2: Analyze and Rank Candidates")
    job_df = st.session_state['job_df']
    job_titles = job_df['Title'].tolist()
    selected_job = st.selectbox("Select a Job Posting", job_titles)
    
    if st.button("⚡ Analyze Synergy"):
        try:
            job_description = job_df[job_df['Title'] == selected_job]['Description'].iloc[0]
            job_skills = job_df[job_df['Title'] == selected_job]['Skills'].iloc[0] if 'Skills' in job_df.columns else []
            job_description_processed = preprocess_text(job_description)
            
            candidates = []
            progress_bar = st.progress(0)
            status_placeholder = st.empty()
            
            for i, uploaded_file in enumerate(uploaded_files):
                resume_text = extract_resume_text(uploaded_file)
                if not resume_text:
                    continue
                processed_resume = preprocess_text(resume_text)
                entities = extract_all_entities(resume_text)
                match_score = compute_match_score(processed_resume, job_description_processed)
                missing_skills = skill_gap_analysis(entities["Skills"], job_skills)
                
                candidates.append({
                    "File Name": uploaded_file.name,
                    "Detected Name": entities["Name"],
                    "Synergy Score (%)": match_score,
                    "Candidate Skills": ", ".join(entities["Skills"]) or "None",
                    "Missing Skills": ", ".join(missing_skills) or "None",
                    "Education": ", ".join(entities["Education"]) or "None",
                    "Experience": ", ".join(entities["Experience"]) or "None",
                    "Designations": ", ".join(entities["Designations"]) or "None",
                    "Contact": f"{entities['Email']}, {entities['Phone']}"
                })
                
                progress = (i + 1) / len(uploaded_files)
                progress_bar.progress(progress)
                status_placeholder.info(f"Processing resume {i+1}/{len(uploaded_files)}...")
            
            status_placeholder.success("Analysis complete!")
            time.sleep(1)
            status_placeholder.empty()
            progress_bar.empty()
            
            ranked_candidates = pd.DataFrame(candidates).sort_values(by="Synergy Score (%)", ascending=False)
            st.session_state['results_df'] = ranked_candidates
            
            st.subheader("Ranked Candidates with Skill-Gap Analysis")
            st.dataframe(
                ranked_candidates.style.background_gradient(cmap='Blues', subset=['Synergy Score (%)']),
                use_container_width=True
            )
            
            # Download results
            csv = ranked_candidates.to_csv(index=False).encode('utf-8')
            st.download_button(
                label="Download Candidate Analysis (CSV)",
                data=csv,
                file_name=f"{selected_job.replace(' ', '_')}_candidate_analysis.csv",
                mime='text/csv'
            )
        except Exception as e:
            st.error(f"Error during analysis: {e}")
    st.markdown('</div>', unsafe_allow_html=True)

st.markdown("---")
st.markdown("**Keywords**: NLP, Resume Parsing, Job Scraping, Skill-Gap Analysis, Python, spaCy, NLTK, Scikit-learn, OCR, Machine Learning")
