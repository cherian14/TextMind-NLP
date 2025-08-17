# 🧠 TextMind-NLP: The Recruiter's Co-Pilot

### From a Mountain of Resumes to a Shortlist of Stars.

![Python](https://img.shields.io/badge/Python-3.8%2B-blue?style=for-the-badge&logo=python)
![spaCy](https://img.shields.io/badge/spaCy-3.5%2B-blueviolet?style=for-the-badge&logo=spacy)
![NLTK](https://img.shields.io/badge/NLTK-3.8%2B-yellow?style=for-the-badge)
![Scikit-learn](https://img.shields.io/badge/Scikit--learn-1.3.0-orange?style=for-the-badge&logo=scikit-learn)

---

Tired of the endless sea of resumes? **TextMind-NLP** is an intelligent engine that cuts through the noise, using Natural Language Processing to find the perfect candidate-job fit. It automates the most time-consuming part of recruiting, allowing you to focus on what matters: connecting with top talent.

![TextMind-NLP Demo GIF](https://raw.githubusercontent.com/cherian14/TextMind-NLP/main/demo.gif)
_**(Pro-Tip: Record a short GIF of your CLI in action, name it `demo.gif`, and upload it for this to work!)**_

## ✨ The Intelligence Behind the Engine

This isn't just a keyword searcher. TextMind-NLP understands context, skills, and experience to deliver a true measure of a candidate's suitability.

*   **🌐 Real-World Job Scraping:** Ethically scrapes 500+ live job descriptions from the web to build a realistic, dynamic dataset for analysis.
*   **📄 Intelligent Resume Deconstruction:** Parses PDF resumes in seconds, extracting key entities like skills, education, and years of experience using `spaCy` and `pdfplumber`.
*   **💡 AI-Powered Fit Scoring:** Leverages `TF-IDF Vectorization` and `Cosine Similarity` from Scikit-learn to calculate a precise percentage match score between a resume and a job description.
*   **🚀 Automated Candidate Ranking:** Processes resumes in bulk and generates a clean, ranked list of the most promising candidates for any given role, dramatically reducing screening time.

## 🛠️ Technology & Architecture

This project showcases a complete NLP pipeline, from data acquisition to intelligent analysis.

| Component                 | Technologies                                         | Purpose                                        |
| ------------------------- | ---------------------------------------------------- | ---------------------------------------------- |
| **Data Acquisition**      | `Python`, `BeautifulSoup`                            | Web scraping for job descriptions              |
| **Document Parsing**      | `pdfplumber`, `spaCy`                                | Extracting clean text and entities from PDFs   |
| **NLP & Text Processing** | `NLTK`, `spaCy`                                      | Tokenization, stop-word removal, lemmatization |
| **Machine Learning Core** | `Scikit-learn` (TfidfVectorizer, Cosine Similarity) | Feature extraction and similarity scoring      |
| **Command-Line Interface**| `Python` (`argparse`)                                | User-friendly controls for running the tool    |

## 🎬 How It Works: The 4-Step Flow

1.  **INGEST:** The `scraper` module gathers hundreds of job descriptions from the web.
2.  **PARSE:** The `resume_utils` module reads a candidate's PDF resume and extracts structured information.
3.  **ANALYZE:** The core logic transforms both job and resume text into numerical vectors.
4.  **SCORE:** Cosine similarity is calculated to produce a simple, intuitive match percentage.

## 🚀 Get Started in 60 Seconds

Run the entire pipeline on your local machine.

**Prerequisites:** Python 3.8+, Git

1.  **Clone the Repository:**
    ```bash
    git clone https://github.com/cherian14/TextMind-NLP.git
    cd TextMind-NLP
    ```

2.  **Install Dependencies:**
    ```bash
    pip install -r requirements.txt
    ```

3.  **Download NLP Models:**
    ```bash
    python -m spacy download en_core_web_sm
    ```

4.  **Run the Analysis:**
    Execute the tool from your command line.
    ```bash
    # Example: Match a sample resume against 10 pages of scraped jobs
    python cli.py sample_resume.txt "https://www.linkedin.com/jobs/search" 10
    ```
    The results, showing the top job matches and their scores, will be printed directly to your console.

---

## 💼 Message to Recruiters & Hiring Managers

> This project is a demonstration of my ability to design and build end-to-end data solutions that solve tangible business problems. It showcases a strong foundation in **Python, Natural Language Processing, and Machine Learning**.
>
> From architecting a web scraper and building a robust parsing pipeline to implementing a similarity-scoring algorithm, this tool highlights my skills in turning unstructured text into actionable intelligence. I am passionate about leveraging AI to create efficiency and drive value. Let's discuss how I can bring that passion to your team.

## 🤝 Connect with Me

**Built with ❤️ by Cherian R**

*   **GitHub:** [@cherian14](https://github.com/cherian14)
*   **LinkedIn:** [Cherian R](https://www.linkedin.com/in/cherian-r-a1bba3292/)
*   **Email:** `cherian262005@gmail.com`

**Star the repo if it sparked your interest! 🌟**
