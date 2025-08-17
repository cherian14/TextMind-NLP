# Resume Matcher (Windows, step-by-step)

## 0) Open Command Prompt (CMD)
Press Start → type `cmd` → Enter.

## 1) Create a project folder and enter it
Example (you can choose any folder):
```
mkdir %USERPROFILE%\Desktop\resume-matcher
cd %USERPROFILE%\Desktop\resume-matcher
```

## 2) Create and activate a virtual environment
```
py -m venv .venv
call .venv\Scripts\activate
```
If you use PowerShell, activate with:
```
.venv\Scripts\Activate.ps1
```

## 3) Copy these files into your folder
- `requirements.txt`
- `scraper_bs4.py`
- `resume_utils.py`
- `cli.py`
- `sample_resume.txt`

(Or unzip `resume_matcher_project.zip` that you downloaded from ChatGPT.)

## 4) Install packages
Prefer using the launcher (`py -m`) so you don't need `pip` on PATH.
```
py -m pip install --upgrade pip
py -m pip install -r requirements.txt
```

If any package shows a compatibility error on Python 3.13, install Python 3.11 alongside and create the venv with it:
- Download Python 3.11 from python.org and install (check "Add to PATH").
- Then: `py -3.11 -m venv .venv` and re-activate the venv.
- Re-run step 4.

## 5) Download NLTK data (non-interactive) and spaCy model
```
py -c "import nltk; nltk.download('punkt'); nltk.download('stopwords')"
py -m spacy download en_core_web_sm
```

## 6) (Optional) Scrape sample job descriptions
This example uses the well-known demo site "Real Python Fake Jobs".
```
py scraper_bs4.py --url https://realpython.github.io/fake-jobs/ --out data\job_descriptions.json
```

## 7) Run the matcher
**Option A: Match a resume against a single JD text**
```
py cli.py --resume sample_resume.txt --jd-text "Seeking Python developer with NLP and web scraping experience."
```

**Option B: Match a resume against a JSON file of many JDs**
```
py cli.py --resume sample_resume.txt --jd-file data\job_descriptions.json --top 5
```

## 8) Notes
- `--resume` accepts `.pdf` or `.txt` files.
- JSON file can be either `[{"title": "...","description": "..."}]` or `{"items":[...]}`.
- The demo extractor uses TF-IDF similarity; it's simple and fast.

Enjoy!
