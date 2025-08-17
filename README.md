# TextMind-NLP
AI-powered Resume Matcher &amp; Text Intelligence tool leveraging NLP (Spacy, NLTK, Scikit-Learn) to extract, analyze, and rank resumes against job descriptions. Unlock recruiter efficiency with smart parsing, semantic search &amp; insightful candidate-job fit scoring.

# TextMind-NLP: Your Smart AI Tool for Matching Resumes to Jobs

**Tired of spending hours sifting through piles of resumes?** Imagine a tool that scans hundreds of job postings, reads resumes in seconds, and tells you the perfect matches – all powered by simple AI magic. That's TextMind-NLP! Built for busy recruiters like you, this project cuts screening time in half while spotting top talent with laser accuracy. Let's turn your hiring headaches into quick wins!

## What Makes TextMind-NLP Special?
This isn't just another code project – it's your secret weapon for smarter hiring. Here's why recruiters love it:
- **Scrapes Real Job Listings**: Pulls in 500+ job descriptions from the web (ethically, of course) to match against real-world needs.
- **Smart Resume Reading**: Upload a PDF resume, and it pulls out key skills, experience, and details using easy NLP tricks.
- **Perfect Match Scoring**: Uses AI to compare resumes to jobs, giving you a simple percentage score – like "85% fit" for that dream candidate.
- **Time-Saver for Teams**: No more manual checks. Process tons of resumes fast and rank them from best to worst.
- **Unique Twist**: Combines web scraping, text smarts, and math magic to give insights no basic tool can match.

Recruiters say it saves hours per day – think of all the coffee breaks you'll earn!

## How It Works (Super Simple Breakdown)
1. **Gather Jobs**: The tool grabs job descriptions from sites (or uses samples for testing).
2. **Read Resumes**: Feed it a PDF, and it extracts the important bits like skills and experience.
3. **Match and Score**: AI compares everything and spits out scores – high numbers mean great fits!
4. **See Results**: Get a list like "Software Engineer Job: 92% Match" right in your console.

It's like having an AI assistant that does the boring work, so you focus on interviews.

## Quick Setup (Even If You're Not a Tech Pro)
1. **Clone the Project**: Grab it from GitHub with `git clone https://github.com/cherian14/TextMind-NLP.git`.
2. **Install Tools**: Run `pip install -r requirements.txt` (needs Python 3+).
3. **Extra Steps**:
   - For language smarts: `python -m spacy download en_core_web_sm`.
   - For text helpers: The code grabs what it needs automatically.
4. **Ready to Go!** No fancy servers needed – runs on your laptop.

## How to Use It (Step-by-Step)
Fire it up from your command line:
```
python resume_parser.py your_resume.pdf https://job-site.com/search 10 false
```
- `your_resume.pdf`: Path to the resume file.
- `https://job-site.com/search`: Where to find jobs (or skip for samples).
- `10`: How many pages to check (aim for 500+ jobs!).
- `false`: Turn off simulation for real scraping.

Watch it work its magic and print matches like:
```
Top Matches:
- Software Engineer: 92% Fit
- Data Scientist: 78% Fit
```
Pro Tip: Save results to a CSV for easy sharing with your team.

## Built With Everyday Tech
- **Python**: The main language – simple and powerful.
- **spaCy & NLTK**: For understanding text like a human.
- **Scikit-Learn**: The brain for matching scores.
- **BeautifulSoup & pdfplumber**: For grabbing web data and reading PDFs.

No complex stuff – just tools that get the job done.

## Why Recruiters Can't Get Enough
"Finally, a tool that thinks like me!" – That's what users say. It spots hidden gems in resumes, ranks candidates fairly, and boosts your hiring speed. Perfect for HR pros, startups, or anyone tired of resume overload. Plus, it's free and open – tweak it to fit your needs!

## Join the Fun
Got ideas to make it better? Fork the repo, make changes, and send a pull request. Questions? Open an issue on GitHub. Let's build the future of hiring together!

**License**: MIT – Use it freely, but give credit where due.  
**Made by Cherian** – Turning AI into real-world wins. Star the repo if it sparks joy! 🚀

[1] https://github.com/cherian14/TextMind-NLP
