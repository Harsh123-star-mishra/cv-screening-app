# CV Screening App (Streamlit)

A privacy-friendly resume screening tool. Upload PDFs/DOCX + paste Job Description, get similarity scores, keyword coverage, and ATS checks.

## Deploy on Streamlit Community Cloud
1. Create a new **public GitHub repo** (e.g., `streamlit-cv-screening-app`).
2. Add these files at repo root:
   - `app.py`
   - `requirements.txt`
   - (optional) `README.md`
3. Go to **https://share.streamlit.io** and connect your GitHub account.
4. Select your repo, set **Main file path** = `app.py`, and **Deploy**.
5. After build completes, your app will be live at `https://<your-username>-<repo-name>-<branch>.streamlit.app/`.

## Local run (optional)
```bash
pip install -r requirements.txt
streamlit run app.py
```

## Features
- Parse **PDF** (pdfplumber) and **DOCX** (docx2txt)
- **TF‑IDF cosine similarity** between JD & each resume (custom implementation)
- **Keyword coverage** (matched/missing; simple phrase detection)
- **ATS checks**: email, phone, LinkedIn, GitHub, sections, bullets
- **Export CSV**

## Notes
- Scanned PDFs (images) won't extract text (no OCR).
- Adjust stopwords/thresholds in `app.py` per role/domain.
