# CV Screening App (Browser-only)

A lightweight, privacy-friendly CV/resume screening tool that runs 100% in the browser. No Python, no server.

## Features
- Parse **PDF** and **DOCX** resumes client-side (PDF.js + mammoth.js)
- Paste **Job Description** and get similarity score (cosine TF-IDF)
- Highlight matched/missing keywords; optional simple phrase matching
- Basic **ATS checks** (email, phone, LinkedIn, sections)
- Export results to **CSV**
- Host with **GitHub Pages** in minutes

## How to run locally
Just open `index.html` in a modern browser (Chrome/Edge/Firefox). To avoid CORS, prefer a local server or GitHub Pages hosting.

## Deploy to GitHub Pages
1. Create a new GitHub repo (e.g., `cv-screening-app`).
2. Upload the contents of this folder (`index.html`, `assets/`).
3. In **Settings → Pages**, set **Branch** to `main` (root) and save.
4. Wait 1–2 minutes; your site will be live at `https://<your-username>.github.io/cv-screening-app/`.

## Notes
- All processing happens client-side. Large or scanned PDFs may reduce accuracy.
- This is a heuristic tool; tweak stopwords/logic in `assets/app.js` as needed.
