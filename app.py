# streamlit app for CV screening
import io, re, math, tempfile
from typing import List, Dict, Any
import streamlit as st
import pandas as pd
import numpy as np
import pdfplumber
import docx2txt

st.set_page_config(page_title="CV Screening App", layout="wide")
st.title("📄 CV Screening App (Streamlit)")
st.caption("Upload multiple resumes (PDF/DOCX) + paste Job Description. All processing on Streamlit Cloud backend. No installs on your machine.")

# ------------------ Utilities ------------------
STOPWORDS = set([s.strip() for s in """
a,an,the,of,in,on,for,with,and,or,to,from,by,at,as,is,are,was,were,be,been,being,this,that,these,those,will,shall,can,could,should,would,may,might,about,into,over,per,via,within,without,across,through,up,down,out,off,so,if,then,than,too,very,more,most,least,not,no,yes,also,using,use,used,including,include,includes,ability,experience,years,year,responsibilities,responsibility,candidate,role,job,work,working
""".split(',')])


def clean_text(t: str) -> str:
    if not t:
        return ""
    t = t.replace("\u00A0", " ")
    t = re.sub(r"[\r\n]+", " ", t)
    t = re.sub(r"[^a-zA-Z0-9+#\./\- ]", " ", t)
    t = re.sub(r"\s{2,}", " ", t)
    return t.strip()


def tokenize(t: str) -> List[str]:
    t = clean_text(t).lower()
    toks = re.split(r"[^a-z0-9+#\./\-]+", t)
    return [w for w in toks if w and w not in STOPWORDS and len(w) > 1]


def extract_phrases(tokens: List[str], max_len: int = 3) -> List[str]:
    phrases = set()
    for n in range(2, max_len + 1):
        for i in range(0, max(0, len(tokens) - n + 1)):
            p = tokens[i:i+n]
            if any(w in STOPWORDS for w in p):
                continue
            phrases.add(" ".join(p))
    return list(phrases)


def term_freq(tokens: List[str]) -> Dict[str, float]:
    tf: Dict[str, float] = {}
    for t in tokens:
        tf[t] = tf.get(t, 0.0) + 1.0
    total = float(len(tokens) or 1)
    for k in list(tf.keys()):
        tf[k] = tf[k] / total
    return tf


def inverse_doc_freq(doc_tokens: List[List[str]]) -> Dict[str, float]:
    N = len(doc_tokens)
    df: Dict[str, int] = {}
    for arr in doc_tokens:
        for t in set(arr):
            df[t] = df.get(t, 0) + 1
    idf: Dict[str, float] = {}
    for t, d in df.items():
        idf[t] = math.log((N + 1) / (d + 0.5)) + 1.0
    return idf


def build_vector(tf: Dict[str, float], idf: Dict[str, float]) -> Dict[str, float]:
    return {t: tf.get(t, 0.0) * idf.get(t, 0.0) for t in tf.keys()}


def cosine_sim(v1: Dict[str, float], v2: Dict[str, float]) -> float:
    keys = set(v1.keys()) | set(v2.keys())
    dot = sum(v1.get(k, 0.0) * v2.get(k, 0.0) for k in keys)
    n1 = math.sqrt(sum((v1.get(k, 0.0))**2 for k in keys))
    n2 = math.sqrt(sum((v2.get(k, 0.0))**2 for k in keys))
    if n1 == 0.0 or n2 == 0.0:
        return 0.0
    return dot / (n1 * n2)


def keyword_coverage(jd_tokens: List[str], resume_tokens: List[str], use_phrases: bool):
    jd_tf = term_freq(jd_tokens)
    sorted_tokens = sorted(jd_tf.items(), key=lambda x: x[1], reverse=True)
    top_tokens = [w for (w, _) in sorted_tokens if not re.match(r"^(year|years|experience|candidate|responsibilities|responsibility)$", w)]
    top_tokens = top_tokens[:30]
    res_set = set(resume_tokens)
    present = [t for t in top_tokens if t in res_set]
    missing = [t for t in top_tokens if t not in res_set]

    phrase_present, phrase_missing = [], []
    if use_phrases:
        phrases = extract_phrases(jd_tokens)
        phrases = [p for p in phrases if len(p.split()) > 1][:20]
        res_text = " ".join(resume_tokens)
        phrase_present = [p for p in phrases if p in res_text]
        phrase_missing = [p for p in phrases if p not in res_text]
    return present, missing, phrase_present, phrase_missing


def ats_checks(text: str, strict: bool = False):
    checks = []
    has_email = bool(re.search(r"[A-Z0-9._%+-]+@[A-Z0-9.-]+\.[A-Z]{2,}", text, re.I))
    # normalize digits for phone
    digits = re.sub(r"\D", " ", text)
    digits = re.sub(r"\s+", " ", digits)
    has_phone = bool(re.search(r"(\+\d{1,3}[- ]?)?\d{10}", digits))
    has_linked = bool(re.search(r"linkedin\.com/[A-Za-z0-9_\-/]+", text, re.I))
    has_github = bool(re.search(r"github\.com/[A-Za-z0-9_\-/]+", text, re.I))
    has_sections = bool(re.search(r"(education|experience|skills|projects|summary|objective)", text, re.I))
    bullets_ok = bool(re.search(r"(\n\s*[\u2022\-*])", text))
    years = re.findall(r"(\d+)\+?\s*(years|yrs)", text, re.I)

    checks.append({"name": "Email", "ok": has_email})
    checks.append({"name": "Phone", "ok": has_phone})
    checks.append({"name": "LinkedIn", "ok": has_linked})
    checks.append({"name": "GitHub", "ok": has_github})
    checks.append({"name": "Sections present", "ok": has_sections})
    checks.append({"name": "Bullets used", "ok": bullets_ok})
    if strict:
        checks.append({"name": "Experience mentioned", "ok": len(years) > 0})
    return checks

# ------------------ File parsing ------------------

def parse_pdf(uploaded_file) -> str:
    # uploaded_file is a streamlit UploadedFile
    with pdfplumber.open(io.BytesIO(uploaded_file.getvalue())) as pdf:
        texts = []
        for page in pdf.pages:
            try:
                texts.append(page.extract_text() or "")
            except Exception:
                pass
    return clean_text("\n".join(texts))


def parse_docx(uploaded_file) -> str:
    # docx2txt expects a file path; write to temp
    with tempfile.NamedTemporaryFile(delete=False, suffix=".docx") as tmp:
        tmp.write(uploaded_file.getbuffer())
        tmp_path = tmp.name
    try:
        text = docx2txt.process(tmp_path) or ""
    finally:
        try:
            import os
            os.remove(tmp_path)
        except Exception:
            pass
    return clean_text(text)


def file_to_text(uploaded_file) -> str:
    name = (uploaded_file.name or "").lower()
    if name.endswith('.pdf'):
        return parse_pdf(uploaded_file)
    elif name.endswith('.docx'):
        return parse_docx(uploaded_file)
    else:
        raise ValueError(f"Unsupported file: {uploaded_file.name}")

# ------------------ UI ------------------
col1, col2 = st.columns([1,1])
with col1:
    jd = st.text_area("Job Description", placeholder="Paste JD here...", height=200)
with col2:
    files = st.file_uploader("Upload resumes (PDF/DOCX)", type=["pdf","docx"], accept_multiple_files=True)

with st.sidebar:
    st.header("Options")
    use_phrases = st.checkbox("Try simple phrase matching", value=True)
    strict_ats = st.checkbox("Strict ATS checks", value=False)
    topn_display = st.slider("Show top N keywords", min_value=10, max_value=50, value=30, step=5)

run = st.button("Analyze")

if run:
    if not jd.strip():
        st.warning("Please paste a Job Description first.")
        st.stop()
    if not files:
        st.warning("Please upload at least one resume.")
        st.stop()

    st.info("Parsing resumes...")
    texts = []
    for f in files:
        try:
            t = file_to_text(f)
        except Exception as e:
            t = ""
        texts.append({"file": f, "text": t})

    st.info("Scoring...")
    jd_tokens = tokenize(jd)
    doc_tokens = [tokenize(x["text"]) for x in [{"text": t["text"]} for t in texts]]
    all_tokens = [jd_tokens] + doc_tokens

    idf = inverse_doc_freq(all_tokens)
    jd_vec = build_vector(term_freq(jd_tokens), idf)

    rows = []
    for i, x in enumerate(texts):
        tokens = doc_tokens[i]
        vec = build_vector(term_freq(tokens), idf)
        score = cosine_sim(jd_vec, vec)
        present, missing, phrase_present, phrase_missing = keyword_coverage(jd_tokens, tokens, use_phrases)
        ats = ats_checks(x["text"], strict_ats)
        row = {
            "Resume": x["file"].name,
            "Score(%)": round(score*100, 1),
            "ATS_OK": f"{sum(1 for c in ats if c['ok'])}/{len(ats)}",
            "MatchedKeywords": ", ".join((present + phrase_present)[:topn_display]),
            "MissingKeywords": ", ".join((missing + phrase_missing)[:topn_display]),
            "_ats": ats
        }
        rows.append(row)

    rows = sorted(rows, key=lambda r: r["Score(%)"], reverse=True)

    df = pd.DataFrame([{k:v for k,v in r.items() if not k.startswith('_')} for r in rows])
    st.subheader("Results")
    st.dataframe(df, use_container_width=True)

    csv = df.to_csv(index=False).encode('utf-8')
    st.download_button("⬇️ Download CSV", data=csv, file_name="cv-screening-results.csv", mime="text/csv")

    with st.expander("Show ATS details per resume"):
        for r in rows:
            st.markdown(f"**{r['Resume']}** — Score: {r['Score(%)']}% | ATS: {r['ATS_OK']}")
            badges = [f"✅ {c['name']}" if c['ok'] else f"❌ {c['name']}" for c in r['_ats']]
            st.write(", ".join(badges))
            st.divider()

st.markdown("""
---
**Notes**
- Image/scanned PDFs are not supported (no OCR). If needed, we can add Tesseract OCR.
- Heuristic matching; tune stopwords/logic for your roles.
""")
