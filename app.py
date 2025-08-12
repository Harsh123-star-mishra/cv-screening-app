# app.py — Polished UI/UX for CV Screening (Streamlit)
import io, re, math, tempfile, time
from typing import List, Dict
import streamlit as st
import pandas as pd
import numpy as np
import pdfplumber
import docx2txt

# ------------------ Page config ------------------
st.set_page_config(
    page_title="HR Mobineers – CV Screening",
    page_icon="📄",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ------------------ Minimal theming via CSS ------------------
STYLES = """
<style>
.block-container { padding-top: 1.1rem; padding-bottom: 2rem; }
header[data-testid="stHeader"] { backdrop-filter: blur(6px); }

.chip { display:inline-block; padding:4px 10px; border-radius:999px; margin:4px 6px 0 0; font-size:12px; background:#eef2ff; color:#3730a3; }
.badge { display:inline-block; padding:2px 8px; border-radius:999px; font-size:12px; margin:0 6px 6px 0; }
.badge.ok { background:#dcfce7; color:#166534; }
.badge.warn { background:#fee2e2; color:#991b1b; }
.kw { display:inline-block; padding:3px 8px; border-radius:10px; margin:3px 6px 0 0; font-size:12px; background:#f1f5f9; color:#0f172a; }
.kw.missing { background:#fff1f2; color:#9f1239; }
.small { color:#64748b; font-size:0.85rem; }
</style>
"""
st.markdown(STYLES, unsafe_allow_html=True)

# ------------------ Header ------------------
left, right = st.columns([0.8, 0.2])
with left:
    st.markdown("### 📄 HR Mobineers – CV Screening")
    st.caption("Upload multiple resumes (PDF/DOCX), paste JD, get similarity scores, keyword coverage & ATS checks.")
with right:
    st.markdown("<div class='small' style='text-align:right;'>Runs on Streamlit Cloud</div>", unsafe_allow_html=True)

# ------------------ Utilities ------------------
STOPWORDS = set([s.strip() for s in """
a,an,the,of,in,on,for,with,and,or,to,from,by,at,as,is,are,was,were,be,been,being,this,that,these,those,will,shall,can,could,should,would,may,might,about,into,over,per,via,within,without,across,through,up,down,out,off,so,if,then,than,too,very,more,most,least,not,no,yes,also,using,use,used,including,include,includes,ability,experience,years,year,responsibilities,responsibility,candidate,role,job,work,working
""".split(',')])

def clean_text(t: str) -> str:
    if not t: return ""
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
            if any(w in STOPWORDS for w in p): continue
            phrases.add(" ".join(p))
    return list(phrases)

def term_freq(tokens: List[str]) -> Dict[str, float]:
    tf: Dict[str, float] = {}
    for t in tokens: tf[t] = tf.get(t, 0.0) + 1.0
    total = float(len(tokens) or 1)
    for k in list(tf.keys()): tf[k] = tf[k] / total
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
    if n1 == 0.0 or n2 == 0.0: return 0.0
    return dot / (n1 * n2)

def keyword_coverage(jd_tokens: List[str], resume_tokens: List[str], use_phrases: bool):
    jd_tf = term_freq(jd_tokens)
    sorted_tokens = sorted(jd_tf.items(), key=lambda x: x[1], reverse=True)
    top_tokens = [w for (w, _) in sorted_tokens if not re.match(r"^(year|years|experience|candidate|responsibilities|responsibility)$", w)][:30]
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
    digits = re.sub(r"\D", " ", text); digits = re.sub(r"\s+", " ", digits)
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

def parse_pdf(uploaded_file) -> str:
    with pdfplumber.open(io.BytesIO(uploaded_file.getvalue())) as pdf:
        texts = []
        for page in pdf.pages:
            try:
                texts.append(page.extract_text() or "")
            except Exception:
                pass
    return clean_text("\n".join(texts))

def parse_docx(uploaded_file) -> str:
    with tempfile.NamedTemporaryFile(delete=False, suffix=".docx") as tmp:
        tmp.write(uploaded_file.getbuffer())
        tmp_path = tmp.name
    try:
        text = docx2txt.process(tmp_path) or ""
    finally:
        try:
            import os; os.remove(tmp_path)
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

# ------------------ Sidebar (sticky) ------------------
with st.sidebar:
    st.header("⚙️ Settings")
    use_phrases = st.checkbox("Simple phrase matching", value=True, help="2–3 word phrases from JD")
    strict_ats = st.checkbox("Strict ATS checks", value=False)
    topn_display = st.slider("Top N keywords (display)", 10, 50, 30, 5)
    score_threshold = st.slider("Shortlist score ≥ (%)", 0, 100, 70, 5)
    sort_by = st.selectbox("Sort by", ["Score desc", "ATS desc", "Name asc"])
    st.divider()
    st.caption("Tip: For scanned PDFs, consider OCR. We can add it on request.")

# ------------------ Main: Input ------------------
c1, c2 = st.columns([1, 1])
with c1:
    jd = st.text_area("Job Description", placeholder="Paste JD here…", height=220)
    st.markdown("<div class='small'>Tip: Include must-have skills & years.</div>", unsafe_allow_html=True)
with c2:
    files = st.file_uploader("Upload resumes (PDF/DOCX)", type=["pdf", "docx"], accept_multiple_files=True)
    if files:
        st.markdown("**Files:** " + " ".join([
            f"<span class='chip'>{('📄 PDF' if f.name.lower().endswith('.pdf') else '📝 DOCX')} — {f.name}</span>" for f in files
        ]), unsafe_allow_html=True)

run_col = st.container()
with run_col:
    run = st.button("🚀 Analyze", type="primary", use_container_width=True)

# ------------------ Analyze ------------------
if run:
    if not jd.strip():
        st.warning("Please paste a Job Description first.")
        st.stop()
    if not files:
        st.warning("Please upload at least one resume.")
        st.stop()

    progress = st.progress(0, text="Parsing resumes…")
    texts = []
    for i, f in enumerate(files, start=1):
        try:
            t = file_to_text(f)
        except Exception:
            t = ""
        texts.append({"file": f, "text": t})
        progress.progress(i / max(1, len(files)), text=f"Parsing: {f.name}")
    st.toast("Parsing complete ✅")

    with st.spinner("Scoring & extracting keywords…"):
        jd_tokens = tokenize(jd)
        doc_tokens = [tokenize(x["text"]) for x in texts]
        all_tokens = [jd_tokens] + doc_tokens
        idf = inverse_doc_freq(all_tokens)
        jd_vec = build_vector(term_freq(jd_tokens), idf)

        rows = []
        for i, x in enumerate(texts):
            tokens = doc_tokens[i]
            vec = build_vector(term_freq(tokens), idf)
            score = cosine_sim(jd_vec, vec)
            present, missing, phr_p, phr_m = keyword_coverage(jd_tokens, tokens, use_phrases)
            ats = ats_checks(x["text"], strict_ats)
            ats_ok = sum(1 for c in ats if c['ok']); ats_total = len(ats)
            ats_pct = round((ats_ok/ats_total)*100, 0) if ats_total else 0
            rows.append({
                "Resume": x["file"].name,
                "Score(%)": round(score*100, 1),
                "ATS_OK": f"{ats_ok}/{ats_total}",
                "ATS(%)": ats_pct,
                "MatchedKeywords": ", ".join((present + phr_p)[:topn_display]),
                "MissingKeywords": ", ".join((missing + phr_m)[:topn_display]),
                "_ats": ats,
                "_present": present, "_missing": missing,
                "_phr_present": phr_p, "_phr_missing": phr_m,
            })

    # Sorting
    if sort_by == "ATS desc":
        rows.sort(key=lambda r: r["ATS(%)"], reverse=True)
    elif sort_by == "Name asc":
        rows.sort(key=lambda r: r["Resume"].lower())
    else:
        rows.sort(key=lambda r: r["Score(%)"], reverse=True)

    # Summary metrics
    scores = [r["Score(%)"] for r in rows]
    best = max(scores) if scores else 0.0
    avg = round(sum(scores)/len(scores), 1) if scores else 0.0
    m1, m2, m3 = st.columns(3)
    m1.metric("Total Resumes", len(rows))
    m2.metric("Best Score", f"{best}%")
    m3.metric("Average Score", f"{avg}%")

    # Tabs
    tab_results, tab_insights, tab_ats = st.tabs(["📊 Results", "🔎 Insights", "✅ ATS Details"])

    # --- Results Tab ---
    with tab_results:
        df = pd.DataFrame([{k: v for k, v in r.items() if not k.startswith("_")} for r in rows])
        st.caption(f"Shortlist resumes with Score ≥ **{score_threshold}%**")
        shortlist_mask = df["Score(%)"] >= score_threshold if not df.empty else pd.Series(dtype=bool)

        # Show shortlisted chips
        if not df.empty:
            shortlisted = df[shortlist_mask]
            st.markdown(
                "**Shortlisted:** " + (" ".join([
                    f"<span class='chip'>⭐ {n} ({s}%)</span>" for n, s in zip(shortlisted["Resume"], shortlisted["Score(%)"])
                ]) if len(shortlisted) else "<span class='small'>No resumes above threshold.</span>"),
                unsafe_allow_html=True
            )

        # Dataframe with progress bars
        try:
            st.dataframe(
                df,
                use_container_width=True,
                column_config={
                    "Score(%)": st.column_config.ProgressColumn("Score", help="Similarity to JD", format="%d%%", min_value=0, max_value=100),
                    "ATS(%)": st.column_config.ProgressColumn("ATS", help="ATS checks passed", format="%d%%", min_value=0, max_value=100),
                    "Resume": st.column_config.TextColumn("Resume", help="File name"),
                    "ATS_OK": st.column_config.TextColumn("ATS OK", help="Checks passed/total"),
                    "MatchedKeywords": st.column_config.TextColumn("Matched Keywords"),
                    "MissingKeywords": st.column_config.TextColumn("Missing Keywords"),
                },
                hide_index=True,
                height=420,
            )
        except Exception:
            # Older Streamlit fallback
            st.dataframe(df, use_container_width=True, height=420)

        # Downloads
        csv = df.to_csv(index=False).encode("utf-8")
        st.download_button("⬇️ Download All Results (CSV)", data=csv, file_name="cv-screening-results.csv", mime="text/csv", use_container_width=True)

        if not df.empty and shortlist_mask.any():
            csv_short = df[shortlist_mask].to_csv(index=False).encode("utf-8")
            st.download_button("⭐ Download Shortlist (CSV)", data=csv_short, file_name="cv-shortlist.csv", mime="text/csv", use_container_width=True)

    # --- Insights Tab ---
    with tab_insights:
        # Aggregate top missing keywords
        agg_missing = {}
        for r in rows:
            for kw in (r["_missing"] or []):
                agg_missing[kw] = agg_missing.get(kw, 0) + 1
        top_missing = sorted(agg_missing.items(), key=lambda x: x[1], reverse=True)[:20]

        if top_missing:
            st.subheader("Top Missing Keywords (across resumes)")
            st.markdown("".join([f"<span class='kw missing'>{w} • {c}</span>" for w, c in top_missing]), unsafe_allow_html=True)
            # Small chart
            try:
                chart_df = pd.DataFrame(top_missing, columns=["Keyword", "Count"]).set_index("Keyword")
                st.bar_chart(chart_df)
            except Exception:
                pass
        else:
            st.info("No missing keywords found across resumes.")

    # --- ATS Details Tab ---
    with tab_ats:
        for r in rows:
            st.markdown(f"**{r['Resume']}** — Score: **{r['Score(%)']}%** | ATS: **{r['ATS_OK']}**")
            badges = " ".join([
                f"<span class='badge {'ok' if c['ok'] else 'warn'}'>{'✅' if c['ok'] else '❌'} {c['name']}</span>" for c in r["_ats"]
            ])
            st.markdown(badges, unsafe_allow_html=True)
            st.markdown("**Matched:** " + "".join([f"<span class='kw'>{w}</span>" for w in (r['_present'] + r['_phr_present'])[:20]]), unsafe_allow_html=True)
            st.markdown("**Missing:** " + "".join([f"<span class='kw missing'>{w}</span>" for w in (r['_missing'] + r['_phr_missing'])[:20]]), unsafe_allow_html=True)
            st.divider()

# Footer
st.markdown("---")
st.caption("© HR Mobineers • Streamlit app. Need Hindi UI or custom branding? Ping to enable.")
