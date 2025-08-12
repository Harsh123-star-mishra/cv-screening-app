// ---- Utility: Stopwords (short list) ----
const STOPWORDS = new Set(`a,an,the,of,in,on,for,with,and,or,to,from,by,at,as,is,are,was,were,be,been,being,this,that,these,those,will,shall,can,could,should,would,may,might,about,into,over,per,via,within,without,across,through,up,down,out,off,so,if,then,than,too,very,more,most,least,not,no,yes,also,using,use,used,using,including,include,includes,ability,experience,years,year` .split(',').map(s=>s.trim()));

// ---- Basic text cleaners ----
function cleanText(t) {
  return (t || '')
    .replace(/ /g, ' ')
    .replace(/[
]+/g, ' ')
    .replace(/[^a-zA-Z0-9+#\.\-/ ]/g, ' ')
    .replace(/\s{2,}/g, ' ')
    .trim();
}

function tokenize(t){
  return cleanText(t).toLowerCase()
    .split(/[^a-z0-9+#\.\-/]+/)
    .filter(w => w && !STOPWORDS.has(w) && w.length > 1);
}

// Naive phrase extraction: keep 2-3grams that appear in JD
function extractPhrases(tokens, maxLen=3){
  const phrases = new Set();
  for(let n=2; n<=maxLen; n++){
    for(let i=0; i<=tokens.length-n; i++){
      const p = tokens.slice(i, i+n).join(' ');
      if(!p.split(' ').some(w => STOPWORDS.has(w))) phrases.add(p);
    }
  }
  return phrases;
}

// TF map
function termFreq(tokens){
  const tf = new Map();
  tokens.forEach(t => tf.set(t, (tf.get(t)||0)+1));
  const total = tokens.length || 1;
  for(const [k,v] of tf) tf.set(k, v/total);
  return tf;
}

// IDF from collection of docs
function inverseDocFreq(docTokenArrays){
  const N = docTokenArrays.length;
  const df = new Map();
  docTokenArrays.forEach(arr => {
    const uniq = new Set(arr);
    uniq.forEach(t => df.set(t, (df.get(t)||0)+1));
  });
  const idf = new Map();
  for(const [t,d] of df) idf.set(t, Math.log((N+1)/(d+0.5))+1); // smooth
  return idf;
}

function buildVector(tf, idf){
  const vec = new Map();
  for(const [t, w] of tf){
    const id = idf.get(t) || 0;
    vec.set(t, w*id);
  }
  return vec;
}

function cosineSim(v1, v2){
  let dot=0, n1=0, n2=0;
  const keys = new Set([...v1.keys(), ...v2.keys()]);
  keys.forEach(k => {
    const a = v1.get(k)||0; const b = v2.get(k)||0;
    dot += a*b; n1 += a*a; n2 += b*b;
  });
  if(n1===0 || n2===0) return 0;
  return dot / (Math.sqrt(n1)*Math.sqrt(n2));
}

// ATS checks
function atsChecks(text, strict=false){
  const checks = [];
  const hasEmail = /[A-Z0-9._%+-]+@[A-Z0-9.-]+\.[A-Z]{2,}/i.test(text);
  const hasPhone = /(\+\d{1,3}[- ]?)?\d{10}/.test(text.replace(/\D/g, ' ').replace(/\s+/g, ' '));
  const hasLinked = /(linkedin\.com\/[A-Za-z0-9_\-/]+)/i.test(text);
  const hasGithub = /(github\.com\/[A-Za-z0-9_\-/]+)/i.test(text);
  const hasSections = /(education|experience|skills|projects|summary|objective)/i.test(text);
  const fileOk = true; // Determined earlier per file type
  const bulletsOk = /•|\-|\*/.test(text) || /
\s*\-/.test(text);
  const years = (text.match(/(\d+)\+?\s*(years|yrs)/gi)||[]);

  checks.push({name:'Email', ok: hasEmail});
  checks.push({name:'Phone', ok: hasPhone});
  checks.push({name:'LinkedIn', ok: hasLinked});
  checks.push({name:'GitHub', ok: hasGithub});
  checks.push({name:'Sections present', ok: hasSections});
  checks.push({name:'Bullets used', ok: bulletsOk});
  if(strict){
    checks.push({name:'Experience mentioned', ok: years.length>0});
  }
  return checks;
}

// Keyword coverage: top-K JD tokens + phrases matched in resume
function keywordCoverage(jdTokens, resumeTokens, usePhrases){
  const jdTF = termFreq(jdTokens);
  const sorted = [...jdTF.entries()].sort((a,b)=>b[1]-a[1]);
  const topTokens = sorted
    .filter(([w])=>!/^(year|years|experience|candidate|responsibilities|responsibility)$/i.test(w))
    .slice(0,30)
    .map(([w])=>w);
  const resSet = new Set(resumeTokens);
  const present = topTokens.filter(t=>resSet.has(t));
  const missing = topTokens.filter(t=>!resSet.has(t));

  let phrasePresent = [], phraseMissing = [];
  if(usePhrases){
    const phrases = [...extractPhrases(jdTokens)].filter(p=>p.split(' ').length>1).slice(0,20);
    const resText = resumeTokens.join(' ');
    phrasePresent = phrases.filter(p => resText.includes(p));
    phraseMissing = phrases.filter(p => !resText.includes(p));
  }
  return {present, missing, phrasePresent, phraseMissing};
}

// ---- File parsing (PDF / DOCX) ----
async function parsePDF(file){
  const arrayBuffer = await file.arrayBuffer();
  const pdf = await pdfjsLib.getDocument({data: arrayBuffer}).promise;
  let fullText = '';
  for(let i=1;i<=pdf.numPages;i++){
    const page = await pdf.getPage(i);
    const content = await page.getTextContent();
    const strings = content.items.map(it=>it.str).join(' ');
    fullText += '
' + strings;
  }
  return cleanText(fullText);
}

async function parseDOCX(file){
  const arrayBuffer = await file.arrayBuffer();
  const result = await window.mammoth.extractRawText({arrayBuffer});
  return cleanText(result.value || '');
}

async function fileToText(file){
  const ext = file.name.toLowerCase().split('.').pop();
  if(ext === 'pdf') return await parsePDF(file);
  if(ext === 'docx') return await parseDOCX(file);
  throw new Error('Unsupported file: '+file.name);
}

// ---- UI logic ----
const fileInput = document.getElementById('fileInput');
const dropZone = document.getElementById('dropZone');
const fileListEl = document.getElementById('fileList');
const analyzeBtn = document.getElementById('analyzeBtn');
const downloadCsvBtn = document.getElementById('downloadCsvBtn');
const statusEl = document.getElementById('status');
const resultsEl = document.getElementById('results');
const jdTextEl = document.getElementById('jdText');
const usePhrasesEl = document.getElementById('usePhrases');
const strictATSEl = document.getElementById('strictATS');

let files = [];
let lastResults = [];

function renderFileList(){
  if(files.length===0){ fileListEl.textContent = 'No files added yet.'; return; }
  fileListEl.innerHTML = files.map(f=>`<div>• ${f.name} <span class="small">(${Math.round(f.size/1024)} KB)</span></div>`).join('');
}

function setStatus(msg){ statusEl.textContent = msg || ''; }

function setResultsTable(rows){
  if(rows.length===0){ resultsEl.innerHTML = '<p class="small">No results yet.</p>'; return; }
  const html = [`<table>`,
    `<thead><tr><th>Resume</th><th>Score</th><th>ATS</th><th>Matched Keywords</th><th>Missing Keywords</th></tr></thead>`,
    `<tbody>`,
  ];
  for(const r of rows){
    const atsBadges = r.ats.map(c=>`<span class="badge ${c.ok?'ok':'warn'}">${c.name}</span>`).join(' ');
    html.push(`<tr>
      <td>${r.name}</td>
      <td class="score">${(r.score*100).toFixed(1)}%</td>
      <td>${atsBadges}</td>
      <td class="kws">${r.present.slice(0,20).map(k=>`<span>${k}</span>`).join(' ')}</td>
      <td class="kws">${r.missing.slice(0,20).map(k=>`<span>${k}</span>`).join(' ')}</td>
    </tr>`);
  }
  html.push(`</tbody></table>`);
  resultsEl.innerHTML = html.join('
');
}

function toCSV(rows){
  const header = ['Resume','Score(0-1)','ATS_OK(Count)','MatchedKeywords','MissingKeywords'];
  const lines = [header.join(',')];
  for(const r of rows){
    const atsOk = r.ats.filter(x=>x.ok).length + '/' + r.ats.length;
    const line = [
      '"'+r.name.replace(/"/g,'""')+'"',
      r.score.toFixed(4),
      '"'+atsOk+'"',
      '"'+r.present.join(' | ').replace(/"/g,'""')+'"',
      '"'+r.missing.join(' | ').replace(/"/g,'""')+'"'
    ].join(',');
    lines.push(line);
  }
  return lines.join('
');
}

function download(filename, text){
  const blob = new Blob([text], {type: 'text/csv;charset=utf-8;'});
  const url = URL.createObjectURL(blob);
  const a = document.createElement('a');
  a.href = url; a.download = filename; a.click();
  setTimeout(()=>URL.revokeObjectURL(url), 1000);
}

// Drag & drop events
['dragenter','dragover'].forEach(evt => dropZone.addEventListener(evt, e=>{ e.preventDefault(); dropZone.classList.add('dragover'); }));
['dragleave','drop'].forEach(evt => dropZone.addEventListener(evt, e=>{ e.preventDefault(); dropZone.classList.remove('dragover'); }));
dropZone.addEventListener('drop', e=>{ files = [...files, ...Array.from(e.dataTransfer.files)]; renderFileList(); });
dropZone.addEventListener('click', ()=> fileInput.click());
fileInput.addEventListener('change', e=>{ files = [...files, ...Array.from(e.target.files)]; renderFileList(); fileInput.value = ''; });

analyzeBtn.addEventListener('click', async ()=>{
  const jd = jdTextEl.value.trim();
  if(!jd){ alert('Please paste a Job Description first.'); return; }
  if(files.length===0){ alert('Please add at least one resume file (PDF/DOCX).'); return; }

  setStatus('Parsing resumes...');
  const usePhrases = usePhrasesEl.checked;
  const strictATS = strictATSEl.checked;

  const jdTokens = tokenize(jd);
  const texts = [];
  for(const f of files){
    try{
      const t = await fileToText(f);
      texts.push({file: f, text: t});
    } catch(err){
      console.error(err);
      texts.push({file: f, text: ''});
    }
  }

  setStatus('Scoring...');
  const allTokens = [jdTokens, ...texts.map(x=>tokenize(x.text))];
  const idf = inverseDocFreq(allTokens);
  const jdVec = buildVector(termFreq(jdTokens), idf);

  const rows = [];
  for(let i=0;i<texts.length;i++){
    const tokens = allTokens[i+1];
    const vec = buildVector(termFreq(tokens), idf);
    const score = cosineSim(jdVec, vec);
    const cov = keywordCoverage(jdTokens, tokens, usePhrases);
    const ats = atsChecks(texts[i].text, strictATS);
    rows.push({
      name: texts[i].file.name,
      score, present: cov.present.concat(cov.phrasePresent).slice(0,30),
      missing: cov.missing.concat(cov.phraseMissing).slice(0,30),
      ats
    });
  }

  rows.sort((a,b)=>b.score-a.score);
  lastResults = rows;
  setResultsTable(rows);
  downloadCsvBtn.disabled = rows.length===0;
  setStatus('Done');
});

downloadCsvBtn.addEventListener('click', ()=>{
  if(!lastResults.length) return;
  const csv = toCSV(lastResults);
  download('cv-screening-results.csv', csv);
});
