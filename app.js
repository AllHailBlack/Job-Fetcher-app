// Full Expanded Version — Daily Job Fetcher + AI Resume Toolkit
// Filename: app.js
// Tech: Node.js, Express
// Features:
//  - Daily job fetch (max 30) from a public API (example: Remotive)
//  - Keyword extraction from job descriptions
//  - Hybrid semantic + keyword resume matching
//  - PDF/DOCX resume upload and parsing
//  - GPT resume tips, job-specific resume tailoring, and cover letter generation
//  - Simple REST API for all features

/*
  USAGE:
  1) Install Node.js
  2) npm init -y
  3) npm install express axios node-cron natural multer pdf-parse mammoth openai cors
  4) Set environment variable OPENAI_API_KEY for GPT features
  5) node app.js

  NOTE: This is an expanded single-file example. For production, split into routes/controllers/services.
*/
require('dotenv').config();

const express = require('express');
const axios = require('axios');
const cron = require('node-cron');
const natural = require('natural');
const cors = require('cors');
const multer = require('multer');
const pdfParse = require('pdf-parse');
const mammoth = require('mammoth');
const fs = require('fs');
const path = require('path');
const { v4: uuidv4 } = require('uuid');

// OpenAI client (official SDK)
const OpenAI = require('openai');
const openai = new OpenAI({ apiKey: process.env.OPENAI_API_KEY });

/* =========================
   CONFIGURATION
   ========================= */
const PORT = process.env.PORT || 3000;
const JOB_TYPE = 'concept artist, 2d character artist';
const MAX_JOBS = 30; // limit per fetch
const JOB_API_URL = 'https://remotive.com/api/remote-jobs?search='; // example public API

/* =========================
   EXPRESS + MIDDLEWARE
   ========================= */
app.use(express.static('public'));
   const app = express();
app.use(cors());
app.use(express.json({ limit: '1mb' }));
app.use(express.urlencoded({ extended: true }));

const upload = multer({ dest: 'uploads/' });

/* =========================
   IN-MEMORY STORAGE (for demo)
   ========================= */
// For production, replace with a database
let storedJobs = [];
let lastFetchAt = null;

/* =========================
   UTIL: TEXT CLEANING & TOKENIZATION
   ========================= */
const tokenizer = new natural.WordTokenizer();
const stopWords = natural.stopwords || [];

function cleanAndTokenize(text) {
  if (!text) return [];
  const lowered = text.replace(/[]+/g, ' ').replace(/[^a-zA-Z0-9 #+\-_.]/g, ' ').toLowerCase();
  const tokens = tokenizer.tokenize(lowered);
  return tokens.filter(t => t.length > 2 && !stopWords.includes(t));
}

/* =========================
   KEYWORD EXTRACTION
   ========================= */
function extractKeywords(text, limit = 20) {
  const ART_KEYWORDS = new Set([
    'concept','artist','character','2d','illustration','digital','painting','anatomy',
    'stylized','environment','sketch','photoshop','procreate','color','lighting',
    'render','visual','design','composition','story','ideation'
  ]);

  const tokens = cleanAndTokenize(text);
  const freq = {};

  tokens.forEach(t => {
    const weight = ART_KEYWORDS.has(t) ? 4 : 1;
    freq[t] = (freq[t] || 0) + weight;
  });

  const sorted = Object.entries(freq).sort((a, b) => b[1] - a[1]);
  return sorted.slice(0, limit).map(([term]) => term);
}


/* =========================
   SEMANTIC SIMILARITY (TF-IDF + COSINE)
   - Implemented locally to avoid external vector APIs
   - Works by building combined vocabulary for two documents,
     computing TF-IDF for each term, then cosine similarity
   ========================= */
function buildTermFreq(docTokens) {
  const tf = {};
  docTokens.forEach(t => (tf[t] = (tf[t] || 0) + 1));
  // convert to term frequency (raw counts). We'll normalize later.
  return tf;
}

function computeIdf(allDocsTokens) {
  const df = {}; // document frequency
  const N = allDocsTokens.length;
  allDocsTokens.forEach(tokens => {
    const seen = new Set(tokens);
    seen.forEach(t => (df[t] = (df[t] || 0) + 1));
  });
  const idf = {};
  Object.keys(df).forEach(term => {
    idf[term] = Math.log((N + 1) / (df[term] + 1)) + 1; // smoothing
  });
  return idf;
}

function tfidfVector(docTokens, idf, vocab) {
  const tf = buildTermFreq(docTokens);
  // build vector in order of vocab
  const vec = vocab.map(term => {
    const termFreq = tf[term] || 0;
    // use raw TF multiplied by IDF
    return termFreq * (idf[term] || 0);
  });
  return vec;
}

function dotProduct(a, b) {
  let s = 0;
  for (let i = 0; i < a.length; i++) s += a[i] * b[i];
  return s;
}

function magnitude(a) {
  let s = 0;
  for (let i = 0; i < a.length; i++) s += a[i] * a[i];
  return Math.sqrt(s);
}

function cosineSimilarity(vecA, vecB) {
  const denom = magnitude(vecA) * magnitude(vecB);
  if (denom === 0) return 0;
  return dotProduct(vecA, vecB) / denom;
}

function semanticSimilarity(resumeText, jobDescription) {
  const resumeTokens = cleanAndTokenize(resumeText);
  const jobTokens = cleanAndTokenize(jobDescription);
  if (resumeTokens.length === 0 || jobTokens.length === 0) return 0;

  const allDocs = [resumeTokens, jobTokens];
  const idf = computeIdf(allDocs);
  // vocabulary is union of tokens
  const vocab = Array.from(new Set([...resumeTokens, ...jobTokens]));

  const vecA = tfidfVector(resumeTokens, idf, vocab);
  const vecB = tfidfVector(jobTokens, idf, vocab);

  const sim = cosineSimilarity(vecA, vecB);
  // map similarity [0,1] to percentage [0,100]
  return Math.round(sim * 100);
}

/* =========================
   HYBRID MATCH SCORE
   - Combines keyword overlap + semantic similarity
   - Tunable weights
   ========================= */
function keywordMatchPercentage(resumeText, jobKeywords) {
  if (!jobKeywords || jobKeywords.length === 0) return 0;
  const resumeTokens = new Set(cleanAndTokenize(resumeText));
  let matches = 0;
  jobKeywords.forEach(k => { if (resumeTokens.has(k)) matches++; });
  return Math.round((matches / jobKeywords.length) * 100);
}

function finalMatchScore(resumeText, job) {
  const kwScore = keywordMatchPercentage(resumeText, job.keywords || []); // 0-100
  const semScore = semanticSimilarity(resumeText, job.description || ''); // 0-100
  const weightKw = 0.4;
  const weightSem = 0.6;
  const final = Math.round(kwScore * weightKw + semScore * weightSem);
  return { final, kwScore, semScore };
}

/* =========================
   JOB FETCHING
   - Fetches job list from external API
   - Extracts keywords
   - Stores up to MAX_JOBS
   ========================= */
async function fetchJobs(jobType = JOB_TYPE) {
  try {
    const url = `${JOB_API_URL}${encodeURIComponent(jobType)}`;
    const resp = await axios.get(url, { timeout: 15000 });
    const jobsRaw = (resp.data && resp.data.jobs) || [];

    const jobs = jobsRaw.slice(0, MAX_JOBS).map(j => {
      const description = j.description || j.job_description || j.contents || '';
      const keywords = extractKeywords(description, 25);
      return {
        id: j.id || uuidv4(),
        title: j.title || j.name || 'Unknown Title',
        company: j.company_name || j.company || 'Unknown Company',
        description,
        url: j.url || j.job_url || null,
        location: j.candidate_required_location || j.location || 'Remote',
        keywords
      };
    });

    storedJobs = jobs;
    lastFetchAt = new Date().toISOString();
    console.log(`[${new Date().toISOString()}] Fetched ${jobs.length} jobs for '${jobType}'`);
    return jobs.length;
  } catch (err) {
    console.error('Job fetch failed:', err.message || err);
    return 0;
  }
}

// initial fetch
fetchJobs().catch(() => {});

// schedule daily fetch at 09:00 server time
cron.schedule('30 10 * * *', () => {
  console.log('Running scheduled job fetch...');
  fetchJobs().catch(() => {});
});

/* =========================
   EXPRESS ROUTES
   ========================= */

// Health
app.get('/health', (req, res) => res.json({ ok: true, lastFetchAt }));

// Get stored jobs
app.get('/jobs', (req, res) => res.json({ count: storedJobs.length, jobs: storedJobs }));

/*
  Resume matching via POST /match
  Body: { resumeText: string }
*/
app.post('/match', (req, res) => {
  try {
    const resumeText = req.body.resumeText || '';
    if (!resumeText) return res.status(400).json({ error: 'resumeText is required' });

    const matches = storedJobs.map(job => {
      const scores = finalMatchScore(resumeText, job);
      return {
        id: job.id,
        title: job.title,
        company: job.company,
        matchScore: `${scores.final}%`,
        keywordScore: `${scores.kwScore}%`,
        semanticScore: `${scores.semScore}%`,
        url: job.url,
        keywords: job.keywords
      };
    }).sort((a, b) => parseInt(b.matchScore) - parseInt(a.matchScore));

    res.json({ results: matches });
  } catch (err) {
    res.status(500).json({ error: 'Matching failed' });
  }
});

/*
  Upload resume file and return matching results
  - Accepts multipart/form-data with file field 'resume'
  - Supports PDF and DOCX (and older Word if needed)
*/
app.post('/upload-resume', upload.single('resume'), async (req, res) => {
  try {
    if (!req.file) return res.status(400).json({ error: 'No file uploaded' });

    const filePath = req.file.path;
    let resumeText = '';

    // Basic mime checks
    const pdfMime = 'application/pdf';
    const docxMime = 'application/vnd.openxmlformats-officedocument.wordprocessingml.document';
    const docMime = 'application/msword';

    if (req.file.mimetype === pdfMime) {
      const buffer = fs.readFileSync(filePath);
      const data = await pdfParse(buffer);
      resumeText = data.text || '';
    } else if (req.file.mimetype === docxMime || req.file.mimetype === docMime) {
      const result = await mammoth.extractRawText({ path: filePath });
      resumeText = result.value || '';
    } else {
      // try to process by extension as a fallback
      const ext = path.extname(req.file.originalname).toLowerCase();
      if (ext === '.pdf') {
        const buffer = fs.readFileSync(filePath);
        const data = await pdfParse(buffer);
        resumeText = data.text || '';
      } else if (ext === '.docx' || ext === '.doc') {
        const result = await mammoth.extractRawText({ path: filePath });
        resumeText = result.value || '';
      } else {
        fs.unlinkSync(filePath);
        return res.status(400).json({ error: 'Unsupported file type' });
      }
    }

    // cleanup temporary upload
    try { fs.unlinkSync(filePath); } catch (e) { /* ignore */ }

    // produce matching results
    const results = storedJobs.map(job => {
      const scores = finalMatchScore(resumeText, job);
      return {
        id: job.id,
        title: job.title,
        company: job.company,
        matchScore: `${scores.final}%`,
        keywordScore: `${scores.kwScore}%`,
        semanticScore: `${scores.semScore}%`,
        url: job.url,
        keywords: job.keywords
      };
    }).sort((a, b) => parseInt(b.matchScore) - parseInt(a.matchScore));

    res.json({ extractedTextPreview: resumeText.substring(0, 300), results });
  } catch (err) {
    console.error('Upload error:', err);
    res.status(500).json({ error: 'Resume processing failed' });
  }
});

/* =========================
   GPT-BASED FEATURES
   - Uses OpenAI (ensure OPENAI_API_KEY set)
   - resume-tips: general improvement tips
   - tailor-resume: job-specific resume rewrite
   - cover-letter: generate cover letter aligned to job
   ========================= */

// helper to call OpenAI Chat Completions (small wrapper)
async function callOpenAI(prompt, system = 'You are an expert career coach.') {
  if (!process.env.OPENAI_API_KEY) {
    throw new Error('OpenAI API key not configured');
  }

  // Use the Chat Completions API
  const response = await openai.chat.completions.create({
    model: 'gpt-4o-mini',
    messages: [
      { role: 'system', content: system },
      { role: 'user', content: prompt }
    ],
    max_tokens: 800
  });

  const text = response.choices && response.choices[0] && response.choices[0].message && response.choices[0].message.content;
  return text || '';
}

// Resume improvement tips
app.post('/resume-tips', async (req, res) => {
  try {
    const resumeText = req.body.resumeText || '';
    if (!resumeText) return res.status(400).json({ error: 'resumeText is required' });

    const prompt = `Analyze the resume below and provide concise, actionable improvement tips. Focus on ATS optimization, strong impact statements, formatting, relevant skill highlighting, and measurable results. Provide bullets and examples where possible.

RESUME:
${resumeText}`;

    const tips = await callOpenAI(prompt);
    res.json({ tips });
  } catch (err) {
    console.error('GPT tips error:', err.message || err);
    res.status(500).json({ error: 'Failed to generate tips' });
  }
});

// Job-specific resume tailoring (rewrite the resume to fit the job)
app.post('/tailor-resume', async (req, res) => {
  try {
    const resumeText = req.body.resumeText || '';
    const jobDescription = req.body.jobDescription || '';
    if (!resumeText || !jobDescription) return res.status(400).json({ error: 'resumeText and jobDescription are required' });

    const prompt = `You are an expert resume writer and ATS specialist. Take the RESUME and rewrite it so that it's optimized for the following JOB DESCRIPTION. Emphasize relevant skills, reorder sections if necessary, convert responsibilities into achievement bullets (with metrics where plausible), and output the final resume in plain text.

JOB DESCRIPTION:
${jobDescription}

RESUME:
${resumeText}`;

    const tailored = await callOpenAI(prompt);
    res.json({ tailoredResume: tailored });
  } catch (err) {
    console.error('Tailor resume error:', err.message || err);
    res.status(500).json({ error: 'Failed to tailor resume' });
  }
});

// Cover letter generator
app.post('/cover-letter', async (req, res) => {
  try {
    const resumeText = req.body.resumeText || '';
    const jobDescription = req.body.jobDescription || '';
    const companyName = req.body.companyName || 'the company';

    if (!resumeText || !jobDescription) return res.status(400).json({ error: 'resumeText and jobDescription are required' });

    const prompt = `Write a professional, personalized cover letter addressed to the hiring manager at ${companyName}. Use the applicant's resume and the job description to highlight the best-fit skills and achievements. Keep it concise (max 4 short paragraphs). Provide a strong closing and call-to-action.`;

    const coverLetter = await callOpenAI(`${prompt}

JOB DESCRIPTION:
${jobDescription}

RESUME:
${resumeText}`);
    res.json({ coverLetter });
  } catch (err) {
    console.error('Cover letter error:', err.message || err);
    res.status(500).json({ error: 'Failed to generate cover letter' });
  }
});

/* =========================
   SIMPLE UI ENDPOINTS (for quick testing)
   ========================= */

app.get('/', (req, res) => {
  res.sendFile(path.join(__dirname, 'public', 'index.html'));
});


/* =========================
   START SERVER
   ========================= */
app.listen(PORT, () => {
  console.log(`App listening on port ${PORT}`);
  console.log('Make sure OPENAI_API_KEY is set for GPT features.');
});
