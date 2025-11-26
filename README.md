# Job-Fetcher-app
AI-powered job matching and resume analysis web app
# AI Job Matcher & Resume Toolkit

An advanced Node.js web application that fetches jobs daily, analyzes job descriptions, matches resumes, extracts keywords, and generates AI-powered resume improvements, tailored resumes, and personalized cover letters. Includes PDF/DOCX resume parsing and semantic matching.

---

## 🚀 Features

### **1. Daily Job Fetching**

* Pulls up to 30 jobs daily at **10:30 AM**
* Fetches "concept artist / 2D character artist" roles by default
* Extracts keywords for matching

### **2. Resume Matching (Hybrid Model)**

* Keyword analysis
* Semantic matching using TF-IDF + cosine similarity
* Combined weighted score

### **3. PDF & DOCX Resume Upload**

* Extracts resume text using pdf-parse and mammoth
* Matches resume automatically to fetched jobs

### **4. AI-Powered Tools**

* Resume improvement suggestions
* Job-specific resume tailoring
* Cover letter generation
* Uses OpenAI API

### **5. REST API Endpoints**

* `/jobs` – List fetched jobs
* `/match` – Match resume text
* `/upload-resume` – Upload file for matching
* `/resume-tips` – Improve resume with AI
* `/tailor-resume` – Tailor resume to job
* `/cover-letter` – Generate cover letter

---

## 📦 Installation

```bash
npm install
```

Required dependencies include:

```
express, axios, node-cron, natural, multer,
pdf-parse, mammoth, openai, cors, uuid
```

---

## 🔧 Environment Variables

Create a `.env` file:

```
OPENAI_API_KEY=your_openai_key_here
```

---

## ▶️ Run the App

Start normally:

```bash
npm start
```

Start with auto-reload (if installed):

```bash
npm run dev
```

Default port: **3000**

---

## 🌐 Deployment

You can host this app using:

* Render (recommended)
* Railway
* Fly.io
* DigitalOcean

Push your project to GitHub and connect your hosting provider.

---

## 🧪 Example Requests

### Match Resume

```json
POST /match
{
  "resumeText": "I create stylized concept art and 2D character designs..."
}
```

### Tailor Resume

```json
POST /tailor-resume
{
  "resumeText": "...",
  "jobDescription": "We are seeking a concept artist..."
}
```

### Generate Cover Letter

```json
POST /cover-letter
{
  "resumeText": "...",
  "jobDescription": "...",
  "companyName": "Riot Games"
}
```

---

## 📁 Project Structure

```
ai-job-matcher/
│
├── app.js
├── package.json
├── package-lock.json
├── uploads/
├── README.md
└── .gitignore
```

---

## 📜 License

MIT

---

## 👤 Author

Your Name

---

If you want, I can also create:

* `render.yaml` for deployment
* A frontend UI
* Dockerfile
* Modular version (routes/controllers/services)
  Just let me know!
