# 🚀 BLIND Platform: Production Deployment & Git Push Guide

This document provides exact step-by-step instructions and code snippets to push your local changes to GitHub and manually deploy the full stack on **Render** (Backend) and **Vercel** (Frontend).

---

## 📦 1. Git Push (Automated or Manual)

### Option A: Use the Automation Script (Recommended for Windows)
Open your PowerShell terminal in the project root and run:
```powershell
.\deploy.ps1
```
This script will automatically stage all modified files, prompt you for a commit message, and push your branch to `origin main`.

### Option B: Manual Git Commands
If you prefer running git commands manually:
```bash
# Stage all safety remediation changes and deployment configs
git add .

# Commit with a descriptive safety case message
git commit -m "feat: safety case remediation, architecture isolation, and production deployment configs"

# Push to your remote GitHub repository
git push origin main
```

---

## 🐍 2. Manual Deployment on Render (Flask / Socket.IO Backend)

We have created [`render.yaml`](file:///c:/Users/ayush/OneDrive/Desktop/BLIND/render.yaml) and [`requirements_prod.txt`](file:///c:/Users/ayush/OneDrive/Desktop/BLIND/requirements_prod.txt) in your repository to configure Gunicorn and Eventlet with thread-locked OpenCV/PyTorch inference.

### Step-by-Step Instructions:
1. Go to your [Render Dashboard](https://dashboard.render.com).
2. **If already linked via GitHub / Blueprint**:
   - Click on your backend web service (`blind-ai-backend`).
   - Click the **Manual Deploy** dropdown button in the top right.
   - Select **Deploy latest commit**.
3. **If setting up for the first time**:
   - Click **+ New** -> **Blueprint** (or **Web Service**).
   - Connect your GitHub repository `FOX-KNIGHT/Blind`.
   - Render will automatically detect [`render.yaml`](file:///c:/Users/ayush/OneDrive/Desktop/BLIND/render.yaml) and apply the following settings:
     - **Build Command**: `pip install --upgrade pip && pip install -r requirements_prod.txt`
     - **Start Command**: `gunicorn -k custom_worker.CustomEventletWorker -w 1 -b 0.0.0.0:$PORT app:app --timeout 120`
4. **Environment Variables on Render**:
   - `PYTHONUNBUFFERED` = `1`
   - `CORS_ORIGINS` = `*` (or your Vercel URL once deployed)
   - `LLM_API_KEY` = `<Your-Gemini-or-OpenAI-Key>` (optional, for LLM spatial reasoning)
5. **Copy Backend URL**: Once deployment turns green, copy your Render URL (e.g., `https://blind-ai-backend.onrender.com`).

---

## ⚡ 3. Manual Deployment on Vercel (Next.js Cyber-Cockpit Frontend)

We have created [`vercel.json`](file:///c:/Users/ayush/OneDrive/Desktop/BLIND/vercel.json) and [`frontend/vercel.json`](file:///c:/Users/ayush/OneDrive/Desktop/BLIND/frontend/vercel.json) to optimize Next.js routing and build caching.

### Option A: Vercel Dashboard (GitHub Integration)
1. Go to your [Vercel Dashboard](https://vercel.com/dashboard).
2. **If project is already imported**:
   - Click on your project (`blind-frontend`).
   - Go to the **Deployments** tab.
   - Find your latest commit from GitHub, click the three dots (`...`) on the right, and click **Redeploy**.
3. **If importing for the first time**:
   - Click **Add New...** -> **Project** -> Import `FOX-KNIGHT/Blind`.
   - **CRITICAL STEP**: Under **Root Directory**, click **Edit** and select `frontend`.
   - Under **Environment Variables**, add:
     - `NEXT_PUBLIC_BACKEND_URL` = `<Your-Render-Backend-URL>` (e.g., `https://blind-ai-backend.onrender.com`)
   - Click **Deploy**.

### Option B: Deploy via Vercel CLI (Direct from Terminal)
You can deploy manually directly from your command line without waiting for GitHub hooks:
```bash
# Navigate into the frontend folder
cd frontend

# Install Vercel CLI if you haven't already
npm install -g vercel

# Trigger production deployment
npx vercel --prod
```
When prompted by the CLI:
- Set up and deploy? **Y**
- Which scope do you want to deploy to? **Select your account**
- Link to existing project? **N** (or Y if already linked)
- What’s your project’s name? **blind-ai-frontend**
- In which directory is your code located? **./**
- Want to override the settings? **N**

---

## 🛡️ 4. Post-Deployment Verification
Once both services are live:
1. Open your Vercel frontend URL in Chrome/Edge.
2. Check the **Live Telemetry Bar** at the top: confirm that `FPS` and `Latency` metrics start streaming from your Render backend.
3. Turn on audio and confirm that the **330Hz Audible Heartbeat** pulses every 5 seconds.
