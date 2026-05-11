# UIU Academic RAG — Deployment Guide

## Option A: GitHub Codespaces (Free, For Testing)

1. Push repo to GitHub
2. Open Codespaces → devcontainer auto-configures
3. Fill in backend/.env with your keys
4. Run: cd infra/docker && docker-compose up -d
5. Open Port 5173 in browser

## Option B: Railway + Vercel (Production, ~$5/month)

### Backend on Railway
1. Go to railway.app → New Project → Deploy from GitHub
2. Select your repo
3. Add these environment variables in Railway dashboard:
   - SECRET_KEY = (any 32+ char random string)
   - GROQ_API_KEY = gsk_your_key
   - POSTGRES_PASSWORD = (Railway auto-sets this)
   - ENVIRONMENT = production
   - RESTRICT_TO_UIU_EMAIL = true
   - UIU_EMAIL_DOMAIN = uiu.ac.bd
   - GOOGLE_OAUTH_CLIENT_ID = your_client_id
   - GOOGLE_OAUTH_CLIENT_SECRET = your_client_secret
   - GOOGLE_OAUTH_REDIRECT_URI = https://your-frontend.vercel.app
4. Add PostgreSQL service → Railway auto-connects
5. Add Redis service → Railway auto-connects
6. Run pgvector: In Railway PostgreSQL shell: CREATE EXTENSION IF NOT EXISTS vector;

### Frontend on Vercel (Free)
1. Go to vercel.com → New Project → Import your repo
2. Set Root Directory: frontend
3. Add environment variable:
   - VITE_API_BASE_URL = https://your-railway-backend.railway.app/api/v1
4. Deploy

### Google OAuth Setup (Required for login)
1. Go to console.cloud.google.com
2. APIs & Services → Credentials → Create OAuth 2.0 Client ID
3. Authorized redirect URIs: https://your-frontend.vercel.app
4. Copy Client ID and Client Secret to Railway env vars

## First Admin User
After deploying, make yourself admin:
In Railway PostgreSQL shell:
  UPDATE users SET is_admin = true WHERE email = 'your@uiu.ac.bd';

## Upload Course Materials
1. Login as admin
2. Click "Upload Material" in sidebar
3. Upload PDF/PPTX/DOCX files
4. System processes automatically in background

## Environment Variables Reference
| Variable | Required | Description |
|---|---|---|
| SECRET_KEY | YES | Random 32+ char string |
| GROQ_API_KEY | YES | Free from console.groq.com |
| POSTGRES_PASSWORD | YES | Database password |
| GOOGLE_OAUTH_CLIENT_ID | YES | For student login |
| GOOGLE_OAUTH_CLIENT_SECRET | YES | For student login |
| RESTRICT_TO_UIU_EMAIL | NO | true = only @uiu.ac.bd |
| UIU_EMAIL_DOMAIN | NO | Default: uiu.ac.bd |
| LLM_MODEL | NO | Default: llama-3.1-8b-instant |
