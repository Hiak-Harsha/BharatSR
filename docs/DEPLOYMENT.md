# BharatSR (SIH26142) — Cloud & Production Deployment Guide

This guide details step-by-step instructions to deploy the complete **BharatSR** satellite super-resolution system across cloud platforms:
- **Backend**: [Render](https://render.com) (Docker Container Runtime)
- **Frontend**: [Vercel](https://vercel.com) (Next.js Edge & Serverless Platform)
- **Unified Local**: Docker Compose (Single-command deployment)

---

## Architecture Overview

```
                      +---------------------------------------+
                      |           Client Browser              |
                      +---------------------------------------+
                                          |
                                          | HTTPS
                                          v
                      +---------------------------------------+
                      |         Vercel (Next.js 16)           |
                      |   - UI, MapLibre GL, React Query      |
                      |   - /api/* Rewrites Proxy             |
                      +---------------------------------------+
                                          |
                                          | Internal HTTPS Rewrite
                                          v
                      +---------------------------------------+
                      |      Render (FastAPI / Docker)        |
                      |   - PyTorch RCAN / SRCNN Inference    |
                      |   - Hann-window Tiling Engine         |
                      |   - SQLite JobStore (Async Jobs)      |
                      |   - Geospatial GeoTIFF CRS Engine     |
                      +---------------------------------------+
```

---

## 1. Backend Deployment on Render

The repository includes a ready-to-use [`render.yaml`](../render.yaml) blueprint and an optimized [`backend/Dockerfile`](../backend/Dockerfile) with GDAL, PyTorch CPU, and OpenCV.

### Step 1.1: Connect Repository to Render
1. Sign in to [Render Dashboard](https://dashboard.render.com).
2. Click **New +** -> **Blueprint**.
3. Connect your GitHub repository (`BharatSR` / `SIH26142`).
4. Render will detect `render.yaml` and configure the web service automatically.

*(Alternatively, create a **Web Service** manually:)*
- **Runtime**: `Docker`
- **Docker Command Context**: `.` (root directory)
- **Dockerfile Path**: `./backend/Dockerfile`
- **Instance Type**: `Free` or `Starter` (512MB–2GB RAM)
- **Health Check Path**: `/api/health`

### Step 1.2: Configure Environment Variables
Set the following environment variables in the Render Service Settings:

| Variable | Recommended Value | Description |
| :--- | :--- | :--- |
| `PYTHONUNBUFFERED` | `1` | Real-time Python log streaming |
| `PORT` | `10000` | Port assigned by Render |
| `BHARATSR_CORS_ORIGINS` | `https://your-frontend.vercel.app,http://localhost:3000` | Allowed origins (comma-separated) |
| `BHARATSR_MAX_WORKERS` | `2` (Free) / `4` (Starter) | Concurrency limit for background inference |

### Step 1.3: Verify Deployment
Once deployment finishes, open:
`https://your-backend-service.onrender.com/api/health`

Expected response:
```json
{"status": "ok", "service": "bharatsr-backend", "version": "2.0.0"}
```

---

## 2. Frontend Deployment on Vercel

The frontend is a modern Next.js 16 application configured with automatic API proxy rewrites.

### Step 2.1: Import Project to Vercel
1. Sign in to [Vercel Dashboard](https://vercel.com).
2. Click **Add New...** -> **Project**.
3. Select your GitHub repository.
4. In the **Project Settings**:
   - **Framework Preset**: `Next.js`
   - **Root Directory**: `frontend` *(Click Edit and select the `frontend` folder)*
   - **Build Command**: `npm run build` (Default)
   - **Output Directory**: `.next` (Default)
   - **Install Command**: `npm install` (Default)

### Step 2.2: Configure Environment Variables
In the Vercel **Environment Variables** section, add:

| Name | Value | Purpose |
| :--- | :--- | :--- |
| `BACKEND_INTERNAL_URL` | `https://your-backend-service.onrender.com` | Target URL for server-side Next.js `/api/*` rewrites |
| `NEXT_PUBLIC_API_URL` | `https://your-backend-service.onrender.com` | Client-side fallback API address |

> [!NOTE]
> Next.js proxies all `/api/*` requests through server rewrites (`frontend/next.config.ts`). This guarantees that your users will **never experience CORS issues** or mixed-content SSL warnings.

### Step 2.3: Deploy & Verify
Click **Deploy**. Once built:
1. Navigate to `https://your-frontend.vercel.app` (Landing Page).
2. Click **Launch Console** -> `https://your-frontend.vercel.app/console`.
3. Check the header status badge — it will display a green pulsing dot with **System Online**.
4. Run 4x Super-Resolution on `sample_1` or upload a GeoTIFF.

---

## 3. Unified Local Deployment with Docker Compose

To run both services locally in isolated containers:

```bash
# Clone the repository
git clone https://github.com/Hiak-Harsha/BharatSR.git
cd BharatSR

# Build and start all services
docker compose up --build
```

- **Frontend Console**: [http://localhost:3000](http://localhost:3000)
- **Backend API Docs**: [http://localhost:8000/docs](http://localhost:8000/docs)
- **Health Endpoint**: [http://localhost:8000/api/health](http://localhost:8000/api/health)

---

## 4. Local Bare Metal Development

### Backend (Terminal 1)
```bash
# Windows
python -m venv venv
venv\Scripts\activate
pip install -r requirements.txt
pip install -r backend/requirements.txt
python -m uvicorn backend.app.main:app --host 127.0.0.1 --port 8000 --reload

# Linux / macOS
python3 -m venv venv
source venv/bin/activate
pip install -r requirements.txt
pip install -r backend/requirements.txt
python -m uvicorn backend.app.main:app --host 127.0.0.1 --port 8000 --reload
```

### Frontend (Terminal 2)
```bash
cd frontend
npm install
npm run dev
```

Visit [http://localhost:3000](http://localhost:3000).

---

## 5. Verification & Test Suite

Before deploying or submitting, execute the full automated test suite:

```bash
# Backend pytest suite (67 passed tests)
pytest backend/tests -v

# Frontend Vitest test suite (7 passed tests)
cd frontend
npm test

# GeoTIFF Engine & Projection Verification
python tools/test_geotiff_roundtrip.py
```
