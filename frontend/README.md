# BharatSR Web Console (SIH26142)

Frontend interface for the **BharatSR** 4x satellite super-resolution system, built with Next.js 16 (React 19), MapLibre GL, Zustand, TanStack Query, and Tailwind CSS.

---

## Getting Started Locally

### 1. Install Dependencies
```bash
npm install
```

### 2. Configure Environment Variables
Copy `.env.example` to `.env.local`:
```bash
cp .env.example .env.local
```

Default variables for local development:
```env
BACKEND_INTERNAL_URL=http://127.0.0.1:8000
NEXT_PUBLIC_WS_URL=ws://127.0.0.1:8000
```

### 3. Run Development Server
```bash
npm run dev
```

Open [http://localhost:3000](http://localhost:3000) for the Landing Page and [http://localhost:3000/console](http://localhost:3000/console) for the Super-Resolution Console.

---

## Testing & Quality Assurance

```bash
# Run Vitest test suite
npm test

# Run Next.js production build verification
npm run build
```

---

## Deployment on Vercel

When importing this repository into Vercel:
1. **Root Directory**: Select `frontend`.
2. **Framework Preset**: `Next.js`.
3. **Environment Variables**:
   - `BACKEND_INTERNAL_URL`: Your live backend URL on Render (e.g. `https://bharatsr-backend.onrender.com`).
   - `NEXT_PUBLIC_WS_URL`: Your live backend WebSocket URL (e.g. `wss://bharatsr-backend.onrender.com`).
4. **Deploy**: All `/api/*` routes are automatically forwarded to your Render backend via Next.js internal server-side rewrites in `next.config.ts`, eliminating browser CORS issues.
