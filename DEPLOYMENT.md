# Deployment Guide

## Local Development

### 1. Start the Backend
```bash
cd backend
pip install -r requirements.txt
python main.py
```
The API will be available at `http://localhost:8000`

### 2. Start the Frontend
```bash
cd frontend
npm install
npm start
```
The frontend will be available at `http://localhost:3000`

### 3. Test the API
```bash
python test_api.py
```

## Vercel Deployment

### Prerequisites
- Vercel account
- Vercel CLI installed (`npm i -g vercel`)

### Steps

1. **Prepare the project**
   ```bash
   # Build the frontend
   cd frontend
   npm run build
   cd ..
   ```

2. **Deploy to Vercel**
   ```bash
   vercel --prod
   ```

3. **Configure environment variables** (if needed)
   - Go to Vercel dashboard
   - Select your project
   - Go to Settings > Environment Variables
   - Add any required environment variables

### Project Structure for Vercel

The project is configured to work with Vercel's serverless functions:

- `backend/main.py` - FastAPI backend (Python serverless function)
- `frontend/build/` - Static React build
- `vercel.json` - Vercel configuration

### API Routes

- `/api/*` - Backend API endpoints
- `/*` - Frontend React app

## Testing Deployment

After deployment, test the endpoints:

1. **Health Check**: `https://your-app.vercel.app/api/health`
2. **API Docs**: `https://your-app.vercel.app/api/docs`
3. **Frontend**: `https://your-app.vercel.app/`

## Troubleshooting

### Common Issues

1. **CORS Errors**: Make sure CORS is enabled in the FastAPI backend
2. **Model Loading**: Ensure model files are in the correct path
3. **Build Errors**: Check that all dependencies are properly installed

### Logs

Check Vercel function logs in the dashboard for debugging.
