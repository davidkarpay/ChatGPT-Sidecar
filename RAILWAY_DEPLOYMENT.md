# Railway Deployment Guide for Sidecar

This guide will help you deploy your Sidecar application to Railway.app with simple email/password authentication.

## 🚀 Quick Deployment Steps

### Step 1: Login to Railway
```bash
npx railway login
```
This will open your browser to authenticate with Railway.

### Step 2: Create New Project
```bash
npx railway project create
```
Choose "sidecar" as your project name when prompted.

### Step 3: Add PostgreSQL Database
```bash
npx railway add postgresql
```
Railway will automatically provision a PostgreSQL database and set the `DATABASE_URL` environment variable.

### Step 4: Set Environment Variables
```bash
# Set your admin credentials (replace with your actual email and secure password)
npx railway variables set ADMIN_EMAIL=your-email@domain.com
npx railway variables set ADMIN_PASSWORD=YourSecurePassword123

# Set a secure API key
npx railway variables set API_KEY=your-secure-api-key-here

# Optional: Configure session lifetime
npx railway variables set SESSION_LIFETIME_HOURS=24
```

### Step 5: Deploy Your Application
```bash
npx railway up
```
Railway will:
- Build your Docker container
- Run database migrations automatically (`alembic upgrade head`)
- Deploy your application
- Provide you with a public URL

### Step 6: Access Your Application
Once deployed, Railway will give you a URL like: `https://sidecar-production-xxxx.up.railway.app`

Access your application:
- **Login**: `https://your-app.railway.app/auth/simple/login`
- **API**: Use your API key with header `X-API-Key: your-secure-api-key`

## 🔧 Alternative Deployment Methods

### Option A: Deploy from GitHub
1. Push your code to GitHub
2. Connect your GitHub repository to Railway
3. Railway will automatically deploy on every push

### Option B: One-Command Deploy
```bash
# Set all variables and deploy in one go
npx railway variables set ADMIN_EMAIL=your-email@domain.com ADMIN_PASSWORD=YourSecurePassword123 API_KEY=your-secure-api-key && npx railway up
```

## 🛠 Post-Deployment Configuration

### Configure ChatGPT Sync (Optional)
Once your app is deployed, you can configure automatic ChatGPT synchronization:

```bash
curl -X POST https://your-app.railway.app/sync/config \
  -H "X-API-Key: your-secure-api-key" \
  -H "Content-Type: application/json" \
  -d '{
    "sync_enabled": true,
    "sync_interval_hours": 24,
    "chatgpt_export_url": "your-chatgpt-export-url"
  }'
```

### View Sync Status
```bash
curl -H "X-API-Key: your-secure-api-key" https://your-app.railway.app/sync/config
```

## 📊 Monitor Your Deployment

### Check Deployment Status
```bash
npx railway status
```

### View Application Logs
```bash
npx railway logs
```

### Access Railway Dashboard
```bash
npx railway open
```

## 🔒 Security Recommendations

1. **Use Strong Passwords**: Generate a secure password for `ADMIN_PASSWORD`
   ```bash
   # Generate a secure password (run this locally)
   python3 -c "import secrets; print(secrets.token_urlsafe(16))"
   ```

2. **Secure API Key**: Generate a secure API key
   ```bash
   # Generate a secure API key (run this locally)
   python3 -c "import secrets; print(secrets.token_urlsafe(32))"
   ```

3. **Environment Variables**: Never commit secrets to your code repository

4. **HTTPS**: Railway automatically provides HTTPS for all deployments

## 🎯 Expected Result

After successful deployment, you'll have:

✅ **Secure Cloud Application**: Accessible only with your credentials  
✅ **PostgreSQL Database**: Automatically managed by Railway  
✅ **Automatic Migrations**: Database schema updates on every deploy  
✅ **HTTPS Security**: SSL/TLS encryption by default  
✅ **Scalable Infrastructure**: Auto-scaling based on demand  
✅ **ChatGPT Sync Ready**: API endpoints for automated data synchronization  

## 🐛 Troubleshooting

### Deployment Failed
```bash
# Check logs for detailed error information
npx railway logs --follow
```

### Database Connection Issues
```bash
# Verify DATABASE_URL is set
npx railway variables
```

### Application Not Starting
- Check that all required environment variables are set
- Verify Dockerfile builds successfully locally: `docker build -t sidecar .`

### Authentication Not Working
- Verify `ADMIN_EMAIL` and `ADMIN_PASSWORD` are set correctly
- Check application logs for authentication errors

## 🔄 Updating Your Deployment

To update your deployed application:

```bash
# Make your code changes, then redeploy
npx railway up
```

Railway will automatically:
- Build the new version
- Run any new database migrations
- Deploy with zero downtime

## 📱 Quick Commands Reference

```bash
# Essential commands
npx railway login                    # Login to Railway
npx railway project create          # Create new project
npx railway add postgresql          # Add database
npx railway variables set KEY=value # Set environment variables
npx railway up                      # Deploy application
npx railway logs                    # View logs
npx railway open                    # Open Railway dashboard
npx railway status                  # Check deployment status
```

## 🎉 You're Done!

Your Sidecar application is now running in the cloud with:
- Simple email/password authentication
- PostgreSQL database
- Automated ChatGPT sync capabilities
- Professional-grade infrastructure

Access your application at the Railway-provided URL and login with your configured credentials at `/auth/simple/login`!