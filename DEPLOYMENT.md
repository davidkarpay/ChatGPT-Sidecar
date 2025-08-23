# Sidecar Cloud Deployment Guide

This guide covers deploying Sidecar to the cloud with Google OAuth authentication and automated ChatGPT data synchronization.

## Features Added for Cloud Deployment

### 1. Google OAuth Authentication
- Secure login using Google accounts
- Session management with automatic expiration
- User-specific data isolation

### 2. ChatGPT Data Synchronization
- Automated downloading and processing of ChatGPT exports
- Configurable sync frequency (hourly, daily, etc.)
- Manual sync triggers via API
- Deduplication to avoid processing same files multiple times
- Comprehensive sync history and status tracking

### 3. Background Task Processing
- Celery integration for async processing
- Redis for task queuing and caching
- Automatic retry and error handling

### 4. Database Migration System
- Alembic for schema versioning
- PostgreSQL support for production
- Automatic migration on deployment

## Deployment Options

### Option 1: Railway (Recommended)

Railway provides a simple deployment platform with automatic PostgreSQL and Redis provisioning.

#### Prerequisites
1. [Railway account](https://railway.app)
2. Google Cloud Console project for OAuth
3. ChatGPT export URL (optional, for automatic sync)

#### Steps

1. **Create Google OAuth Application**
   - Go to [Google Cloud Console](https://console.cloud.google.com)
   - Create a new project or select existing one
   - Enable Google+ API
   - Go to "Credentials" → "Create Credentials" → "OAuth 2.0 Client IDs"
   - Set application type to "Web application"
   - Add authorized redirect URI: `https://your-app.railway.app/auth/callback`
   - Note down Client ID and Client Secret

2. **Deploy to Railway**
   ```bash
   # Install Railway CLI
   npm install -g @railway/cli

   # Login to Railway
   railway login

   # Deploy from your project directory
   railway up
   ```

3. **Configure Environment Variables in Railway**
   ```bash
   # Database (automatically configured by Railway)
   DB_URL=postgresql://user:pass@host:port/dbname

   # Redis (add Redis service in Railway)
   REDIS_URL=redis://host:port

   # Google OAuth
   GOOGLE_CLIENT_ID=your_google_client_id
   GOOGLE_CLIENT_SECRET=your_google_client_secret
   REDIRECT_URI=https://your-app.railway.app/auth/callback

   # Application settings
   API_KEY=your_secure_api_key
   SESSION_LIFETIME_HOURS=24
   ENVIRONMENT=production

   # ChatGPT Sync (optional)
   CHATGPT_SYNC_ENABLED=true
   CHATGPT_EXPORT_URL=https://your-chatgpt-export-url.zip
   CHATGPT_SYNC_INTERVAL_HOURS=24
   ```

4. **Add Additional Services**
   - Add PostgreSQL service in Railway dashboard
   - Add Redis service in Railway dashboard
   - Services will auto-configure environment variables

### Option 2: Docker Deployment

Use the provided Docker configuration for any container platform.

#### Local Development with Docker
```bash
# Start all services
docker-compose up -d

# View logs
docker-compose logs -f

# Stop services
docker-compose down
```

#### Production Docker Deployment
```bash
# Build production image
docker build -t sidecar:latest .

# Run with environment variables
docker run -d \
  -p 8088:8088 \
  -e DB_URL=postgresql://... \
  -e REDIS_URL=redis://... \
  -e GOOGLE_CLIENT_ID=... \
  -e GOOGLE_CLIENT_SECRET=... \
  -e API_KEY=... \
  sidecar:latest
```

## Configuration

### Environment Variables

| Variable | Required | Default | Description |
|----------|----------|---------|-------------|
| `DB_URL` | Yes | SQLite | PostgreSQL connection string |
| `REDIS_URL` | Yes | localhost:6379 | Redis connection string |
| `GOOGLE_CLIENT_ID` | For OAuth | - | Google OAuth client ID |
| `GOOGLE_CLIENT_SECRET` | For OAuth | - | Google OAuth client secret |
| `REDIRECT_URI` | For OAuth | localhost | OAuth callback URL |
| `API_KEY` | Yes | "change-me" | API authentication key |
| `SESSION_LIFETIME_HOURS` | No | 24 | Session expiration time |
| `CHATGPT_SYNC_ENABLED` | No | false | Enable automatic sync |
| `CHATGPT_EXPORT_URL` | For sync | - | URL to download ChatGPT export |
| `CHATGPT_SYNC_INTERVAL_HOURS` | No | 24 | Sync frequency |

### Database Setup

The application automatically runs migrations on startup. For manual migration:

```bash
# Run migrations
alembic upgrade head

# Create new migration
alembic revision -m "Description"
```

## API Endpoints

### Authentication
- `GET /auth/login` - Initiate Google OAuth login
- `GET /auth/callback` - OAuth callback handler  
- `POST /auth/logout` - Logout and invalidate session
- `GET /auth/user` - Get current user info

### Sync Configuration
- `POST /sync/config` - Configure sync settings
- `GET /sync/config` - Get current sync configuration
- `POST /sync/manual` - Trigger manual sync
- `GET /sync/history` - View sync history
- `GET /sync/status/{task_id}` - Check sync task status

### Original Sidecar APIs
All existing search, ingest, and chat APIs remain available.

## Usage Examples

### Configure Automatic Sync
```bash
curl -X POST https://your-app.railway.app/sync/config \
  -H "X-API-Key: your_api_key" \
  -H "Content-Type: application/json" \
  -d '{
    "sync_enabled": true,
    "chatgpt_export_url": "https://example.com/export.zip",
    "sync_frequency_hours": 24
  }'
```

### Trigger Manual Sync
```bash
curl -X POST https://your-app.railway.app/sync/manual \
  -H "X-API-Key: your_api_key" \
  -H "Content-Type: application/json" \
  -d '{"export_url": "https://example.com/export.zip"}'
```

### Check Sync History
```bash
curl https://your-app.railway.app/sync/history \
  -H "X-API-Key: your_api_key"
```

## Background Tasks

The application uses Celery for background processing:

### Running Background Services

**Development:**
```bash
# Start Redis
redis-server

# Start Celery worker
celery -A app.tasks worker --loglevel=info

# Start Celery beat (scheduler)
celery -A app.tasks beat --loglevel=info

# Or use convenience scripts
./scripts/start_celery.sh
./scripts/stop_celery.sh
```

**Production:**
Background tasks are automatically handled by the deployment platform when properly configured.

### Monitoring Tasks
```bash
# Install flower for monitoring
pip install flower

# Start flower
celery -A app.tasks flower
```

## Troubleshooting

### Common Issues

1. **OAuth not working**
   - Verify Google OAuth credentials
   - Check redirect URI matches exactly
   - Ensure HTTPS in production

2. **Sync failing**
   - Check export URL is accessible
   - Verify Redis is running
   - Check Celery worker logs

3. **Database migration errors**
   - Ensure PostgreSQL is accessible
   - Check database permissions
   - Review migration logs

### Logs and Monitoring

```bash
# Application logs
docker-compose logs app

# Celery worker logs
docker-compose logs worker

# Database logs
docker-compose logs db
```

### Health Checks

- `GET /healthz` - Basic health check
- `GET /healthz/chat` - Chat model health check
- Check Redis: `redis-cli ping`
- Check PostgreSQL: `psql $DB_URL -c "SELECT 1"`

## Security Considerations

1. **Use strong API keys** in production
2. **Enable HTTPS** for OAuth and session security
3. **Secure Redis** with authentication
4. **Use environment variables** for all secrets
5. **Regular backups** of PostgreSQL database
6. **Monitor access logs** for suspicious activity

## Scaling

- **Horizontal scaling**: Add more Celery workers
- **Database scaling**: Use PostgreSQL read replicas
- **Caching**: Redis clustering for high availability
- **Load balancing**: Use Railway's automatic load balancing

For questions or issues, refer to the main README.md or create an issue in the repository.