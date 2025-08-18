# 🔒 Production Security Guide

## 🚨 **Critical Security Steps**

### **1. Environment Configuration**

**Generate Secure API Key:**
```bash
# Generate a cryptographically secure API key
python -c "import secrets; print(secrets.token_urlsafe(32))"
```

**Required Environment Variables:**
```bash
# Copy example and customize
cp .env.example .env

# Update .env with:
API_KEY=your_secure_random_key_here
DB_PATH=/secure/path/to/production.db
```

### **2. Database Security**

**File Permissions:**
```bash
# Secure database file permissions
chmod 600 sidecar.db
chown app_user:app_user sidecar.db
```

**Location:**
- Store database outside web root
- Use absolute paths in production
- Regular backups with encryption

### **3. Network Security**

**API Authentication:**
- All endpoints except `/healthz` require `X-API-Key` header
- Use HTTPS in production (reverse proxy recommended)
- Rate limiting recommended (not built-in)

**CORS Configuration:**
- Configure allowed origins for production domains
- Restrict to necessary HTTP methods

### **4. File System Security**

**Protected Directories:**
```
data/indexes/     # Vector index files - sensitive
.env             # Environment variables - secret
sidecar.db       # Database - contains all data
```

**Permissions:**
```bash
chmod 700 data/
chmod 600 .env
chmod 600 data/indexes/*
```

### **5. Model Security**

**GPT-J Model Files:**
- Models cached in `~/.cache/huggingface/`
- Large files (~24GB) - monitor disk space
- Cached models are not sensitive but consume storage

### **6. Input Validation**

**Already Implemented:**
- Pydantic schema validation for all API requests
- SQL injection protection via SQLAlchemy
- File path sanitization for uploaded content

**Additional Recommendations:**
- Input size limits configured in environment
- Content filtering for uploaded documents
- Regular security updates for dependencies

### **7. Monitoring & Logging**

**Health Checks:**
```bash
curl -f http://localhost:8088/healthz
```

**Log Monitoring:**
- Application logs via uvicorn
- Monitor for authentication failures
- Track API usage patterns

### **8. Deployment Checklist**

**Before Deployment:**
- [ ] Generate secure API key
- [ ] Configure environment variables
- [ ] Set proper file permissions
- [ ] Test health check endpoint
- [ ] Verify HTTPS configuration
- [ ] Run security scan on dependencies

**After Deployment:**
- [ ] Verify API authentication works
- [ ] Test all endpoints with production config
- [ ] Monitor resource usage (CPU, memory, disk)
- [ ] Set up log aggregation
- [ ] Configure backup strategy

### **9. Container Security** (Optional)

**Dockerfile Best Practices:**
```dockerfile
# Use non-root user
RUN adduser --disabled-password --gecos '' appuser
USER appuser

# Copy only necessary files
COPY requirements.txt .
COPY app/ ./app/
COPY static/ ./static/

# Set environment
ENV API_KEY=""
ENV DB_PATH="/data/sidecar.db"
```

### **10. Regular Maintenance**

**Security Updates:**
```bash
# Update dependencies regularly
pip install --upgrade -r requirements.txt

# Check for vulnerabilities
pip audit
```

**Monitoring:**
- Disk space (models + database growth)
- Memory usage (GPT-J model loading)
- API response times
- Authentication failure rates

## ⚠️ **Security Warnings**

### **Never Commit:**
- `.env` files with real credentials
- Database files with user data
- API keys or secrets
- User-uploaded content
- Model cache files

### **Production Hardening:**
- Use reverse proxy (nginx/Apache) with HTTPS
- Implement rate limiting
- Set up log rotation
- Configure firewall rules
- Use process manager (systemd/supervisor)
- Monitor for security vulnerabilities

### **Data Privacy:**
- User data in database requires GDPR compliance
- Chat history contains personal information
- Implement data retention policies
- Provide data export/deletion capabilities

## 🛡️ **Incident Response**

**If Security Breach Suspected:**
1. Immediately rotate API keys
2. Check access logs for unauthorized requests
3. Backup and secure database
4. Update all dependencies
5. Review file permissions
6. Consider temporary service shutdown

**Recovery:**
1. Generate new secure credentials
2. Restore from clean backup if needed
3. Update security configurations
4. Monitor for continued threats
5. Document incident and improvements

---

**Remember**: Security is an ongoing process, not a one-time setup. Regular reviews and updates are essential for production systems.