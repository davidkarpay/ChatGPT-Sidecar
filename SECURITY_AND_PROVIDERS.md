# Security Improvements & LLM Provider Support

This document outlines the new security features and LLM provider abstraction implemented in Sidecar.

## 🔐 Security Improvements

### Role-Based API Keys

Sidecar now supports multiple API keys with different access levels:

- **READ_KEY**: Search, chat, document access (lower risk operations)
- **ADMIN_KEY**: Ingestion, reindexing, training (system modifications)
- **SUPER_KEY**: User management, system configuration (highest privilege)
- **API_KEY**: Legacy single key (provides admin access for backward compatibility)

#### Configuration
```bash
# .env file
READ_KEY=your-read-only-key-here
ADMIN_KEY=your-admin-key-here
SUPER_KEY=your-super-admin-key-here
```

### Client-Side Security

- **sessionStorage**: API keys are stored in sessionStorage by default (cleared on browser close)
- **Optional Persistence**: "Remember me" checkbox stores keys securely in localStorage
- **Rate Limiting**: Built-in client and server-side rate limiting (60 requests/minute for READ keys)
- **Audit Logging**: All authentication attempts are logged in `auth_audit_log` table

### Authentication Monitoring

View authentication logs:
```sql
SELECT * FROM auth_audit_log ORDER BY timestamp DESC LIMIT 100;
```

## 🤖 LLM Provider Support

Sidecar now supports multiple LLM providers with hot-swappable configuration:

### Supported Providers

1. **OpenAI** (GPT-3.5, GPT-4)
2. **Anthropic** (Claude 3)
3. **Ollama** (Local models)
4. **GPT-J** (Legacy, local)

### Quick Setup

#### OpenAI Configuration
```bash
# .env
LLM_PROVIDER=openai
LLM_MODEL=gpt-3.5-turbo
OPENAI_API_KEY=your-openai-api-key
```

#### Anthropic Configuration  
```bash
# .env
LLM_PROVIDER=anthropic
LLM_MODEL=claude-3-sonnet-20240229
ANTHROPIC_API_KEY=your-anthropic-api-key
```

#### Ollama Configuration
```bash
# .env
LLM_PROVIDER=ollama
LLM_MODEL=llama2
LLM_BASE_URL=http://localhost:11434  # Default Ollama URL
```

### Provider Management Endpoints

- `GET /providers/available` - List all providers and their availability
- `GET /providers/current` - Get current provider details
- `POST /providers/test` - Test provider connection (requires ADMIN_KEY)

### Configuration Options

All providers support these common settings:

```bash
LLM_PROVIDER=openai           # Provider type
LLM_MODEL=gpt-3.5-turbo      # Model name
LLM_MAX_TOKENS=512           # Max response tokens
LLM_TEMPERATURE=0.7          # Response creativity (0.0-1.0)
LLM_TIMEOUT=30               # Request timeout (seconds)
LLM_STREAMING=true           # Enable streaming responses
```

## 🚀 Usage Examples

### Switching to OpenAI
1. Set environment variables:
   ```bash
   export LLM_PROVIDER=openai
   export OPENAI_API_KEY=your-key-here
   ```
2. Restart Sidecar
3. Chat endpoints automatically use OpenAI

### Testing Provider Connection
```bash
curl -X POST http://localhost:8088/providers/test \
  -H "X-API-Key: your-admin-key" 
```

### Checking Available Providers
```bash
curl http://localhost:8088/providers/available \
  -H "X-API-Key: your-read-key"
```

## 🛡️ Security Best Practices

1. **Use Role-Based Keys**: Don't give admin keys to read-only applications
2. **Rotate Keys Regularly**: Change API keys periodically
3. **Monitor Audit Logs**: Check `auth_audit_log` for suspicious activity
4. **Enable HTTPS**: Use HTTPS in production to protect API keys in transit
5. **Limit Key Scope**: Use READ_KEY for most operations, ADMIN_KEY only when needed

## 🔄 Migration Guide

### From Single API Key
If you're using the old `API_KEY` system:

1. **Immediate**: Continue using `API_KEY` - it still provides admin access
2. **Recommended**: Create role-based keys:
   ```bash
   READ_KEY=new-read-key
   ADMIN_KEY=new-admin-key
   # Keep API_KEY for backward compatibility
   ```

### Client Updates
Update your client applications to:
1. Use appropriate access level keys
2. Handle rate limiting errors (429 status)
3. Implement proper error handling for auth failures

## 📊 Monitoring

### API Usage by Key
```sql
SELECT api_key_hash, COUNT(*) as requests, 
       SUM(CASE WHEN success THEN 1 ELSE 0 END) as successful
FROM auth_audit_log 
GROUP BY api_key_hash;
```

### Failed Authentication Attempts
```sql
SELECT * FROM auth_audit_log 
WHERE success = 0 
ORDER BY timestamp DESC;
```

### Rate Limit Violations
```sql
SELECT * FROM auth_audit_log 
WHERE event_type = 'rate_limit_exceeded' 
ORDER BY timestamp DESC;
```

## 🆘 Troubleshooting

### Provider Connection Issues
1. Check API keys are correctly set
2. Verify network connectivity
3. Test with `/providers/test` endpoint
4. Check logs for detailed error messages

### Authentication Problems
1. Verify API key format and permissions
2. Check audit logs for failed attempts
3. Ensure correct `X-API-Key` header format
4. Consider rate limiting if requests fail with 429

### Client Issues
1. Clear browser storage if keys are outdated
2. Check console for JavaScript errors
3. Verify "remember me" setting if keys don't persist

This new architecture provides enterprise-grade security while maintaining ease of use and flexibility for different deployment scenarios.