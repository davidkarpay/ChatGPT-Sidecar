# Authentication Options for Sidecar

You have three authentication options for your private Sidecar deployment:

## Option 1: Simple Email/Password (Recommended for Private Use)

The simplest and most secure option for a private application. No third-party dependencies.

### Setup
1. Set environment variables:
   ```bash
   ADMIN_EMAIL=your-email@domain.com
   ADMIN_PASSWORD=your-secure-password
   ```

2. Access your application:
   - Go to `/auth/simple/login` for the login form
   - Use your configured email and password
   - Sessions last 24 hours by default

### Pros:
- ✅ No third-party OAuth setup required
- ✅ Completely private - no external dependencies
- ✅ Simple to configure and maintain
- ✅ Full control over user access

### Cons:
- ❌ Only supports one user (easily extendable if needed)
- ❌ Manual password management

## Option 2: Zoho OAuth (Great for Existing Zoho Users)

Perfect if you already use Zoho services and want seamless integration.

### Setup
1. **Create Zoho OAuth Application:**
   - Go to [Zoho Developer Console](https://api-console.zoho.com/)
   - Create a new "Self Client" application
   - Set redirect URI: `https://your-app.railway.app/auth/zoho/callback`
   - Note down Client ID and Client Secret

2. **Configure Environment Variables:**
   ```bash
   ZOHO_CLIENT_ID=your_zoho_client_id
   ZOHO_CLIENT_SECRET=your_zoho_client_secret
   ZOHO_DOMAIN=accounts.zoho.com  # or accounts.zoho.eu, accounts.zoho.in
   REDIRECT_URI=https://your-app.railway.app/auth/zoho/callback
   ```

3. **Access your application:**
   - Go to `/auth/zoho/login` to start OAuth flow
   - Login with your Zoho account
   - Get redirected back to your application

### Pros:
- ✅ Uses your existing Zoho account
- ✅ No public audience required
- ✅ Professional OAuth implementation
- ✅ Automatic user profile information

### Cons:
- ❌ Requires Zoho account setup
- ❌ More configuration than simple auth

## Option 3: API Key Only (Development/Automation)

Keep using the existing API key authentication for development or API access.

### Setup
```bash
API_KEY=your-secure-api-key
```

### Usage
```bash
curl -H "X-API-Key: your-secure-api-key" https://your-app/search
```

### Pros:
- ✅ Simple for API access
- ✅ Good for automation/scripts
- ✅ No browser required

### Cons:
- ❌ No web UI login
- ❌ Not suitable for browser-based access

## Deployment Examples

### Railway with Simple Auth
```bash
# In Railway environment variables
ADMIN_EMAIL=david@yourdomain.com
ADMIN_PASSWORD=YourSecurePassword123!
SESSION_LIFETIME_HOURS=24
```

### Railway with Zoho OAuth
```bash
# In Railway environment variables
ZOHO_CLIENT_ID=1000.ABC123XYZ
ZOHO_CLIENT_SECRET=your_secret_here
ZOHO_DOMAIN=accounts.zoho.com
REDIRECT_URI=https://your-app.railway.app/auth/zoho/callback
SESSION_LIFETIME_HOURS=24
```

### Docker with Simple Auth
```yaml
# docker-compose.yml
environment:
  - ADMIN_EMAIL=admin@localhost
  - ADMIN_PASSWORD=SecurePassword123
  - SESSION_LIFETIME_HOURS=24
```

## Authentication Endpoints

### Simple Auth
- `GET /auth/simple/login` - Login form
- `POST /auth/simple/login` - Submit credentials
- `POST /auth/simple/logout` - Logout

### Zoho OAuth
- `GET /auth/zoho/login` - Start OAuth flow
- `GET /auth/zoho/callback` - OAuth callback
- `POST /auth/zoho/logout` - Logout

### Generic
- `GET /auth/user` - Get current user info (works with any auth method)

## Security Considerations

1. **Use HTTPS in production** - Essential for session security
2. **Strong passwords** - Use a password manager
3. **Secure environment variables** - Never commit secrets to code
4. **Session timeouts** - Configure appropriate session lifetime
5. **Regular updates** - Keep dependencies updated

## Switching Between Methods

You can enable multiple authentication methods simultaneously:

```bash
# Enable both simple and Zoho auth
ADMIN_EMAIL=admin@localhost
ADMIN_PASSWORD=SecurePassword
ZOHO_CLIENT_ID=your_client_id
ZOHO_CLIENT_SECRET=your_secret
```

Users can then choose which method to use:
- `/auth/simple/login` for email/password
- `/auth/zoho/login` for Zoho OAuth

## Recommendation

For your private use case, **Simple Email/Password authentication** is the best choice because:

1. **No external dependencies** - Works completely offline
2. **No OAuth setup hassle** - Just set email and password
3. **Maximum privacy** - No data shared with third parties
4. **Easy deployment** - Works immediately on any platform
5. **Secure** - Uses bcrypt password hashing and secure sessions

Simply set these environment variables and you're ready to go:
```bash
ADMIN_EMAIL=your-email@domain.com
ADMIN_PASSWORD=your-secure-password
```

Then access your application at `/auth/simple/login`!