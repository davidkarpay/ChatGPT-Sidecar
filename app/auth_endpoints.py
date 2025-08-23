"""Alternative authentication endpoints."""

import json
import secrets
from fastapi import APIRouter, HTTPException, Request, Response, Form, Depends
from fastapi.responses import HTMLResponse, RedirectResponse
from pydantic import BaseModel

# Import authentication methods
from .auth_zoho import zoho_oauth_handler, zoho_session_manager, ZOHO_OAUTH_ENABLED
from .auth_simple import simple_auth, SIMPLE_AUTH_ENABLED

router = APIRouter()

class LoginRequest(BaseModel):
    email: str
    password: str

# Simple Auth Endpoints
@router.get("/auth/simple/login")
async def simple_login_form():
    """Serve simple login form."""
    if not SIMPLE_AUTH_ENABLED:
        raise HTTPException(status_code=501, detail="Simple authentication not configured")
    
    html_content = """
    <!DOCTYPE html>
    <html>
    <head>
        <title>Sidecar Login</title>
        <style>
            body { font-family: Arial, sans-serif; max-width: 400px; margin: 100px auto; padding: 20px; }
            .login-form { background: #f5f5f5; padding: 30px; border-radius: 8px; }
            input { width: 100%; padding: 10px; margin: 10px 0; border: 1px solid #ddd; border-radius: 4px; }
            button { width: 100%; padding: 12px; background: #007cba; color: white; border: none; border-radius: 4px; cursor: pointer; }
            button:hover { background: #005a87; }
            .error { color: red; margin: 10px 0; }
        </style>
    </head>
    <body>
        <div class="login-form">
            <h2>Sidecar Login</h2>
            <form action="/auth/simple/login" method="post">
                <input type="email" name="email" placeholder="Email" required>
                <input type="password" name="password" placeholder="Password" required>
                <button type="submit">Login</button>
            </form>
        </div>
    </body>
    </html>
    """
    return HTMLResponse(content=html_content)

@router.post("/auth/simple/login")
async def simple_login(email: str = Form(...), password: str = Form(...)):
    """Handle simple login form submission."""
    if not SIMPLE_AUTH_ENABLED:
        raise HTTPException(status_code=501, detail="Simple authentication not configured")
    
    user = simple_auth.authenticate_user(email, password)
    if not user:
        raise HTTPException(status_code=401, detail="Invalid email or password")
    
    session_token = simple_auth.create_session(user)
    
    response = RedirectResponse(url="/", status_code=302)
    response.set_cookie("session_token", session_token, httponly=True, secure=True, samesite="lax")
    return response

@router.post("/auth/simple/logout")
async def simple_logout(request: Request):
    """Logout from simple auth."""
    if not SIMPLE_AUTH_ENABLED:
        raise HTTPException(status_code=501, detail="Simple authentication not configured")
    
    session_token = request.cookies.get("session_token")
    if session_token:
        simple_auth.invalidate_session(session_token)
    
    response = Response(content=json.dumps({"message": "Logged out successfully"}))
    response.delete_cookie("session_token")
    return response

# Zoho OAuth Endpoints
@router.get("/auth/zoho/login")
async def zoho_login(request: Request):
    """Initiate Zoho OAuth login flow."""
    if not ZOHO_OAUTH_ENABLED:
        raise HTTPException(status_code=501, detail="Zoho OAuth authentication not configured")
    
    state = secrets.token_urlsafe(32)
    authorization_url = zoho_oauth_handler.get_authorization_url(state)
    
    response = RedirectResponse(url=authorization_url)
    response.set_cookie("oauth_state", state, httponly=True, secure=True, samesite="lax")
    return response

@router.get("/auth/zoho/callback")
async def zoho_callback(request: Request, code: str = None, state: str = None, error: str = None):
    """Handle Zoho OAuth callback."""
    if not ZOHO_OAUTH_ENABLED:
        raise HTTPException(status_code=501, detail="Zoho OAuth authentication not configured")
    
    if error:
        raise HTTPException(status_code=400, detail=f"OAuth error: {error}")
    
    if not code or not state:
        raise HTTPException(status_code=400, detail="Missing authorization code or state")
    
    # Validate state
    stored_state = request.cookies.get("oauth_state")
    if stored_state != state:
        raise HTTPException(status_code=400, detail="Invalid state parameter")
    
    try:
        # Exchange code for user info
        user_info = await zoho_oauth_handler.exchange_code_for_token(code, state)
        
        # Create session
        session_token = zoho_session_manager.create_session(user_info)
        
        # Redirect to main app with session
        response = RedirectResponse(url="/")
        response.set_cookie("session_token", session_token, httponly=True, secure=True, samesite="lax")
        response.delete_cookie("oauth_state")
        
        return response
        
    except Exception as e:
        raise HTTPException(status_code=400, detail="Authentication failed")

@router.post("/auth/zoho/logout")
async def zoho_logout(request: Request):
    """Logout from Zoho OAuth."""
    if not ZOHO_OAUTH_ENABLED:
        raise HTTPException(status_code=501, detail="Zoho OAuth authentication not configured")
    
    session_token = request.cookies.get("session_token")
    if session_token:
        zoho_session_manager.invalidate_session(session_token)
    
    response = Response(content=json.dumps({"message": "Logged out successfully"}))
    response.delete_cookie("session_token")
    return response

# Generic user info endpoint
@router.get("/auth/user")
async def get_auth_user_info(request: Request):
    """Get current user information from any auth method."""
    session_token = request.cookies.get("session_token")
    
    if not session_token:
        return {"authenticated": False}
    
    # Try simple auth first
    if SIMPLE_AUTH_ENABLED:
        user = simple_auth.get_user_from_session(session_token)
        if user:
            return {
                "authenticated": True,
                "auth_method": "simple",
                "user": {
                    "email": user["email"],
                    "display_name": user["display_name"]
                }
            }
    
    # Try Zoho OAuth
    if ZOHO_OAUTH_ENABLED:
        user = zoho_session_manager.get_user_from_session(session_token)
        if user:
            return {
                "authenticated": True,
                "auth_method": "zoho",
                "user": {
                    "email": user["email"],
                    "display_name": user["display_name"],
                    "picture": user.get("picture"),
                    "verified_email": user.get("verified_email", False)
                }
            }
    
    return {"authenticated": False}