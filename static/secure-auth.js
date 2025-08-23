/**
 * Secure authentication module for Sidecar API
 * Replaces localStorage with sessionStorage + optional secure persistence
 */

class SecureAuth {
    constructor() {
        this.API_KEY_SESSION = 'SIDECAR_API_KEY';
        this.API_KEY_REMEMBER = 'SIDECAR_API_KEY_REMEMBER';
        this.REMEMBER_FLAG = 'SIDECAR_REMEMBER_ME';
        
        // Initialize from stored values
        this.apiKey = this.getStoredApiKey();
        this.rememberMe = this.getRememberPreference();
        
        // Rate limiting protection
        this.requestCount = 0;
        this.requestWindow = Date.now();
        this.maxRequestsPerMinute = 60;
    }
    
    /**
     * Get API key from storage (sessionStorage first, then localStorage if remember me is enabled)
     */
    getStoredApiKey() {
        // Always check sessionStorage first (temporary session key)
        let key = sessionStorage.getItem(this.API_KEY_SESSION);
        if (key) {
            return key;
        }
        
        // Check if user has "remember me" enabled
        const remember = localStorage.getItem(this.REMEMBER_FLAG);
        if (remember === 'true') {
            key = localStorage.getItem(this.API_KEY_REMEMBER);
            if (key) {
                // Copy to session storage for current session
                sessionStorage.setItem(this.API_KEY_SESSION, key);
                return key;
            }
        }
        
        return '';
    }
    
    /**
     * Get remember preference
     */
    getRememberPreference() {
        return localStorage.getItem(this.REMEMBER_FLAG) === 'true';
    }
    
    /**
     * Set API key with optional persistence
     * @param {string} key - API key
     * @param {boolean} remember - Whether to persist across sessions
     */
    setApiKey(key, remember = false) {
        this.apiKey = key;
        this.rememberMe = remember;
        
        // Always store in session storage for current session
        if (key) {
            sessionStorage.setItem(this.API_KEY_SESSION, key);
        } else {
            sessionStorage.removeItem(this.API_KEY_SESSION);
        }
        
        // Handle persistence preference
        if (remember && key) {
            localStorage.setItem(this.API_KEY_REMEMBER, key);
            localStorage.setItem(this.REMEMBER_FLAG, 'true');
        } else {
            localStorage.removeItem(this.API_KEY_REMEMBER);
            localStorage.removeItem(this.REMEMBER_FLAG);
        }
    }
    
    /**
     * Clear all stored API keys
     */
    clearApiKey() {
        this.apiKey = '';
        this.rememberMe = false;
        
        sessionStorage.removeItem(this.API_KEY_SESSION);
        localStorage.removeItem(this.API_KEY_REMEMBER);
        localStorage.removeItem(this.REMEMBER_FLAG);
    }
    
    /**
     * Get current API key
     */
    getApiKey() {
        return this.apiKey;
    }
    
    /**
     * Check if rate limiting should be applied (client-side protection)
     */
    checkRateLimit() {
        const now = Date.now();
        
        // Reset window if more than a minute has passed
        if (now - this.requestWindow > 60000) {
            this.requestCount = 0;
            this.requestWindow = now;
        }
        
        this.requestCount++;
        
        if (this.requestCount > this.maxRequestsPerMinute) {
            throw new Error('Rate limit exceeded. Please wait before making more requests.');
        }
    }
    
    /**
     * Create authenticated fetch request with automatic API key injection
     * @param {string} url - Request URL
     * @param {object} options - Fetch options
     * @returns {Promise} - Fetch promise
     */
    async authenticatedFetch(url, options = {}) {
        // Check rate limiting
        this.checkRateLimit();
        
        if (!this.apiKey) {
            throw new Error('No API key configured. Please enter your API key.');
        }
        
        // Ensure headers exist
        options.headers = options.headers || {};
        
        // Add API key header
        options.headers['X-API-Key'] = this.apiKey;
        
        // Add default content type if not specified
        if (!options.headers['Content-Type'] && 
            (options.method === 'POST' || options.method === 'PUT')) {
            options.headers['Content-Type'] = 'application/json';
        }
        
        try {
            const response = await fetch(url, options);
            
            // Handle authentication errors
            if (response.status === 401) {
                this.clearApiKey();
                throw new Error('Invalid API key. Please check your credentials and try again.');
            }
            
            if (response.status === 403) {
                throw new Error('Insufficient permissions for this operation.');
            }
            
            if (response.status === 429) {
                throw new Error('Rate limit exceeded. Please wait before making more requests.');
            }
            
            return response;
            
        } catch (error) {
            // Network errors or other issues
            if (error.name === 'TypeError' && error.message.includes('fetch')) {
                throw new Error('Network error. Please check your connection and try again.');
            }
            throw error;
        }
    }
    
    /**
     * Initialize API key input field with secure handlers
     * @param {HTMLInputElement} inputElement - API key input field
     * @param {HTMLInputElement} rememberCheckbox - Remember me checkbox (optional)
     */
    initializeApiKeyInput(inputElement, rememberCheckbox = null) {
        if (!inputElement) return;
        
        // Set initial values
        inputElement.value = this.apiKey;
        
        if (rememberCheckbox) {
            rememberCheckbox.checked = this.rememberMe;
            
            // Handle remember me checkbox changes
            rememberCheckbox.addEventListener('change', (e) => {
                const remember = e.target.checked;
                this.setApiKey(this.apiKey, remember);
            });
        }
        
        // Handle API key input changes
        let debounceTimer;
        inputElement.addEventListener('input', (e) => {
            clearTimeout(debounceTimer);
            debounceTimer = setTimeout(() => {
                const key = e.target.value.trim();
                const remember = rememberCheckbox ? rememberCheckbox.checked : false;
                this.setApiKey(key, remember);
            }, 500); // Debounce to avoid excessive storage writes
        });
        
        // Clear key on focus if it's a placeholder
        inputElement.addEventListener('focus', () => {
            if (inputElement.value === 'Enter your API key...') {
                inputElement.value = '';
            }
        });
        
        // Provide visual feedback for key validation
        inputElement.addEventListener('blur', () => {
            if (!inputElement.value.trim()) {
                inputElement.style.borderColor = '#ef4444';  // Red border
                inputElement.title = 'API key is required';
            } else {
                inputElement.style.borderColor = '#10b981';  // Green border
                inputElement.title = 'API key configured';
            }
        });
    }
    
    /**
     * Show API key management UI (for advanced users)
     */
    showKeyManagement() {
        const currentKey = this.getApiKey();
        const maskedKey = currentKey ? 
            currentKey.substring(0, 8) + '...' + currentKey.substring(currentKey.length - 4) : 
            'None';
        
        const message = `Current API Key: ${maskedKey}\n\n` +
                       `Remember setting: ${this.rememberMe ? 'Enabled' : 'Disabled'}\n\n` +
                       `Choose an action:`;
        
        const action = prompt(message + '\n\n1. Clear all keys\n2. Cancel', '2');
        
        if (action === '1') {
            this.clearApiKey();
            location.reload(); // Refresh page to reflect changes
        }
    }
    
    /**
     * Get access level display name from auth info
     */
    getAccessLevelDisplay(authInfo) {
        if (!authInfo || !authInfo.level) return 'Unknown';
        
        const levels = {
            'read': '🔍 Read Access',
            'admin': '⚙️ Admin Access',
            'super': '🔑 Super Admin'
        };
        
        return levels[authInfo.level] || authInfo.level;
    }
}

// Global instance
window.secureAuth = new SecureAuth();