import streamlit as st
import secrets
import hashlib
import string
import time
import json
from datetime import datetime, timedelta
from typing import Dict, Optional, Tuple

class AuthenticationManager:
    """Enhanced authentication manager with improved security features."""
    
    def __init__(self):
        self.max_login_attempts = 5
        self.lockout_duration = 300  # 5 minutes in seconds
        self.session_timeout = 3600  # 1 hour in seconds
        self.min_password_length = 8
        self.max_password_length = 128
        
        # Initialize session state variables
        self._initialize_session_state()
    
    def _initialize_session_state(self):
        """Initialize all necessary session state variables."""
        if "user_db" not in st.session_state:
            st.session_state.user_db = {}
        
        if "failed_attempts" not in st.session_state:
            st.session_state.failed_attempts = {}
        
        if "lockout_times" not in st.session_state:
            st.session_state.lockout_times = {}
        
        if "last_activity" not in st.session_state:
            st.session_state.last_activity = None
        
        if "password_change_required" not in st.session_state:
            st.session_state.password_change_required = {}

    def _hash_password_with_salt(self, password: str, salt: str = None) -> Tuple[str, str]:
        """Hash password with salt for better security."""
        if salt is None:
            salt = secrets.token_hex(32)
        
        # Use PBKDF2 for better security than simple SHA256
        password_hash = hashlib.pbkdf2_hmac('sha256', 
                                          password.encode('utf-8'), 
                                          salt.encode('utf-8'), 
                                          100000)  # 100,000 iterations
        return password_hash.hex(), salt

    def _verify_password(self, password: str, stored_hash: str, salt: str) -> bool:
        """Verify password against stored hash and salt."""
        password_hash, _ = self._hash_password_with_salt(password, salt)
        return password_hash == stored_hash

    def _is_strong_password(self, password: str) -> Tuple[bool, str]:
        """Validate password strength."""
        if len(password) < self.min_password_length:
            return False, f"Password must be at least {self.min_password_length} characters long."
        
        if len(password) > self.max_password_length:
            return False, f"Password must be no more than {self.max_password_length} characters long."
        
        checks = {
            'uppercase': any(c.isupper() for c in password),
            'lowercase': any(c.islower() for c in password),
            'digit': any(c.isdigit() for c in password),
            'special': any(c in string.punctuation for c in password)
        }
        
        missing = [key for key, value in checks.items() if not value]
        
        if len(missing) > 1:  # Allow missing one category
            return False, f"Password must contain at least 3 of: uppercase, lowercase, digits, special characters."
        
        return True, "Password is strong."

    def generate_secure_password(self, length: int = 16) -> str:
        """Generate a cryptographically secure password."""
        if length < self.min_password_length:
            length = self.min_password_length
        
        # Ensure at least one character from each category
        password = [
            secrets.choice(string.ascii_uppercase),
            secrets.choice(string.ascii_lowercase),
            secrets.choice(string.digits),
            secrets.choice(string.punctuation)
        ]
        
        # Fill the rest randomly
        all_chars = string.ascii_letters + string.digits + string.punctuation
        for _ in range(length - 4):
            password.append(secrets.choice(all_chars))
        
        # Shuffle the password
        secrets.SystemRandom().shuffle(password)
        return ''.join(password)

    def _is_account_locked(self, username: str) -> bool:
        """Check if account is locked due to too many failed attempts."""
        if username not in st.session_state.lockout_times:
            return False
        
        lockout_time = st.session_state.lockout_times[username]
        if time.time() - lockout_time < self.lockout_duration:
            return True
        else:
            # Lockout period expired, reset
            del st.session_state.lockout_times[username]
            st.session_state.failed_attempts[username] = 0
            return False

    def _record_failed_attempt(self, username: str):
        """Record a failed login attempt."""
        if username not in st.session_state.failed_attempts:
            st.session_state.failed_attempts[username] = 0
        
        st.session_state.failed_attempts[username] += 1
        
        if st.session_state.failed_attempts[username] >= self.max_login_attempts:
            st.session_state.lockout_times[username] = time.time()

    def _reset_failed_attempts(self, username: str):
        """Reset failed attempts after successful login."""
        if username in st.session_state.failed_attempts:
            st.session_state.failed_attempts[username] = 0
        if username in st.session_state.lockout_times:
            del st.session_state.lockout_times[username]

    def _check_session_timeout(self) -> bool:
        """Check if the current session has timed out."""
        if not st.session_state.get("authenticated", False):
            return False
        
        if st.session_state.last_activity is None:
            st.session_state.last_activity = time.time()
            return False
        
        if time.time() - st.session_state.last_activity > self.session_timeout:
            self.logout()
            return True
        
        # Update last activity
        st.session_state.last_activity = time.time()
        return False

    def create_user(self, username: str, password: str = None) -> Tuple[bool, str, str]:
        """Create a new user account."""
        username = username.strip()
        
        if not username:
            return False, "Username cannot be empty.", ""
        
        if len(username) < 3:
            return False, "Username must be at least 3 characters long.", ""
        
        if username in st.session_state.user_db:
            return False, f"Username '{username}' already exists.", ""
        
        if password is None:
            password = self.generate_secure_password()
            password_generated = True
        else:
            password_generated = False
            is_strong, message = self._is_strong_password(password)
            if not is_strong:
                return False, message, ""
        
        # Hash the password with salt
        password_hash, salt = self._hash_password_with_salt(password)
        
        # Store user data
        st.session_state.user_db[username] = {
            'hash': password_hash,
            'salt': salt,
            'created_at': datetime.now().isoformat(),
            'last_login': None,
            'password_generated': password_generated
        }
        
        # Mark for password change if generated
        if password_generated:
            st.session_state.password_change_required[username] = True
        
        return True, "User created successfully!", password if password_generated else ""

    def authenticate_user(self, username: str, password: str) -> Tuple[bool, str]:
        """Authenticate user login."""
        username = username.strip()
        
        if not username or not password:
            return False, "Please enter both username and password."
        
        if self._is_account_locked(username):
            remaining_time = int(self.lockout_duration - (time.time() - st.session_state.lockout_times[username]))
            return False, f"Account locked due to too many failed attempts. Try again in {remaining_time} seconds."
        
        if username not in st.session_state.user_db:
            self._record_failed_attempt(username)
            return False, "Invalid username or password."
        
        user_data = st.session_state.user_db[username]
        
        if self._verify_password(password, user_data['hash'], user_data['salt']):
            # Successful login
            self._reset_failed_attempts(username)
            st.session_state.authenticated = True
            st.session_state.username = username
            st.session_state.last_activity = time.time()
            
            # Update last login
            st.session_state.user_db[username]['last_login'] = datetime.now().isoformat()
            
            return True, "Login successful!"
        else:
            self._record_failed_attempt(username)
            return False, "Invalid username or password."

    def change_password(self, username: str, old_password: str, new_password: str) -> Tuple[bool, str]:
        """Change user password."""
        if username not in st.session_state.user_db:
            return False, "User not found."
        
        user_data = st.session_state.user_db[username]
        
        # Verify old password
        if not self._verify_password(old_password, user_data['hash'], user_data['salt']):
            return False, "Current password is incorrect."
        
        # Validate new password
        is_strong, message = self._is_strong_password(new_password)
        if not is_strong:
            return False, message
        
        # Hash new password
        new_hash, new_salt = self._hash_password_with_salt(new_password)
        
        # Update password
        st.session_state.user_db[username]['hash'] = new_hash
        st.session_state.user_db[username]['salt'] = new_salt
        st.session_state.user_db[username]['password_changed_at'] = datetime.now().isoformat()
        
        # Remove password change requirement
        if username in st.session_state.password_change_required:
            del st.session_state.password_change_required[username]
        
        return True, "Password changed successfully!"

    def logout(self):
        """Log out the current user."""
        keys_to_keep = ['user_db', 'failed_attempts', 'lockout_times', 'password_change_required']
        keys_to_remove = [key for key in st.session_state.keys() if key not in keys_to_keep]
        
        for key in keys_to_remove:
            del st.session_state[key]

    def is_authenticated(self) -> bool:
        """Check if user is authenticated and session is valid."""
        if self._check_session_timeout():
            return False
        return st.session_state.get("authenticated", False)

    def requires_password_change(self) -> bool:
        """Check if current user needs to change password."""
        username = st.session_state.get("username")
        return username and st.session_state.password_change_required.get(username, False)


def authentication_page():
    """Enhanced authentication page with improved security and UX."""
    auth_manager = AuthenticationManager()
    
    # Apply custom CSS styling
    st.markdown("""
    <style>
    /* Main container styling */
    .main .block-container {
        padding-top: 2rem;
        padding-bottom: 2rem;
        max-width: 800px;
    }
    
    /* Authentication header styling */
    .auth-header {
        text-align: center;
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        background-clip: text;
        font-size: 2.5rem;
        font-weight: 700;
        margin-bottom: 2rem;
        text-shadow: 2px 2px 4px rgba(0,0,0,0.1);
    }
    
    /* Card-like containers */
    .auth-card {
        background: white;
        padding: 2rem;
        border-radius: 15px;
        box-shadow: 0 10px 30px rgba(0,0,0,0.1);
        border: 1px solid #e0e0e0;
        margin: 1rem 0;
    }
    
    /* Tab styling */
    .stTabs [data-baseweb="tab-list"] {
        gap: 8px;
        background-color: #f8f9fa;
        border-radius: 10px;
        padding: 5px;
    }
    
    .stTabs [data-baseweb="tab"] {
        height: 50px;
        padding: 10px 20px;
        background-color: transparent;
        border-radius: 8px;
        color: #666;
        font-weight: 500;
        border: none;
        transition: all 0.3s ease;
    }
    
    .stTabs [aria-selected="true"] {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        color: white !important;
        box-shadow: 0 4px 15px rgba(102, 126, 234, 0.3);
    }
    
    /* Form styling */
    .stForm {
        background: #f8f9fa;
        padding: 1.5rem;
        border-radius: 12px;
        border: 2px solid #e9ecef;
        margin: 1rem 0;
    }
    
    /* Input field styling */
    .stTextInput > div > div > input {
        border-radius: 8px;
        border: 2px solid #e9ecef;
        padding: 12px 16px;
        font-size: 16px;
        transition: all 0.3s ease;
    }
    
    .stTextInput > div > div > input:focus {
        border-color: #667eea;
        box-shadow: 0 0 0 3px rgba(102, 126, 234, 0.1);
        outline: none;
    }
    
    /* Button styling */
    .stButton > button {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        color: white;
        border: none;
        border-radius: 8px;
        padding: 12px 24px;
        font-weight: 600;
        font-size: 16px;
        transition: all 0.3s ease;
        box-shadow: 0 4px 15px rgba(102, 126, 234, 0.3);
    }
    
    .stButton > button:hover {
        transform: translateY(-2px);
        box-shadow: 0 6px 20px rgba(102, 126, 234, 0.4);
    }
    
    /* Secondary button styling */
    .stButton > button[kind="secondary"] {
        background: #6c757d;
        box-shadow: 0 4px 15px rgba(108, 117, 125, 0.3);
    }
    
    .stButton > button[kind="secondary"]:hover {
        background: #5a6268;
        box-shadow: 0 6px 20px rgba(108, 117, 125, 0.4);
    }
    
    /* Success message styling */
    .stSuccess {
        background: linear-gradient(135deg, #56ab2f 0%, #a8e6cf 100%);
        border: none;
        border-radius: 8px;
        padding: 1rem;
        color: #155724;
    }
    
    /* Error message styling */
    .stError {
        background: linear-gradient(135deg, #ff6b6b 0%, #ffa8a8 100%);
        border: none;
        border-radius: 8px;
        padding: 1rem;
        color: #721c24;
    }
    
    /* Warning message styling */
    .stWarning {
        background: linear-gradient(135deg, #ffeaa7 0%, #fdcb6e 100%);
        border: none;
        border-radius: 8px;
        padding: 1rem;
        color: #856404;
    }
    
    /* Info message styling */
    .stInfo {
        background: linear-gradient(135deg, #74b9ff 0%, #a29bfe 100%);
        border: none;
        border-radius: 8px;
        padding: 1rem;
        color: #0c5460;
    }
    
    /* Code block styling for generated passwords */
    .stCode {
        background: #2d3748;
        border: 2px solid #667eea;
        border-radius: 8px;
        padding: 1rem;
        font-family: 'Courier New', Monaco, monospace;
        font-size: 18px;
        letter-spacing: 2px;
        text-align: center;
        color: #00ff88;
    }
    
    /* Expander styling */
    .streamlit-expanderHeader {
        background: #f8f9fa;
        border-radius: 8px;
        border: 2px solid #e9ecef;
        padding: 0.5rem 1rem;
        font-weight: 600;
    }
    
    .streamlit-expanderContent {
        background: #ffffff;
        border: 2px solid #e9ecef;
        border-top: none;
        border-radius: 0 0 8px 8px;
        padding: 1rem;
    }
    
    /* Radio button styling */
    .stRadio > div {
        background: #f8f9fa;
        padding: 1rem;
        border-radius: 8px;
        border: 2px solid #e9ecef;
    }
    
    /* Column styling for welcome section */
    .welcome-section {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        color: white;
        padding: 2rem;
        border-radius: 15px;
        margin: 1rem 0;
        box-shadow: 0 10px 30px rgba(102, 126, 234, 0.3);
    }
    
    /* Logout button specific styling */
    .logout-btn {
        background: #dc3545 !important;
        box-shadow: 0 4px 15px rgba(220, 53, 69, 0.3) !important;
    }
    
    .logout-btn:hover {
        background: #c82333 !important;
        box-shadow: 0 6px 20px rgba(220, 53, 69, 0.4) !important;
    }
    
    /* Account info styling */
    .account-info {
        background: #f8f9fa;
        padding: 1.5rem;
        border-radius: 10px;
        border-left: 4px solid #667eea;
        margin: 1rem 0;
    }
    
    /* Password strength indicator */
    .password-strength {
        padding: 0.5rem;
        border-radius: 5px;
        margin-top: 0.5rem;
        font-weight: 500;
    }
    
    .strength-weak {
        background-color: #ffebee;
        color: #c62828;
        border-left: 4px solid #f44336;
    }
    
    .strength-medium {
        background-color: #fff3e0;
        color: #ef6c00;
        border-left: 4px solid #ff9800;
    }
    
    .strength-strong {
        background-color: #e8f5e8;
        color: #2e7d32;
        border-left: 4px solid #4caf50;
    }
    
    /* Animation for loading states */
    @keyframes pulse {
        0% { opacity: 1; }
        50% { opacity: 0.5; }
        100% { opacity: 1; }
    }
    
    .loading {
        animation: pulse 1.5s infinite;
    }
    
    /* Mobile responsiveness */
    @media (max-width: 768px) {
        .main .block-container {
            padding: 1rem;
        }
        
        .auth-header {
            font-size: 2rem;
        }
        
        .auth-card {
            padding: 1.5rem;
            margin: 0.5rem 0;
        }
        
        .stButton > button {
            width: 100%;
            margin: 0.5rem 0;
        }
    }
    
    /* Dark theme support */
    @media (prefers-color-scheme: dark) {
        .auth-card {
            background: #1e1e1e;
            border-color: #333;
        }
        
        .stForm {
            background: #2d2d2d;
            border-color: #444;
        }
        
        .account-info {
            background: #2d2d2d;
            color: white;
        }
    }
    </style>
    """, unsafe_allow_html=True)
    
    # Only set title if not already set by main app
    if 'page_title_set' not in st.session_state:
        st.markdown('<h1 class="auth-header">🔐 Secure Authentication</h1>', unsafe_allow_html=True)
    else:
        st.markdown('<h2 class="auth-header">🔐 Authentication</h2>', unsafe_allow_html=True)
    
    # Check session timeout
    if auth_manager._check_session_timeout():
        st.warning("Your session has expired. Please log in again.")
        st.rerun()
    
    # If authenticated, show user dashboard
    if auth_manager.is_authenticated():
        username = st.session_state.username
        user_data = st.session_state.user_db[username]
        
        # Welcome section with enhanced styling
        st.markdown(f"""
        <div class="welcome-section">
            <h2 style="margin: 0; font-size: 1.8rem;">👋 Welcome back, {username}!</h2>
            <p style="margin: 0.5rem 0 0 0; opacity: 0.9;">You are successfully logged in to your secure account.</p>
        </div>
        """, unsafe_allow_html=True)
        
        col1, col2 = st.columns([3, 1])
        with col2:
            if st.button("🚪 Logout", type="secondary", key="logout_btn"):
                auth_manager.logout()
                st.rerun()
        
        # Show user info in styled container
        with st.expander("👤 Account Information", expanded=False):
            st.markdown(f"""
            <div class="account-info">
                <h4 style="margin-top: 0; color: #667eea;">📋 Account Details</h4>
                <p><strong>Username:</strong> {username}</p>
                {f'<p><strong>Account Created:</strong> {datetime.fromisoformat(user_data["created_at"]).strftime("%B %d, %Y at %H:%M")}</p>' if user_data.get('created_at') else ''}
                {f'<p><strong>Last Login:</strong> {datetime.fromisoformat(user_data["last_login"]).strftime("%B %d, %Y at %H:%M")}</p>' if user_data.get('last_login') else ''}
                <p><strong>Account Status:</strong> <span style="color: #28a745;">✅ Active & Secure</span></p>
            </div>
            """, unsafe_allow_html=True)
        
        # Password change section with enhanced styling
        if auth_manager.requires_password_change():
            st.markdown("""
            <div style="background: linear-gradient(135deg, #ffeaa7 0%, #fdcb6e 100%); 
                        padding: 1rem; border-radius: 8px; margin: 1rem 0; 
                        border-left: 4px solid #f39c12;">
                <strong>⚠️ Security Notice:</strong> You are using a generated password. 
                Please change it to something memorable and secure.
            </div>
            """, unsafe_allow_html=True)
        
        with st.expander("🔑 Change Password"):
            with st.form("change_password_form"):
                old_password = st.text_input("Current Password", type="password")
                new_password = st.text_input("New Password", type="password")
                confirm_password = st.text_input("Confirm New Password", type="password")
                
                if st.form_submit_button("Change Password"):
                    if new_password != confirm_password:
                        st.error("New passwords do not match.")
                    else:
                        success, message = auth_manager.change_password(username, old_password, new_password)
                        if success:
                            st.success(message)
                        else:
                            st.error(message)
        
        return True
    
    # Authentication forms
    st.info("Please log in or create a new account to continue.")
    
    tab1, tab2 = st.tabs(["🆕 Create Account", "🔑 Login"])
    
    with tab1:
        st.subheader("Create New Account")
        
        account_type = st.radio(
            "Choose account setup type:",
            ["Generate secure password for me", "I'll set my own password"],
            index=0
        )
        
        with st.form("create_account_form"):
            new_username = st.text_input("Choose a Username", help="At least 3 characters")
            
            if account_type == "I'll set my own password":
                new_password = st.text_input("Password", type="password")
                confirm_password = st.text_input("Confirm Password", type="password")
            else:
                new_password = None
                st.info("A secure password will be generated for you.")
            
            submit_new = st.form_submit_button("Create Account", type="primary")
            
            if submit_new:
                if account_type == "I'll set my own password":
                    if new_password != confirm_password:
                        st.error("Passwords do not match.")
                        st.stop()
                
                password_to_use = None if account_type == "Generate secure password for me" else new_password

                success, message, generated_password = auth_manager.create_user(new_username, password_to_use)

                if success:
                    st.success(message)

                    if generated_password:
                        st.markdown("---")
                        st.subheader("🔑 Your Generated Password")
                        st.code(generated_password, language=None)
                        st.warning("⚠️ **Important:** Save this password securely! You'll need it to log in.")
                        st.info("💡 You can change this password after logging in.")

                    # Auto-login
                    auth_success, _ = auth_manager.authenticate_user(new_username, generated_password or new_password)
                    # if auth_success:
                    #     st.rerun()
                else:
                    st.error(message)

    
    with tab2:
        st.subheader("Login to Your Account")
        
        with st.form("login_form"):
            username = st.text_input("Username")
            password = st.text_input("Password", type="password")
            submit_login = st.form_submit_button("Login", type="primary")
            
            if submit_login:
                success, message = auth_manager.authenticate_user(username, password)
                if success:
                    st.success(f"✅ {message}")
                    st.rerun()
                else:
                    st.error(f"❌ {message}")
        
        # Show failed attempts warning with enhanced styling
        if username and username in st.session_state.failed_attempts:
            attempts = st.session_state.failed_attempts[username]
            if attempts > 0:
                remaining = auth_manager.max_login_attempts - attempts
                if remaining > 0:
                    st.markdown(f"""
                    <div style="background: linear-gradient(135deg, #ffeaa7 0%, #fdcb6e 100%); 
                                padding: 1rem; border-radius: 8px; margin: 1rem 0;">
                        <strong>⚠️ Security Alert:</strong> {attempts} failed attempt(s) detected. 
                        <strong>{remaining} attempts remaining</strong> before account lockout.
                    </div>
                    """, unsafe_allow_html=True)
    
    return False