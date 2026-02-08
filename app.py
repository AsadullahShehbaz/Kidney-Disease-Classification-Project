"""
🤖 Agentic RAG with Knowledge Base 
Enterprise-grade interface with advanced UI/UX design + ENHANCED LOGGING

Features:
✅ Full dark mode interface
✅ Enhanced code rendering with syntax highlighting
✅ User Authentication with enhanced security feedback
✅ Long Term Memory (Semantic Memory)
✅ Advanced Thread Management with visual indicators
✅ Document Upload & Management with drag-and-drop feel
✅ Real-time Streaming Chat with typing indicators
✅ Comprehensive Analytics Dashboard
✅ Responsive design with smooth animations
✅ DETAILED LOGGING FOR DEBUGGING
"""

import streamlit as st
import requests
import json
import time
from datetime import datetime
from typing import Optional, Dict, List, Tuple
import logging

# ========================================
# LOGGING CONFIGURATION
# ========================================
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# ========================================
# CONFIGURATION
# ========================================
API_BASE_URL = "https://enterprise-ai-assistant-with-custom-knowledge-ba-production.up.railway.app/"

st.set_page_config(
    page_title="Agentic RAG with Knowledge Base | Enterprise AI Platform",
    page_icon="🤖",
    layout="wide",
    initial_sidebar_state="expanded",
    menu_items={
        'Get Help': 'https://github.com/asadullahshehbaz',
        'Report a bug': "https://github.com/asadullahshehbaz/issues",
        'About': "# Agentic RAG with Knowledge Base\nYour intelligent AI companion powered by advanced RAG technology."
    }
)

# ========================================
# CUSTOM CSS STYLING - FULL DARK MODE
# ========================================
def inject_custom_css():
    """Inject professional custom CSS for enhanced UI/UX with full dark mode"""
    st.markdown("""
    <style>
    /* Import Google Fonts - Professional Typography */
    @import url('https://fonts.googleapis.com/css2?family=Outfit:wght@300;400;500;600;700&family=JetBrains+Mono:wght@400;500;600&display=swap');
    
    /* Root Variables - Dark Mode Design System */
    :root {
        --primary-color: #667eea;
        --primary-light: #8b9cf5;
        --primary-dark: #4c63d2;
        --accent-color: #f093fb;
        --success-color: #10b981;
        --warning-color: #f59e0b;
        --error-color: #ef4444;
        --info-color: #3b82f6;
        
        /* Dark theme colors */
        --bg-primary: #0f0f1e;
        --bg-secondary: #1a1a2e;
        --bg-tertiary: #252538;
        --bg-hover: #2d2d44;
        --text-primary: #e5e7eb;
        --text-secondary: #9ca3af;
        --text-muted: #6b7280;
        --border-color: #374151;
        --border-light: #4b5563;
        
        /* Shadows */
        --shadow-sm: 0 2px 8px rgba(0,0,0,0.3);
        --shadow-md: 0 4px 16px rgba(0,0,0,0.4);
        --shadow-lg: 0 8px 32px rgba(0,0,0,0.5);
        --shadow-glow: 0 0 20px rgba(102, 126, 234, 0.3);
        
        --transition-speed: 0.3s;
    }
    
    /* Global Styles */
    * {
        font-family: 'Outfit', -apple-system, BlinkMacSystemFont, sans-serif;
    }
    
    /* Main Container - Dark */
    .main {
        background: var(--bg-primary);
        padding: 0 !important;
        color: var(--text-primary);
    }
    
    .block-container {
        padding-top: 2rem !important;
        padding-bottom: 2rem !important;
        max-width: 1400px !important;
    }
    
    /* Headers Enhancement */
    h1, h2, h3, h4, h5, h6 {
        font-weight: 700 !important;
        letter-spacing: -0.02em !important;
        color: var(--text-primary) !important;
    }
    
    h1 {
        font-size: 2.5rem !important;
        margin-bottom: 0.5rem !important;
        background: linear-gradient(135deg, var(--primary-color) 0%, var(--accent-color) 100%);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        background-clip: text;
    }
    
    /* Sidebar Styling - DARK MODE */
    [data-testid="stSidebar"] {
        background: var(--bg-secondary) !important;
        border-right: 1px solid var(--border-color);
        padding: 0 !important;
    }
    
    [data-testid="stSidebar"] > div:first-child {
        padding: 1.5rem 1rem !important;
        background: var(--bg-secondary) !important;
    }
    
    [data-testid="stSidebar"] .element-container {
        margin-bottom: 0.5rem;
    }
    
    /* Sidebar text color */
    [data-testid="stSidebar"] * {
        color: var(--text-primary) !important;
    }
    
    [data-testid="stSidebar"] .caption {
        color: var(--text-secondary) !important;
    }
    
    /* User Profile Card - Dark Mode */
    .user-profile-card {
        background: linear-gradient(135deg, var(--primary-color) 0%, var(--primary-dark) 100%);
        padding: 2rem 1.5rem;
        border-radius: 16px;
        margin-bottom: 1.5rem;
        text-align: center;
        color: white;
        box-shadow: var(--shadow-glow);
        border: 1px solid rgba(102, 126, 234, 0.3);
    }
    
    .user-avatar {
        width: 80px;
        height: 80px;
        background: rgba(255,255,255,0.15);
        border-radius: 50%;
        margin: 0 auto 1rem;
        display: flex;
        align-items: center;
        justify-content: center;
        font-size: 2.5rem;
        border: 3px solid rgba(255,255,255,0.2);
        backdrop-filter: blur(10px);
    }
    
    .user-name {
        font-size: 1.25rem;
        font-weight: 600;
        margin: 0 0 0.25rem 0;
        color: white;
        text-shadow: 0 2px 4px rgba(0,0,0,0.2);
    }
    
    .user-role {
        font-size: 0.9rem;
        opacity: 0.9;
        margin: 0;
        color: rgba(255,255,255,0.9);
    }
    
    /* Section Headers - Dark Mode */
    .section-header {
        font-size: 0.75rem;
        font-weight: 700;
        color: var(--text-secondary);
        margin: 1.5rem 0 0.75rem 0;
        padding: 0;
        text-transform: uppercase;
        letter-spacing: 0.1em;
    }
    
    /* Button Enhancements - Dark Mode */
    .stButton > button {
        border-radius: 12px !important;
        font-weight: 500 !important;
        transition: all var(--transition-speed) cubic-bezier(0.4, 0, 0.2, 1) !important;
        border: none !important;
        box-shadow: var(--shadow-sm) !important;
        letter-spacing: 0.01em !important;
        height: 44px !important;
        font-size: 0.95rem !important;
        color: var(--text-primary) !important;
    }
    
    .stButton > button:hover {
        transform: translateY(-2px) !important;
        box-shadow: var(--shadow-md) !important;
    }
    
    .stButton > button[kind="primary"] {
        background: linear-gradient(135deg, var(--primary-color) 0%, var(--primary-dark) 100%) !important;
        color: white !important;
        box-shadow: var(--shadow-glow) !important;
    }
    
    .stButton > button[kind="primary"]:hover {
        box-shadow: 0 0 25px rgba(102, 126, 234, 0.5) !important;
    }
    
    .stButton > button[kind="secondary"] {
        background: var(--bg-tertiary) !important;
        color: var(--text-primary) !important;
        border: 1px solid var(--border-color) !important;
    }
    
    .stButton > button[kind="secondary"]:hover {
        background: var(--bg-hover) !important;
        border-color: var(--border-light) !important;
    }
    
    /* Small Buttons */
    .stButton > button[data-testid*="delete"],
    .stButton > button:has(> div:first-child:last-child) {
        min-width: 44px !important;
        padding: 0 0.75rem !important;
    }
    
    /* Input Fields - Dark Mode */
    .stTextInput > div > div > input,
    .stTextArea > div > div > textarea {
        border-radius: 12px !important;
        border: 2px solid var(--border-color) !important;
        padding: 0.75rem 1rem !important;
        font-size: 0.95rem !important;
        transition: all var(--transition-speed) !important;
        background: var(--bg-tertiary) !important;
        color: var(--text-primary) !important;
    }
    
    .stTextInput > div > div > input:focus,
    .stTextArea > div > div > textarea:focus {
        border-color: var(--primary-color) !important;
        box-shadow: 0 0 0 3px rgba(102, 126, 234, 0.2) !important;
        background: var(--bg-secondary) !important;
    }
    
    .stTextInput > div > div > input::placeholder,
    .stTextArea > div > div > textarea::placeholder {
        color: var(--text-muted) !important;
    }
    
    /* Label colors */
    .stTextInput > label,
    .stTextArea > label {
        color: var(--text-primary) !important;
    }
    
    /* Chat Message Styling - Enhanced Dark Mode */
    .stChatMessage {
        border-radius: 16px !important;
        padding: 1.25rem !important;
        margin-bottom: 1rem !important;
        box-shadow: var(--shadow-sm) !important;
        transition: all var(--transition-speed) !important;
        border: 1px solid var(--border-color) !important;
    }
    
    .stChatMessage:hover {
        box-shadow: var(--shadow-md) !important;
        border-color: var(--border-light) !important;
    }
    
    .stChatMessage[data-testid*="user"] {
        background: linear-gradient(135deg, rgba(102, 126, 234, 0.15) 0%, rgba(118, 75, 162, 0.15) 100%) !important;
        border-left: 4px solid var(--primary-color) !important;
    }
    
    .stChatMessage[data-testid*="assistant"] {
        background: var(--bg-secondary) !important;
        border-left: 4px solid var(--success-color) !important;
    }
    
    .stChatMessage * {
        color: var(--text-primary) !important;
    }
    
    /* Chat Input - Dark Mode */
    .stChatInputContainer {
        border-top: 1px solid var(--border-color) !important;
        padding-top: 1rem !important;
        background: transparent !important;
    }
    
    .stChatInput > div {
        border-radius: 24px !important;
        border: 2px solid var(--border-color) !important;
        background: var(--bg-secondary) !important;
        box-shadow: var(--shadow-md) !important;
        transition: all var(--transition-speed) !important;
    }
    
    .stChatInput > div:focus-within {
        border-color: var(--primary-color) !important;
        box-shadow: 0 0 0 4px rgba(102, 126, 234, 0.2), var(--shadow-md) !important;
        background: var(--bg-tertiary) !important;
    }
    
    .stChatInput input {
        color: var(--text-primary) !important;
        background: transparent !important;
    }
    
    .stChatInput input::placeholder {
        color: var(--text-muted) !important;
    }
    
    /* File Uploader - Dark Mode */
    [data-testid="stFileUploader"] {
        border: 2px dashed var(--border-color) !important;
        border-radius: 16px !important;
        padding: 2rem 1.5rem !important;
        background: var(--bg-tertiary) !important;
        transition: all var(--transition-speed) !important;
        text-align: center !important;
    }
    
    [data-testid="stFileUploader"]:hover {
        border-color: var(--primary-color) !important;
        background: var(--bg-hover) !important;
    }
    
    [data-testid="stFileUploader"] section {
        border: none !important;
        padding: 0 !important;
    }
    
    [data-testid="stFileUploader"] button {
        background: linear-gradient(135deg, var(--primary-color) 0%, var(--primary-dark) 100%) !important;
        color: white !important;
        border: none !important;
        border-radius: 8px !important;
        padding: 0.5rem 1.5rem !important;
        font-weight: 500 !important;
        margin-top: 0.5rem !important;
    }
    
    /* Expander Styling - Dark Mode */
    .streamlit-expanderHeader {
        border-radius: 12px !important;
        background: var(--bg-tertiary) !important;
        border: 1px solid var(--border-color) !important;
        font-weight: 500 !important;
        padding: 0.75rem 1rem !important;
        transition: all var(--transition-speed) !important;
        color: var(--text-primary) !important;
    }
    
    .streamlit-expanderHeader:hover {
        background: var(--bg-hover) !important;
        border-color: var(--primary-color) !important;
    }
    
    .streamlit-expanderContent {
        background: var(--bg-secondary) !important;
        border: 1px solid var(--border-color) !important;
        border-top: none !important;
    }
    
    /* Metrics Enhancement - Dark Mode */
    [data-testid="stMetric"] {
        background: var(--bg-tertiary) !important;
        padding: 1rem !important;
        border-radius: 12px !important;
        border: 1px solid var(--border-color) !important;
        box-shadow: var(--shadow-sm) !important;
        transition: all var(--transition-speed) !important;
    }
    
    [data-testid="stMetric"]:hover {
        box-shadow: var(--shadow-md) !important;
        transform: translateY(-2px) !important;
        border-color: var(--border-light) !important;
    }
    
    [data-testid="stMetricValue"] {
        font-size: 1.75rem !important;
        font-weight: 700 !important;
        background: linear-gradient(135deg, var(--primary-color) 0%, var(--accent-color) 100%);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        background-clip: text;
    }
    
    [data-testid="stMetricLabel"] {
        color: var(--text-secondary) !important;
        font-weight: 500 !important;
    }
    
    /* Info/Warning/Success/Error Boxes - Dark Mode */
    .stAlert {
        border-radius: 12px !important;
        border: 1px solid !important;
        padding: 1rem 1.25rem !important;
        font-weight: 500 !important;
        box-shadow: var(--shadow-sm) !important;
    }
    
    /* Info */
    [data-testid="stNotification"][kind="info"],
    .stAlert[kind="info"] {
        background: rgba(59, 130, 246, 0.1) !important;
        border-color: var(--info-color) !important;
        color: #93c5fd !important;
    }
    
    /* Success */
    [data-testid="stNotification"][kind="success"],
    .stAlert[kind="success"] {
        background: rgba(16, 185, 129, 0.1) !important;
        border-color: var(--success-color) !important;
        color: #6ee7b7 !important;
    }
    
    /* Warning */
    [data-testid="stNotification"][kind="warning"],
    .stAlert[kind="warning"] {
        background: rgba(245, 158, 11, 0.1) !important;
        border-color: var(--warning-color) !important;
        color: #fcd34d !important;
    }
    
    /* Error */
    [data-testid="stNotification"][kind="error"],
    .stAlert[kind="error"] {
        background: rgba(239, 68, 68, 0.1) !important;
        border-color: var(--error-color) !important;
        color: #fca5a5 !important;
    }
    
    /* Divider */
    hr {
        margin: 1rem 0 !important;
        border: none !important;
        height: 1px !important;
        background: var(--border-color) !important;
    }
    
    /* Tabs Enhancement - Dark Mode */
    .stTabs [data-baseweb="tab-list"] {
        gap: 8px;
        background: var(--bg-tertiary) !important;
        border-radius: 12px !important;
        padding: 4px !important;
    }
    
    .stTabs [data-baseweb="tab"] {
        border-radius: 8px !important;
        padding: 0.75rem 1.5rem !important;
        font-weight: 500 !important;
        transition: all var(--transition-speed) !important;
        color: var(--text-secondary) !important;
        background: transparent !important;
    }
    
    .stTabs [data-baseweb="tab"]:hover {
        background: var(--bg-hover) !important;
        color: var(--text-primary) !important;
    }
    
    .stTabs [aria-selected="true"] {
        background: linear-gradient(135deg, var(--primary-color) 0%, var(--primary-dark) 100%) !important;
        color: white !important;
    }
    
    /* Spinner Enhancement */
    .stSpinner > div {
        border-color: var(--primary-color) !important;
        border-right-color: transparent !important;
    }
    
    /* Code blocks - ENHANCED RENDERING */
    pre {
        background: #1e1e2e !important;
        border: 1px solid var(--border-color) !important;
        border-radius: 12px !important;
        padding: 1rem !important;
        overflow-x: auto !important;
        box-shadow: var(--shadow-sm) !important;
    }
    
    code {
        font-family: 'JetBrains Mono', 'Courier New', monospace !important;
        font-size: 0.9em !important;
        font-weight: 500 !important;
        line-height: 1.6 !important;
    }
    
    /* Inline code */
    :not(pre) > code {
        background: rgba(102, 126, 234, 0.15) !important;
        padding: 0.2rem 0.5rem !important;
        border-radius: 6px !important;
        border: 1px solid rgba(102, 126, 234, 0.3) !important;
        color: #93c5fd !important;
    }
    
    /* Code block */
    pre code {
        background: transparent !important;
        padding: 0 !important;
        border: none !important;
        color: #e5e7eb !important;
        display: block !important;
    }
    
    /* Syntax highlighting for Python */
    .language-python .hljs-keyword { color: #c792ea !important; }
    .language-python .hljs-string { color: #c3e88d !important; }
    .language-python .hljs-number { color: #f78c6c !important; }
    .language-python .hljs-comment { color: #676e95 !important; font-style: italic !important; }
    .language-python .hljs-function { color: #82aaff !important; }
    .language-python .hljs-class { color: #ffcb6b !important; }
    .language-python .hljs-built_in { color: #89ddff !important; }
    
    /* Caption text */
    .caption {
        color: var(--text-secondary) !important;
        font-size: 0.875rem !important;
        font-weight: 500 !important;
    }
    
    /* Status badges - Dark Mode */
    .status-badge {
        display: inline-block;
        padding: 0.3rem 0.85rem;
        border-radius: 20px;
        font-size: 0.75rem;
        font-weight: 700;
        letter-spacing: 0.05em;
        margin: 0.5rem 0;
        text-transform: uppercase;
    }
    
    .status-pending {
        background: rgba(245, 158, 11, 0.2);
        color: #fcd34d;
        border: 1px solid rgba(245, 158, 11, 0.3);
    }
    
    .status-processing {
        background: rgba(59, 130, 246, 0.2);
        color: #93c5fd;
        border: 1px solid rgba(59, 130, 246, 0.3);
    }
    
    .status-completed {
        background: rgba(16, 185, 129, 0.2);
        color: #6ee7b7;
        border: 1px solid rgba(16, 185, 129, 0.3);
    }
    
    .status-failed {
        background: rgba(239, 68, 68, 0.2);
        color: #fca5a5;
        border: 1px solid rgba(239, 68, 68, 0.3);
    }
    
    /* Smooth scroll */
    html {
        scroll-behavior: smooth;
    }
    
    /* Custom scrollbar - Dark Mode */
    ::-webkit-scrollbar {
        width: 10px;
        height: 10px;
    }
    
    ::-webkit-scrollbar-track {
        background: var(--bg-secondary);
        border-radius: 10px;
    }
    
    ::-webkit-scrollbar-thumb {
        background: var(--border-light);
        border-radius: 10px;
        transition: background var(--transition-speed);
    }
    
    ::-webkit-scrollbar-thumb:hover {
        background: var(--primary-color);
    }
    
    /* Animation keyframes */
    @keyframes fadeIn {
        from {
            opacity: 0;
            transform: translateY(10px);
        }
        to {
            opacity: 1;
            transform: translateY(0);
        }
    }
    
    @keyframes slideIn {
        from {
            transform: translateX(-100%);
            opacity: 0;
        }
        to {
            transform: translateX(0);
            opacity: 1;
        }
    }
    
    @keyframes pulse {
        0%, 100% {
            opacity: 1;
        }
        50% {
            opacity: 0.5;
        }
    }
    
    @keyframes glow {
        0%, 100% {
            box-shadow: 0 0 20px rgba(102, 126, 234, 0.3);
        }
        50% {
            box-shadow: 0 0 30px rgba(102, 126, 234, 0.5);
        }
    }
    
    /* Apply animations */
    .element-container {
        animation: fadeIn 0.4s ease-out;
    }
    
    .user-profile-card {
        animation: glow 3s ease-in-out infinite;
    }
    
    /* Hero gradient background - Dark */
    .hero-section {
        background: linear-gradient(135deg, rgba(102, 126, 234, 0.2) 0%, rgba(118, 75, 162, 0.2) 100%);
        border: 1px solid rgba(102, 126, 234, 0.3);
        border-radius: 20px;
        padding: 2rem;
        color: var(--text-primary);
        box-shadow: var(--shadow-glow);
        margin-bottom: 2rem;
        backdrop-filter: blur(10px);
    }
    
    /* Footer styling */
    footer {
        text-align: center;
        padding: 2rem 0;
        color: var(--text-muted);
        font-size: 0.9rem;
        border-top: 1px solid var(--border-color);
        margin-top: 2rem;
    }
    
    /* Checkbox - Dark Mode */
    [data-testid="stCheckbox"] {
        color: var(--text-primary) !important;
    }
    
    /* Select box - Dark Mode */
    .stSelectbox > div > div {
        background: var(--bg-tertiary) !important;
        border-color: var(--border-color) !important;
        color: var(--text-primary) !important;
    }
    
    /* Markdown links */
    a {
        color: var(--primary-light) !important;
        text-decoration: none !important;
        transition: color var(--transition-speed) !important;
    }
    
    a:hover {
        color: var(--accent-color) !important;
        text-decoration: underline !important;
    }
    
    /* Responsive design */
    @media (max-width: 768px) {
        .block-container {
            padding: 1rem !important;
        }
        
        h1 {
            font-size: 1.75rem !important;
        }
        
        .stButton > button {
            height: 40px !important;
            font-size: 0.9rem !important;
        }
        
        .user-avatar {
            width: 60px !important;
            height: 60px !important;
            font-size: 2rem !important;
        }
    }
    </style>
    """, unsafe_allow_html=True)

inject_custom_css()

# ========================================
# SESSION STATE INITIALIZATION
# ========================================
def init_session_state():
    """Initialize all session state variables with type safety"""
    defaults = {
        "token": None,
        "username": None,
        "current_thread_id": None,
        "messages": [],
        "threads": [],
        "documents": [],
        "show_welcome": True,
        "stats_cache": None,
        "last_stats_update": None,
        "debug_logs": [],  # NEW: Store debug logs
    }
    
    for key, value in defaults.items():
        if key not in st.session_state:
            st.session_state[key] = value

init_session_state()

# ========================================
# LOGGING HELPER
# ========================================
def log_debug(message: str):
    """Add debug log to session state and print to console"""
    timestamp = datetime.now().strftime("%H:%M:%S.%f")[:-3]
    log_entry = f"[{timestamp}] {message}"
    logger.info(message)
    st.session_state.debug_logs.append(log_entry)
    # Keep only last 100 logs
    if len(st.session_state.debug_logs) > 100:
        st.session_state.debug_logs = st.session_state.debug_logs[-100:]

# ========================================
# API HELPER FUNCTIONS
# ========================================

def get_headers() -> Dict[str, str]:
    """Get authorization headers"""
    if st.session_state.token:
        return {"Authorization": f"Bearer {st.session_state.token}"}
    return {}

def api_call(method: str, endpoint: str, **kwargs) -> Tuple[bool, any]:
    """
    Generic API call function with error handling
    
    Args:
        method: HTTP method (GET, POST, DELETE, PATCH)
        endpoint: API endpoint path
        **kwargs: Additional arguments for requests
    
    Returns:
        tuple: (success: bool, data: dict/list/str)
    """
    try:
        url = f"{API_BASE_URL}{endpoint}"
        log_debug(f"API {method} {endpoint}")
        
        response = requests.request(method, url, timeout=30, **kwargs)
        log_debug(f"Response: {response.status_code}")
        
        if response.status_code in [200, 201]:
            try:
                return True, response.json()
            except:
                return True, response.text
        else:
            try:
                error_msg = response.json().get("detail", f"Error {response.status_code}")
            except:
                error_msg = f"Error {response.status_code}"
            log_debug(f"API Error: {error_msg}")
            return False, error_msg
            
    except requests.exceptions.Timeout:
        log_debug("⏱️ Request timeout")
        return False, "Request timeout. Please try again."
    except requests.exceptions.ConnectionError:
        log_debug("🔌 Connection error")
        return False, "Connection error. Please check your internet connection."
    except Exception as e:
        log_debug(f"💥 Exception: {str(e)}")
        return False, f"Unexpected error: {str(e)}"

# ========================================
# AUTHENTICATION FUNCTIONS
# ========================================

def register_user(username: str, email: str, password: str) -> Tuple[bool, str]:
    """Register a new user"""
    return api_call(
        "POST",
        "/api/auth/register",
        json={"username": username, "email": email, "password": password}
    )

def login_user(username: str, password: str) -> Tuple[bool, str]:
    """Login user and store token"""
    success, data = api_call(
        "POST",
        "/api/auth/login",
        data={"username": username, "password": password}
    )
    
    if success:
        st.session_state.token = data["access_token"]
        st.session_state.username = username
        log_debug(f"✅ User logged in: {username}")
        return True, "Login successful!"
    
    return False, data

def logout():
    """Clear session and logout"""
    log_debug("🚪 User logged out")
    keys_to_clear = ["token", "username", "current_thread_id", "messages", 
                     "threads", "documents", "stats_cache", "last_stats_update"]
    for key in keys_to_clear:
        if key in ["token", "username", "current_thread_id", "stats_cache", "last_stats_update"]:
            st.session_state[key] = None
        else:
            st.session_state[key] = []
    st.session_state.show_welcome = True

# ========================================
# THREAD MANAGEMENT FUNCTIONS
# ========================================

def get_threads() -> List[Dict]:
    """Fetch all threads"""
    success, data = api_call("GET", "/api/chat/threads", headers=get_headers())
    if success:
        st.session_state.threads = data
        log_debug(f"📚 Loaded {len(data)} threads")
        return data
    return []

def create_thread() -> Optional[str]:
    """Create a new thread"""
    success, data = api_call("POST", "/api/chat/threads/new", headers=get_headers())
    if success:
        log_debug(f"➕ Created thread: {data['thread_id']}")
        return data["thread_id"]
    return None

def get_thread_history(thread_id: str) -> List[Dict]:
    """Get thread history"""
    log_debug(f"📖 Loading history for: {thread_id}")
    success, data = api_call(
        "GET",
        f"/api/chat/threads/{thread_id}",
        headers=get_headers()
    )
    if success:
        history = data.get("history", [])
        log_debug(f"✅ Loaded {len(history)} messages")
        return history
    return []

def delete_thread(thread_id: str) -> bool:
    """Delete a thread"""
    success, _ = api_call(
        "DELETE",
        f"/api/chat/threads/{thread_id}",
        headers=get_headers()
    )
    if success:
        log_debug(f"🗑️ Deleted thread: {thread_id}")
    return success

def update_thread_title(thread_id: str, new_title: str) -> bool:
    """Update thread title"""
    success, _ = api_call(
        "PATCH",
        f"/api/chat/threads/{thread_id}",
        headers=get_headers(),
        json={"title": new_title}
    )
    if success:
        log_debug(f"✏️ Updated title: {new_title}")
    return success

# ========================================
# CHAT FUNCTIONS (STREAMING WITH LOGGING)
# ========================================

def stream_message(message: str, thread_id: Optional[str] = None):
    """
    Stream chat response with DETAILED LOGGING
    
    Yields:
        dict: Chunk with type and data
    """
    try:
        log_debug(f"🚀 Stream START - Msg: '{message[:40]}...'")
        log_debug(f"🔖 Thread: {thread_id}")
        
        response = requests.post(
            f"{API_BASE_URL}/api/chat/stream",
            headers=get_headers(),
            json={"message": message, "thread_id": thread_id},
            stream=True,
            timeout=120
        )
        
        log_debug(f"📡 HTTP {response.status_code}")
        
        if response.status_code != 200:
            error_msg = f"Error: {response.status_code}"
            log_debug(f"❌ {error_msg}")
            yield {"type": "error", "message": error_msg}
            return
        
        chunk_count = 0
        for line in response.iter_lines(decode_unicode=True):
            if not line or not line.startswith("data: "):
                continue
            
            data_str = line[6:]
            
            if data_str == "[DONE]":
                log_debug(f"✅ Stream DONE - {chunk_count} chunks")
                break
            
            try:
                chunk = json.loads(data_str)
                chunk_count += 1
                chunk_type = chunk.get("type")
                
                # Log chunk details
                if chunk_type == "content":
                    content_len = len(chunk.get("data", ""))
                    log_debug(f"💬 Content #{chunk_count} ({content_len} chars)")
                elif chunk_type == "status":
                    log_debug(f"📊 Status: {chunk.get('status')}")
                elif chunk_type == "tool_start":
                    log_debug(f"🔧 Tool: {chunk.get('tool')}")
                elif chunk_type == "sources":
                    log_debug(f"📚 Sources: {len(chunk.get('sources', []))}")
                elif chunk_type == "error":
                    log_debug(f"❌ Error: {chunk.get('message')}")
                
                yield chunk
                
            except json.JSONDecodeError as e:
                log_debug(f"⚠️ JSON error: {str(e)}")
                continue
                
    except requests.exceptions.Timeout:
        log_debug("⏱️ Stream timeout")
        yield {"type": "error", "message": "Request timeout. Please try again."}
    except Exception as e:
        log_debug(f"💥 Stream error: {str(e)}")
        yield {"type": "error", "message": str(e)}

# ========================================
# DOCUMENT FUNCTIONS
# ========================================

def upload_document(file) -> Tuple[bool, str]:
    """Upload a PDF document"""
    log_debug(f"📤 Uploading: {file.name}")
    files = {"file": (file.name, file.getvalue(), "application/pdf")}
    return api_call(
        "POST",
        "/api/documents/upload",
        headers=get_headers(),
        files=files
    )

def get_documents() -> List[Dict]:
    """Get all documents"""
    success, data = api_call("GET", "/api/documents/", headers=get_headers())
    if success:
        st.session_state.documents = data
        log_debug(f"📄 Loaded {len(data)} documents")
        return data
    return []

def delete_document(doc_id: str) -> bool:
    """Delete a document"""
    success, _ = api_call(
        "DELETE",
        f"/api/documents/{doc_id}",
        headers=get_headers()
    )
    if success:
        log_debug(f"🗑️ Deleted document: {doc_id}")
    return success

def get_upload_stats() -> Optional[Dict]:
    """Get document upload statistics with caching"""
    current_time = time.time()
    
    # Cache for 30 seconds
    if (st.session_state.last_stats_update and 
        current_time - st.session_state.last_stats_update < 30 and
        st.session_state.stats_cache):
        return st.session_state.stats_cache
    
    success, data = api_call(
        "GET",
        "/api/documents/stats/summary",
        headers=get_headers()
    )
    
    if success:
        st.session_state.stats_cache = data
        st.session_state.last_stats_update = current_time
        return data
    return None

# ========================================
# UI COMPONENTS - DARK MODE DESIGN
# ========================================

def render_hero_section():
    """Render hero section for login page"""
    st.markdown("""
    <div class="hero-section">
        <h1 style="font-size: 3rem; margin-bottom: 1rem; -webkit-text-fill-color: initial; color: var(--text-primary);">
            🤖 Agentic RAG Assistant
        </h1>
        <p style="font-size: 1.25rem; opacity: 0.9; margin-bottom: 0; color: var(--text-secondary);">
            Experience the future of AI-powered conversations with advanced RAG technology
        </p>
    </div>
    """, unsafe_allow_html=True)

def render_login_page():
    """Render enhanced login/register page with dark mode"""
    
    render_hero_section()
    
    # Feature highlights
    col1, col2, col3 = st.columns(3)
    
    features = [
        ("🧠", "Smart Memory", "Long-term semantic memory for context-aware conversations"),
        ("📚", "Document RAG", "Upload and chat with your PDF documents seamlessly"),
        ("⚡", "Real-time AI", "Streaming responses with multi-tool integration")
    ]
    
    for col, (icon, title, desc) in zip([col1, col2, col3], features):
        with col:
            st.markdown(f"""
            <div style="text-align: center; padding: 1.5rem; background: var(--bg-secondary); 
                        border-radius: 12px; box-shadow: var(--shadow-sm); 
                        border: 1px solid var(--border-color); transition: all 0.3s;">
                <h3 style="font-size: 2.5rem; margin: 0; color: var(--primary-color);">{icon}</h3>
                <h4 style="margin: 0.5rem 0; color: var(--text-primary);">{title}</h4>
                <p style="color: var(--text-secondary); font-size: 0.9rem; margin: 0;">{desc}</p>
            </div>
            """, unsafe_allow_html=True)
    
    st.markdown("<br>", unsafe_allow_html=True)
    
    # Login/Register Tabs
    tab1, tab2 = st.tabs(["🔐 Login", "📝 Register"])
    
    # LOGIN TAB
    with tab1:
        st.markdown("### Welcome Back!")
        st.caption("Sign in to access your AI assistant")
        
        with st.form("login_form", clear_on_submit=False):
            username = st.text_input(
                "Username",
                placeholder="Enter your username",
                help="Your registered username"
            )
            password = st.text_input(
                "Password",
                type="password",
                placeholder="Enter your password",
                help="Your account password"
            )
            
            col1, col2 = st.columns([3, 1])
            with col1:
                submit = st.form_submit_button("🚀 Sign In", use_container_width=True, type="primary")
            with col2:
                st.form_submit_button("❓ Help", use_container_width=True)
            
            if submit:
                if username and password:
                    with st.spinner("🔐 Authenticating..."):
                        success, message = login_user(username, password)
                        if success:
                            st.success("✅ " + message)
                            st.balloons()
                            time.sleep(0.8)
                            st.rerun()
                        else:
                            st.error("❌ " + message)
                else:
                    st.warning("⚠️ Please fill in all fields")
    
    # REGISTER TAB
    with tab2:
        st.markdown("### Create Your Account")
        st.caption("Join thousands of users leveraging AI for productivity")
        
        with st.form("register_form", clear_on_submit=True):
            col1, col2 = st.columns(2)
            
            with col1:
                reg_username = st.text_input(
                    "Username",
                    placeholder="Choose a unique username",
                    help="4-20 characters, letters and numbers only"
                )
            
            with col2:
                reg_email = st.text_input(
                    "Email",
                    placeholder="your.email@example.com",
                    help="A valid email address"
                )
            
            reg_password = st.text_input(
                "Password",
                type="password",
                placeholder="Create a strong password",
                help="Minimum 8 characters, mix of letters and numbers"
            )
            reg_password2 = st.text_input(
                "Confirm Password",
                type="password",
                placeholder="Confirm your password"
            )
            
            agree_terms = st.checkbox("I agree to the Terms of Service and Privacy Policy")
            
            submit_reg = st.form_submit_button("📝 Create Account", use_container_width=True, type="primary")
            
            if submit_reg:
                if not agree_terms:
                    st.error("❌ Please agree to the terms and conditions")
                elif reg_username and reg_email and reg_password:
                    if reg_password == reg_password2:
                        if len(reg_password) < 8:
                            st.error("❌ Password must be at least 8 characters long")
                        else:
                            with st.spinner("📝 Creating your account..."):
                                success, message = register_user(reg_username, reg_email, reg_password)
                                if success:
                                    st.success("✅ Account created successfully! Please login.")
                                    st.balloons()
                                else:
                                    st.error("❌ " + str(message))
                    else:
                        st.error("❌ Passwords don't match")
                else:
                    st.warning("⚠️ Please fill in all fields")

def render_sidebar():
    """Render enhanced sidebar - REORGANIZED LAYOUT"""
    
    with st.sidebar:
        # 1. USER PROFILE SECTION
        st.markdown(f"""
        <div class="user-profile-card">
            <div class="user-avatar">👤</div>
            <div class="user-name">{st.session_state.username.upper()}</div>
            <div class="user-role">AI Power User</div>
        </div>
        """, unsafe_allow_html=True)
        
        if st.button("🚪 Logout", use_container_width=True, type="secondary", key="logout_btn"):
            logout()
            st.rerun()
        
        st.divider()
        
        # 2. KNOWLEDGE BASE / RAG SECTION (Below Conversations)
        st.markdown('<div class="section-header">📚 Knowledge Base</div>', unsafe_allow_html=True)
        
        uploaded_file = st.file_uploader(
            "Upload PDF Document",
            type=["pdf"],
            help="Upload PDF documents to enhance AI responses",
            label_visibility="collapsed",
            key="file_uploader"
        )
        
        if uploaded_file:
            if st.button("📤 Upload Document", use_container_width=True, type="primary", key="upload_doc"):
                with st.spinner("📤 Uploading and processing..."):
                    success, message = upload_document(uploaded_file)
                    if success:
                        st.success("✅ Document uploaded!")
                        time.sleep(1)
                        st.rerun()
                    else:
                        st.error(f"❌ {message}")
        
        # DOCUMENT LIST
        st.markdown("<br>", unsafe_allow_html=True)
        st.caption("**Your Documents:**")
        
        documents = get_documents()
        
        if documents:
            for doc in documents:
                status_emoji = {
                    "pending": "⏳",
                    "processing": "⚙️",
                    "completed": "✅",
                    "failed": "❌"
                }.get(doc['status'], "❓")
                
                status_class = f"status-{doc['status']}"
                
                with st.expander(f"{status_emoji} {doc['filename'][:25]}...", expanded=False):
                    st.markdown(f"""
                    <div style="padding: 0.5rem 0;">
                        <span class="status-badge {status_class}">{doc['status'].upper()}</span>
                    </div>
                    """, unsafe_allow_html=True)
                    
                    st.caption(f"**Size:** {doc['file_size']:,} bytes")
                    st.caption(f"**Uploaded:** {doc['uploaded_at'][:16]}")
                    
                    if doc.get('error_message'):
                        st.error(f"⚠️ {doc['error_message']}")
                    
                    if st.button("🗑️ Remove", key=f"del_doc_{doc['id']}", use_container_width=True):
                        if delete_document(doc['id']):
                            st.success("✅ Removed!")
                            time.sleep(0.5)
                            st.rerun()
        else:
            st.info("💡 No documents yet. Upload PDFs to get started!")
        
        st.divider()
        
        # 4. STATISTICS (As Dropdown - Below RAG)
        with st.expander("📊 Statistics", expanded=False):
            stats = get_upload_stats()
            if stats:
                col1, col2 = st.columns(2)
                with col1:
                    st.metric("Total", stats.get('total_uploads', 0))
                    st.metric("Processing", stats.get('processing', 0))
                
                with col2:
                    st.metric("Ready", stats.get('completed', 0))
                    failed_count = stats.get('failed', 0)
                    st.metric("Failed", failed_count)
            else:
                st.info("No statistics available")
        # 2. CONVERSATIONS SECTION (Below User Profile)
        st.markdown('<div class="section-header">💬 Conversations</div>', unsafe_allow_html=True)
        
        col1, col2 = st.columns([3, 1])
        with col1:
            if st.button("➕ New Chat", use_container_width=True, type="primary", key="new_chat_btn"):
                with st.spinner("Creating new conversation..."):
                    new_thread_id = create_thread()
                    if new_thread_id:
                        st.session_state.current_thread_id = new_thread_id
                        st.session_state.messages = []
                        st.success("✅ New chat created!")
                        time.sleep(0.5)
                        st.rerun()
                    else:
                        st.error("❌ Failed to create chat")
        
        with col2:
            if st.button("🔄", use_container_width=True, help="Refresh conversations", key="refresh_threads"):
                get_threads()
                st.rerun()
        
        # THREAD LIST
        threads = get_threads()
        
        if threads:
            st.caption(f"**{len(threads)} active conversations**")
            st.markdown("<br>", unsafe_allow_html=True)
            
            for idx, thread in enumerate(threads):
                title = thread.get('title', 'Untitled')
                if len(title) > 28:
                    title = title[:28] + "..."
                
                is_current = thread['id'] == st.session_state.current_thread_id
                
                # Thread container
                col1, col2 = st.columns([5, 1])
                
                with col1:
                    button_type = "primary" if is_current else "secondary"
                    emoji = "📌" if is_current else "💬"
                    
                    if st.button(
                        f"{emoji} {title}",
                        key=f"thread_{thread['id']}",
                        use_container_width=True,
                        type=button_type
                    ):
                        if not is_current:
                            st.session_state.current_thread_id = thread['id']
                            st.session_state.messages = get_thread_history(thread['id'])
                            st.rerun()
                
                with col2:
                    if st.button("🗑️", key=f"delete_{thread['id']}", help="Delete"):
                        if delete_thread(thread['id']):
                            if thread['id'] == st.session_state.current_thread_id:
                                st.session_state.current_thread_id = None
                                st.session_state.messages = []
                            st.success("✅ Deleted!")
                            time.sleep(0.5)
                            st.rerun()
        else:
            st.info("💡 No conversations yet. Start a new one!")
        
        st.divider()
        
        # RENAME CURRENT THREAD
        if st.session_state.current_thread_id:
            with st.expander("✏️ Rename Current Chat", expanded=False):
                new_title = st.text_input(
                    "New title",
                    key="rename_input",
                    placeholder="Enter a descriptive title"
                )
                if st.button("💾 Save Title", use_container_width=True, key="save_title"):
                    if new_title:
                        if update_thread_title(st.session_state.current_thread_id, new_title):
                            st.success("✅ Title updated!")
                            time.sleep(0.5)
                            st.rerun()
                    else:
                        st.warning("⚠️ Please enter a title")
        
        st.divider()
        


def render_chat_interface():
    """Render main chat interface with streaming preserved"""
    
    # Header
    st.markdown("# 💬 Agentic RAG with Knowledge Base")
    
    if st.session_state.current_thread_id:
        st.markdown(f"""
        <div style="background: linear-gradient(135deg, rgba(102, 126, 234, 0.15) 0%, rgba(118, 75, 162, 0.15) 100%); 
                    padding: 1rem 1.5rem; border-radius: 12px; margin-bottom: 1.5rem;
                    border-left: 4px solid var(--primary-color); border: 1px solid rgba(102, 126, 234, 0.3);">
            <p style="margin: 0; color: var(--primary-light); font-weight: 600;">
                🔖 Active Thread: <code style="background: rgba(102, 126, 234, 0.2); padding: 0.2rem 0.5rem; 
                border-radius: 6px; font-size: 0.85rem; color: #93c5fd;">{st.session_state.current_thread_id}</code>
            </p>
        </div>
        """, unsafe_allow_html=True)
    else:
        st.info("👋 Start a new conversation or select one from the sidebar to begin chatting!")
    
    # Debug logs expander
    with st.expander("🐛 Debug Logs", expanded=False):
        if st.session_state.debug_logs:
            col1, col2 = st.columns([4, 1])
            with col2:
                if st.button("🗑️ Clear", key="clear_logs"):
                    st.session_state.debug_logs = []
                    st.rerun()
            
            log_text = "\n".join(st.session_state.debug_logs[-50:])
            st.code(log_text, language="log")
        else:
            st.info("No logs yet. Start chatting to see debug information.")
    
    st.divider()
    
    # Display message history
    log_debug(f"📝 Displaying {len(st.session_state.messages)} messages")
    for msg in st.session_state.messages:
        with st.chat_message(msg["role"], avatar="👤" if msg["role"] == "user" else "🤖"):
            st.markdown(msg["content"])
            
            # Show sources if available
            if msg.get("sources"):
                with st.expander("📚 Sources Referenced", expanded=False):
                    for idx, source in enumerate(msg["sources"], 1):
                        st.markdown(f"**{idx}.** {source}")
    
    # Chat input - STREAMING LOGIC PRESERVED EXACTLY
    user_input = st.chat_input("💭 Ask me anything... (e.g., 'Summarize my documents', 'Calculate 25 * 48')")
    
    if user_input:
        log_debug(f"👤 User: '{user_input[:50]}...'")
        
        # Add user message
        st.session_state.messages.append({
            "role": "user",
            "content": user_input
        })
        
        # Display user message
        with st.chat_message("user", avatar="👤"):
            st.markdown(user_input)
        
        # Get and display AI response with EXACT streaming logic
        with st.chat_message("assistant", avatar="🤖"):
            status_placeholder = st.empty()
            message_placeholder = st.empty()
            full_response = ""
            is_generating = False
            sources = []
            
            # Stream the response - EXACT LOGIC PRESERVED
            for chunk in stream_message(user_input, st.session_state.current_thread_id):
                chunk_type = chunk.get("type")
                
                # Handle different chunk types - EXACT LOGIC
                if chunk_type == "status":
                    status = chunk.get("status")
                    message = chunk.get("message", "Processing...")
                    
                    if status == "retrieving":
                        status_placeholder.info(f"🔍 {message}")
                    elif status == "complete":
                        status_placeholder.success(f"✅ {message}")
                        time.sleep(1)
                        status_placeholder.empty()
                    elif status == "started":
                        status_placeholder.info(f"⚡ {message}")
                        time.sleep(1)
                        status_placeholder.empty()
                
                elif chunk_type == "tool_start":
                    tool_name = chunk.get("tool", "unknown")
                    
                    tool_display = {
                        "search_my_documents": "🔍 Searching your documents",
                        "calculator": "🧮 Calculating",
                        "google_web_search": "🌐 Searching the web",
                        "web_scrape": "📄 Fetching webpage"
                    }
                    
                    display_name = tool_display.get(tool_name, f"🔧 Using {tool_name}")
                    status_placeholder.info(display_name + "...")
                
                elif chunk_type == "tool_complete":
                    message = chunk.get("message", "Tool complete")
                    status_placeholder.success(f" {message}")
                    time.sleep(1)
                    status_placeholder.empty()
                
                elif chunk_type == "content":
                    if not is_generating:
                        status_placeholder.empty()
                        is_generating = True
                    
                    full_response += chunk.get("data", "")
                    message_placeholder.markdown(full_response + "▌")
                
                elif chunk_type == "sources":
                    sources = chunk.get("sources", [])
                
                elif chunk_type == "error":
                    status_placeholder.error(f"❌ {chunk.get('message')}")
                    break
            
            # Clear status and show final response
            status_placeholder.empty()
            message_placeholder.markdown(full_response)
            
            log_debug(f"🤖 Response: {len(full_response)} chars, {len(sources)} sources")
            
            # Show sources if available
            if sources:
                with st.expander("📚 Sources Referenced", expanded=False):
                    for idx, source in enumerate(sources, 1):
                        st.markdown(f"**{idx}.** {source}")
        
        # Add assistant message to history
        st.session_state.messages.append({
            "role": "assistant",
            "content": full_response,
            "sources": sources
        })
        
        log_debug("💾 Message saved to history")
        
        # Auto-refresh to update thread list
        st.rerun()

def render_footer():
    """Render professional footer"""
    st.divider()
    st.markdown("""
    <div style="text-align: center; padding: 2rem 0 1rem 0; color: var(--text-muted);">
        <p style="margin: 0 0 0.5rem 0; font-size: 0.9rem;">
            Built with ❤️ using <strong>Streamlit</strong> | Powered by <strong>LangGraph</strong> & <strong>FastAPI</strong>
        </p>
        <p style="margin: 0; font-size: 0.85rem; opacity: 0.8;">
            © 2026 Agentic RAG with Knowledge Base | Enterprise AI Platform
        </p>
    </div>
    """, unsafe_allow_html=True)

# ========================================
# MAIN APP
# ========================================

def main():
    """Main application logic with enhanced UI"""
    
    log_debug("🎬 App started")
    
    # Check if user is logged in
    if not st.session_state.token:
        render_login_page()
        render_footer()
    else:
        # Auto-load thread history if needed
        if st.session_state.current_thread_id and not st.session_state.messages:
            st.session_state.messages = get_thread_history(st.session_state.current_thread_id)
        
        # Render sidebar and chat
        render_sidebar()
        render_chat_interface()
        render_footer()

# ========================================
# RUN APP
# ========================================

if __name__ == "__main__":
    main()