import os
import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
from sklearn.manifold import TSNE
import json
from datetime import datetime
from typing import Dict, Any

from config import SpamClassifierConfig
from data_loader import DataLoader
from embedding_generator import EmbeddingGenerator
from evaluator import ModelEvaluator
from spam_classifier import SpamClassifierPipeline
from email_handler import GmailHandler

# --- Cấu hình trang và CSS tùy chỉnh ---
st.set_page_config(page_title="Bảng điều khiển Email Classifier", layout="centered")
st.markdown("""
<style>
/* Royal Green Theme - Professional Design */
:root {
    --royal-green: #1a4d2e;
    --royal-green-light: #2d5a3d;
    --royal-green-dark: #0f2e1a;
    --royal-green-accent: #4ade80;
    --royal-green-accent-dark: #22c55e;
    --text-primary: #f8fafc;
    --text-secondary: #cbd5e1;
    --text-muted: #94a3b8;
    --bg-primary: #0f172a;
    --bg-secondary: #1e293b;
    --bg-tertiary: #334155;
    --border-color: #475569;
    --success-color: #10b981;
    --warning-color: #f59e0b;
    --error-color: #ef4444;
    --info-color: #3b82f6;
}

/* Global Theme */
html, body, [class*="css"] {
    font-family: 'Inter', 'Segoe UI', -apple-system, BlinkMacSystemFont, sans-serif;
    color: var(--text-primary);
    background: linear-gradient(135deg, var(--bg-primary) 0%, var(--royal-green-dark) 100%);
    line-height: 1.6;
}

/* Enhanced Button Styles */
.stButton > button {
    background: linear-gradient(135deg, var(--royal-green) 0%, var(--royal-green-light) 100%);
    color: var(--text-primary);
    border: none;
    border-radius: 12px;
    padding: 0.75rem 1.5rem;
    font-weight: 600;
    font-size: 0.95rem;
    transition: all 0.3s cubic-bezier(0.4, 0, 0.2, 1);
    box-shadow: 0 4px 6px rgba(26, 77, 46, 0.2);
    position: relative;
    overflow: hidden;
}

.stButton > button:hover {
    background: linear-gradient(135deg, var(--royal-green-light) 0%, var(--royal-green-accent-dark) 100%);
    transform: translateY(-2px);
    box-shadow: 0 8px 25px rgba(26, 77, 46, 0.3);
}

.stButton > button:active {
    transform: translateY(0);
    box-shadow: 0 2px 4px rgba(26, 77, 46, 0.2);
}

/* Navigation Buttons - Uniform Sizing */
.stButton > button[data-testid="baseButton-secondary"] {
    min-height: 3.5rem;
    min-width: 100%;
    display: flex;
    align-items: center;
    justify-content: center;
    text-align: center;
    white-space: nowrap;
    font-size: 1rem;
    font-weight: 600;
    padding: 1rem 1.5rem;
}

/* Ensure all navigation buttons have consistent height */
div[data-testid="column"] .stButton > button {
    height: 3.5rem;
    min-height: 3.5rem;
    display: flex;
    align-items: center;
    justify-content: center;
    text-align: center;
    white-space: nowrap;
}

/* Professional Title and Description */
.main-title {
    font-size: 3.2rem;
    font-weight: 800;
    margin: 2rem 0 1rem 0;
    text-align: center;
    color: var(--royal-green-accent);
    text-shadow: 0 2px 4px rgba(74, 222, 128, 0.3);
    letter-spacing: -0.02em;
    display: flex;
    align-items: center;
    justify-content: center;
    gap: 0.5rem;
}

.main-title span {
    background: linear-gradient(135deg, var(--royal-green-accent) 0%, var(--royal-green-accent-dark) 100%);
    -webkit-background-clip: text;
    -webkit-text-fill-color: transparent;
    background-clip: text;
}

.subtext {
    font-size: 1.2rem;
    line-height: 1.7;
    color: var(--text-secondary);
    max-width: 800px;
    margin: 0 auto 2rem auto;
    text-align: center;
    font-weight: 400;
}

/* Enhanced Folder Boxes */
.folder-box {
    background: linear-gradient(135deg, var(--bg-secondary) 0%, var(--royal-green-dark) 100%);
    border-radius: 16px;
    padding: 1.5rem;
    box-shadow: 0 8px 32px rgba(0, 0, 0, 0.2);
    margin-bottom: 1.5rem;
    border: 1px solid var(--border-color);
    backdrop-filter: blur(10px);
}

.folder-title {
    font-size: 1.3rem;
    font-weight: 700;
    color: var(--royal-green-accent);
    display: flex;
    align-items: center;
    margin-bottom: 1rem;
    text-transform: uppercase;
    letter-spacing: 0.05em;
}

.folder-count {
    background: linear-gradient(135deg, var(--royal-green-accent) 0%, var(--royal-green-accent-dark) 100%);
    color: var(--text-primary);
    border-radius: 20px;
    padding: 0.3rem 0.8rem;
    margin-left: 0.8rem;
    font-size: 0.85rem;
    font-weight: 600;
    box-shadow: 0 2px 8px rgba(74, 222, 128, 0.3);
}

/* Enhanced Folder Buttons */
.folder-box .stButton > button {
    background: linear-gradient(135deg, var(--bg-tertiary) 0%, var(--royal-green-dark) 100%);
    border: 1px solid var(--border-color);
    margin: 0.5rem 0;
    width: 100%;
    text-align: left;
    border-radius: 12px;
    padding: 1rem;
    font-size: 0.9rem;
    line-height: 1.4;
    white-space: normal !important; 
    word-wrap: break-word !important; 
    overflow-wrap: break-word !important; 
    min-height: auto !important;
    height: auto !important; 
    display: block !important; 
    text-overflow: unset !important; 
    overflow: visible !important; 
    max-width: none !important;
    flex-wrap: wrap !important;
    align-items: flex-start !important;
    justify-content: flex-start !important;
}

/* Force text wrapping for email buttons specifically */
.folder-box .stButton > button[data-testid="baseButton-secondary"] {
    white-space: pre-wrap !important;
    word-break: break-word !important;
    overflow-wrap: break-word !important;
    min-height: auto !important;
    height: auto !important;
    max-height: none !important;
    display: block !important;
    text-overflow: unset !important;
    overflow: visible !important;
    line-height: 1.5 !important;
    padding: 1rem !important;
    text-align: left !important;
    align-items: flex-start !important;
    justify-content: flex-start !important;
    flex-direction: column !important;
}

/* Additional CSS to force text wrapping on button content */
.folder-box .stButton > button * {
    white-space: pre-wrap !important;
    word-wrap: break-word !important;
    overflow-wrap: break-word !important;
    word-break: break-word !important;
}

/* Target the button text content specifically */
.folder-box .stButton > button p {
    white-space: pre-wrap !important;
    word-wrap: break-word !important;
    overflow-wrap: break-word !important;
    word-break: break-word !important;
    margin: 0 !important;
    padding: 0 !important;
}

.folder-box .stButton > button:hover {
    background: linear-gradient(135deg, var(--royal-green-light) 0%, var(--royal-green) 100%);
    border-color: var(--royal-green-accent);
    transform: translateY(-2px);
    box-shadow: 0 8px 25px rgba(26, 77, 46, 0.3);
}

/* Enhanced Content Container */
.content-container {
    background: linear-gradient(135deg, var(--bg-secondary) 0%, var(--royal-green-dark) 100%);
    border-radius: 16px;
    padding: 1.5rem;
    min-height: 350px;
    display: flex;
    flex-direction: column;
    align-items: stretch;
    justify-content: flex-start;
    color: var(--text-secondary);
    overflow-y: auto;
    max-width: 100%;
    word-wrap: break-word;
    white-space: pre-wrap;
    border: 1px solid var(--border-color);
    box-shadow: 0 8px 32px rgba(0, 0, 0, 0.2);
    backdrop-filter: blur(10px);
}

.placeholder {
    color: var(--text-muted);
    font-style: italic;
    text-align: center;
    flex-grow: 1;
    display: flex;
    align-items: center;
    justify-content: center;
    font-size: 1.1rem;
}

/* Enhanced Auth Box */
.auth-box {
    background: linear-gradient(135deg, var(--bg-secondary) 0%, var(--royal-green-dark) 100%);
    border-radius: 20px;
    padding: 2.5rem;
    text-align: center;
    margin: 2rem 0;
    border: 2px solid var(--royal-green-accent);
    box-shadow: 0 12px 40px rgba(26, 77, 46, 0.2);
    backdrop-filter: blur(10px);
}

/* Enhanced Email Items */
.email-item {
    background: linear-gradient(135deg, var(--bg-tertiary) 0%, var(--royal-green-dark) 100%);
    border-radius: 12px;
    padding: 1rem;
    margin: 0.8rem 0;
    border-left: 4px solid var(--royal-green-accent);
    cursor: pointer;
    transition: all 0.3s cubic-bezier(0.4, 0, 0.2, 1);
    box-shadow: 0 4px 12px rgba(0, 0, 0, 0.1);
}

.email-item:hover {
    background: linear-gradient(135deg, var(--royal-green-light) 0%, var(--royal-green) 100%);
    transform: translateX(6px);
    box-shadow: 0 8px 25px rgba(26, 77, 46, 0.2);
}

.email-subject {
    font-weight: 700;
    font-size: 1.1rem;
    color: var(--text-primary);
    margin-bottom: 0.4rem;
}

.email-sender {
    font-size: 0.9rem;
    color: var(--text-secondary);
    margin-bottom: 0.3rem;
}

.email-snippet {
    font-size: 0.85rem;
    color: var(--text-muted);
    opacity: 0.9;
}

/* Enhanced User Profile */
.user-profile {
    background: linear-gradient(135deg, var(--bg-secondary) 0%, var(--royal-green-dark) 100%);
    border-radius: 16px;
    padding: 1.5rem;
    margin-bottom: 1.5rem;
    border: 1px solid var(--border-color);
    box-shadow: 0 8px 32px rgba(0, 0, 0, 0.2);
    backdrop-filter: blur(10px);
}

/* Enhanced Relabel Buttons */
.relabel-buttons {
    background: linear-gradient(135deg, var(--bg-secondary) 0%, var(--royal-green-dark) 100%);
    border-radius: 16px;
    padding: 1.5rem;
    margin: 1.5rem 0;
    border: 2px solid var(--warning-color);
    box-shadow: 0 8px 32px rgba(245, 158, 11, 0.2);
    backdrop-filter: blur(10px);
}

.relabel-title {
    color: var(--warning-color);
    font-weight: 700;
    margin-bottom: 1rem;
    font-size: 1.2rem;
    text-transform: uppercase;
    letter-spacing: 0.05em;
}

/* Enhanced Badges */
.correction-badge {
    background: linear-gradient(135deg, var(--error-color) 0%, #dc2626 100%);
    color: var(--text-primary);
    border-radius: 8px;
    padding: 0.3rem 0.8rem;
    font-size: 0.8rem;
    font-weight: 600;
    margin-left: 0.5rem;
    box-shadow: 0 2px 8px rgba(239, 68, 68, 0.3);
}

.corrected-badge {
    background: linear-gradient(135deg, var(--success-color) 0%, var(--royal-green-accent-dark) 100%);
    color: var(--text-primary);
    border-radius: 8px;
    padding: 0.3rem 0.8rem;
    font-size: 0.8rem;
    font-weight: 600;
    margin-left: 0.5rem;
    box-shadow: 0 2px 8px rgba(16, 185, 129, 0.3);
}

/* Enhanced Metrics */
[data-testid="metric-container"] {
    background: linear-gradient(135deg, var(--bg-secondary) 0%, var(--royal-green-dark) 100%);
    border-radius: 16px;
    padding: 1.5rem;
    border: 1px solid var(--border-color);
    box-shadow: 0 8px 32px rgba(0, 0, 0, 0.2);
    backdrop-filter: blur(10px);
}

/* Enhanced Expanders */
.streamlit-expanderHeader {
    background: linear-gradient(135deg, var(--bg-secondary) 0%, var(--royal-green-dark) 100%) !important;
    border-radius: 12px !important;
    border: 1px solid var(--border-color) !important;
    color: var(--royal-green-accent) !important;
    font-weight: 600 !important;
}

/* Enhanced Selectbox */
.stSelectbox > div > div {
    background: linear-gradient(135deg, var(--bg-secondary) 0%, var(--royal-green-dark) 100%);
    border: 1px solid var(--border-color);
    border-radius: 12px;
}

/* Enhanced Text Input */
.stTextInput > div > div > input {
    background: linear-gradient(135deg, var(--bg-secondary) 0%, var(--royal-green-dark) 100%);
    border: 1px solid var(--border-color);
    border-radius: 12px;
    color: var(--text-primary);
}

/* Enhanced Number Input */
.stNumberInput > div > div > input {
    background: linear-gradient(135deg, var(--bg-secondary) 0%, var(--royal-green-dark) 100%);
    border: 1px solid var(--border-color);
    border-radius: 12px;
    color: var(--text-primary);
}

/* Enhanced Progress Bar */
.stProgress > div > div > div {
    background: linear-gradient(135deg, var(--royal-green-accent) 0%, var(--royal-green-accent-dark) 100%);
    border-radius: 8px;
}

/* Enhanced Success/Error Messages */
.element-container .stAlert {
    border-radius: 12px;
    border: none;
    box-shadow: 0 4px 12px rgba(0, 0, 0, 0.1);
}

/* Custom Scrollbar */
::-webkit-scrollbar {
    width: 8px;
}

::-webkit-scrollbar-track {
    background: var(--bg-primary);
    border-radius: 4px;
}

::-webkit-scrollbar-thumb {
    background: linear-gradient(135deg, var(--royal-green) 0%, var(--royal-green-light) 100%);
    border-radius: 4px;
}

::-webkit-scrollbar-thumb:hover {
    background: linear-gradient(135deg, var(--royal-green-light) 0%, var(--royal-green-accent) 100%);
}

/* Responsive Design */
@media (max-width: 768px) {
    .main-title {
        font-size: 2.5rem;
    }
    
    .subtext {
        font-size: 1rem;
    }
    
    .folder-box {
        padding: 1rem;
    }
    
    .content-container {
        padding: 1rem;
    }
}
</style>
""", unsafe_allow_html=True)

# --- Khởi tạo Gmail Handler ---
@st.cache_resource
def get_gmail_handler():
    """Khởi tạo Gmail Handler"""
    credentials_path = "./cache/input/credentials.json"
    if not os.path.exists(credentials_path):
        st.error(f"Không tìm thấy file credentials.json tại: {credentials_path}")
        st.info("Vui lòng đặt file credentials.json vào thư mục ./cache/input/")
        st.stop()
    return GmailHandler(credentials_path)

# --- Tải và cache pipeline để tái sử dụng ---
@st.cache_resource
def load_pipeline(classifier_type: str = 'knn'):
    """
    Khởi tạo và train pipeline phân loại spam.
    Kết quả được cache để không train lại mỗi lần rerun.
    
    Args:
        classifier_type: Loại bộ phân loại ('knn' hoặc 'tfidf')
    """
    cfg = SpamClassifierConfig()
    pipeline = SpamClassifierPipeline(cfg, classifier_type=classifier_type)
    pipeline.train()
    return pipeline

@st.cache_resource
def load_pipeline_with_classifier(classifier_type: str = 'knn'):
    """
    Khởi tạo pipeline với bộ phân loại được chỉ định.
    Cache riêng cho từng loại classifier.
    """
    return load_pipeline(classifier_type)

@st.cache_resource
def load_pipeline_with_corrections(classifier_type: str = 'knn'):
    """
    🆕 Khởi tạo và train pipeline với corrections.
    Cache riêng cho pipeline với corrections.
    
    Args:
        classifier_type: Loại bộ phân loại ('knn' hoặc 'tfidf')
    """
    cfg = SpamClassifierConfig()
    pipeline = SpamClassifierPipeline(cfg, classifier_type=classifier_type)
    training_info = pipeline.train_with_corrections()
    return pipeline, training_info

def retrain_model_with_corrections(classifier_type: str = 'knn') -> Dict[str, Any]:
    """
    🆕 Retrain model với corrections và clear cache.
    
    Args:
        classifier_type: Loại classifier
        
    Returns:
        Dict chứa thông tin training
    """
    # Clear tất cả cache để force retrain hoàn toàn
    load_pipeline_with_corrections.clear()
    load_pipeline_with_classifier.clear()
    load_pipeline.clear()
    get_embeddings_cached.clear()
    compute_tsne_cached.clear()
    
    # Thiết lập regenerate_embeddings = True để force regenerate embeddings
    cfg = SpamClassifierConfig()
    cfg.regenerate_embeddings = True
    
    # Retrain với corrections và regenerate embeddings
    pipeline = SpamClassifierPipeline(cfg, classifier_type=classifier_type)
    training_info = pipeline.train_with_corrections()
    
    # Update session state
    st.session_state['current_pipeline'] = pipeline
    st.session_state['training_info'] = training_info
    
    return training_info

# --- Tải dữ liệu mẫu vào session_state ---
@st.cache_data
def load_sample_data(path: str) -> pd.DataFrame:
    """Đọc file CSV chứa dữ liệu email (Category, Message)."""
    return pd.read_csv(path)

@st.cache_data
def get_embeddings_cached(messages: list) -> np.ndarray:
    """Cache embeddings generation."""
    cfg = SpamClassifierConfig()
    eg = EmbeddingGenerator(cfg)
    
    # The EmbeddingGenerator already handles .npy file caching internally
    # It will automatically load from cache if available, or generate and save if not
    return eg.generate_embeddings(messages)

@st.cache_data
def compute_tsne_cached(sub_emb: np.ndarray) -> np.ndarray:
    """Cache t-SNE computation."""
    return TSNE(
        n_components=2, 
        init="random", 
        learning_rate="auto"
    ).fit_transform(sub_emb)

def show_authentication_ui():
    """UI mới với auto authentication"""
    st.markdown('<div class="auth-box">', unsafe_allow_html=True)
    st.markdown("### 🔐 Cần xác thực Gmail")
    
    auth_tab = st.selectbox(
        "Chọn phương thức đăng nhập:",
        ["🚀 Tự động (Khuyến nghị)", "🔗 Thủ công"]
    )
    
    if auth_tab == "🚀 Tự động (Khuyến nghị)":
        st.info("🎯 Phương thức này sẽ tự động mở browser và nhận token")
        
        if st.button("🚀 Đăng nhập Tự động", type="primary", use_container_width=True):
            with st.spinner("🔄 Đang khởi tạo OAuth flow..."):
                try:
                    success = gmail_handler.authenticate_auto()
                    if success:
                        st.toast("🎉 Đăng nhập thành công! Bắt đầu phân loại Email.", icon="✅")
                        st.rerun()
                except Exception as e:
                    st.error(f"❌ Auto auth failed: {str(e)}")
                    st.info("💡 Thử phương thức thủ công")
    
    else:
        email_hint = st.text_input("Nhập email:", 
                                  placeholder="user@gmail.com", 
                                  key="email_hint_new")
        
        try:
            if email_hint:
                auth_url = gmail_handler.get_authorization_url_with_hint(email_hint)
            else:
                auth_url = gmail_handler.get_authorization_url()
            
            st.markdown(f'<a href="{auth_url}" target="_blank" style="background: var(--royal-green); color: white; padding: 0.8rem 1.5rem; text-decoration: none; border-radius: 8px; display: inline-block; margin: 1rem 0;">🔑 Đăng nhập Gmail</a>', unsafe_allow_html=True)
        except Exception as e:
            st.error(f"Lỗi tạo auth URL: {str(e)}")

        # Hướng dẫn
        st.markdown("---")
        st.markdown("**Hướng dẫn:**")
        st.markdown("0. Phương thức này dùng cho lần đăng nhập đầu tiên")
        st.markdown("1. Click vào nút đăng nhập bên trên")
        st.markdown("2. Chọn tài khoản Gmail và cho phép quyền truy cập")
        st.markdown("3. Copy authorization code từ URL redirect")
        st.markdown("4. Paste code vào ô bên dưới")
        
        st.markdown("---")
        st.markdown("**Nhập authorization code:**")

        auth_code = st.text_input("Authorization code:", 
                                 placeholder="Paste code từ Google...",
                                 key="auth_code_new")
        
        if st.button("🔐 Xác thực", type="primary", key="auth_submit_new") and auth_code:
            with st.spinner("Đang xác thực..."):
                if gmail_handler.handle_oauth_callback(auth_code.strip()):
                    st.success("✅ Xác thực thành công!")
                    st.balloons()
                    st.rerun()
    
    st.markdown('</div>', unsafe_allow_html=True)

# 🆕 --- Hàm quản lý correction data ---
def load_corrections():
    """Load correction data từ file JSON"""
    correction_file = "./cache/corrections.json"
    if os.path.exists(correction_file):
        try:
            with open(correction_file, 'r', encoding='utf-8') as f:
                return json.load(f)
        except:
            return {}
    return {}

def save_corrections(corrections):
    """Save correction data vào file JSON"""
    correction_file = "./cache/corrections.json"
    os.makedirs("./cache", exist_ok=True)
    try:
        with open(correction_file, 'w', encoding='utf-8') as f:
            json.dump(corrections, f, ensure_ascii=False, indent=2)
        return True
    except Exception as e:
        st.error(f"Lỗi lưu corrections: {str(e)}")
        return False

def add_correction(email_id, original_prediction, corrected_label, email_data):
    """Thêm correction mới"""
    corrections = load_corrections()
    corrections[email_id] = {
        'original_prediction': original_prediction,
        'corrected_label': corrected_label,
        'timestamp': datetime.now().isoformat(),
        'subject': email_data.get('subject', ''),
        'sender': email_data.get('sender', ''),
        'snippet': email_data.get('snippet', '')[:100]
    }
    return save_corrections(corrections)

def get_correction_stats():
    """Lấy thống kê về corrections"""
    corrections = load_corrections()
    total = len(corrections)
    spam_to_ham = sum(1 for c in corrections.values() 
                     if c['original_prediction'] == 'spam' and c['corrected_label'] == 'ham')
    ham_to_spam = sum(1 for c in corrections.values() 
                     if c['original_prediction'] == 'ham' and c['corrected_label'] == 'spam')
    
    return {
        'total': total,
        'spam_to_ham': spam_to_ham,
        'ham_to_spam': ham_to_spam
    }

# Khởi tạo các components
try:
    gmail_handler = get_gmail_handler()
    # Khởi tạo pipeline mặc định (KNN)
    pipeline = load_pipeline()
except Exception as e:
    st.error(f"Lỗi khởi tạo ứng dụng: {str(e)}")
    st.stop()

if "df" not in st.session_state:
    config = SpamClassifierConfig()
    st.session_state["df"] = load_sample_data(config.dataset_path)
df = st.session_state["df"]

# --- Quản lý trạng thái trang ---
if "page" not in st.session_state:
    st.session_state.page = "🏠 Tổng quan"

if st.session_state.page != "🏠 Tổng quan":
    col1, col2 = st.columns([1, 4])
    with col1:
        if st.button("🏠 Tổng quan"):
            st.session_state.page = "🏠 Tổng quan"
            st.rerun()
    with col2:
        # 🆕 Hiển thị correction stats
        correction_stats = get_correction_stats()
        if correction_stats['total'] > 0:
            st.info(f"📝 Corrections: {correction_stats['total']} total | "
                   f"🗑️→📥 {correction_stats['spam_to_ham']} | "
                   f"📥→🗑️ {correction_stats['ham_to_spam']}")

# --- Xử lý OAuth callback từ URL parameters ---
query_params = st.query_params
if "code" in query_params and "state" in query_params:
    with st.spinner("Đang xử lý xác thực..."):
        if gmail_handler.handle_oauth_callback(query_params["code"]):
            st.success("✅ Xác thực thành công!")
            st.query_params.clear()
            st.rerun()
        else:
            st.error("❌ Lỗi xác thực! Vui lòng thử lại.")

# --- Trang Tổng quan (Overview) ---
if st.session_state.page == "🏠 Tổng quan":
    # Hiển thị messages từ session state
    if 'success_message' in st.session_state:
        st.success(st.session_state['success_message'])
        del st.session_state['success_message']
    
    if 'warning_message' in st.session_state:
        st.warning(st.session_state['warning_message'])
        del st.session_state['warning_message']
    
    if 'error_message' in st.session_state:
        st.error(st.session_state['error_message'])
        del st.session_state['error_message']
    
    # Hiển thị debug info nếu có
    if 'debug_info' in st.session_state:
        st.info(f"🔍 Debug: {st.session_state['debug_info']}")
        del st.session_state['debug_info']
    
    # Trigger rerun nếu cần
    if st.session_state.get('needs_rerun', False):
        del st.session_state['needs_rerun']
        st.rerun()
    
    st.markdown('<h1 class="main-title">📧 <span>Email Classifier</span></h1>', unsafe_allow_html=True)
    st.markdown('<div class="subtext">Khám phá và phân loại email với giao diện tương tác!</div>', unsafe_allow_html=True)

    # Thống kê nhanh
    total = len(df)
    spam_cnt = len(df[df["Category"] == "spam"])
    ham_cnt  = len(df[df["Category"] == "ham"])
    
    # 🆕 Thêm correction stats
    correction_stats = get_correction_stats()
    
    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Tổng số Email", total)
    c2.metric("Email Spam", spam_cnt, f"{spam_cnt/total*100:.1f}%")
    c3.metric("Email Ham", ham_cnt, f"{ham_cnt/total*100:.1f}%")
    c4.metric("🔧 Corrections", correction_stats['total'])

    st.markdown("### Tính năng:")

    # Nút chuyển đến từng page
    col1, col2, col3, col4 = st.columns(4)
    with col1:
        if st.button("📊 Phân tích Dữ liệu", use_container_width=True):
            st.session_state.page = "📊 Phân tích Dữ liệu"
            st.rerun()
    with col2:
        if st.button("📈 Đánh giá Bộ phân loại", use_container_width=True):
            st.session_state.page = "📈 Đánh giá Bộ phân loại"
            st.rerun()
    with col3:
        if st.button("✉️ Quét Gmail", use_container_width=True):
            st.session_state.page = "✉️ Quét Gmail"
            st.rerun()
    with col4:
        if st.button("🔧 Quản lý Corrections", use_container_width=True):
            st.session_state.page = "🔧 Quản lý Corrections"
            st.rerun()

# --- Trang Phân tích Dữ liệu ---
elif st.session_state.page == "📊 Phân tích Dữ liệu":
    st.header("📊 Phân tích Dữ liệu")

    # 1) Biểu đồ cột Spam vs Ham kèm số lượng trên đỉnh cột
    st.subheader("Phân phối Spam vs Ham")
    counts = df["Category"].value_counts().reset_index()
    counts.columns = ["Nhóm", "Số lượng"]
    fig1 = px.bar(
        counts,
        x="Nhóm",
        y="Số lượng",
        text="Số lượng",
        text_auto=True,
        title="Phân phối Email Spam và Ham",
        color="Nhóm",
        color_discrete_map={"spam":"#ef4444","ham":"#22c55e"},
        labels={"Nhóm":"Loại","Số lượng":"Số lượng Email"}
    )
    st.plotly_chart(fig1, use_container_width=True)

    # 2) t-SNE visualization trên 1.000 mẫu
    st.subheader("Minh họa embedding với t-SNE (1.000 mẫu)")
    messages = df["Message"].tolist()
    embeddings = get_embeddings_cached(messages)

    n_samples = min(1000, embeddings.shape[0])
    idx = np.random.choice(embeddings.shape[0], size=n_samples, replace=False)
    sub_emb = embeddings[idx]
    sub_lbl = [df["Category"].iloc[i] for i in idx]

    with st.spinner("Đang tính toán t-SNE…"):
        proj = compute_tsne_cached(sub_emb)

    df_vis = pd.DataFrame(proj, columns=["Dim 1","Dim 2"])
    df_vis["Nhóm"] = sub_lbl
    fig2 = px.scatter(
        df_vis,
        x="Dim 1",
        y="Dim 2",
        color="Nhóm",
        title="Phân tán embedding qua t-SNE",
        color_discrete_map={"spam":"#ef4444","ham":"#22c55e"},
        hover_data=["Nhóm"]
    )
    st.plotly_chart(fig2, use_container_width=True)

# --- Trang Đánh giá Bộ phân loại ---
elif st.session_state.page == "📈 Đánh giá Bộ phân loại":
    st.header("📈 Đánh giá Bộ phân loại")

    # Khởi tạo DataLoader, Embedding và Pipeline
    cfg = SpamClassifierConfig()
    loader = DataLoader(cfg)
    messages, labels = loader.load_data()
    emb = EmbeddingGenerator(cfg).generate_embeddings(messages)
    train_idx, test_idx, _, _ = loader.split_data(messages, labels)
    test_emb = emb[test_idx]
    test_meta = []
    encoded = loader.label_encoder.transform(labels)
    metadata = loader.create_metadata(messages, labels, encoded)
    for i in test_idx:
        test_meta.append(metadata[i])

    # Train hai mô hình KNN và TF-IDF
    pipe_knn = SpamClassifierPipeline(cfg, classifier_type="knn")

    pipe_tfidf = SpamClassifierPipeline(cfg, classifier_type="tfidf")
    
    # Train models only when needed (deferred training)
    with st.spinner("Đang huấn luyện mô hình KNN..."):
        pipe_knn.train()
    with st.spinner("Đang huấn luyện mô hình TF-IDF..."):
        pipe_tfidf.train()

    evaluator = ModelEvaluator(cfg)

    # Chạy evaluate_accuracy
    with st.spinner("Đang đánh giá mô hình, xin chờ…"):
        combined_results, knn_errors, combined_cms = evaluator.evaluate_accuracy(
            test_embeddings=test_emb,
            test_metadata=test_meta,
            knn_classifier=pipe_knn.classifier,
            tfidf_classifier=pipe_tfidf.classifier,
            k_values=cfg.k_values
        )
    st.success("✅ Đánh giá đã hoàn tất!")

    # Lấy data từ results và cms
    knn_results = combined_results['knn']
    tfidf_results = combined_results['tfidf']
    best_k = combined_results['best_k']
    knn_best = knn_results[best_k]
    knn_cms = combined_cms['knn']
    tfidf_cm = combined_cms['tfidf']

    # 1. Thông báo giá trị K tốt nhất và Accuracy tương ứng
    best_acc = knn_best["accuracy"]
    st.info(f"🔎 K tốt nhất: **k = {best_k}**, Accuracy = **{best_acc:.4f}**")

    # 2. Lineplot KNN metrics
    st.subheader("So sánh chỉ số KNN theo k")
    fig_metrics = evaluator.plot_knn_metrics(knn_results, cfg.k_values)
    st.pyplot(fig_metrics)

    # 3. Heatmaps KNN per k (dùng columns để ngang hàng)
    st.subheader("Confusion Matrix KNN theo k")
    cols = st.columns(len(cfg.k_values))
    for idx, k in enumerate(cfg.k_values):
        with cols[idx]:
            fig_cm = evaluator.plot_knn_confusion(knn_cms[k], k)
            st.pyplot(fig_cm)

    # 4. Barplot phân bố labels
    st.subheader("Phân bố nhãn")
    fig_dist = evaluator.plot_label_distribution(labels)
    st.pyplot(fig_dist)

    # 5. Grouped bar so sánh TF-IDF vs best KNN
    st.subheader("So sánh TF-IDF vs Best KNN")
    fig_comp = evaluator.plot_comparison(knn_best, tfidf_results, best_k)
    st.pyplot(fig_comp)

    # 6. Heatmap TF-IDF
    st.subheader("Confusion Matrix TF-IDF")
    fig_tfidf_cm = evaluator.plot_tfidf_confusion(tfidf_cm)
    st.pyplot(fig_tfidf_cm)

# --- Trang Quản lý Corrections ---
elif st.session_state.page == "🔧 Quản lý Corrections":
    st.header("🔧 Quản lý Corrections")
    
    corrections = load_corrections()
    
    if not corrections:
        st.info("📝 Chưa có correction nào được thực hiện.")
        st.markdown("Để thêm corrections, hãy:")
        st.markdown("1. Vào trang **✉️ Quét Gmail**")
        st.markdown("2. Quét emails và xem kết quả phân loại")
        st.markdown("3. Sử dụng nút **🔄 Đánh dấu lại** để sửa lỗi phân loại")
    else:
        # Thống kê
        stats = get_correction_stats()
        col1, col2, col3 = st.columns(3)
        col1.metric("Tổng Corrections", stats['total'])
        col2.metric("Spam → Ham", stats['spam_to_ham'])
        col3.metric("Ham → Spam", stats['ham_to_spam'])
        
        # Danh sách corrections
        st.subheader("📋 Danh sách Corrections")
        
        # Convert to DataFrame for display
        corrections_data = []
        for email_id, correction in corrections.items():
            corrections_data.append({
                'Email ID': email_id[:8] + '...',
                'Subject': correction['subject'][:50] + '...' if len(correction['subject']) > 50 else correction['subject'],
                'Sender': correction['sender'][:30] + '...' if len(correction['sender']) > 30 else correction['sender'],
                'Original': correction['original_prediction'],
                'Corrected': correction['corrected_label'],
                'Timestamp': correction['timestamp'][:19]  # Remove microseconds
            })
        
        if corrections_data:
            df_corrections = pd.DataFrame(corrections_data)
            st.dataframe(df_corrections, use_container_width=True)
            
            # Export corrections
            col1, col2, col3 = st.columns(3)
            with col1:
                if st.button("📥 Export Corrections (JSON)", use_container_width=True):
                    st.download_button(
                        label="💾 Download corrections.json",
                        data=json.dumps(corrections, indent=2, ensure_ascii=False),
                        file_name=f"corrections_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json",
                        mime="application/json"
                    )
            
            with col2:
                # 🆕 Retrain Model button
                if st.button("🔄 Retrain Model", type="primary", use_container_width=True):
                    st.session_state.page = "🔄 Retrain Model"
                    st.rerun()
            
            with col3:
                # Sử dụng session state để quản lý trạng thái xác nhận
                if 'show_delete_confirmation' not in st.session_state:
                    st.session_state.show_delete_confirmation = False
                
                if not st.session_state.show_delete_confirmation:
                    if st.button("🗑️ Xóa tất cả Corrections", use_container_width=True):
                        st.session_state.show_delete_confirmation = True
                        st.rerun()
                else:
                    st.warning("⚠️ Bạn có chắc chắn muốn xóa tất cả corrections?")
                    col_confirm1, col_confirm2 = st.columns(2)
                    with col_confirm1:
                        if st.button("✅ Xác nhận xóa", type="primary", use_container_width=True):
                            if save_corrections({}):
                                # Clear session state variables
                                st.session_state.show_delete_confirmation = False
                                if 'selected_email' in st.session_state:
                                    st.session_state['selected_email'] = None
                                st.success("✅ Đã xóa tất cả corrections!")
                                st.rerun()
                    with col_confirm2:
                        if st.button("❌ Hủy", use_container_width=True):
                            st.session_state.show_delete_confirmation = False
                            st.rerun()

# --- Trang Retrain Model ---
elif st.session_state.page == "🔄 Retrain Model":
    st.header("🔄 Retrain Model với Corrections")
    
    st.markdown("""
    ### 📝 Hướng dẫn:
    - Model sẽ được retrain với dataset gốc + corrections
    - Corrections sẽ được merge vào training data
    - Cache sẽ được clear để force retrain
    - Quá trình có thể mất vài phút
    """)
    
    # Hiển thị thống kê corrections
    correction_stats = get_correction_stats()
    
    col1, col2, col3 = st.columns(3)
    with col1:
        st.metric("📝 Total Corrections", correction_stats['total'])
    with col2:
        st.metric("🗑️→📥 Spam→Ham", correction_stats['spam_to_ham'])
    with col3:
        st.metric("📥→🗑️ Ham→Spam", correction_stats['ham_to_spam'])
    
    if correction_stats['total'] > 0:
        st.success(f"✅ Có {correction_stats['total']} corrections có thể dùng để retrain")
        
        # Form retrain
        st.markdown("### ⚙️ Cài đặt Retrain")
        
        col1, col2 = st.columns([2, 1])
        with col1:
            classifier_type = st.selectbox(
                "🔍 Chọn loại classifier:",
                ["knn", "tfidf"],
                help="KNN: Dùng embeddings, TF-IDF: Dùng text features"
            )
            
            # Hiển thị thông tin classifier
            classifier_info = {
                'knn': {
                    'name': 'KNN với Embeddings',
                    'description': 'Sử dụng mô hình đa ngôn ngữ E5 để tạo embeddings, sau đó dùng K-Nearest Neighbors để phân loại',
                    'pros': '✅ Độ chính xác cao, hỗ trợ đa ngôn ngữ',
                    'cons': '⚠️ Tốc độ chậm hơn, cần nhiều bộ nhớ'
                },
                'tfidf': {
                    'name': 'TF-IDF + SVM',
                    'description': 'Sử dụng TF-IDF để vector hóa văn bản, kết hợp với SVM để phân loại',
                    'pros': '⚡ Tốc độ nhanh, ít tốn bộ nhớ',
                    'cons': '⚠️ Có thể kém chính xác hơn với văn bản phức tạp'
                }
            }
            
            info = classifier_info[classifier_type]
            with st.expander(f"ℹ️ Thông tin classifier: {info['name']}", expanded=False):
                st.markdown(f"**Mô tả:** {info['description']}")
                st.markdown(f"**Ưu điểm:** {info['pros']}")
                st.markdown(f"**Nhược điểm:** {info['cons']}")
        
        with col2:
            if st.button("🔄 Bắt đầu Retrain", type="primary", use_container_width=True):
                with st.spinner("🔄 Đang retrain model với corrections..."):
                    try:
                        # Hiển thị progress
                        progress_bar = st.progress(0)
                        status_text = st.empty()
                        
                        status_text.text("📊 Đang load dataset gốc...")
                        progress_bar.progress(20)
                        
                        status_text.text("📝 Đang merge corrections...")
                        progress_bar.progress(40)
                        
                        status_text.text("🔧 Đang train model...")
                        progress_bar.progress(60)
                        
                        # Thực hiện retrain
                        training_info = retrain_model_with_corrections(classifier_type)
                        
                        status_text.text("✅ Hoàn tất!")
                        progress_bar.progress(100)
                        
                        st.success("🎉 Retrain thành công!")
                        
                        # Hiển thị thông tin chi tiết
                        st.markdown("### 📊 Thông tin Training")
                        
                        col1, col2 = st.columns(2)
                        with col1:
                            st.metric("📊 Tổng samples", training_info['total_samples'])
                            st.metric("📝 Original samples", training_info['original_samples'])
                        with col2:
                            st.metric("🔧 Correction samples", training_info['correction_samples'])
                            st.metric("🔍 Classifier", classifier_type.upper())
                        
                        # Hiển thị phân bố labels
                        if 'label_distribution' in training_info:
                            st.markdown("### 📈 Phân bố Labels")
                            for label, count in training_info['label_distribution'].items():
                                percentage = (count / training_info['total_samples']) * 100
                                st.write(f"- **{label}**: {count} samples ({percentage:.1f}%)")
                        
                        # Hiển thị thông tin cache
                        st.info("💡 Model đã được cache và sẵn sàng sử dụng!")
                        
                    except Exception as e:
                        st.error(f"❌ Lỗi retrain: {str(e)}")
                        st.info("💡 Kiểm tra lại corrections hoặc thử classifier khác")
    else:
        st.warning("⚠️ Chưa có corrections nào.")
        st.info("💡 Hãy sửa một số predictions trong trang '🔧 Quản lý Corrections' trước khi retrain.")
        
        # Nút chuyển đến trang corrections
        if st.button("🔧 Đi đến Quản lý Corrections", use_container_width=True):
            st.session_state.page = "🔧 Quản lý Corrections"
            st.rerun()

# --- Trang Quét Gmail ---
elif st.session_state.page == "✉️ Quét Gmail":
    # Hiển thị messages từ session state
    if 'success_message' in st.session_state:
        st.success(st.session_state['success_message'])
        del st.session_state['success_message']
    
    if 'warning_message' in st.session_state:
        st.warning(st.session_state['warning_message'])
        del st.session_state['warning_message']
    
    if 'error_message' in st.session_state:
        st.error(st.session_state['error_message'])
        del st.session_state['error_message']
    
    # Hiển thị debug info nếu có
    if 'debug_info' in st.session_state:
        st.info(f"🔍 Debug: {st.session_state['debug_info']}")
        del st.session_state['debug_info']
    
    # Trigger rerun nếu cần
    if st.session_state.get('needs_rerun', False):
        del st.session_state['needs_rerun']
        st.rerun()
    
    st.header("✉️ Quét Gmail")

    # Kiểm tra xác thực
    if 'gmail_credentials' not in st.session_state:
        show_authentication_ui()
    
    else:
        # Đã xác thực, hiển thị giao diện quét email
        gmail_handler.initialize_service_from_session()
        
        # Hiển thị thông tin user
        try:
            user_profile = gmail_handler.get_user_profile()
            st.markdown('<div class="user-profile">', unsafe_allow_html=True)
            st.markdown(f"**👤 Đăng nhập với:** {user_profile['email']}")
            st.markdown(f"**📊 Tổng emails:** {user_profile['total_messages']:,}")
            st.markdown('</div>', unsafe_allow_html=True)
        except Exception as e:
            st.warning(f"Không thể lấy thông tin profile: {str(e)}")
        
        # Controls cho quét email
        st.markdown("### ⚙️ Cài đặt quét email")
        
        # Chọn bộ phân loại
        classifier_option = st.selectbox(
            "🔍 Chọn bộ phân loại:",
            [
                ("knn", "KNN với Embeddings (Độ chính xác cao)"),
                ("tfidf", "TF-IDF (Tốc độ nhanh)")
            ],
            format_func=lambda x: x[1],
            help="KNN: Sử dụng embeddings từ mô hình đa ngôn ngữ, độ chính xác cao nhưng chậm hơn. TF-IDF: Phân loại dựa trên tần suất từ, nhanh nhưng có thể kém chính xác hơn."
        )
        selected_classifier = classifier_option[0]
        
        # Load pipeline với classifier được chọn
        try:
            current_pipeline = load_pipeline_with_classifier(selected_classifier)
            
            # Hiển thị thông tin về bộ phân loại
            classifier_info = {
                'knn': {
                    'name': 'KNN với Embeddings',
                    'description': 'Sử dụng mô hình đa ngôn ngữ E5 để tạo embeddings, sau đó dùng K-Nearest Neighbors để phân loại',
                    'pros': '✅ Độ chính xác cao, hỗ trợ đa ngôn ngữ',
                    'cons': '⚠️ Tốc độ chậm hơn, cần nhiều bộ nhớ'
                },
                'tfidf': {
                    'name': 'TF-IDF + SVM',
                    'description': 'Sử dụng TF-IDF để vector hóa văn bản, kết hợp với SVM để phân loại',
                    'pros': '⚡ Tốc độ nhanh, ít tốn bộ nhớ',
                    'cons': '⚠️ Có thể kém chính xác hơn với văn bản phức tạp'
                }
            }
            
            info = classifier_info[selected_classifier]
            
            with st.expander(f"ℹ️ Thông tin bộ phân loại: {info['name']}", expanded=False):
                st.markdown(f"**Mô tả:** {info['description']}")
                st.markdown(f"**Ưu điểm:** {info['pros']}")
                st.markdown(f"**Nhược điểm:** {info['cons']}")
            
            st.success(f"✅ Đã tải bộ phân loại: {info['name']}")
            
            # Lưu thông tin classifier vào session state
            st.session_state['current_classifier'] = selected_classifier
            st.session_state['current_classifier_name'] = info['name']
            
        except Exception as e:
            st.error(f"❌ Lỗi tải bộ phân loại: {str(e)}")
            st.stop()
        
        col1, col2 = st.columns([1, 1])
        
        with col1:
            max_emails = st.number_input("Số email tối đa:", min_value=1, max_value=50, value=10)
        
        with col2:
            email_query = st.selectbox(
                "Loại email:",
                [
                    "is:unread",
                    "is:inbox", 
                    "label:spam",
                    "from:noreply",
                    "subject:promotion",
                    "is:important",
                    "has:attachment"
                ]
            )
        
        # Custom query option
        custom_query = st.text_input("Hoặc nhập custom query:", placeholder="VD: from:example.com OR subject:urgent")
        final_query = custom_query if custom_query.strip() else email_query
        
        if st.button("🔄 Quét Emails", use_container_width=True, type="primary"):
            with st.spinner(f"Đang quét {max_emails} emails với query: {final_query}..."):
                try:
                    emails = gmail_handler.fetch_emails(max_emails, final_query)
                    
                    if not emails:
                        st.warning("Không tìm thấy email nào!")
                    else:
                        # Phân loại emails
                        classified_emails = []
                        progress_bar = st.progress(0)
                        progress_text = st.empty()
                        
                        for i, email in enumerate(emails):
                            progress_text.text(f"Đang phân loại email {i+1}/{len(emails)}: {email['subject'][:50]}...")
                            
                            # Sử dụng subject + body để phân loại
                            text_to_classify = f"{email['subject']} {email['body']}"
                            
                            result = current_pipeline.predict(text_to_classify)
                            prediction = result['prediction']
                            
                            # Xử lý confidence scores khác nhau cho từng classifier
                            if selected_classifier == 'knn':
                                confidence_scores = result.get('label_distribution', {})
                                confidence = max(confidence_scores.values()) if confidence_scores else 0.5
                            else:  # tfidf
                                confidence_scores = result.get('probabilities', {})
                                confidence = max(confidence_scores.values()) if confidence_scores else 0.5
                            
                            # Lưu confidence scores với tên phù hợp cho từng classifier
                            if selected_classifier == 'knn':
                                email['confidence_scores'] = confidence_scores
                            else:  # tfidf
                                email['confidence_scores'] = confidence_scores
                            
                            email['prediction'] = prediction
                            email['confidence'] = confidence
                            email['confidence_scores'] = confidence_scores
                            
                            # 🆕 Check if this email has been corrected
                            corrections = load_corrections()
                            email['is_corrected'] = email['id'] in corrections
                            if email['is_corrected']:
                                email['corrected_label'] = corrections[email['id']]['corrected_label']
                            
                            classified_emails.append(email)
                            
                            progress_bar.progress((i + 1) / len(emails))
                        
                        # Lưu vào session state
                        st.session_state['classified_emails'] = classified_emails
                        st.session_state['inbox_emails'] = [e for e in classified_emails if e.get('corrected_label', e['prediction']) == 'ham']
                        st.session_state['spam_emails'] = [e for e in classified_emails if e.get('corrected_label', e['prediction']) == 'spam']
                        
                        progress_bar.empty()
                        progress_text.empty()
                        
                        # Thống kê
                        total_emails = len(classified_emails)
                        spam_count = len(st.session_state['spam_emails'])
                        ham_count = len(st.session_state['inbox_emails'])
                        corrected_count = len([e for e in classified_emails if e.get('is_corrected', False)])
                        
                        classifier_name = "KNN với Embeddings" if selected_classifier == 'knn' else "TF-IDF"
                        st.success(f"✅ Đã quét và phân loại {total_emails} emails bằng {classifier_name}!")
                        
                        col1, col2, col3, col4 = st.columns(4)
                        col1.metric("Tổng số", total_emails)
                        col2.metric("Ham", ham_count, f"{ham_count/total_emails*100:.1f}%")
                        col3.metric("Spam", spam_count, f"{spam_count/total_emails*100:.1f}%")
                        col4.metric("🔧 Corrected", corrected_count)
                        
                except Exception as e:
                    st.error(f"❌ Lỗi khi quét emails: {str(e)}")
                    st.info("Vui lòng kiểm tra kết nối mạng và quyền truy cập Gmail.")
        
        # Hiển thị emails đã phân loại
        if 'classified_emails' in st.session_state and st.session_state['classified_emails']:
            st.markdown("---")
            st.subheader("📬 Emails đã phân loại")
            
            # Khởi tạo selected_email nếu chưa có
            if 'selected_email' not in st.session_state:
                st.session_state['selected_email'] = None
            
            # Layout 3 cột
            col_inbox, col_content, col_spam = st.columns([1, 2, 1])
            
            # Cột Inbox (Ham)
            with col_inbox:
                st.markdown('<div class="folder-box">', unsafe_allow_html=True)
                inbox_count = len(st.session_state.get('inbox_emails', []))
                st.markdown(f'<div class="folder-title">📥 Inbox <span class="folder-count">{inbox_count}</span></div>', unsafe_allow_html=True)
                
                for i, email in enumerate(st.session_state.get('inbox_emails', [])):
                    # Tạo preview
                    subject_preview = email['subject'][:35] + "..." if len(email['subject']) > 35 else email['subject']
                    sender_preview = email['sender'].split('<')[0].strip()[:20] if '<' in email['sender'] else email['sender'][:20]
                    confidence = email.get('confidence', 0)
                    
                    # 🆕 Tạo badge cho corrected emails
                    correction_badge = ""
                    if email.get('is_corrected', False):
                        if email['prediction'] != email['corrected_label']:
                            correction_badge = " ✅" 
                    
                    # Button cho inbox emails
                    with st.container():
                        # Hiển thị tiêu đề mail trên nhiều dòng nếu dài
                        subject_full = email['subject']
                        # Tạo nội dung nút với tiêu đề đầy đủ, xuống dòng nếu dài
                        # Chia tiêu đề thành nhiều dòng nếu quá dài
                        subject_lines = []
                        subject = subject_full
                        while len(subject) > 30:
                            subject_lines.append(subject[:30])
                            subject = subject[30:]
                        if subject:
                            subject_lines.append(subject)
                        
                        subject_display = "\n".join(subject_lines)
                        
                        # Tạo HTML button với text wrapping hoàn toàn
                        button_html = f"""
                        <div style="
                            background: linear-gradient(135deg, var(--bg-tertiary) 0%, var(--royal-green-dark) 100%);
                            border: 1px solid var(--border-color);
                            border-radius: 12px;
                            padding: 1rem;
                            margin: 0.5rem 0;
                            cursor: pointer;
                            transition: all 0.3s ease;
                            white-space: pre-wrap;
                            word-wrap: break-word;
                            overflow-wrap: break-word;
                            text-align: left;
                            font-size: 0.9rem;
                            line-height: 1.5;
                            color: var(--text-primary);
                            width: 100%;
                            box-sizing: border-box;
                            min-height: auto;
                            height: auto;
                        " onmouseover="this.style.background='linear-gradient(135deg, var(--royal-green-light) 0%, var(--royal-green) 100%)'; this.style.transform='translateY(-2px)';" 
                           onmouseout="this.style.background='linear-gradient(135deg, var(--bg-tertiary) 0%, var(--royal-green-dark) 100%)'; this.style.transform='translateY(0)';"
                           onclick="window.parent.postMessage({{type: 'streamlit:setComponentValue', key: 'inbox_{email['id']}_{i}', value: true}}, '*')">
                            📧 {subject_full}{correction_badge}<br>
                            👤 {sender_preview}<br>
                            📄 {email['snippet'][:40]}...<br>
                            ✓ Confidence: {confidence:.2f}
                        </div>
                        """
                        
                        # Hiển thị button HTML
                        st.markdown(button_html, unsafe_allow_html=True)
                        
                        # Hidden button để trigger action
                        if st.button(
                            "Select",
                            key=f"inbox_{email['id']}_{i}",
                            use_container_width=True
                        ):
                            st.session_state['selected_email'] = email
                            st.rerun()
                
                st.markdown('</div>', unsafe_allow_html=True)
            
            # Cột nội dung (giữa)
            with col_content:
                if st.session_state['selected_email'] is None:
                    content_html = (
                        '<div class="content-container">'
                        '  <div class="placeholder">'
                        '    <div style="text-align: left; padding: 2rem;">'
                        '      <div style="font-size: 2.2rem; margin-bottom: 0.5rem;">📧</div>'
                        '      <div style="font-size: 1.15rem; font-weight: 600; margin-bottom: 0.2rem;'
                        ' color: var(--royal-green-accent); text-align: left;">'
                        '        Chọn một email'
                        '      </div>'
                        '      <div style="font-size: 1rem; color: var(--text-secondary); text-align: left;">'
                        '        Từ Inbox hoặc Spam để xem nội dung chi tiết'
                        '      </div>'
                        '    </div>'
                        '  </div>'
                        '</div>'
                    )
                else:
                    email = st.session_state['selected_email']
                    from html import escape

                    # Truncate body nếu quá dài
                    body_display = (
                        email['body'][:1500] + "..."
                        if len(email['body']) > 1500 else email['body']
                    )
                    confidence_scores = email.get('confidence_scores', {})
                    confidence_display = ", ".join(
                        [f"{k}: {v:.2f}" for k, v in confidence_scores.items()]
                    )

                    # 🆕 Determine current label (corrected or original)
                    current_label = email.get('corrected_label', email['prediction'])
                    original_prediction = email['prediction']
                    is_corrected = email.get('is_corrected', False)

                    # 🆕 Status badge
                    status_badge = ""
                    if is_corrected:
                        if original_prediction != current_label:
                            status_badge = (
                                f'<span class="corrected-badge">'
                                f'Đã sửa: {original_prediction} → {current_label}'
                                f'</span>'
                            )
                        else:
                            status_badge = (
                                f'<span class="corrected-badge">'
                                f'Đã xác nhận: {current_label}'
                                f'</span>'
                            )

                    # Sử dụng div bọc ngoài, mọi thứ đều căn trái, header không giãn dòng nhiều
                    content_html = (
                        '<div class="content-container" style="text-align: left;">'
                        '  <div style="margin-bottom: 1.1rem; padding-bottom: 1.1rem;'
                        ' border-bottom: 2px solid var(--border-color); text-align: left;">'
                        '    <div style="display: flex; justify-content: flex-start; align-items: center;'
                        ' margin-bottom: 0.3rem;">'
                        '      <span style="font-weight: 700; color: var(--royal-green-accent);'
                        ' font-size: 1.05rem; margin-right: 1.2rem;">'
                        f'        {"📥 HAM" if current_label == "ham" else "🗑️ SPAM"}'
                        '      </span>'
                        '      <span style="font-size: 0.95rem; color: var(--text-secondary); font-weight: 600;">'
                        f'        Confidence: {email.get("confidence", 0):.2f}'
                        '      </span>'
                        '    </div>'
                        '    <div style="font-size: 1.13rem; font-weight: 700; color: var(--text-primary);'
                        ' margin-bottom: 0.3rem; line-height: 1.15; text-align: left;">'
                        f'      {escape(email["subject"])}'
                        '    </div>'
                        '    <div style="font-size: 1rem; color: var(--text-secondary); margin-bottom: 0.2rem;'
                        ' font-weight: 500; text-align: left;">'
                        f'      From: {escape(email["sender"])}'
                        '    </div>'
                        '    <div style="font-size: 0.9rem; color: var(--text-muted); margin-bottom: 0.3rem;'
                        ' font-family: \'Courier New\', monospace; text-align: left;">'
                        f'      {confidence_display}'
                        '    </div>'
                        f'    {status_badge}'
                        '  </div>'
                        '  <div style="line-height: 1.7; color: var(--text-secondary); font-size: 1rem; text-align: left;">'
                        f'    {escape(body_display)}'
                        '  </div>'
                        '</div>'
                    )

                st.markdown(content_html, unsafe_allow_html=True)

                # 🆕 Relabel buttons
                if st.session_state['selected_email'] is not None:
                    email = st.session_state['selected_email']
                    current_label = email.get('corrected_label', email['prediction'])

                    st.markdown(
                        '<div class="relabel-title" style="text-align: left;">'
                        '🔄 Đánh dấu lại phân loại</div>',
                        unsafe_allow_html=True
                    )

                    # Thêm CSS cho nút nền royal green
                    st.markdown(
                        """
                        <style>
                        .stButton>button.relabel-green {
                            background: var(--royal-green-accent, #2ecc40) !important;
                            color: white !important;
                            border: none !important;
                            font-weight: 600;
                        }
                        .stButton>button.relabel-green:disabled {
                            background: #b7e5c2 !important;
                            color: #f0f0f0 !important;
                        }
                        .stButton>button.relabel-green:hover:enabled {
                            background: #1e9e2c !important;
                        }
                        </style>
                        """,
                        unsafe_allow_html=True
                    )

                    col1, col2, col3 = st.columns(3)

                    # Tạo các hàm callback với closure để capture email data
                    def create_relabel_ham_action(email_data):
                        def relabel_ham_action():
                            # Debug logging
                            st.session_state['debug_info'] = f"Bắt đầu relabel_ham_action cho email {email_data['id']}"
                            
                            # Lưu correction vào file local
                            if add_correction(
                                email_data['id'], email_data['prediction'], 'ham', email_data
                            ):
                                # Gọi Gmail API để cập nhật label thật
                                try:
                                    gmail_handler = get_gmail_handler()
                                    st.session_state['debug_info'] = f"Đang gọi Gmail API cho email {email_data['id']}..."
                                    
                                    # Thực hiện correction
                                    result = gmail_handler.apply_single_correction(
                                        email_data['id'], 'ham', email_data['prediction']
                                    )
                                    
                                    if result:
                                        st.session_state['success_message'] = "✅ Đã đánh dấu lại thành HAM và cập nhật Gmail!"
                                        st.session_state['debug_info'] = f"Gmail API thành công cho email {email_data['id']}"
                                    else:
                                        st.session_state['warning_message'] = "⚠️ Đã lưu correction nhưng không thể cập nhật Gmail"
                                        st.session_state['debug_info'] = f"Gmail API thất bại cho email {email_data['id']}"
                                except Exception as e:
                                    st.session_state['error_message'] = f"❌ Lỗi cập nhật Gmail: {str(e)}"
                                    st.session_state['debug_info'] = f"Exception khi gọi Gmail API: {str(e)}"
                                
                                # Cập nhật session state
                                for e in st.session_state['classified_emails']:
                                    if e['id'] == email_data['id']:
                                        e['corrected_label'] = 'ham'
                                        e['is_corrected'] = True
                                        break
                                st.session_state['inbox_emails'] = [
                                    e for e in st.session_state['classified_emails']
                                    if e.get('corrected_label', e['prediction']) == 'ham'
                                ]
                                st.session_state['spam_emails'] = [
                                    e for e in st.session_state['classified_emails']
                                    if e.get('corrected_label', e['prediction']) == 'spam'
                                ]
                                st.session_state['needs_rerun'] = True
                        return relabel_ham_action

                    def create_relabel_spam_action(email_data):
                        def relabel_spam_action():
                            # Debug logging
                            st.session_state['debug_info'] = f"Bắt đầu relabel_spam_action cho email {email_data['id']}"
                            
                            # Lưu correction vào file local
                            if add_correction(
                                email_data['id'], email_data['prediction'], 'spam', email_data
                            ):
                                # Gọi Gmail API để cập nhật label thật
                                try:
                                    gmail_handler = get_gmail_handler()
                                    st.session_state['debug_info'] = f"Đang gọi Gmail API cho email {email_data['id']}..."
                                    
                                    # Thực hiện correction
                                    result = gmail_handler.apply_single_correction(
                                        email_data['id'], 'spam', email_data['prediction']
                                    )
                                    
                                    if result:
                                        st.session_state['success_message'] = "✅ Đã đánh dấu lại thành SPAM và cập nhật Gmail!"
                                        st.session_state['debug_info'] = f"Gmail API thành công cho email {email_data['id']}"
                                    else:
                                        st.session_state['warning_message'] = "⚠️ Đã lưu correction nhưng không thể cập nhật Gmail"
                                        st.session_state['debug_info'] = f"Gmail API thất bại cho email {email_data['id']}"
                                except Exception as e:
                                    st.session_state['error_message'] = f"❌ Lỗi cập nhật Gmail: {str(e)}"
                                    st.session_state['debug_info'] = f"Exception khi gọi Gmail API: {str(e)}"
                                
                                # Cập nhật session state
                                for e in st.session_state['classified_emails']:
                                    if e['id'] == email_data['id']:
                                        e['corrected_label'] = 'spam'
                                        e['is_corrected'] = True
                                        break
                                st.session_state['inbox_emails'] = [
                                    e for e in st.session_state['classified_emails']
                                    if e.get('corrected_label', e['prediction']) == 'ham'
                                ]
                                st.session_state['spam_emails'] = [
                                    e for e in st.session_state['classified_emails']
                                    if e.get('corrected_label', e['prediction']) == 'spam'
                                ]
                                st.session_state['needs_rerun'] = True
                        return relabel_spam_action

                    def create_confirm_action(email_data):
                        def confirm_action():
                            # Debug logging
                            st.session_state['debug_info'] = f"Bắt đầu confirm_action cho email {email_data['id']}"
                            
                            # Lưu correction vào file local
                            if add_correction(
                                email_data['id'], email_data['prediction'], email_data['prediction'], email_data
                            ):
                                # Gọi Gmail API để cập nhật label thật
                                try:
                                    gmail_handler = get_gmail_handler()
                                    st.session_state['debug_info'] = f"Đang gọi Gmail API cho email {email_data['id']}..."
                                    
                                    # Thực hiện correction
                                    result = gmail_handler.apply_single_correction(
                                        email_data['id'], email_data['prediction'], email_data['prediction']
                                    )
                                    
                                    if result:
                                        st.session_state['success_message'] = "✅ Đã xác nhận phân loại và cập nhật Gmail!"
                                        st.session_state['debug_info'] = f"Gmail API thành công cho email {email_data['id']}"
                                    else:
                                        st.session_state['warning_message'] = "⚠️ Đã lưu correction nhưng không thể cập nhật Gmail"
                                        st.session_state['debug_info'] = f"Gmail API thất bại cho email {email_data['id']}"
                                except Exception as e:
                                    st.session_state['error_message'] = f"❌ Lỗi cập nhật Gmail: {str(e)}"
                                    st.session_state['debug_info'] = f"Exception khi gọi Gmail API: {str(e)}"
                                
                                # Cập nhật session state
                                for e in st.session_state['classified_emails']:
                                    if e['id'] == email_data['id']:
                                        e['corrected_label'] = email_data['prediction']
                                        e['is_corrected'] = True
                                        break
                                st.session_state['needs_rerun'] = True
                        return confirm_action

                    # Tạo các callback với email data
                    relabel_ham_action = create_relabel_ham_action(email)
                    relabel_spam_action = create_relabel_spam_action(email)
                    confirm_action = create_confirm_action(email)

                    with col1:
                        st.button(
                            "📥 Đánh dấu là HAM",
                            key=f"ham_{email['id']}",
                            on_click=relabel_ham_action,
                            disabled=(current_label == 'ham'),
                            use_container_width=True,
                            help="Chuyển email này thành HAM",
                            type="secondary",
                            kwargs={},
                        )
                        st.markdown(
                            f"""
                            <style>
                            [data-testid="stButton"][key="ham_{email['id']}"] button {{
                                background: var(--royal-green-accent, #2ecc40) !important;
                                color: white !important;
                                border: none !important;
                                font-weight: 600;
                            }}
                            [data-testid="stButton"][key="ham_{email['id']}"] button:disabled {{
                                background: #b7e5c2 !important;
                                color: #f0f0f0 !important;
                            }}
                            [data-testid="stButton"][key="ham_{email['id']}"] button:hover:enabled {{
                                background: #1e9e2c !important;
                            }}
                            </style>
                            """,
                            unsafe_allow_html=True
                        )
                    with col2:
                        st.button(
                            "🗑️ Đánh dấu là SPAM",
                            key=f"spam_{email['id']}",
                            on_click=relabel_spam_action,
                            disabled=(current_label == 'spam'),
                            use_container_width=True,
                            help="Chuyển email này thành SPAM",
                            type="secondary",
                            kwargs={},
                        )
                        st.markdown(
                            f"""
                            <style>
                            [data-testid="stButton"][key="spam_{email['id']}"] button {{
                                background: var(--royal-green-accent, #2ecc40) !important;
                                color: white !important;
                                border: none !important;
                                font-weight: 600;
                            }}
                            [data-testid="stButton"][key="spam_{email['id']}"] button:disabled {{
                                background: #b7e5c2 !important;
                                color: #f0f0f0 !important;
                            }}
                            [data-testid="stButton"][key="spam_{email['id']}"] button:hover:enabled {{
                                background: #1e9e2c !important;
                            }}
                            </style>
                            """,
                            unsafe_allow_html=True
                        )
                    with col3:
                        st.button(
                            "✅ Xác nhận đúng",
                            key=f"confirm_{email['id']}",
                            on_click=confirm_action,
                            disabled=False,
                            use_container_width=True,
                            help="Xác nhận phân loại hiện tại là đúng",
                            type="secondary",
                            kwargs={},
                        )
                        st.markdown(
                            f"""
                            <style>
                            [data-testid="stButton"][key="confirm_{email['id']}"] button {{
                                background: var(--royal-green-accent, #2ecc40) !important;
                                color: white !important;
                                border: none !important;
                                font-weight: 600;
                            }}
                            [data-testid="stButton"][key="confirm_{email['id']}"] button:disabled {{
                                background: #b7e5c2 !important;
                                color: #f0f0f0 !important;
                            }}
                            [data-testid="stButton"][key="confirm_{email['id']}"] button:hover:enabled {{
                                background: #1e9e2c !important;
                            }}
                            </style>
                            """,
                            unsafe_allow_html=True
                        )
            # Cột Spam
            with col_spam:
                st.markdown('<div class="folder-box">', unsafe_allow_html=True)
                spam_count = len(st.session_state.get('spam_emails', []))
                st.markdown(f'<div class="folder-title">🗑️ Spam <span class="folder-count">{spam_count}</span></div>', unsafe_allow_html=True)
                
                for i, email in enumerate(st.session_state.get('spam_emails', [])):
                    # Tạo preview
                    sender_preview = email['sender'].split('<')[0].strip()[:20] if '<' in email['sender'] else email['sender'][:20]
                    confidence = email.get('confidence', 0)
                    
                    # 🆕 Tạo badge cho corrected emails
                    correction_badge = ""
                    if email.get('is_corrected', False):
                        if email['prediction'] != email['corrected_label']:
                            correction_badge = " ✅"  # Đã được sửa
                    
                    # Button cho spam emails
                    with st.container():
                        # Hiển thị tiêu đề mail trên nhiều dòng nếu dài
                        subject_full = email['subject']
                        # Chia tiêu đề thành nhiều dòng nếu quá dài
                        subject_lines = []
                        subject = subject_full
                        while len(subject) > 30:
                            subject_lines.append(subject[:30])
                            subject = subject[30:]
                        if subject:
                            subject_lines.append(subject)
                        
                        subject_display = "\n".join(subject_lines)
                        
                        # Tạo HTML button với text wrapping hoàn toàn cho spam
                        spam_button_html = f"""
                        <div style="
                            background: linear-gradient(135deg, var(--bg-tertiary) 0%, var(--royal-green-dark) 100%);
                            border: 1px solid var(--border-color);
                            border-radius: 12px;
                            padding: 1rem;
                            margin: 0.5rem 0;
                            cursor: pointer;
                            transition: all 0.3s ease;
                            white-space: pre-wrap;
                            word-wrap: break-word;
                            overflow-wrap: break-word;
                            text-align: left;
                            font-size: 0.9rem;
                            line-height: 1.5;
                            color: var(--text-primary);
                            width: 100%;
                            box-sizing: border-box;
                            min-height: auto;
                            height: auto;
                        " onmouseover="this.style.background='linear-gradient(135deg, var(--royal-green-light) 0%, var(--royal-green) 100%)'; this.style.transform='translateY(-2px)';" 
                           onmouseout="this.style.background='linear-gradient(135deg, var(--bg-tertiary) 0%, var(--royal-green-dark) 100%)'; this.style.transform='translateY(0)';"
                           onclick="window.parent.postMessage({{type: 'streamlit:setComponentValue', key: 'spam_{email['id']}_{i}', value: true}}, '*')">
                            🗑️ {subject_full}{correction_badge}<br>
                            👤 {sender_preview}<br>
                            📄 {email['snippet'][:40]}...<br>
                            ⚠️ Confidence: {confidence:.2f}
                        </div>
                        """
                        
                        # Hiển thị button HTML
                        st.markdown(spam_button_html, unsafe_allow_html=True)
                        
                        # Hidden button để trigger action
                        if st.button(
                            "Select",
                            key=f"spam_{email['id']}_{i}",
                            use_container_width=True
                        ):
                            st.session_state['selected_email'] = email
                            st.rerun()
                
                st.markdown('</div>', unsafe_allow_html=True)
        
        # Logout button
        st.markdown("---")
        if st.button("🔓 Đăng xuất", use_container_width=True):
            # Clear session state
            if 'gmail_credentials' in st.session_state:
                del st.session_state['gmail_credentials']
            if 'oauth_flow' in st.session_state:
                del st.session_state['oauth_flow']
            if 'oauth_state' in st.session_state:
                del st.session_state['oauth_state']
            st.success("✅ Đã đăng xuất!")
            st.rerun()
