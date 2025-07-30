import os
import streamlit as st
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import numpy as np
import plotly.express as px
from sklearn.manifold import TSNE

from config import SpamClassifierConfig
from data_loader import DataLoader
from embedding_generator import EmbeddingGenerator
from evaluator import ModelEvaluator
from spam_classifier import SpamClassifierPipeline
from email_handler import GmailHandler  # Import updated Gmail handler

# --- Cấu hình trang và CSS tùy chỉnh ---
st.set_page_config(page_title="Bảng điều khiển Spam Mail", layout="centered")
st.markdown("""
<style>
/* Chỉnh theme tối toàn cục */
html, body, [class*="css"] {
    font-family: 'Segoe UI', sans-serif;
    color: #f0f0f0;
    background-color: #111827;
}
/* Style cho button */
.stButton > button {
    background-color: #3b82f6;
    color: white;
    border-radius: 8px;
    padding: 0.5rem 1rem;
    transition: all 0.3s ease;
}
.stButton > button:hover {
    background-color: #2563eb;
    transform: translateY(-2px);
}
/* Tiêu đề lớn và đoạn mô tả */
.main-title {
    font-size: 2.8rem;
    font-weight: bold;
    margin-top: 1rem;
    color: #f9fafb;
    text-align: center;
    text-shadow: 1px 1px 5px #3b82f6;
}
.subtext {
    font-size: 1.1rem;
    line-height: 1.6;
    color: #d1d5db;
    max-width: 800px;
    margin: auto;
}
footer {
    color: #9ca3af;
    font-size: 0.85rem;
    text-align: center;
    margin-top: 2rem;
}
/* Hộp thư */
.folder-box {
    background-color: #1f2937;
    border-radius: 8px;
    padding: 0.2rem 0.5rem;
    box-shadow: 0 4px 6px rgba(0, 0, 0, 0.1);
    margin-bottom: 1rem;
}
.folder-title {
    font-size: 1.2rem;
    font-weight: bold;
    color: #3b82f6;
    display: flex;
    align-items: center;
    margin-top: 0;
}
.folder-count {
    background-color: #ef4444;
    color: white;
    border-radius: 50%;
    padding: 0.2rem 0.6rem;
    margin-left: 0.5rem;
    font-size: 0.9rem;
}
/* Style trực tiếp cho button trong list */
.folder-box .stButton > button {
    background-color: #374151;
    margin: 0.3rem 0;
    width: 100%;
    text-align: left;
}
.folder-box .stButton > button:hover {
    background-color: #4b5563;
    transform: translateY(-2px);
}
/* Nội dung mails */
.content-container {
    background-color: #1f2937;
    border-radius: 8px;
    padding: 1rem;
    min-height: 300px;
    display: flex;
    flex-direction: column;
    align-items: stretch;
    justify-content: flex-start;
    color: #d1d5db;
    overflow-y: auto;
    max-width: 100%;
    word-wrap: break-word;
    white-space: pre-wrap;
}
.placeholder {
    color: #9ca3af;
    font-style: italic;
    text-align: center;
    flex-grow: 1;
    display: flex;
    align-items: center;
    justify-content: center;
}
.auth-box {
    background-color: #1f2937;
    border-radius: 8px;
    padding: 2rem;
    text-align: center;
    margin: 2rem 0;
    border: 2px solid #3b82f6;
}
.email-item {
    background-color: #374151;
    border-radius: 6px;
    padding: 0.8rem;
    margin: 0.5rem 0;
    border-left: 4px solid #3b82f6;
    cursor: pointer;
    transition: all 0.2s ease;
}
.email-item:hover {
    background-color: #4b5563;
    transform: translateX(4px);
}
.email-subject {
    font-weight: bold;
    font-size: 1rem;
    color: #f9fafb;
    margin-bottom: 0.3rem;
}
.email-sender {
    font-size: 0.85rem;
    color: #9ca3af;
    margin-bottom: 0.2rem;
}
.email-snippet {
    font-size: 0.8rem;
    color: #d1d5db;
    opacity: 0.8;
}
.user-profile {
    background-color: #1f2937;
    border-radius: 8px;
    padding: 1rem;
    margin-bottom: 1rem;
    border: 1px solid #374151;
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
def load_pipeline():
    """
    Khởi tạo và train pipeline phân loại spam.
    Kết quả được cache để không train lại mỗi lần rerun.
    """
    cfg = SpamClassifierConfig()
    pipeline = SpamClassifierPipeline(cfg)
    pipeline.train()
    return pipeline

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
    return eg.generate_embeddings(messages)

@st.cache_data
def compute_tsne_cached(sub_emb: np.ndarray) -> np.ndarray:
    """Cache t-SNE computation."""
    return TSNE(n_components=2, init="random", learning_rate="auto").fit_transform(sub_emb)

# Khởi tạo các components
try:
    gmail_handler = get_gmail_handler()
    pipeline = load_pipeline()
except Exception as e:
    st.error(f"Lỗi khởi tạo ứng dụng: {str(e)}")
    st.stop()

if "df" not in st.session_state:
    st.session_state["df"] = load_sample_data(SpamClassifierConfig().dataset_path)
df = st.session_state["df"]

# --- Quản lý trạng thái trang ---
if "page" not in st.session_state:
    st.session_state.page = "🏠 Tổng quan"

if st.session_state.page != "🏠 Tổng quan":
    if st.button("🏠 Trở về Tổng quan"):
        st.session_state.page = "🏠 Tổng quan"
        st.rerun()

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
    st.markdown('<h1 class="main-title">📧 Bộ phân loại Spam/Ham Mail</h1>', unsafe_allow_html=True)
    st.markdown('<div class="subtext">Khám phá và phân loại email với giao diện tương tác!</div>', unsafe_allow_html=True)

    # Thống kê nhanh
    total = len(df)
    spam_cnt = len(df[df["Category"] == "spam"])
    ham_cnt  = len(df[df["Category"] == "ham"])
    c1, c2, c3 = st.columns(3)
    c1.metric("Tổng số Email", total)
    c2.metric("Email Spam", spam_cnt, f"{spam_cnt/total*100:.1f}%")
    c3.metric("Email Ham", ham_cnt, f"{ham_cnt/total*100:.1f}%")

    st.markdown("### Tính năng:")

    # Nút chuyển đến từng page
    col1, col2, col3 = st.columns(3)
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
    pipe_knn.train()
    pipe_tfidf = SpamClassifierPipeline(cfg, classifier_type="tfidf")
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

# --- Trang Quét Gmail ---
# Thay thế phần Gmail authentication trong app.py:

elif st.session_state.page == "✉️ Quét Gmail":
    st.header("✉️ Quét Gmail")

    # Kiểm tra xác thực
    if 'gmail_credentials' not in st.session_state:
        st.markdown('<div class="auth-box">', unsafe_allow_html=True)
        st.markdown("### 🔐 Cần xác thực Gmail")
        st.markdown("Để quét email từ Gmail, bạn cần đăng nhập với tài khoản Google của mình.")
        
        # 🔥 FIXED: Không gọi function, dùng URL trực tiếp
        st.markdown("**Nếu Chrome không cho chọn tài khoản:**")
        col_clear, col_normal = st.columns(2)
        
        with col_clear:
            # 🔧 FIX: Dùng URL trực tiếp thay vì function call
            logout_url = "https://accounts.google.com/logout"
            st.markdown(f'<a href="{logout_url}" target="_blank" style="background-color: #ef4444; color: white; padding: 0.6rem 1rem; text-decoration: none; border-radius: 6px; display: inline-block; margin: 0.5rem 0; font-size: 0.9rem;">🚪 Clear Google Session</a>', unsafe_allow_html=True)
            st.caption("Click này trước để logout Google")
        
        with col_normal:
            # 🔧 FIX: Dùng function có sẵn
            try:
                auth_url_fresh = gmail_handler.get_authorization_url()
                st.markdown(f'<a href="{auth_url_fresh}" target="_blank" style="background-color: #f59e0b; color: white; padding: 0.6rem 1rem; text-decoration: none; border-radius: 6px; display: inline-block; margin: 0.5rem 0; font-size: 0.9rem;">🔥 Đăng nhập Gmail</a>', unsafe_allow_html=True)
                st.caption("Force login với account selection")
            except Exception as e:
                st.error(f"Lỗi tạo auth URL: {str(e)}")
        
        st.markdown("---")
        st.markdown("**Tùy chọn khác:**")
        
        col1, col2 = st.columns(2)
        
        with col1:
            st.markdown("**Với email cụ thể:**")
            email_hint = st.text_input("Nhập email:", placeholder="user@gmail.com", key="email_hint")
            if email_hint:
                try:
                    auth_url_hint = gmail_handler.get_authorization_url_with_hint(email_hint)
                    st.markdown(f'<a href="{auth_url_hint}" target="_blank" style="background-color: #22c55e; color: white; padding: 0.6rem 1rem; text-decoration: none; border-radius: 6px; display: inline-block; margin: 0.5rem 0; font-size: 0.9rem;">🎯 Login {email_hint[:20]}...</a>', unsafe_allow_html=True)
                except Exception as e:
                    st.error(f"Lỗi tạo auth URL với hint: {str(e)}")
        
        with col2:
            st.markdown("**Incognito Mode:**")
            st.info("💡 Mở Incognito (`Ctrl+Shift+N`) và copy link bên trái vào đó")
        
        # Troubleshooting guide
        with st.expander("🔧 Vẫn không chọn được tài khoản?"):
            st.markdown("""
            **Thử theo thứ tự:**
            
            1. **Clear Chrome Cache:**
               - `Ctrl+Shift+Delete` → Chọn "All time" → Clear data
            
            2. **Clear Google Cookies cụ thể:**
               - Chrome Settings → Privacy → Cookies → "See all cookies"
               - Tìm và xóa: `accounts.google.com`, `oauth2.googleapis.com`
            
            3. **Incognito Mode (Khuyên dùng):**
               - `Ctrl+Shift+N` → Copy link "🔥 Đăng nhập Gmail" vào incognito
               - Incognito sẽ hiện account selector 100%
            
            4. **Browser khác:**
               - Firefox, Edge thường không có vấn đề cache này
            
            5. **Manual logout:**
               - Vào gmail.com → Logout tất cả tài khoản
               - Sau đó thử lại
            """)
        
        st.markdown("---")
        st.markdown("**Nhập authorization code:**")
        auth_code = st.text_input("Authorization code từ Google:", placeholder="Paste code từ Google tại đây...")
        
        if st.button("🔐 Xác thực", use_container_width=True, type="primary") and auth_code:
            with st.spinner("Đang xác thực..."):
                try:
                    if gmail_handler.handle_oauth_callback(auth_code):
                        st.success("✅ Xác thực thành công!")
                        st.balloons()
                        st.rerun()
                    else:
                        st.error("❌ Lỗi xác thực! Vui lòng kiểm tra code và thử lại.")
                except Exception as e:
                    st.error(f"❌ Lỗi xác thực: {str(e)}")
        
        st.markdown('</div>', unsafe_allow_html=True)
    
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
                            
                            result = pipeline.predict(text_to_classify)
                            prediction = result['prediction']
                            confidence_scores = result.get('label_distribution', {})
                            confidence = max(confidence_scores.values()) if confidence_scores else 0.5
                            
                            email['prediction'] = prediction
                            email['confidence'] = confidence
                            email['confidence_scores'] = confidence_scores
                            classified_emails.append(email)
                            
                            progress_bar.progress((i + 1) / len(emails))
                        
                        # Lưu vào session state
                        st.session_state['classified_emails'] = classified_emails
                        st.session_state['inbox_emails'] = [e for e in classified_emails if e['prediction'] == 'ham']
                        st.session_state['spam_emails'] = [e for e in classified_emails if e['prediction'] == 'spam']
                        
                        progress_bar.empty()
                        progress_text.empty()
                        
                        # Thống kê
                        total_emails = len(classified_emails)
                        spam_count = len(st.session_state['spam_emails'])
                        ham_count = len(st.session_state['inbox_emails'])
                        
                        st.success(f"✅ Đã quét và phân loại {total_emails} emails!")
                        
                        col1, col2, col3 = st.columns(3)
                        col1.metric("Tổng số", total_emails)
                        col2.metric("Ham", ham_count, f"{ham_count/total_emails*100:.1f}%")
                        col3.metric("Spam", spam_count, f"{spam_count/total_emails*100:.1f}%")
                        
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
                
                for email in st.session_state.get('inbox_emails', []):
                    # Tạo preview
                    subject_preview = email['subject'][:35] + "..." if len(email['subject']) > 35 else email['subject']
                    sender_preview = email['sender'].split('<')[0].strip()[:20] if '<' in email['sender'] else email['sender'][:20]
                    confidence = email.get('confidence', 0)
                    
                    # HTML cho email item với confidence
                    email_html = f"""
                    <div class="email-item" onclick="document.getElementById('inbox_{email['id']}').click()">
                        <div class="email-subject">{subject_preview}</div>
                        <div class="email-sender">{sender_preview}</div>
                        <div class="email-snippet">{email['snippet'][:40]}...</div>
                        <div style="font-size: 0.7rem; color: #22c55e; margin-top: 0.2rem;">
                            Confidence: {confidence:.2f}
                        </div>
                    </div>
                    """
                    st.markdown(email_html, unsafe_allow_html=True)
                    
                    # Hidden button để handle click
                    if st.button("Select", key=f"inbox_{email['id']}", help="Click to select email"):
                        st.session_state['selected_email'] = email
                        st.rerun()
                
                st.markdown('</div>', unsafe_allow_html=True)
            
            # Cột nội dung (giữa)
            with col_content:
                if st.session_state['selected_email'] is None:
                    content_html = """
                    <div class="content-container">
                        <div class="placeholder">
                            📧 Chọn một email từ Inbox hoặc Spam để xem nội dung
                        </div>
                    </div>
                    """
                else:
                    email = st.session_state['selected_email']
                    from html import escape
                    
                    # Truncate body nếu quá dài
                    body_display = email['body'][:1500] + "..." if len(email['body']) > 1500 else email['body']
                    confidence_scores = email.get('confidence_scores', {})
                    confidence_display = ", ".join([f"{k}: {v:.2f}" for k, v in confidence_scores.items()])
                    
                    content_html = f"""
                    <div class="content-container">
                        <div style="margin-bottom: 1rem; padding-bottom: 1rem; border-bottom: 1px solid #374151;">
                            <div style="display: flex; justify-content: space-between; align-items: center; margin-bottom: 0.5rem;">
                                <span style="font-weight: bold; color: #3b82f6;">
                                    {'📥 HAM' if email['prediction'] == 'ham' else '🗑️ SPAM'}
                                </span>
                                <span style="background-color: {'#22c55e' if email['prediction'] == 'ham' else '#ef4444'}; 
                                            color: white; padding: 0.2rem 0.6rem; border-radius: 12px; font-size: 0.8rem;">
                                    {email['prediction'].upper()}
                                </span>
                            </div>
                            <div style="font-size: 0.85rem; color: #9ca3af; margin-bottom: 0.3rem;">
                                <strong>From:</strong> {escape(email['sender'])}
                            </div>
                            <div style="font-size: 0.85rem; color: #9ca3af; margin-bottom: 0.3rem;">
                                <strong>Date:</strong> {escape(email['date'])}
                            </div>
                            <div style="font-size: 0.85rem; color: #9ca3af; margin-bottom: 0.3rem;">
                                <strong>Confidence:</strong> {confidence_display}
                            </div>
                            <div style="font-size: 1.1rem; font-weight: bold; color: #f9fafb;">
                                {escape(email['subject'])}
                            </div>
                        </div>
                        <div style="line-height: 1.6; white-space: pre-wrap; overflow-wrap: break-word;">
                            {escape(body_display).replace(chr(10), '<br>')}
                        </div>
                    </div>
                    """
                
                st.markdown(content_html, unsafe_allow_html=True)
                
                # Actions cho email đã chọn
                if st.session_state['selected_email'] is not None:
                    st.markdown("**🔧 Thao tác:**")
                    col1, col2, col3 = st.columns(3)
                    
                    with col1:
                        if st.button("✅ Đánh dấu đã đọc", use_container_width=True):
                            email_id = st.session_state['selected_email']['id']
                            if gmail_handler.mark_as_read(email_id):
                                st.success("Đã đánh dấu đã đọc!")
                            else:
                                st.error("Lỗi đánh dấu đã đọc")
                    
                    with col2:
                        if st.button("🏷️ Gắn nhãn Spam", use_container_width=True):
                            email_id = st.session_state['selected_email']['id']
                            if gmail_handler.move_to_label(email_id, "AI_Detected_Spam"):
                                st.success("Đã gắn nhãn Spam!")
                            else:
                                st.error("Lỗi gắn nhãn")
                    
                    with col3:
                        if st.button("🏷️ Gắn nhãn Ham", use_container_width=True):
                            email_id = st.session_state['selected_email']['id']
                            if gmail_handler.move_to_label(email_id, "AI_Detected_Ham"):
                                st.success("Đã gắn nhãn Ham!")
                            else:
                                st.error("Lỗi gắn nhãn")
            
            # Cột Spam (phải)
            with col_spam:
                st.markdown('<div class="folder-box">', unsafe_allow_html=True)
                spam_count = len(st.session_state.get('spam_emails', []))
                st.markdown(f'<div class="folder-title">🗑️ Spam <span class="folder-count">{spam_count}</span></div>', unsafe_allow_html=True)
                
                for email in st.session_state.get('spam_emails', []):
                    # Tạo preview
                    subject_preview = email['subject'][:35] + "..." if len(email['subject']) > 35 else email['subject']
                    sender_preview = email['sender'].split('<')[0].strip()[:20] if '<' in email['sender'] else email['sender'][:20]
                    confidence = email.get('confidence', 0)
                    
                    # HTML cho email item với confidence
                    email_html = f"""
                    <div class="email-item" onclick="document.getElementById('spam_{email['id']}').click()" 
                         style="border-left-color: #ef4444;">
                        <div class="email-subject">{subject_preview}</div>
                        <div class="email-sender">{sender_preview}</div>
                        <div class="email-snippet">{email['snippet'][:40]}...</div>
                        <div style="font-size: 0.7rem; color: #ef4444; margin-top: 0.2rem;">
                            Confidence: {confidence:.2f}
                        </div>
                    </div>
                    """
                    st.markdown(email_html, unsafe_allow_html=True)
                    
                    # Hidden button để handle click
                    if st.button("Select", key=f"spam_{email['id']}", help="Click to select email"):
                        st.session_state['selected_email'] = email
                        st.rerun()
                
                st.markdown('</div>', unsafe_allow_html=True)
        
        # Logout button
        st.markdown("---")
        if st.button("🚪 Đăng xuất Gmail", type="secondary"):
            # Clear tất cả session data liên quan đến Gmail
            keys_to_clear = [
                'gmail_credentials', 'oauth_flow', 'oauth_state',
                'classified_emails', 'inbox_emails', 'spam_emails', 'selected_email'
            ]
            for key in keys_to_clear:
                if key in st.session_state:
                    del st.session_state[key]
            st.success("Đã đăng xuất!")
            st.rerun()

# --- Footer ---
st.markdown("<footer>Được xây dựng với Streamlit | Vận hành bởi pipeline AI của bạn.</footer>", unsafe_allow_html=True)