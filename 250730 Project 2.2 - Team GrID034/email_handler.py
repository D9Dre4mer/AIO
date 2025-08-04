"""
Gmail API Handler cho Streamlit App với AUTO TOKEN flow - COMPATIBLE VERSION
Tương thích hoàn toàn với app.py hiện tại và thêm tính năng tự động nhận token
"""
import os
import base64
import logging
from typing import List, Dict, Any, Optional, Tuple
from googleapiclient.discovery import build
from google_auth_oauthlib.flow import Flow, InstalledAppFlow
from google.oauth2.credentials import Credentials
from google.auth.transport.requests import Request
from googleapiclient.errors import HttpError
try:
    import streamlit as st
    STREAMLIT_AVAILABLE = True
except ImportError:
    STREAMLIT_AVAILABLE = False
    # Fallback cho non-streamlit environment
    class MockStreamlit:
        def __getattr__(self, name):
            return lambda *args, **kwargs: None
    st = MockStreamlit()

from datetime import datetime
import json
import threading
import webbrowser
import socket
from urllib.parse import urlparse, parse_qs

logger = logging.getLogger(__name__)

class GmailHandler:
    """Class để xử lý Gmail API trong Streamlit với AUTO TOKEN flow và correction features"""
    
    def __init__(self, pipeline=None, config=None, 
                 credentials_path: str = None):
        """
        Khởi tạo GmailHandler.
        
        Args:
            pipeline: SpamClassifierPipeline object (optional)
            config: SpamClassifierConfig object (optional)
            credentials_path: Đường dẫn đến file credentials.json
        """
        self.pipeline = pipeline
        self.config = config
        
        # Xác định đường dẫn credentials
        if credentials_path:
            self.credentials_path = credentials_path
        elif config and hasattr(config, 'credentials_path'):
            self.credentials_path = config.credentials_path
        else:
            self.credentials_path = './cache/input/credentials.json'
        
        self.SCOPES = [
            'https://www.googleapis.com/auth/gmail.readonly',
            'https://www.googleapis.com/auth/gmail.modify',
            'https://www.googleapis.com/auth/gmail.labels'
        ]
        self.service = None
        
        # 🆕 Cache cho labels để tránh API calls không cần thiết
        self._label_cache = {}
        self._correction_label_id = None
        
        # Kiểm tra file credentials
        if not os.path.exists(self.credentials_path):
            raise FileNotFoundError(f"Không tìm thấy file credentials.json tại: {self.credentials_path}")
    
    # 🆕 ===== AUTO AUTHENTICATION METHODS =====
    
    def authenticate_auto(self, port: int = 8080) -> bool:
        """
        🆕 Tự động authenticate với Google mà không cần copy token thủ công.
        
        Args:
            port: Port để chạy local server (default: 8080)
            
        Returns:
            True nếu authenticate thành công, False nếu thất bại
        """
        try:
            # Kiểm tra xem đã có credentials trong session chưa
            if self.initialize_service_from_session():
                logger.info("Đã có credentials hợp lệ trong session")
                return True
            
            # Kiểm tra file token.json nếu có
            token_file = 'token.json'
            if os.path.exists(token_file):
                if self._load_from_token_file(token_file):
                    logger.info("Đã load credentials từ token.json")
                    return True
            
            # Thực hiện OAuth flow mới
            logger.info(f"Bắt đầu OAuth flow trên port {port}...")
            
            # Kiểm tra port có available không
            if not self._is_port_available(port):
                logger.warning(f"Port {port} đang được sử dụng, thử port khác...")
                port = self._find_available_port()
                logger.info(f"Sử dụng port {port}")
            
            # Khởi tạo flow
            flow = InstalledAppFlow.from_client_secrets_file(
                self.credentials_path,
                scopes=self.SCOPES
            )
            
            # 🎯 TỰ ĐỘNG MỞ BROWSER VÀ NHẬN TOKEN
            credentials = flow.run_local_server(
                port=port,
                prompt='select_account',  # Cho phép chọn account
                open_browser=True,       # Tự động mở browser
                success_message='✅ Authentication thành công! Bạn có thể đóng tab này.',
                access_type='offline'    # Để lấy refresh token
            )
            
            # Lưu credentials vào session
            st.session_state['gmail_credentials'] = {
                'token': credentials.token,
                'refresh_token': credentials.refresh_token,
                'token_uri': credentials.token_uri,
                'client_id': credentials.client_id,
                'client_secret': credentials.client_secret,
                'scopes': credentials.scopes
            }
            
            # Lưu vào file token.json để dùng lần sau
            self._save_to_token_file(credentials, token_file)
            
            # Khởi tạo Gmail service
            self.service = build('gmail', 'v1', credentials=credentials)
            
            # 🆕 Initialize correction system
            self._initialize_correction_system()
            
            logger.info("✅ Authentication thành công!")
            st.success("🎉 Đã kết nối Gmail thành công!")
            
            return True
            
        except Exception as e:
            error_msg = str(e)
            logger.error(f"Lỗi authentication: {error_msg}")
            
            # Xử lý lỗi cụ thể
            if "invalid_client" in error_msg:
                st.error("❌ File credentials.json không hợp lệ. Vui lòng kiểm tra lại.")
            elif "access_denied" in error_msg:
                st.error("❌ Người dùng từ chối cấp quyền. Vui lòng thử lại.")
            elif "Connection refused" in error_msg or "WinError 10061" in error_msg:
                st.error(f"❌ Không thể mở local server trên port {port}. Vui lòng thử port khác.")
            else:
                st.error(f"❌ Lỗi authentication: {error_msg}")
            
            return False
    
    def _is_port_available(self, port: int) -> bool:
        """Kiểm tra port có available không."""
        try:
            with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
                s.bind(('localhost', port))
                return True
        except OSError:
            return False
    
    def _find_available_port(self, start_port: int = 8080, max_attempts: int = 10) -> int:
        """Tìm port available."""
        for port in range(start_port, start_port + max_attempts):
            if self._is_port_available(port):
                return port
        
        # Fallback: let system choose
        with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
            s.bind(('localhost', 0))
            return s.getsockname()[1]
    
    def _load_from_token_file(self, token_file: str) -> bool:
        """
        Load credentials từ file token.json.
        
        Args:
            token_file: Đường dẫn file token.json
            
        Returns:
            True nếu load thành công
        """
        try:
            with open(token_file, 'r') as f:
                cred_info = json.load(f)
            
            credentials = Credentials.from_authorized_user_info(cred_info, self.SCOPES)
            
            # Refresh nếu expired
            if credentials.expired and credentials.refresh_token:
                try:
                    credentials.refresh(Request())
                    # Update file với token mới
                    self._save_to_token_file(credentials, token_file)
                    logger.info("Đã refresh token từ file")
                except Exception as e:
                    logger.warning(f"Không thể refresh token từ file: {str(e)}")
                    return False
            
            # Lưu vào session
            st.session_state['gmail_credentials'] = {
                'token': credentials.token,
                'refresh_token': credentials.refresh_token,
                'token_uri': credentials.token_uri,
                'client_id': credentials.client_id,
                'client_secret': credentials.client_secret,
                'scopes': credentials.scopes
            }
            
            # Khởi tạo service
            self.service = build('gmail', 'v1', credentials=credentials)
            self._initialize_correction_system()
            
            return True
            
        except Exception as e:
            logger.warning(f"Không thể load từ token file: {str(e)}")
            return False
    
    def _save_to_token_file(self, credentials: Credentials, token_file: str):
        """
        Lưu credentials vào file token.json.
        
        Args:
            credentials: Google credentials object
            token_file: Đường dẫn file để lưu
        """
        try:
            cred_info = {
                'token': credentials.token,
                'refresh_token': credentials.refresh_token,
                'token_uri': credentials.token_uri,
                'client_id': credentials.client_id,
                'client_secret': credentials.client_secret,
                'scopes': credentials.scopes
            }
            
            with open(token_file, 'w') as f:
                json.dump(cred_info, f, indent=2)
            
            logger.info(f"Đã lưu credentials vào {token_file}")
            
        except Exception as e:
            logger.warning(f"Không thể lưu token file: {str(e)}")
    
    def logout(self):
        """🆕 Logout và xóa tất cả credentials."""
        try:
            # Xóa từ session
            if 'gmail_credentials' in st.session_state:
                del st.session_state['gmail_credentials']
            
            # Xóa các oauth states nếu có
            if 'oauth_flow' in st.session_state:
                del st.session_state['oauth_flow']
            if 'oauth_state' in st.session_state:
                del st.session_state['oauth_state']
            if 'oauth_flow_manual' in st.session_state:
                del st.session_state['oauth_flow_manual']
            if 'oauth_state_manual' in st.session_state:
                del st.session_state['oauth_state_manual']
            
            # Xóa file token.json
            token_file = 'token.json'
            if os.path.exists(token_file):
                os.remove(token_file)
                logger.info("Đã xóa token.json")
            
            # Reset service
            self.service = None
            self._label_cache = {}
            self._correction_label_id = None
            
            st.success("✅ Đã logout thành công!")
            logger.info("Logout thành công")
            
        except Exception as e:
            logger.error(f"Lỗi logout: {str(e)}")
            st.error(f"❌ Lỗi logout: {str(e)}")
    
    def get_auth_status(self) -> Dict[str, Any]:
        """
        🆕 Lấy trạng thái authentication hiện tại.
        
        Returns:
            Dict chứa thông tin auth status
        """
        status = {
            'is_authenticated': False,
            'has_service': False,
            'user_email': None,
            'token_expires': None,
            'scopes': []
        }
        
        try:
            if self.service:
                status['has_service'] = True
                
                # Lấy user profile
                try:
                    profile = self.get_user_profile()
                    status['user_email'] = profile.get('email')
                    status['is_authenticated'] = True
                except:
                    pass
            
            # Kiểm tra credentials trong session
            cred_info = st.session_state.get('gmail_credentials')
            if cred_info:
                status['scopes'] = cred_info.get('scopes', [])
                
                # Check token expiry (nếu có thông tin)
                try:
                    credentials = Credentials.from_authorized_user_info(cred_info, self.SCOPES)
                    if hasattr(credentials, 'expiry') and credentials.expiry:
                        status['token_expires'] = credentials.expiry.isoformat()
                except:
                    pass
            
            return status
            
        except Exception as e:
            logger.error(f"Lỗi get auth status: {str(e)}")
            return status
    
    # ===== MANUAL AUTHENTICATION METHODS (TƯƠNG THÍCH VỚI APP.PY CŨ) =====
    
    def get_authorization_url(self) -> str:
        """
        Tạo URL để authorize với Google với account selector.
        
        Returns:
            URL authorization để user click
        """
        try:
            flow = Flow.from_client_secrets_file(
                self.credentials_path,
                scopes=self.SCOPES,
                redirect_uri='http://localhost:8080'
            )
            
            # ✅ FIX: Chỉ dùng 'prompt', bỏ 'approval_prompt'
            auth_url, state = flow.authorization_url(
                access_type='offline',
                include_granted_scopes='true',
                prompt='consent select_account',  # ✅ Sử dụng prompt thay vì approval_prompt
            )
            
            # Lưu flow và state vào session để dùng sau
            st.session_state['oauth_flow'] = flow
            st.session_state['oauth_state'] = state
            
            logger.info("Đã tạo authorization URL với account selector")
            return auth_url
            
        except Exception as e:
            logger.error(f"Lỗi tạo authorization URL: {str(e)}")
            raise
    
    def get_authorization_url_with_hint(self, email_hint: str = None) -> str:
        """
        Tạo URL với hint email cụ thể (alternative method).
        
        Args:
            email_hint: Email gợi ý để pre-select
            
        Returns:
            URL authorization với email hint
        """
        try:
            flow = Flow.from_client_secrets_file(
                self.credentials_path,
                scopes=self.SCOPES,
                redirect_uri='http://localhost:8080'
            )
            
            auth_params = {
                'access_type': 'offline',
                'include_granted_scopes': 'true',
                'prompt': 'select_account',  # ✅ FIX: Chỉ dùng prompt
            }
            
            # Thêm login_hint nếu có
            if email_hint:
                auth_params['login_hint'] = email_hint
            
            auth_url, state = flow.authorization_url(**auth_params)
            
            # Lưu flow và state vào session để dùng sau
            st.session_state['oauth_flow'] = flow
            st.session_state['oauth_state'] = state
            
            logger.info(f"Đã tạo authorization URL với email hint: {email_hint}")
            return auth_url
            
        except Exception as e:
            logger.error(f"Lỗi tạo authorization URL với hint: {str(e)}")
            raise
    
    def handle_oauth_callback(self, authorization_code: str) -> bool:
        """
        Xử lý callback từ OAuth với authorization code.
        
        Args:
            authorization_code: Code từ Google OAuth
            
        Returns:
            True nếu thành công, False nếu thất bại
        """
        try:
            flow = st.session_state.get('oauth_flow')
            if not flow:
                logger.error("Không tìm thấy OAuth flow trong session")
                return False
            
            # Clean authorization code (remove whitespace)
            authorization_code = authorization_code.strip()
            
            # Validate code format
            if not authorization_code or len(authorization_code) < 10:
                logger.error("Authorization code không hợp lệ")
                return False
                
            # Fetch token từ authorization code
            flow.fetch_token(code=authorization_code)
            
            # Lưu credentials vào session
            credentials = flow.credentials
            st.session_state['gmail_credentials'] = {
                'token': credentials.token,
                'refresh_token': credentials.refresh_token,
                'token_uri': credentials.token_uri,
                'client_id': credentials.client_id,
                'client_secret': credentials.client_secret,
                'scopes': credentials.scopes
            }
            
            # Lưu vào file token.json để dùng lần sau
            self._save_to_token_file(credentials, 'token.json')
            
            # Khởi tạo service
            self.service = build('gmail', 'v1', credentials=credentials)
            
            # 🆕 Initialize correction system
            self._initialize_correction_system()
            
            logger.info("OAuth callback xử lý thành công")
            return True
            
        except Exception as e:
            error_msg = str(e)
            logger.error(f"Lỗi OAuth callback: {error_msg}")
            
            # Specific error handling
            if "invalid_grant" in error_msg:
                logger.error("Authorization code đã hết hạn hoặc đã được sử dụng")
                st.error("❌ Authorization code đã hết hạn hoặc đã được sử dụng. Vui lòng tạo mới.")
            elif "invalid_request" in error_msg:
                logger.error("Request không hợp lệ")
                st.error("❌ Request không hợp lệ. Vui lòng kiểm tra lại authorization code.")
            else:
                st.error(f"❌ Lỗi xác thực: {error_msg}")
            
            return False
    
    # ===== CORE GMAIL METHODS (GIỮ NGUYÊN) =====
    
    def initialize_service_from_session(self) -> bool:
        """
        Khởi tạo Gmail service từ credentials trong session.
        
        Returns:
            True nếu thành công, False nếu thất bại
        """
        try:
            cred_info = st.session_state.get('gmail_credentials')
            if not cred_info:
                logger.warning("Không tìm thấy credentials trong session")
                return False
                
            # Tạo credentials object từ thông tin đã lưu
            credentials = Credentials.from_authorized_user_info(cred_info, self.SCOPES)
            
            # Refresh token nếu expired
            if credentials.expired and credentials.refresh_token:
                try:
                    credentials.refresh(Request())
                    # Cập nhật token mới vào session
                    st.session_state['gmail_credentials']['token'] = credentials.token
                    logger.info("Đã refresh token thành công")
                except Exception as e:
                    logger.error(f"Lỗi refresh token: {str(e)}")
                    return False
            
            # Khởi tạo service
            self.service = build('gmail', 'v1', credentials=credentials)
            
            # 🆕 Initialize correction system
            self._initialize_correction_system()
            
            logger.info("Đã khởi tạo Gmail service từ session")
            return True
            
        except Exception as e:
            logger.error(f"Lỗi khởi tạo service từ session: {str(e)}")
            return False
    
    def _initialize_correction_system(self):
        """🆕 Khởi tạo hệ thống correction labels"""
        try:
            # Refresh label cache
            self._refresh_label_cache()
            
            # Tạo correction label nếu chưa có
            self._correction_label_id = self._get_or_create_label("AI_CORRECTED")
            
            logger.info("Đã khởi tạo correction system")
        except Exception as e:
            logger.warning(f"Không thể khởi tạo correction system: {str(e)}")
    
    def _refresh_label_cache(self):
        """🆕 Refresh cache của labels"""
        try:
            labels_result = self.service.users().labels().list(userId='me').execute()
            labels = labels_result.get('labels', [])
            
            self._label_cache = {label['name']: label['id'] for label in labels}
            logger.info(f"Đã cache {len(self._label_cache)} labels")
            
        except Exception as e:
            logger.error(f"Lỗi refresh label cache: {str(e)}")
    
    def fetch_emails(self, max_results: int = 10, query: str = "is:unread") -> List[Dict[str, Any]]:
        """
        Fetch emails từ Gmail.
        
        Args:
            max_results: Số lượng email tối đa
            query: Query string để filter emails
            
        Returns:
            List các email với thông tin chi tiết
        """
        if not self.service:
            raise ValueError("Gmail service chưa được khởi tạo. Vui lòng authenticate trước.")
            
        try:
            # Lấy danh sách message IDs
            logger.info(f"Fetching emails với query: {query}, max_results: {max_results}")
            
            results = self.service.users().messages().list(
                userId='me',
                q=query,
                maxResults=max_results
            ).execute()
            
            messages = results.get('messages', [])
            
            if not messages:
                logger.info("Không tìm thấy email nào")
                return []
            
            emails = []
            for msg in messages:
                try:
                    # Lấy chi tiết message
                    message = self.service.users().messages().get(
                        userId='me',
                        id=msg['id'],
                        format='full'
                    ).execute()
                    
                    # Extract thông tin email
                    email_data = self._extract_email_data(message)
                    emails.append(email_data)
                    
                except HttpError as e:
                    logger.warning(f"Lỗi khi lấy email {msg['id']}: {str(e)}")
                    continue
                    
            logger.info(f"Đã fetch {len(emails)} emails thành công")
            return emails
            
        except HttpError as e:
            logger.error(f"Lỗi Gmail API khi fetch emails: {str(e)}")
            raise
        except Exception as e:
            logger.error(f"Lỗi không xác định khi fetch emails: {str(e)}")
            raise
    
    def _extract_email_data(self, message: Dict) -> Dict[str, Any]:
        """
        Extract dữ liệu từ Gmail message object.
        
        Args:
            message: Gmail message object
            
        Returns:
            Dict chứa thông tin email đã extract
        """
        headers = message['payload'].get('headers', [])
        
        # Extract header information
        subject = self._get_header_value(headers, 'Subject') or 'No Subject'
        sender = self._get_header_value(headers, 'From') or 'Unknown Sender'
        date = self._get_header_value(headers, 'Date') or 'Unknown Date'
        
        # Extract body
        body = self._extract_body(message['payload'])
        
        # Get snippet (preview text)
        snippet = message.get('snippet', '')
        
        # 🆕 Check if email has correction label
        label_ids = message.get('labelIds', [])
        is_corrected = self._correction_label_id in label_ids if self._correction_label_id else False
        
        return {
            'id': message['id'],
            'subject': subject,
            'sender': sender,
            'date': date,
            'body': body,
            'snippet': snippet,
            'thread_id': message.get('threadId', ''),
            'label_ids': label_ids,
            'is_corrected_in_gmail': is_corrected  # 🆕 Gmail-level correction status
        }
    
    def _get_header_value(self, headers: List[Dict], name: str) -> str:
        """
        Lấy giá trị header theo tên.
        
        Args:
            headers: List các header
            name: Tên header cần lấy
            
        Returns:
            Giá trị header hoặc None nếu không tìm thấy
        """
        for header in headers:
            if header.get('name', '').lower() == name.lower():
                return header.get('value', '')
        return None
    
    def _extract_body(self, payload: Dict) -> str:
        """
        Extract body từ email payload với HTML parsing.
        
        Args:
            payload: Email payload từ Gmail API
            
        Returns:
            Text body của email (đã parse HTML)
        """
        body = ""
        
        # Xử lý email có nhiều parts (multipart)
        if 'parts' in payload:
            for part in payload['parts']:
                # Ưu tiên text/plain
                if part.get('mimeType') == 'text/plain':
                    data = part.get('body', {}).get('data', '')
                    if data:
                        try:
                            body = base64.urlsafe_b64decode(data).decode('utf-8')
                            break
                        except Exception as e:
                            logger.warning(f"Lỗi decode body part: {str(e)}")
                            continue
                            
                # Fallback to text/html nếu không có text/plain
                elif part.get('mimeType') == 'text/html' and not body:
                    data = part.get('body', {}).get('data', '')
                    if data:
                        try:
                            html_content = base64.urlsafe_b64decode(data).decode('utf-8')
                            body = self._html_to_text(html_content)
                        except Exception as e:
                            logger.warning(f"Lỗi decode HTML body: {str(e)}")
                            continue
        
        # Xử lý email đơn giản (không có parts)
        else:
            if payload.get('mimeType') == 'text/plain':
                data = payload.get('body', {}).get('data', '')
                if data:
                    try:
                        body = base64.urlsafe_b64decode(data).decode('utf-8')
                    except Exception as e:
                        logger.warning(f"Lỗi decode single body: {str(e)}")
            elif payload.get('mimeType') == 'text/html':
                data = payload.get('body', {}).get('data', '')
                if data:
                    try:
                        html_content = base64.urlsafe_b64decode(data).decode('utf-8')
                        body = self._html_to_text(html_content)
                    except Exception as e:
                        logger.warning(f"Lỗi decode HTML single body: {str(e)}")
        
        # Fallback to snippet nếu không extract được body
        if not body or len(body.strip()) < 10:
            body = payload.get('snippet', 'No content available')
            
        return body
    
    def _html_to_text(self, html_content: str) -> str:
        """
        Convert HTML content thành plain text.
        
        Args:
            html_content: HTML string
            
        Returns:
            Plain text đã được làm sạch
        """
        try:
            from bs4 import BeautifulSoup
            
            # Parse HTML
            soup = BeautifulSoup(html_content, 'html.parser')
            
            # Remove script và style tags
            for script in soup(["script", "style"]):
                script.decompose()
            
            # Get text và clean up
            text = soup.get_text()
            
            # Clean up whitespace
            lines = (line.strip() for line in text.splitlines())
            chunks = (phrase.strip() for line in lines for phrase in line.split("  "))
            text = '\n'.join(chunk for chunk in chunks if chunk)
            
            return text
            
        except ImportError:
            # Fallback nếu không có BeautifulSoup
            logger.warning("BeautifulSoup không có, sử dụng regex để parse HTML")
            return self._html_to_text_regex(html_content)
        except Exception as e:
            logger.warning(f"Lỗi parse HTML với BeautifulSoup: {str(e)}")
            return self._html_to_text_regex(html_content)
    
    def _html_to_text_regex(self, html_content: str) -> str:
        """
        Fallback HTML parser sử dụng regex (không cần BeautifulSoup).
        
        Args:
            html_content: HTML string
            
        Returns:
            Plain text
        """
        import re
        
        # Remove HTML tags
        text = re.sub(r'<[^>]+>', '', html_content)
        
        # Decode HTML entities
        import html
        text = html.unescape(text)
        
        # Clean up whitespace
        text = re.sub(r'\s+', ' ', text)
        text = text.strip()
        
        return text
    
    def mark_as_read(self, message_id: str) -> bool:
        """
        Đánh dấu email đã đọc.
        
        Args:
            message_id: ID của email
            
        Returns:
            True nếu thành công
        """
        if not self.service:
            logger.error("Gmail service chưa được khởi tạo")
            return False
            
        try:
            self.service.users().messages().modify(
                userId='me',
                id=message_id,
                body={'removeLabelIds': ['UNREAD']}
            ).execute()
            
            logger.info(f"Đã đánh dấu email {message_id} là đã đọc")
            return True
            
        except HttpError as e:
            logger.error(f"Lỗi Gmail API khi mark as read: {str(e)}")
            return False
        except Exception as e:
            logger.error(f"Lỗi không xác định khi mark as read: {str(e)}")
            return False
    
    def move_to_label(self, message_id: str, label_name: str) -> bool:
        """
        Di chuyển email tới label (tạo label nếu chưa có).
        
        Args:
            message_id: ID của email
            label_name: Tên label
            
        Returns:
            True nếu thành công
        """
        if not self.service:
            logger.error("Gmail service chưa được khởi tạo")
            return False
            
        try:
            # Tạo hoặc lấy label ID
            label_id = self._get_or_create_label(label_name)
            
            # Add label to message
            self.service.users().messages().modify(
                userId='me',
                id=message_id,
                body={'addLabelIds': [label_id]}
            ).execute()
            
            logger.info(f"Đã thêm label '{label_name}' cho email {message_id}")
            return True
            
        except Exception as e:
            logger.error(f"Lỗi khi thêm label: {str(e)}")
            return False
    
    def _get_or_create_label(self, label_name: str) -> str:
        """
        Lấy hoặc tạo label và trả về label ID.
        
        Args:
            label_name: Tên label
            
        Returns:
            Label ID
        """
        try:
            # Check cache trước
            if label_name in self._label_cache:
                return self._label_cache[label_name]
            
            # Refresh cache và check lại
            self._refresh_label_cache()
            if label_name in self._label_cache:
                return self._label_cache[label_name]
            
            # Tạo label mới nếu không tìm thấy
            new_label = {
                'name': label_name,
                'labelListVisibility': 'labelShow',
                'messageListVisibility': 'show'
            }
            
            created_label = self.service.users().labels().create(
                userId='me',
                body=new_label
            ).execute()
            
            # Update cache
            label_id = created_label['id']
            self._label_cache[label_name] = label_id
            
            logger.info(f"Đã tạo label mới: {label_name}")
            return label_id
            
        except HttpError as e:
            logger.error(f"Lỗi Gmail API khi tạo/lấy label: {str(e)}")
            raise
        except Exception as e:
            logger.error(f"Lỗi không xác định khi tạo/lấy label: {str(e)}")
            raise
    
    def get_user_profile(self) -> Dict[str, str]:
        """
        Lấy thông tin profile của user.
        
        Returns:
            Dict chứa thông tin user
        """
        if not self.service:
            raise ValueError("Gmail service chưa được khởi tạo")
            
        try:
            profile = self.service.users().getProfile(userId='me').execute()
            return {
                'email': profile.get('emailAddress', 'Unknown'),
                'total_messages': profile.get('messagesTotal', 0),
                'total_threads': profile.get('threadsTotal', 0)
            }
        except HttpError as e:
            logger.error(f"Lỗi Gmail API khi lấy profile: {str(e)}")
            raise
        except Exception as e:
            logger.error(f"Lỗi không xác định khi lấy profile: {str(e)}")
            raise
    
    # 🆕 ===== CORRECTION MANAGEMENT METHODS =====
    
    def apply_single_correction(self, email_id: str, corrected_label: str, 
                              original_prediction: str = None) -> bool:
        """
        🆕 Áp dụng correction cho một email cụ thể
        
        Args:
            email_id: ID của email
            corrected_label: Label đã được sửa ('spam' hoặc 'ham')
            original_prediction: Prediction gốc của AI (để logging)
            
        Returns:
            True nếu thành công
        """
        if not self.service:
            logger.error("Gmail service chưa được khởi tạo")
            return False
            
        try:
            # Add correction label
            if self._correction_label_id:
                self.service.users().messages().modify(
                    userId='me',
                    id=email_id,
                    body={'addLabelIds': [self._correction_label_id]}
                ).execute()
            
            # Apply corrected classification
            if corrected_label.lower() == 'spam':
                # Move to spam và remove inbox
                success = self._move_to_spam(email_id)
            else:
                # Move to inbox và remove spam
                success = self._move_to_inbox(email_id)
            
            if success:
                logger.info(f"Đã apply correction cho email {email_id}: {original_prediction} → {corrected_label}")
            
            return success
            
        except Exception as e:
            logger.error(f"Lỗi apply correction cho email {email_id}: {str(e)}")
            return False
    
    def _move_to_spam(self, email_id: str) -> bool:
        """🆕 Di chuyển email vào spam folder"""
        try:
            self.service.users().messages().modify(
                userId='me',
                id=email_id,
                body={
                    'addLabelIds': ['SPAM'],
                    'removeLabelIds': ['INBOX']
                }
            ).execute()
            return True
        except Exception as e:
            logger.error(f"Lỗi move to spam: {str(e)}")
            return False
    
    def _move_to_inbox(self, email_id: str) -> bool:
        """🆕 Di chuyển email vào inbox"""
        try:
            self.service.users().messages().modify(
                userId='me',
                id=email_id,
                body={
                    'addLabelIds': ['INBOX'],
                    'removeLabelIds': ['SPAM']
                }
            ).execute()
            return True
        except Exception as e:
            logger.error(f"Lỗi move to inbox: {str(e)}")
            return False
    
    def bulk_apply_corrections(self, corrections_data: Dict[str, Dict]) -> Dict[str, bool]:
        """
        🆕 Áp dụng corrections cho nhiều emails cùng lúc
        
        Args:
            corrections_data: Dict {email_id: {corrected_label, original_prediction, ...}}
            
        Returns:
            Dict với kết quả apply cho từng email
        """
        results = {}
        
        if not self.service:
            logger.error("Gmail service chưa được khởi tạo")
            return {email_id: False for email_id in corrections_data.keys()}
        
        logger.info(f"Bắt đầu bulk apply {len(corrections_data)} corrections...")
        
        for email_id, correction_info in corrections_data.items():
            try:
                corrected_label = correction_info.get('corrected_label')
                original_prediction = correction_info.get('original_prediction')
                
                success = self.apply_single_correction(
                    email_id, 
                    corrected_label, 
                    original_prediction
                )
                results[email_id] = success
                
            except Exception as e:
                logger.error(f"Lỗi bulk apply cho email {email_id}: {str(e)}")
                results[email_id] = False
        
        success_count = sum(1 for success in results.values() if success)
        logger.info(f"Bulk apply hoàn tất: {success_count}/{len(corrections_data)} thành công")
        
        return results
    
    def get_correction_statistics(self) -> Dict[str, int]:
        """
        🆕 Lấy thống kê emails đã corrected từ Gmail labels
        
        Returns:
            Dict chứa thống kê
        """
        if not self.service:
            logger.error("Gmail service chưa được khởi tạo")
            return {'total_corrected': 0, 'spam_corrections': 0, 'ham_corrections': 0}
            
        try:
            # Count emails with correction label
            if not self._correction_label_id:
                return {'total_corrected': 0, 'spam_corrections': 0, 'ham_corrections': 0}
            
            # Query với correction label
            results = self.service.users().messages().list(
                userId='me',
                q='label:AI_CORRECTED'
            ).execute()
            
            total_corrected = len(results.get('messages', []))
            
            if total_corrected == 0:
                return {'total_corrected': 0, 'spam_corrections': 0, 'ham_corrections': 0}
            
            # Count spam corrections
            spam_results = self.service.users().messages().list(
                userId='me', 
                q='label:AI_CORRECTED label:SPAM'
            ).execute()
            
            spam_count = len(spam_results.get('messages', []))
            ham_count = total_corrected - spam_count
            
            return {
                'total_corrected': total_corrected,
                'spam_corrections': spam_count,
                'ham_corrections': ham_count
            }
            
        except Exception as e:
            logger.error(f"Lỗi lấy correction statistics: {str(e)}")
            return {'total_corrected': 0, 'spam_corrections': 0, 'ham_corrections': 0}
    
    def export_corrected_emails(self, max_results: int = 100) -> List[Dict[str, Any]]:
        """
        🆕 Export danh sách emails đã được corrected
        
        Args:
            max_results: Số lượng email tối đa để export
            
        Returns:
            List emails đã corrected với thông tin chi tiết
        """
        if not self.service or not self._correction_label_id:
            logger.warning("Service hoặc correction label chưa sẵn sàng")
            return []
            
        try:
            # Fetch corrected emails
            results = self.service.users().messages().list(
                userId='me',
                q='label:AI_CORRECTED',
                maxResults=max_results
            ).execute()
            
            messages = results.get('messages', [])
            if not messages:
                return []
            
            corrected_emails = []
            for msg in messages:
                try:
                    # Get full message details
                    message = self.service.users().messages().get(
                        userId='me',
                        id=msg['id'],
                        format='full'
                    ).execute()
                    
                    # Extract email data
                    email_data = self._extract_email_data(message)
                    
                    # Add correction metadata
                    email_data['correction_timestamp'] = datetime.now().isoformat()
                    email_data['current_labels'] = [
                        self._get_label_name_by_id(label_id) 
                        for label_id in email_data.get('label_ids', [])
                    ]
                    
                    corrected_emails.append(email_data)
                    
                except Exception as e:
                    logger.warning(f"Lỗi export email {msg['id']}: {str(e)}")
                    continue
            
            logger.info(f"Đã export {len(corrected_emails)} corrected emails")
            return corrected_emails
            
        except Exception as e:
            logger.error(f"Lỗi export corrected emails: {str(e)}")
            return []
    
    def _get_label_name_by_id(self, label_id: str) -> str:
        """🆕 Lấy tên label từ ID (sử dụng cache)"""
        for name, cached_id in self._label_cache.items():
            if cached_id == label_id:
                return name
        
        # Fallback: query trực tiếp nếu không có trong cache
        try:
            label = self.service.users().labels().get(userId='me', id=label_id).execute()
            return label.get('name', label_id)
        except:
            return label_id
    
    def sync_corrections_with_local_file(self, local_corrections_path: str) -> Dict[str, Any]:
        """
        🆕 Đồng bộ corrections giữa Gmail và file local
        
        Args:
            local_corrections_path: Đường dẫn đến file corrections.json local
            
        Returns:
            Dict chứa kết quả sync
        """
        sync_results = {
            'applied_to_gmail': 0,
            'already_synced': 0,
            'failed': 0,
            'errors': []
        }
        
        if not os.path.exists(local_corrections_path):
            logger.warning(f"File corrections không tồn tại: {local_corrections_path}")
            return sync_results
        
        try:
            # Load local corrections
            with open(local_corrections_path, 'r', encoding='utf-8') as f:
                local_corrections = json.load(f)
            
            if not local_corrections:
                logger.info("Không có corrections để sync")
                return sync_results
            
            logger.info(f"Bắt đầu sync {len(local_corrections)} corrections với Gmail...")
            
            for email_id, correction_data in local_corrections.items():
                try:
                    # Check xem email có đã được corrected trong Gmail chưa
                    message = self.service.users().messages().get(
                        userId='me',
                        id=email_id,
                        format='minimal'
                    ).execute()
                    
                    label_ids = message.get('labelIds', [])
                    already_corrected = self._correction_label_id in label_ids
                    
                    if already_corrected:
                        sync_results['already_synced'] += 1
                        continue
                    
                    # Apply correction to Gmail
                    corrected_label = correction_data.get('corrected_label')
                    original_prediction = correction_data.get('original_prediction')
                    
                    success = self.apply_single_correction(
                        email_id, 
                        corrected_label, 
                        original_prediction
                    )
                    
                    if success:
                        sync_results['applied_to_gmail'] += 1
                    else:
                        sync_results['failed'] += 1
                        
                except HttpError as e:
                    if e.resp.status == 404:
                        # Email không tồn tại (có thể đã bị xóa)
                        sync_results['errors'].append(f"Email {email_id} không tồn tại")
                    else:
                        sync_results['errors'].append(f"Lỗi HTTP cho {email_id}: {str(e)}")
                    sync_results['failed'] += 1
                    
                except Exception as e:
                    sync_results['errors'].append(f"Lỗi sync {email_id}: {str(e)}")
                    sync_results['failed'] += 1
            
            logger.info(f"Sync hoàn tất: {sync_results['applied_to_gmail']} applied, "
                       f"{sync_results['already_synced']} already synced, "
                       f"{sync_results['failed']} failed")
            
            return sync_results
            
        except Exception as e:
            logger.error(f"Lỗi sync corrections: {str(e)}")
            sync_results['errors'].append(f"Lỗi general: {str(e)}")
            return sync_results
    
    # 🆕 ===== EMAIL PROCESSING METHODS FOR MAIN.PY COMPATIBILITY =====
    
    def process_emails(self, max_results: int = 10) -> Dict[str, Any]:
        """
        🆕 Xử lý emails mới và phân loại spam (tương thích với main.py)
        
        Args:
            max_results: Số lượng email tối đa để xử lý
            
        Returns:
            Dict chứa kết quả xử lý
        """
        if not self.service:
            logger.error("Gmail service chưa được khởi tạo")
            return {'processed': 0, 'spam_count': 0, 'ham_count': 0, 'errors': []}
        
        if not self.pipeline:
            logger.error("Spam classifier pipeline chưa được khởi tạo")
            return {'processed': 0, 'spam_count': 0, 'ham_count': 0, 'errors': []}
        
        try:
            # Fetch emails mới
            emails = self.fetch_emails(max_results=max_results, query="is:unread")
            
            if not emails:
                logger.info("Không có email mới để xử lý")
                return {'processed': 0, 'spam_count': 0, 'ham_count': 0, 'errors': []}
            
            results = {
                'processed': 0,
                'spam_count': 0,
                'ham_count': 0,
                'errors': []
            }
            
            for email in emails:
                try:
                    # Phân loại email
                    prediction_result = self.pipeline.predict(email['body'], k=3)
                    prediction = prediction_result['prediction']
                    confidence = prediction_result.get('confidence', 0.0)
                    
                    # Log kết quả
                    logger.info(f"Email {email['id']}: {prediction} (confidence: {confidence:.2f})")
                    
                    # Áp dụng phân loại vào Gmail
                    if prediction.lower() == 'spam':
                        success = self._move_to_spam(email['id'])
                        if success:
                            results['spam_count'] += 1
                        else:
                            results['errors'].append(f"Không thể move email {email['id']} vào spam")
                    else:
                        # Đánh dấu đã đọc cho ham emails
                        success = self.mark_as_read(email['id'])
                        if success:
                            results['ham_count'] += 1
                        else:
                            results['errors'].append(f"Không thể mark email {email['id']} as read")
                    
                    results['processed'] += 1
                    
                except Exception as e:
                    error_msg = f"Lỗi xử lý email {email['id']}: {str(e)}"
                    logger.error(error_msg)
                    results['errors'].append(error_msg)
            
            logger.info(f"Đã xử lý {results['processed']} emails: {results['spam_count']} spam, {results['ham_count']} ham")
            return results
            
        except Exception as e:
            error_msg = f"Lỗi trong process_emails: {str(e)}"
            logger.error(error_msg)
            return {'processed': 0, 'spam_count': 0, 'ham_count': 0, 'errors': [error_msg]}
    
    def get_unread_emails(self, max_results: int = 10) -> List[Dict[str, Any]]:
        """
        🆕 Lấy danh sách emails chưa đọc
        
        Args:
            max_results: Số lượng email tối đa
            
        Returns:
            List emails chưa đọc
        """
        return self.fetch_emails(max_results=max_results, query="is:unread")
    
    def get_spam_emails(self, max_results: int = 10) -> List[Dict[str, Any]]:
        """
        🆕 Lấy danh sách emails trong spam folder
        
        Args:
            max_results: Số lượng email tối đa
            
        Returns:
            List emails trong spam
        """
        return self.fetch_emails(max_results=max_results, query="label:spam")
    
    def get_inbox_emails(self, max_results: int = 10) -> List[Dict[str, Any]]:
        """
        🆕 Lấy danh sách emails trong inbox
        
        Args:
            max_results: Số lượng email tối đa
            
        Returns:
            List emails trong inbox
        """
        return self.fetch_emails(max_results=max_results, query="label:inbox")
    
    def initialize_for_main(self) -> bool:
        """
        🆕 Khởi tạo Gmail service cho main.py (tương thích với main.py)
        
        Returns:
            True nếu thành công, False nếu thất bại
        """
        try:
            # Thử load từ token file trước
            token_file = 'token.json'
            if os.path.exists(token_file):
                if self._load_from_token_file(token_file):
                    logger.info("Đã khởi tạo service từ token file")
                    return True
            
            # Nếu không có token, thử authenticate auto
            if self.authenticate_auto():
                logger.info("Đã khởi tạo service qua auto authentication")
                return True
            
            logger.error("Không thể khởi tạo Gmail service")
            return False
            
        except Exception as e:
            logger.error(f"Lỗi khởi tạo service cho main.py: {str(e)}")
            return False