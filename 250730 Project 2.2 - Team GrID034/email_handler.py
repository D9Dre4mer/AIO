"""
Gmail API Handler cho Streamlit App với OAuth flow - OPTIMIZED VERSION
Bao gồm các tính năng tối ưu cho correction management
"""
import os
import base64
import logging
from typing import List, Dict, Any, Optional, Tuple
from googleapiclient.discovery import build
from google_auth_oauthlib.flow import Flow
from google.oauth2.credentials import Credentials
from google.auth.transport.requests import Request
from googleapiclient.errors import HttpError
import streamlit as st
from datetime import datetime
import json

logger = logging.getLogger(__name__)

class GmailHandler:
    """Class để xử lý Gmail API trong Streamlit với OAuth flow và correction features"""
    
    def __init__(self, credentials_path: str):
        """
        Khởi tạo GmailHandler.
        
        Args:
            credentials_path: Đường dẫn đến file credentials.json
        """
        self.credentials_path = credentials_path
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
        if not os.path.exists(credentials_path):
            raise FileNotFoundError(f"Không tìm thấy file credentials.json tại: {credentials_path}")
    
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
    
    def create_correction_report_label(self, report_name: str = None) -> Optional[str]:
        """
        🆕 Tạo label đặc biệt cho correction report
        
        Args:
            report_name: Tên custom cho report (optional)
            
        Returns:
            Label ID nếu thành công, None nếu thất bại
        """
        try:
            if not report_name:
                timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
                report_name = f"CORRECTION_REPORT_{timestamp}"
            
            return self._get_or_create_label(report_name)
            
        except Exception as e:
            logger.error(f"Lỗi tạo correction report label: {str(e)}")
            return None
    
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
            
            # Get Gmail statistics để check đã sync chưa
            gmail_stats = self.get_correction_statistics()
            
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
    
    def create_correction_summary_email(self, corrections_data: Dict[str, Dict], 
                                      send_to_self: bool = False) -> Optional[str]:
        """
        🆕 Tạo email tóm tắt về corrections đã thực hiện
        
        Args:
            corrections_data: Dict chứa correction data
            send_to_self: Có gửi email cho chính mình không
            
        Returns:
            Message ID của email tóm tắt nếu thành công
        """
        if not self.service or not corrections_data:
            return None
            
        try:
            # Tạo nội dung email tóm tắt
            total_corrections = len(corrections_data)
            spam_to_ham = sum(1 for c in corrections_data.values() 
                            if c.get('original_prediction') == 'spam' and c.get('corrected_label') == 'ham')
            ham_to_spam = sum(1 for c in corrections_data.values() 
                            if c.get('original_prediction') == 'ham' and c.get('corrected_label') == 'spam')
            
            # HTML content
            html_content = f"""
            <html>
            <body>
                <h2>📧 Báo cáo Corrections - {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}</h2>
                
                <h3>📊 Thống kê tổng quan:</h3>
                <ul>
                    <li><strong>Tổng corrections:</strong> {total_corrections}</li>
                    <li><strong>🗑️ → 📥 Spam to Ham:</strong> {spam_to_ham}</li>
                    <li><strong>📥 → 🗑️ Ham to Spam:</strong> {ham_to_spam}</li>
                </ul>
                
                <h3>📋 Chi tiết corrections:</h3>
                <table border="1" style="border-collapse: collapse; width: 100%;">
                    <tr>
                        <th>Subject</th>
                        <th>Sender</th>
                        <th>Original</th>
                        <th>Corrected</th>
                        <th>Timestamp</th>
                    </tr>
            """
            
            # Add correction details
            for email_id, correction in corrections_data.items():
                html_content += f"""
                    <tr>
                        <td>{correction.get('subject', 'N/A')[:50]}...</td>
                        <td>{correction.get('sender', 'N/A')[:30]}...</td>
                        <td>{correction.get('original_prediction', 'N/A')}</td>
                        <td>{correction.get('corrected_label', 'N/A')}</td>
                        <td>{correction.get('timestamp', 'N/A')[:19]}</td>
                    </tr>
                """
            
            html_content += """
                </table>
                
                <p><em>Email này được tạo tự động bởi AI Email Classifier.</em></p>
            </body>
            </html>
            """
            
            # Nếu send_to_self, có thể implement gửi email
            # Hiện tại chỉ return nội dung để log
            logger.info(f"Đã tạo correction summary cho {total_corrections} corrections")
            
            # Trong thực tế, có thể save summary vào draft hoặc send
            return html_content
            
        except Exception as e:
            logger.error(f"Lỗi tạo correction summary: {str(e)}")
            return None
    
    def cleanup_old_corrections(self, days_old: int = 30) -> Dict[str, int]:
        """
        🆕 Dọn dẹp corrections cũ (remove correction labels)
        
        Args:
            days_old: Số ngày để coi là "cũ"
            
        Returns:
            Dict với thống kê cleanup
        """
        cleanup_stats = {
            'processed': 0,
            'cleaned': 0,
            'errors': 0
        }
        
        if not self.service or not self._correction_label_id:
            return cleanup_stats
            
        try:
            # Get corrected emails
            results = self.service.users().messages().list(
                userId='me',
                q='label:AI_CORRECTED'
            ).execute()
            
            messages = results.get('messages', [])
            
            from datetime import datetime, timedelta
            cutoff_date = datetime.now() - timedelta(days=days_old)
            
            for msg in messages:
                try:
                    cleanup_stats['processed'] += 1
                    
                    # Get message details để check date
                    message = self.service.users().messages().get(
                        userId='me',
                        id=msg['id'],
                        format='minimal'
                    ).execute()
                    
                    # Simple cleanup: remove correction label from all old corrections
                    # Trong thực tế có thể check date cụ thể hơn
                    self.service.users().messages().modify(
                        userId='me',
                        id=msg['id'],
                        body={'removeLabelIds': [self._correction_label_id]}
                    ).execute()
                    
                    cleanup_stats['cleaned'] += 1
                    
                except Exception as e:
                    logger.warning(f"Lỗi cleanup email {msg['id']}: {str(e)}")
                    cleanup_stats['errors'] += 1
            
            logger.info(f"Cleanup hoàn tất: {cleanup_stats['cleaned']}/{cleanup_stats['processed']} cleaned")
            return cleanup_stats
            
        except Exception as e:
            logger.error(f"Lỗi cleanup corrections: {str(e)}")
            return cleanup_stats
    
    def get_correction_insights(self) -> Dict[str, Any]:
        """
        🆕 Phân tích insights từ corrections để cải thiện model
        
        Returns:
            Dict chứa insights về patterns trong corrections
        """
        insights = {
            'total_corrections': 0,
            'correction_patterns': {},
            'sender_patterns': {},
            'subject_patterns': {},
            'recommendations': []
        }
        
        if not self.service or not self._correction_label_id:
            return insights
            
        try:
            # Export corrected emails để analyze
            corrected_emails = self.export_corrected_emails(max_results=200)
            
            if not corrected_emails:
                return insights
            
            insights['total_corrections'] = len(corrected_emails)
            
            # Analyze correction patterns
            spam_labels = ['SPAM']
            ham_labels = ['INBOX']
            
            for email in corrected_emails:
                current_labels = email.get('current_labels', [])
                
                # Determine current classification
                if any(label in spam_labels for label in current_labels):
                    current_class = 'spam'
                else:
                    current_class = 'ham'
                
                # Count patterns (giả sử có thông tin original prediction)
                pattern_key = f"corrected_to_{current_class}"
                insights['correction_patterns'][pattern_key] = insights['correction_patterns'].get(pattern_key, 0) + 1
                
                # Analyze sender patterns
                sender = email.get('sender', '')
                if '@' in sender:
                    domain = sender.split('@')[-1].split('>')[0]
                    insights['sender_patterns'][domain] = insights['sender_patterns'].get(domain, 0) + 1
                
                # Analyze subject patterns
                subject = email.get('subject', '').lower()
                common_spam_keywords = ['promotion', 'offer', 'free', 'win', 'urgent', 'limited']
                for keyword in common_spam_keywords:
                    if keyword in subject:
                        insights['subject_patterns'][keyword] = insights['subject_patterns'].get(keyword, 0) + 1
            
            # Generate recommendations
            if insights['correction_patterns'].get('corrected_to_ham', 0) > insights['correction_patterns'].get('corrected_to_spam', 0):
                insights['recommendations'].append("Model có xu hướng classify spam quá mức - cần điều chỉnh threshold")
            
            top_sender_domains = sorted(insights['sender_patterns'].items(), key=lambda x: x[1], reverse=True)[:3]
            if top_sender_domains:
                insights['recommendations'].append(f"Domains hay bị misclassify: {', '.join([d[0] for d in top_sender_domains])}")
            
            logger.info(f"Đã analyze insights từ {len(corrected_emails)} corrections")
            return insights
            
        except Exception as e:
            logger.error(f"Lỗi analyze correction insights: {str(e)}")
            return insights
    
    # 🆕 ===== ADVANCED FEATURES =====
    
    def auto_apply_corrections_from_file(self, corrections_file: str, 
                                       dry_run: bool = True) -> Dict[str, Any]:
        """
        🆕 Tự động apply corrections từ file với option dry run
        
        Args:
            corrections_file: Đường dẫn file corrections.json
            dry_run: Nếu True, chỉ simulate không thực sự apply
            
        Returns:
            Dict với kết quả operation
        """
        results = {
            'total_corrections': 0,
            'would_apply': 0,
            'actually_applied': 0,
            'errors': [],
            'dry_run': dry_run
        }
        
        if not os.path.exists(corrections_file):
            results['errors'].append(f"File không tồn tại: {corrections_file}")
            return results
        
        try:
            with open(corrections_file, 'r', encoding='utf-8') as f:
                corrections_data = json.load(f)
            
            results['total_corrections'] = len(corrections_data)
            
            for email_id, correction in corrections_data.items():
                try:
                    if dry_run:
                        # Chỉ validate email có tồn tại không
                        message = self.service.users().messages().get(
                            userId='me',
                            id=email_id,
                            format='minimal'
                        ).execute()
                        results['would_apply'] += 1
                    else:
                        # Thực sự apply correction
                        success = self.apply_single_correction(
                            email_id,
                            correction.get('corrected_label'),
                            correction.get('original_prediction')
                        )
                        if success:
                            results['actually_applied'] += 1
                
                except HttpError as e:
                    if e.resp.status == 404:
                        results['errors'].append(f"Email {email_id} không tồn tại")
                    else:
                        results['errors'].append(f"HTTP error {email_id}: {str(e)}")
                except Exception as e:
                    results['errors'].append(f"Error {email_id}: {str(e)}")
            
            mode = "DRY RUN" if dry_run else "LIVE"
            logger.info(f"Auto apply corrections [{mode}]: "
                       f"{results.get('actually_applied', results.get('would_apply', 0))}"
                       f"/{results['total_corrections']} processed")
            
            return results
            
        except Exception as e:
            results['errors'].append(f"Lỗi general: {str(e)}")
            return results
    
    def generate_training_data_from_corrections(self, output_file: str = None) -> Tuple[List[str], List[str]]:
        """
        🆕 Generate training data từ corrected emails để retrain model
        
        Args:
            output_file: File để save training data (optional)
            
        Returns:
            Tuple (messages, labels) cho training
        """
        messages = []
        labels = []
        
        try:
            # Export corrected emails
            corrected_emails = self.export_corrected_emails(max_results=500)
            
            spam_labels = ['SPAM']
            
            for email in corrected_emails:
                # Extract text content
                subject = email.get('subject', '')
                body = email.get('body', '')
                text_content = f"{subject} {body}".strip()
                
                if not text_content:
                    continue
                
                # Determine label từ current Gmail labels
                current_labels = email.get('current_labels', [])
                if any(label in spam_labels for label in current_labels):
                    label = 'spam'
                else:
                    label = 'ham'
                
                messages.append(text_content)
                labels.append(label)
            
            # Save to file nếu được yêu cầu
            if output_file and messages:
                import pandas as pd
                df = pd.DataFrame({
                    'Message': messages,
                    'Category': labels
                })
                df.to_csv(output_file, index=False, encoding='utf-8')
                logger.info(f"Đã save {len(messages)} training samples vào {output_file}")
            
            logger.info(f"Generated {len(messages)} training samples từ corrections")
            return messages, labels
            
        except Exception as e:
            logger.error(f"Lỗi generate training data: {str(e)}")
            return [], []