"""
Gmail API Handler cho Streamlit App với OAuth flow
"""
import os
import base64
import logging
from typing import List, Dict, Any
from googleapiclient.discovery import build
from google_auth_oauthlib.flow import Flow
from google.oauth2.credentials import Credentials
from google.auth.transport.requests import Request
from googleapiclient.errors import HttpError
import streamlit as st

logger = logging.getLogger(__name__)

"""
Gmail API Handler cho Streamlit App với OAuth flow
"""
import os
import base64
import logging
from typing import List, Dict, Any
from googleapiclient.discovery import build
from google_auth_oauthlib.flow import Flow
from google.oauth2.credentials import Credentials
from google.auth.transport.requests import Request
from googleapiclient.errors import HttpError
import streamlit as st

logger = logging.getLogger(__name__)

class GmailHandler:
    """Class để xử lý Gmail API trong Streamlit với OAuth flow"""
    
    def __init__(self, credentials_path: str):
        """
        Khởi tạo GmailHandler.
        
        Args:
            credentials_path: Đường dẫn đến file credentials.json
        """
        self.credentials_path = credentials_path
        self.SCOPES = [
            'https://www.googleapis.com/auth/gmail.readonly',
            'https://www.googleapis.com/auth/gmail.modify'
        ]
        self.service = None
        
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
            
            auth_url, state = flow.authorization_url(
                access_type='offline',
                include_granted_scopes='true',
                prompt='select_account',  # 🔥 QUAN TRỌNG: Bắt buộc chọn tài khoản
                # prompt='consent'  # Thay bằng 'select_account'
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
                'prompt': 'select_account',
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
            
            logger.info("OAuth callback xử lý thành công")
            return True
            
        except Exception as e:
            logger.error(f"Lỗi OAuth callback: {str(e)}")
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
            logger.info("Đã khởi tạo Gmail service từ session")
            return True
            
        except Exception as e:
            logger.error(f"Lỗi khởi tạo service từ session: {str(e)}")
            return False
    
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
        
        return {
            'id': message['id'],
            'subject': subject,
            'sender': sender,
            'date': date,
            'body': body,
            'snippet': snippet,
            'thread_id': message.get('threadId', ''),
            'label_ids': message.get('labelIds', [])
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
        Extract body từ email payload.
        
        Args:
            payload: Email payload từ Gmail API
            
        Returns:
            Text body của email
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
                            body = base64.urlsafe_b64decode(data).decode('utf-8')
                            # TODO: Có thể thêm HTML parsing ở đây nếu cần
                        except Exception as e:
                            logger.warning(f"Lỗi decode HTML body: {str(e)}")
                            continue
        
        # Xử lý email đơn giản (không có parts)
        else:
            if payload.get('mimeType') in ['text/plain', 'text/html']:
                data = payload.get('body', {}).get('data', '')
                if data:
                    try:
                        body = base64.urlsafe_b64decode(data).decode('utf-8')
                    except Exception as e:
                        logger.warning(f"Lỗi decode single body: {str(e)}")
        
        # Fallback to snippet nếu không extract được body
        if not body:
            body = payload.get('snippet', 'No content available')
            
        return body
    
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
            # Lấy danh sách labels hiện có
            labels_result = self.service.users().labels().list(userId='me').execute()
            labels = labels_result.get('labels', [])
            
            # Tìm label theo tên
            for label in labels:
                if label['name'] == label_name:
                    return label['id']
            
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
            
            logger.info(f"Đã tạo label mới: {label_name}")
            return created_label['id']
            
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
        except Exception as e:
            logger.error(f"Lỗi khi lấy user profile: {str(e)}")
            return {'email': 'Unknown', 'total_messages': 0, 'total_threads': 0}