# RUN_DEMO_STEPS (Windows PowerShell)

## 0) Chuẩn bị nhanh (chạy một lần)
```powershell
cd "AIO/250730 Project 2.2 - Team GrID034"
python --version
pip install -r requirements.txt
python -c "import nltk; nltk.download('stopwords'); nltk.download('wordnet')"
```

---

## 1) CLI Demo (chạy trên Terminal)

### 1.1 Train + tạo embeddings (lần đầu)
```powershell
python .\main.py
Get-Content .\logs\spam_classifier.log -Tail 20
```

### 1.2 Đánh giá hiệu năng (tuỳ chọn)
```powershell
python .\main.py --evaluate --k-values "1,3,5,7"
```

### 1.3 Gộp email local rồi tái tạo embeddings (tuỳ chọn)
```powershell
python .\main.py --merge-emails --regenerate
```

### 1.4 Chạy phân loại Gmail real-time (tuỳ chọn)
- Đặt `credentials.json` vào `cache\input\`
```powershell
# Nếu cần reset token lần trước (bỏ qua nếu chưa tạo token)
Remove-Item .\cache\input\token.json -ErrorAction SilentlyContinue

python .\main.py --run-email-classifier
# Dừng an toàn: Ctrl + C
```

### 1.5 Tái tạo embeddings thủ công (khi cần)
```powershell
python .\main.py --regenerate
```

---

## 2) Streamlit UI Demo

### 2.1 Khởi động giao diện Web
```powershell
streamlit run .\app.py
```

### 2.2 Thứ tự thao tác tối giản trên UI (không cần câu lệnh)
- Dashboard: xem trạng thái model/metrics/cache
- Classify: nhập nội dung email → Predict
- Batch: upload CSV → Run
- Corrections: chỉnh `cache\corrections.json` → Retrain with Corrections
- Visualization: mở biểu đồ/TSNE
- Gmail: nếu có nút Start/Stop, bật để watch realtime

---

## 3) Quick Fix (khi gặp lỗi thường gặp)

### 3.1 Cài lại/thêm gói
```powershell
pip install -r requirements.txt
```

### 3.2 Xoá cache rồi tạo lại
```powershell
Remove-Item .\cache\embeddings\* -Force -ErrorAction SilentlyContinue
Remove-Item .\cache\faiss_index\* -Force -ErrorAction SilentlyContinue
python .\main.py --regenerate
```

### 3.3 Reset Gmail token
```powershell
Remove-Item .\cache\input\token.json -ErrorAction SilentlyContinue
python .\main.py --run-email-classifier
```

### 3.4 Dừng/khởi động lại Streamlit
```powershell
# Dừng: đóng cửa sổ hoặc
Get-Process streamlit -ErrorAction SilentlyContinue | Stop-Process
# Chạy lại
streamlit run .\app.py
```
