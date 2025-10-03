# Step 4 Training - Debugging Guide

## Nếu vẫn bị ngắt kết nối sau khi training

### 1. Kiểm tra Console Logs

Mở terminal và xem logs khi Streamlit chạy:

```bash
streamlit run app.py
```

Tìm các error messages:
- `ScriptRunContext`
- `MemoryError`
- `ConnectionError`
- `RecursionError`

### 2. Kiểm tra Session State

Thêm debug log vào code để kiểm tra session state:

```python
# Thêm vào cuối finally block (sau dòng 4400)
st.write("DEBUG - Session State:")
st.write(f"training_completed: {st.session_state.get('training_completed', 'NOT_SET')}")
st.write(f"results_displayed: {st.session_state.get('results_displayed', 'NOT_SET')}")
st.write(f"training_in_progress: {st.session_state.get('training_in_progress', 'NOT_SET')}")
```

### 3. Kiểm tra Memory Usage

Nếu dataset quá lớn, có thể gây out of memory:

```python
# Thêm vào đầu training (sau dòng 3933)
import psutil
process = psutil.Process()
memory_before = process.memory_info().rss / 1024 / 1024  # MB
st.info(f"💾 Memory before training: {memory_before:.2f} MB")
```

```python
# Thêm vào sau training xong (sau dòng 4115)
memory_after = process.memory_info().rss / 1024 / 1024  # MB
st.info(f"💾 Memory after training: {memory_after:.2f} MB")
st.info(f"💾 Memory increase: {memory_after - memory_before:.2f} MB")
```

### 4. Giảm Dataset Size

Nếu memory quá cao, giảm dataset size trong Step 1:
- Giảm số samples
- Giảm số features
- Sử dụng sampling

### 5. Disable Cache nếu cần

Nếu cache gây vấn đề, disable nó tạm thời:

```python
# Comment out cache save (dòng 4255-4283)
# try:
#     from cache_manager import training_results_cache
#     ...
# except Exception as cache_error:
#     ...
```

### 6. Kiểm tra Streamlit Version

Đảm bảo dùng Streamlit version tương thích:

```bash
pip show streamlit
```

Nên dùng version >= 1.28.0

### 7. Tăng Server Timeout

Thêm vào `.streamlit/config.toml`:

```toml
[server]
maxUploadSize = 1000
maxMessageSize = 1000
enableCORS = false
enableXsrfProtection = false

[browser]
serverAddress = "localhost"
gatherUsageStats = false
serverPort = 8501

[runner]
magicEnabled = false
fastReruns = true
```

### 8. Disable Auto-rerun

Nếu vẫn có vấn đề với rerun, disable nó hoàn toàn:

```python
# Comment out st.rerun() (dòng 4400)
# if not st.session_state.get('results_displayed', False):
#     st.rerun()
```

### 9. Check Network Connection

Nếu chạy trên remote server, kiểm tra:
- WebSocket connection
- Network latency
- Firewall settings

### 10. Alternative: Run without Spinner

Disable spinner để tránh UI blocking:

```python
# Comment out spinner (dòng 3932)
# with st.spinner("🔄 Starting training pipeline..."):
```

## Common Error Messages & Solutions

### Error: "ScriptRunContext not found"
**Solution**: Không dùng threading, dùng direct execution (đã fix)

### Error: "Memory Error"
**Solution**: Giảm dataset size hoặc tăng RAM

### Error: "Connection lost"
**Solution**: 
1. Check Streamlit server logs
2. Tăng timeout trong config
3. Disable auto-rerun

### Error: "Recursion Error"
**Solution**: Có vòng lặp rerun, check logic `results_displayed`

### Error: "Widget not found"
**Solution**: Widget đã bị cleanup, check timing của `.empty()`

## Debug Mode

Để enable full debug mode, thêm vào đầu file:

```python
import os
os.environ['STREAMLIT_DEBUG'] = '1'
```

Hoặc chạy với flag:

```bash
streamlit run app.py --logger.level=debug
```

## Performance Monitoring

Monitor performance với:

```python
import time

start_time = time.time()
# ... training code ...
elapsed_time = time.time() - start_time

st.info(f"⏱️ Total elapsed time: {elapsed_time:.2f}s")
```

## Liên hệ Support

Nếu vẫn gặp vấn đề, cung cấp thông tin sau:
1. Streamlit version
2. Python version
3. OS (Windows/Linux/Mac)
4. Dataset size (rows × columns)
5. Selected models
6. Error logs từ console
7. Memory usage
8. Screenshot của error

## Quick Fixes Checklist

- [ ] Restart Streamlit server
- [ ] Clear browser cache (Ctrl+Shift+R)
- [ ] Clear session state (click hamburger menu → Clear cache)
- [ ] Giảm dataset size
- [ ] Giảm số models
- [ ] Disable Optuna (trials = 0)
- [ ] Disable ensemble methods
- [ ] Check memory available
- [ ] Update Streamlit version
- [ ] Check Python version (>= 3.8)
