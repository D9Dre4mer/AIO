# Step 4 - Quick Fix Reference Card

## 🐛 Vấn đề: Streamlit ngắt kết nối sau training

### ✅ ĐÃ FIX (2025-10-03)

| Vấn đề | Nguyên nhân | Fix |
|--------|-------------|-----|
| **Ngắt sau hiển thị results** | Rerun ngay sau display | Thêm flag `results_displayed` |
| **Memory leak** | Progress widgets không cleanup | Cleanup `progress_bar`, `status_text` |
| **UI freeze** | Spinner/progress không clear | Clear trong exception handler |

## 🔧 3 Fixes chính

### Fix #1: Ngăn Rerun Loop
```python
# TRƯỚC (dòng 4396) - SAI
st.rerun()  # ❌ Rerun ngay sau display → crash

# SAU (dòng 4397-4400) - ĐÚNG
if not st.session_state.get('results_displayed', False):
    st.rerun()  # ✅ Chỉ rerun nếu chưa display
```

### Fix #2: Set Flag sau Display
```python
# Dòng 4256 - Thêm sau st.dataframe()
st.session_state.results_displayed = True  # ✅ Đánh dấu đã display
```

### Fix #3: Cleanup Widgets
```python
# Dòng 4117-4122 - Cleanup sau success
try:
    progress_bar.empty()      # ✅ Clear progress bar
    status_text.empty()       # ✅ Clear status text
except:
    pass
```

## 🎯 Test Nhanh

1. ✅ Run training
2. ✅ Xem "Detailed Results" hiển thị
3. ✅ **KHÔNG** bị disconnect
4. ✅ Click "Next ▶" → Chuyển sang Step 5

## 🚨 Nếu vẫn lỗi

### Option 1: Check Session State
```python
# Thêm vào cuối finally block
st.write(f"results_displayed: {st.session_state.get('results_displayed')}")
```

### Option 2: Disable Auto-rerun
```python
# Comment dòng 4400
# st.rerun()
```

### Option 3: Reduce Dataset
- Giảm samples trong Step 1
- Giảm số models trong Step 3

## 📊 Flow hoạt động

```
User click "Start Training"
    ↓
Set results_displayed = False
    ↓
Training chạy...
    ↓
Display "Detailed Results"
    ↓
Set results_displayed = True
    ↓
Finally block check results_displayed
    ↓
    ├─ False → st.rerun() (refresh UI)
    └─ True → SKIP rerun (tránh crash)
    ↓
Cleanup widgets
    ↓
UI ổn định ✅
```

## 🔍 Debug Commands

### Check memory
```bash
pip install psutil
```

### Check Streamlit version
```bash
pip show streamlit
```

### Run debug mode
```bash
streamlit run app.py --logger.level=debug
```

## ⚡ Performance Tips

1. **Giảm Optuna trials**: 50 → 10
2. **Disable ensemble**: Voting OFF, Stacking OFF
3. **Sample data**: Dùng 10K samples thay vì 100K
4. **Fewer models**: Chọn 2-3 models thay vì 10+

## 📝 Files Modified

- `app.py` (6 vị trí):
  - Dòng 3905: Reset flag (Restart)
  - Dòng 3923: Init flag (Start)
  - Dòng 4117-4122: Cleanup widgets (success)
  - Dòng 4126-4129: Cleanup widgets (error)
  - Dòng 4256: Set flag (after display)
  - Dòng 4397-4400: Conditional rerun

## 💡 Key Points

1. ✅ **results_displayed flag** prevents rerun loop
2. ✅ **Cleanup widgets** prevents memory leak
3. ✅ **Conditional rerun** prevents crash
4. ✅ **Exception handling** ensures cleanup

## 🎉 Expected Behavior

- Training completes ✅
- Results display ✅
- UI stays stable ✅
- No disconnect ✅
- Can navigate to Step 5 ✅

## 📞 Need Help?

Check these files:
1. `STEP4_FIX_SUMMARY.md` - Chi tiết fixes
2. `STEP4_DEBUGGING_GUIDE.md` - Debug instructions
3. `STEP4_QUICK_FIX_REFERENCE.md` - This file
