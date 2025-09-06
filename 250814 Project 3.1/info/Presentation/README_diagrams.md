# 📊 AIO Classifier - Presentation Diagrams

Thư mục này chứa các file LaTeX để tạo sơ đồ cho bài thuyết trình AIO Classifier.

## 📁 Các file có sẵn:

### 1. **system_architecture.tex**
- Sơ đồ kiến trúc hệ thống cơ bản
- Hiển thị 6 components chính và luồng dữ liệu
- Phù hợp cho slide tổng quan

### 2. **detailed_architecture.tex**
- Sơ đồ kiến trúc chi tiết với core components
- Bao gồm BaseModel, Ensemble, Caching
- Phù hợp cho slide kỹ thuật

### 3. **performance_comparison.tex**
- So sánh hiệu suất các models
- Top 3 models với accuracy và training time
- Phù hợp cho slide kết quả

### 4. **presentation_diagrams.tex**
- File tổng hợp tất cả sơ đồ
- Bao gồm workflow 5 bước
- Phù hợp để compile thành PDF hoàn chỉnh

## 🚀 Cách sử dụng:

### Compile từng sơ đồ riêng lẻ:
```bash
# Sơ đồ kiến trúc cơ bản
pdflatex system_architecture.tex

# Sơ đồ kiến trúc chi tiết
pdflatex detailed_architecture.tex

# So sánh hiệu suất
pdflatex performance_comparison.tex
```

### Compile tất cả sơ đồ:
```bash
pdflatex presentation_diagrams.tex
```

## 📋 Yêu cầu:

- **LaTeX Distribution**: TeXLive, MiKTeX, hoặc MacTeX
- **Packages cần thiết**:
  - tikz
  - babel (vietnamese)
  - geometry
  - inputenc

## 🎨 Tùy chỉnh:

### Thay đổi màu sắc:
- Sửa `draw=blue!60` thành màu khác
- Sửa `fill=blue!10` cho màu nền

### Thay đổi kích thước:
- Sửa `text width=3cm` cho độ rộng
- Sửa `minimum height=2cm` cho chiều cao

### Thêm components:
- Copy style từ component có sẵn
- Đặt vị trí với `at (x,y)`
- Thêm arrows với `\draw[arrow]`

## 📊 Sơ đồ có sẵn:

1. **Kiến trúc Modular** - 6 components chính
2. **Kiến trúc Chi tiết** - Core components + features
3. **So sánh Hiệu suất** - Top 3 models + metrics
4. **Workflow 5 Bước** - Quy trình sử dụng

## 🔧 Troubleshooting:

### Lỗi compile:
- Kiểm tra packages đã cài đặt
- Sử dụng `pdflatex -interaction=nonstopmode`
- Kiểm tra encoding UTF-8

### Sơ đồ không hiển thị đúng:
- Kiểm tra kích thước page
- Điều chỉnh `geometry` settings
- Sử dụng `standalone` class cho từng sơ đồ

## 📝 Ghi chú:

- Tất cả sơ đồ được thiết kế cho slide 16:9
- Màu sắc phù hợp với theme xanh-dương
- Font size tối ưu cho presentation
- Có thể export sang PNG/PDF cho PowerPoint

---

**Tác giả**: Project 3.1 - AIO Classifier  
**Ngày tạo**: $(date)  
**Phiên bản**: 1.0
