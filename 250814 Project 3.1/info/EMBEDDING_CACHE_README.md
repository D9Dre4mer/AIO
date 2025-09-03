# Embedding Cache System

## Tổng quan

Hệ thống cache embeddings cho phép lưu trữ embeddings đã tạo vào disk để tái sử dụng trong các lần training tiếp theo, giúp tiết kiệm thời gian đáng kể (từ 30+ phút xuống vài giây).

## Cách hoạt động

### 1. **Cache Key Generation**
- Tự động tạo cache key dựa trên:
  - Số lượng samples (train/val/test)
  - Loại embeddings được chọn
  - Cấu hình sampling
  - Tên cột text/label
  - Categories được chọn

### 2. **Cache Storage**
- Embeddings được lưu trong: `./cache/embeddings/`
- Format: `embeddings_[hash].pkl`
- Tự động tạo thư mục nếu chưa tồn tại

### 3. **Cache Loading**
- Tự động kiểm tra cache khi bắt đầu training
- Load từ disk nếu tìm thấy cache phù hợp
- Fallback tạo mới nếu không có cache

## Sử dụng

### Tự động (Mặc định)
```python
# Chương trình sẽ tự động:
# 1. Kiểm tra cache khi bắt đầu
# 2. Load cache nếu có
# 3. Tạo mới và cache nếu không có
# 4. Hiển thị thông báo cache status
```

### Quản lý Cache thủ công
```bash
# Xem trạng thái cache
python manage_embedding_cache.py

# Chọn option 1 để xem cache status
# Chọn option 2 để xóa tất cả cache
# Chọn option 3 để xóa cache cụ thể
```

## Thông báo Cache

### Lần đầu tạo embeddings:
```
🔤 Creating new embeddings (will be cached for future use)...
💾 Embeddings cached to: ./cache/embeddings/embeddings_abc123.pkl
```

### Lần sau sử dụng cache:
```
✅ Loaded embeddings from persistent cache!
📂 Loaded embeddings from cache: ./cache/embeddings/embeddings_abc123.pkl
```

### Sử dụng cache trong memory:
```
🔄 Reusing embeddings from memory...
```

## Cache Key Examples

```
embeddings_7b8f2a39c860  # 1000 samples, Word Embeddings, 3 categories
embeddings_a1b2c3d4e5f6  # 5000 samples, BoW+TF-IDF, 5 categories
embeddings_f6e5d4c3b2a1  # 10000 samples, All embeddings, 2 categories
```

## Lợi ích

### ⏱️ **Tiết kiệm thời gian**
- Lần đầu: 30+ phút tạo embeddings
- Lần sau: Vài giây load từ cache

### 💾 **Tiết kiệm tài nguyên**
- Không cần tải lại model sentence-transformers
- Không cần xử lý text lại
- Giảm CPU/GPU usage

### 🔄 **Tái sử dụng linh hoạt**
- Cache theo cấu hình cụ thể
- Tự động detect cache phù hợp
- Dễ dàng quản lý và xóa cache

## Quản lý Cache

### Xem cache status:
```python
evaluator = ComprehensiveEvaluator()
evaluator.show_embedding_cache_status()
```

### Xóa cache:
```python
# Xóa tất cả cache
evaluator.clear_embedding_cache()

# Xóa cache cụ thể
evaluator.clear_embedding_cache("embeddings_abc123")
```

### Cache directory:
```
./cache/embeddings/
├── embeddings_7b8f2a39c860.pkl  # 2.3 GB
├── embeddings_a1b2c3d4e5f6.pkl  # 1.8 GB
└── embeddings_f6e5d4c3b2a1.pkl  # 3.1 GB
```

## Lưu ý

1. **Cache size**: Mỗi file cache có thể từ 1-5 GB tùy số samples
2. **Cache validity**: Cache chỉ hợp lệ với cùng cấu hình data
3. **Manual cleanup**: Có thể xóa cache cũ để tiết kiệm disk space
4. **Cross-session**: Cache được lưu persistent, có thể dùng qua các session

## Troubleshooting

### Cache không được tạo:
- Kiểm tra quyền ghi vào thư mục `./cache/`
- Kiểm tra disk space còn đủ

### Cache không được load:
- Kiểm tra cấu hình data có thay đổi không
- Xóa cache cũ và tạo mới

### Cache bị corrupt:
- Xóa file cache bị lỗi
- Chạy lại để tạo cache mới
