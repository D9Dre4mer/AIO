"""
Module phân loại sử dụng K-Nearest Neighbors với FAISS.
"""
import faiss
import numpy as np
import os
import pickle
from typing import List, Dict, Tuple, Any
from collections import Counter


class KNNClassifier:
    """Class phân loại KNN sử dụng FAISS."""
    
    def __init__(self, embedding_dim: int):
        """
        Khởi tạo KNN Classifier.
        
        Args:
            embedding_dim: Số chiều của embeddings
        """
        self.embedding_dim = embedding_dim
        self.index = faiss.IndexFlatIP(embedding_dim)  # Inner product
        self.train_metadata = None
    
    def fit(self, 
            train_embeddings: np.ndarray, 
            train_metadata: List[Dict[str, Any]]) -> None:
        """
        Huấn luyện classifier với dữ liệu train.
        
        Args:
            train_embeddings: Embeddings của dữ liệu train
            train_metadata: Metadata của dữ liệu train
        """
        self.train_metadata = train_metadata
        self.index.add(train_embeddings.astype('float32'))
        print(f"FAISS index đã tạo với {self.index.ntotal} vectors")
    
    def save_index(self, cache_suffix: str = "") -> str:
        """
        🆕 Lưu FAISS index và metadata vào cache.
        
        Args:
            cache_suffix: Suffix cho tên file cache
            
        Returns:
            Đường dẫn file cache đã lưu
        """
        # Tạo thư mục cache nếu chưa có
        cache_dir = os.path.join('cache', 'faiss_index')
        os.makedirs(cache_dir, exist_ok=True)
        
        # Tên file cache
        index_file = os.path.join(cache_dir, f"faiss_index{cache_suffix}.faiss")
        metadata_file = os.path.join(cache_dir, f"faiss_metadata{cache_suffix}.pkl")
        
        print(f"FAISS SAVE: {cache_suffix} index with {self.index.ntotal} vectors")
        
        # Lưu FAISS index
        faiss.write_index(self.index, index_file)
        
        # Lưu metadata
        with open(metadata_file, 'wb') as f:
            pickle.dump(self.train_metadata, f)
        
        return index_file
    
    def load_index(self, cache_suffix: str = "") -> bool:
        """
        🆕 Load FAISS index và metadata từ cache.
        
        Args:
            cache_suffix: Suffix cho tên file cache
            
        Returns:
            True nếu load thành công, False nếu không
        """
        # Tên file cache
        cache_dir = os.path.join('cache', 'faiss_index')
        index_file = os.path.join(cache_dir, f"faiss_index{cache_suffix}.faiss")
        metadata_file = os.path.join(cache_dir, f"faiss_metadata{cache_suffix}.pkl")
        
        # Kiểm tra file tồn tại
        if not os.path.exists(index_file) or not os.path.exists(metadata_file):
            print(f"FAISS LOAD: {cache_suffix} cache not found")
            return False
        
        try:
            # Load FAISS index
            self.index = faiss.read_index(index_file)
            
            # Load metadata
            with open(metadata_file, 'rb') as f:
                self.train_metadata = pickle.load(f)
            
            print(f"FAISS LOAD: {cache_suffix} index with {self.index.ntotal} vectors")
            return True
            
        except Exception as e:
            print(f"FAISS LOAD: Error loading {cache_suffix} index - {e}")
            return False
    
    def predict(self, 
                query_embedding: np.ndarray, 
                k: int = 1) -> Tuple[str, List[Dict[str, Any]]]:
        """
        Phân loại một query sử dụng k-nearest neighbors.
        
        Args:
            query_embedding: Embedding của query
            k: Số lượng neighbors
            
        Returns:
            Tuple chứa prediction và thông tin neighbors
        """
        # Debug: Kiểm tra kích thước embedding
        print(f"Debug: Query embedding shape: {query_embedding.shape}")
        print(f"Debug: FAISS index dimension: {self.index.d}")
        print(f"Debug: Query embedding dtype: {query_embedding.dtype}")
        
        # Đảm bảo kích thước đúng
        if query_embedding.shape[1] != self.index.d:
            raise ValueError(
                f"Embedding dimension mismatch! "
                f"Query: {query_embedding.shape[1]}, "
                f"Index: {self.index.d}"
            )
        
        # Tìm kiếm trong FAISS index
        scores, indices = self.index.search(query_embedding, k)
        
        # Lấy predictions từ top-k neighbors
        predictions = []
        neighbor_info = []
        
        for i in range(k):
            neighbor_idx = indices[0][i]
            neighbor_score = scores[0][i]
            neighbor_data = self.train_metadata[neighbor_idx]
            
            predictions.append(neighbor_data['label'])
            neighbor_info.append({
                'score': float(neighbor_score),
                'label': neighbor_data['label'],
                'message': self._truncate_message(neighbor_data['message'])
            })
        
        # Majority vote cho prediction cuối cùng
        final_prediction = Counter(predictions).most_common(1)[0][0]
        
        return final_prediction, neighbor_info
    
    def _truncate_message(self, message: str, max_length: int = 100) -> str:
        """
        Cắt ngắn message để hiển thị.
        
        Args:
            message: Tin nhắn gốc
            max_length: Độ dài tối đa
            
        Returns:
            Tin nhắn đã cắt ngắn
        """
        if len(message) > max_length:
            return message[:max_length] + "..."
        return message
