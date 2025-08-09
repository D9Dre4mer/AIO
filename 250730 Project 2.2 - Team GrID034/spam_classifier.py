"""
Pipeline chính cho spam classification.
"""
import numpy as np
import os
import logging
import json
import pickle
from typing import Dict, Any, List, Optional, Tuple
from config import SpamClassifierConfig
from data_loader import DataLoader
from embedding_generator import EmbeddingGenerator
from knn_classifier import KNNClassifier
from tfidf_classifier import TFIDFClassifier

logger = logging.getLogger(__name__)


class SpamClassifierPipeline:
    """Pipeline hoàn chỉnh cho spam classification."""
    
    def __init__(self,
                 config: SpamClassifierConfig = None,
                 classifier_type: str = 'knn'):
        """
        Khởi tạo SpamClassifierPipeline.
        
        Args:
            config: Cấu hình hệ thống
        """
        self.config = config or SpamClassifierConfig()
        self.data_loader = DataLoader(self.config)
        self.embedding_generator = (
            EmbeddingGenerator(self.config)
            if classifier_type == 'knn' else None
        )
        self.classifier = None
        self.classifier_type = classifier_type
        
    def load_corrections(self) -> Dict[str, Any]:
        """Load correction data từ file JSON"""
        correction_file = "./cache/corrections.json"
        if os.path.exists(correction_file):
            try:
                with open(correction_file, 'r', encoding='utf-8') as f:
                    return json.load(f)
            except Exception as e:
                logger.warning(f"Lỗi load corrections: {e}")
                return {}
        return {}

    # Corrections dataset cache helpers
    def _get_corrections_dataset_path(self) -> str:
        """Đường dẫn file cache cho merged corrections dataset."""
        cache_dir = os.path.join('cache', 'datasets')
        os.makedirs(cache_dir, exist_ok=True)
        # Lưu theo model để tránh xung đột giữa các model khác nhau
        model_name_safe = self.config.model_name.replace('/', '_')
        return os.path.join(
            cache_dir, f"with_corrections_dataset_{model_name_safe}.pkl"
        )

    def _save_corrections_dataset(
        self, messages: List[str], labels: List[str]
    ) -> None:
        """Lưu merged corrections dataset (đã preprocess) vào cache."""
        dataset_path = self._get_corrections_dataset_path()
        try:
            with open(dataset_path, 'wb') as f:
                pickle.dump({'messages': messages, 'labels': labels}, f)
            logger.info(
                f"Đã lưu merged corrections dataset: {dataset_path} | "
                f"samples: {len(messages)}"
            )
        except Exception as e:
            logger.warning(f"Không thể lưu corrections dataset: {e}")

    def _load_corrections_dataset(
        self,
    ) -> Optional[Tuple[List[str], List[str]]]:
        """Thử load merged corrections dataset từ cache, nếu có."""
        dataset_path = self._get_corrections_dataset_path()
        if not os.path.exists(dataset_path):
            return None
        try:
            with open(dataset_path, 'rb') as f:
                data = pickle.load(f)
            messages = data.get('messages', [])
            labels = data.get('labels', [])
            if len(messages) != len(labels):
                logger.warning(
                    "Corrections dataset không đồng nhất (messages != labels). Bỏ qua cache này."
                )
                return None
            logger.info(
                "Đã load merged corrections dataset từ cache: %s | samples: %d",
                dataset_path,
                len(messages),
            )
            return messages, labels
        except Exception as e:
            logger.warning(f"Không thể load corrections dataset: {e}")
            return None
        
    def train_with_corrections(self) -> Dict[str, Any]:
        """
        🆕 Huấn luyện mô hình với dữ liệu gốc + corrections.
        
        Returns:
            Dict chứa thông tin training với corrections
        """
        logger.info("Đang tải dữ liệu gốc...")
        messages, labels = self.data_loader.load_data()
        
        corrections = self.load_corrections()
        logger.info(f"Tìm thấy {len(corrections)} corrections")
        
        if corrections:
            logger.info("Đang merge corrections vào dataset...")
            
            extended_messages = messages.copy()
            extended_labels = labels.copy()
            
            for email_id, correction in corrections.items():
                subject = correction.get('subject', '')
                sender = correction.get('sender', '')
                snippet = correction.get('snippet', '')
                
                correction_text = (
                    f"Subject: {subject}\nFrom: {sender}\n{snippet}"
                )
                
                corrected_label = correction.get('corrected_label', 'ham')
                # Bảo đảm preprocess cho corrections để thống nhất với CSV
                preprocessed_text = self.data_loader.preprocess_text(correction_text)
                extended_messages.append(preprocessed_text)
                extended_labels.append(corrected_label)
                
                logger.info(f"Thêm correction: {email_id} -> {corrected_label}")
            
            messages = extended_messages
            labels = extended_labels
            
            logger.info(f"Dataset sau merge: {len(messages)} samples")

            # Lưu merged corrections dataset để dùng ổn định giữa các lần train
            self._save_corrections_dataset(messages, labels)
        
        return self._train_with_data(messages, labels, use_corrections=True)
        
    def train(self) -> None:
        """Huấn luyện mô hình với dữ liệu gốc."""
        logger.info("Đang tải dữ liệu...")
        messages, labels = self.data_loader.load_data()
        
        self._train_with_data(messages, labels, use_corrections=False)
        
    def _train_with_data(self, messages: list, labels: list,
                         use_corrections: bool = False) -> Dict[str, Any]:
        """
        🆕 Internal method để train với data được cung cấp.
        
        Args:
            messages: List messages
            labels: List labels
            use_corrections: Có sử dụng corrections không
            
        Returns:
            Dict chứa thông tin training
        """
        # Nếu train với corrections và có dữ liệu được truyền vào từ bên ngoài,
        # thì sử dụng dữ liệu đó thay vì load từ cache
        original_messages_count = None
        if use_corrections:
            # Chỉ load cache nếu messages rỗng (tức là được gọi trực tiếp)
            if len(messages) == 0:
                cached = self._load_corrections_dataset()
                if cached is not None:
                    messages, labels = cached
                    logger.info("Đã load merged corrections dataset từ cache")
            
            # Ghi nhận số mẫu gốc để thống kê
            try:
                orig_msgs, _ = self.data_loader.load_data()
                original_messages_count = len(orig_msgs)
            except Exception:
                original_messages_count = None

        total_samples = len(messages)
        training_info = {
            'total_samples': total_samples,
            'original_samples': (
                original_messages_count if (use_corrections and original_messages_count is not None)
                else (len(messages) if not use_corrections else None)
            ),
            'correction_samples': (
                (total_samples - original_messages_count)
                if (use_corrections and original_messages_count is not None)
                else (0 if not use_corrections else None)
            ),
            'use_corrections': use_corrections
        }
        
        if self.classifier_type == 'knn':
            # 🆕 Tạo tên file cache dựa trên loại training
            cache_suffix = "_with_corrections" if use_corrections else "_original"
            emb_file = os.path.join(
                'cache', 'embeddings',
                f"embeddings_{self.config.model_name.replace('/', '_')}{cache_suffix}.npy"
            )
            dataset_count = len(messages)
            
            logger.info(f"Training type: {'with corrections' if use_corrections else 'original'}")
            logger.info(f"Dataset count: {dataset_count}")
            logger.info(f"Cache file: {emb_file}")

            if os.path.exists(emb_file):
                try:
                    cache = np.load(emb_file)
                    cache_count = cache.shape[0]
                    logger.info(f"Cache count: {cache_count}")
                    
                    if (cache_count != dataset_count
                            and not self.config.regenerate_embeddings):
                        msg = (
                            f"Số dòng trong dataset ({dataset_count}) "
                            f"không khớp với embeddings cache ({cache_count}). "
                            f"Training type: {'with corrections' if use_corrections else 'original'}. "
                            "Chạy lại với --regenerate để cập nhật."
                        )
                        raise ValueError(msg)
                    if (cache_count != dataset_count
                            and self.config.regenerate_embeddings):
                        logger.info(
                            f"Xóa cache cũ: {emb_file} "
                            f"(flag regenerate_embeddings=True, "
                            f"training type: {'with corrections' if use_corrections else 'original'})"
                        )
                        os.remove(emb_file)
                except Exception as e:
                    logger.error(f"Lỗi khi kiểm tra cache: {e}")
                    raise

        logger.info(f"Các lớp: {self.data_loader.get_class_names()}")

        # Chia dữ liệu
        train_idx, _, y_train, _ = (
            self.data_loader.split_data(messages, labels)
        )
        train_msgs = [messages[i] for i in train_idx]
        train_lbls = [labels[i] for i in train_idx]
        
        if self.classifier_type == 'knn':
            # Tạo embeddings
            logger.info(f"Tạo embeddings cho {len(messages)} tin nhắn...")
            cache_suffix = "_with_corrections" if use_corrections else "_original"
            embeddings = self.embedding_generator.generate_embeddings(
                messages, cache_suffix=cache_suffix
            )
            logger.info(f"Kích thước embeddings: {embeddings.shape}")

            # Tạo metadata
            encoded = self.data_loader.label_encoder.transform(labels)
            metadata = self.data_loader.create_metadata(
                messages, labels, encoded
            )

            # Chia embeddings và metadata
            train_emb = embeddings[train_idx]
            train_meta = [metadata[i] for i in train_idx]

            logger.info(f"Kích thước tập train: {len(train_emb)}")
            logger.info(f"Phân bố nhãn train: {np.bincount(y_train)}")

            # 🆕 Tạo và huấn luyện classifier với cache FAISS index
            self.classifier = KNNClassifier(train_emb.shape[1])
            
            # Thử load FAISS index từ cache trước
            cache_suffix = "_with_corrections" if use_corrections else "_original"
            if not self.classifier.load_index(cache_suffix):
                # Nếu không có cache, train mới và lưu cache
                self.classifier.fit(train_emb, train_meta)
                self.classifier.save_index(cache_suffix)

        elif self.classifier_type == 'tfidf':
            self.classifier = TFIDFClassifier()
            self.classifier.fit(train_msgs, train_lbls)

        else:
            raise ValueError(
                f"Loại classifier không hợp lệ: {self.classifier_type}"
            )
            
        training_info['train_samples'] = len(train_msgs)
        training_info['label_distribution'] = {
            label: train_lbls.count(label) for label in set(train_lbls)
        }
        
        logger.info(f"Training hoàn tất: {training_info}")
        return training_info
            
    def predict(self, text: str, k: int = None) -> Dict[str, Any]:
        """
        Phân loại một văn bản.
        
        Args:
            text: Văn bản cần phân loại
            k: Số lượng neighbors (mặc định từ config)
            
        Returns:
            Dict chứa kết quả phân loại
        """
        if self.classifier is None:
            raise ValueError("Mô hình chưa được huấn luyện. "
                           "Hãy gọi train() trước.")

        pre_text = self.data_loader.preprocess_text(text)

        if self.classifier_type == 'knn':
            k = k or self.config.default_k
            
            logger.info(f"\n***Đang phân loại: '{text}'")
            logger.info(f"\n***Sử dụng top-{k} nearest neighbors")
            
            # Tạo query embedding
            query_embedding = self.embedding_generator.generate_query_embedding(
                text
            )
            
            # Phân loại
            prediction, neighbors = self.classifier.predict(
                query_embedding, k=k
            )
            
            # Hiển thị kết quả
            logger.info(f"\n***Dự đoán: {prediction.upper()}")
            logger.info("\n***Top neighbors:")
            for i, neighbor in enumerate(neighbors, 1):
                logger.info(f"{i}. Nhãn: {neighbor['label']} | "
                            f"Điểm: {neighbor['score']:.4f}")
                logger.info(f"Tin nhắn: {neighbor['message']}")
            
            # Đếm phân bố nhãn
            labels = [n['label'] for n in neighbors]
            label_counts = {
                label: labels.count(label) for label in set(labels)
            }
            
            
            return {
                'prediction': prediction,
                'neighbors': neighbors,
                'label_distribution': label_counts
            }
            
        if self.classifier_type == 'tfidf':
            return self.classifier.predict(pre_text, return_proba=True)

        raise ValueError(
            f"Loại classifier không hợp lệ: {self.classifier_type}"
        )