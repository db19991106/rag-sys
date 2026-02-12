from typing import List, Optional, Dict, Tuple, Any
import numpy as np
import time
from pathlib import Path
import json
from models import VectorDBType, VectorDBConfig, VectorStatus
from utils.logger import logger
from config import settings


class VectorDatabase:
    """向量数据库基类"""

    def __init__(self, dimension: int):
        self.dimension = dimension
        self.total_vectors = 0

    def add_vectors(self, vectors: np.ndarray, metadata: List[Dict]):
        """添加向量"""
        raise NotImplementedError

    def search(
        self, query_vector: np.ndarray, top_k: int = 5
    ) -> Tuple[np.ndarray, List[Dict]]:
        """搜索向量"""
        raise NotImplementedError

    def delete_vectors(self, ids: List[str]):
        """删除向量"""
        raise NotImplementedError

    def get_status(self) -> VectorStatus:
        """获取状态"""
        raise NotImplementedError

    def save(self):
        """保存数据库"""
        raise NotImplementedError

    def load(self):
        """加载数据库"""
        raise NotImplementedError


class FAISSDatabase(VectorDatabase):
    """FAISS 向量数据库"""

    def __init__(
        self, dimension: int, index_type: str = "HNSW", index_path: str = None
    ):
        super().__init__(dimension)
        self.index_type = index_type
        self.index = None
        self.metadata: Dict[str, Dict] = {}
        # 使用传入的路径或默认路径
        if index_path:
            db_dir = Path(index_path)
        else:
            db_dir = Path(settings.vector_db_dir)
        self.db_path = db_dir / "faiss_index"
        self.metadata_path = db_dir / "faiss_metadata.json"
        self._init_index()

    def _init_index(self):
        """初始化 FAISS 索引"""
        import faiss

        if self.index_type == "HNSW":
            # HNSW 索引 - 高性能
            # 优化参数: M=16 (图的连接数), efConstruction=200 (构建时的搜索宽度)
            self.index = faiss.IndexHNSWFlat(self.dimension, 16)
            # 设置构建参数
            self.index.hnsw.efConstruction = 200
            # 设置搜索参数
            self.index.hnsw.efSearch = 128
        elif self.index_type == "IVF":
            # IVF 索引 - 倒排文件
            # 优化参数: nlist=400 (聚类中心数)
            quantizer = faiss.IndexFlatL2(self.dimension)
            nlist = min(400, self.dimension * 2)  # 根据维度动态调整
            self.index = faiss.IndexIVFFlat(quantizer, self.dimension, nlist)
        elif self.index_type == "PQ":
            # PQ 索引 - 乘积量化
            # 优化参数: M=8 (子向量数量), nbits=8 (每个子向量的位数)
            M = min(8, self.dimension // 32)  # 根据维度动态调整
            self.index = faiss.IndexPQ(self.dimension, M, 8)
        else:
            # 默认使用 Flat 索引
            self.index = faiss.IndexFlatL2(self.dimension)

        logger.info(f"初始化 FAISS 索引: {self.index_type} (维度: {self.dimension}) ")

    def add_vectors(self, vectors: np.ndarray, metadata: List[Dict]):
        """添加向量"""
        import faiss
        import time

        if not isinstance(vectors, np.ndarray):
            vectors = np.array(vectors, dtype=np.float32)

        # 确保 vectors 是二维数组
        if vectors.ndim == 1:
            vectors = vectors.reshape(1, -1)

        # 训练索引 (如果需要)
        if hasattr(self.index, "is_trained") and not self.index.is_trained:
            logger.info(f"训练 IVF 索引，样本数: {len(vectors)}")
            start_train = time.time()
            self.index.train(vectors)
            train_time = time.time() - start_train
            logger.info(f"✓ FAISS 索引训练完成，耗时: {train_time:.2f}s")

            # 训练后需要重建索引以启用搜索
            logger.info("重建索引以启用搜索功能...")
            self.index.reset()
            self.index.train(vectors)
            logger.info("✓ 索引重建完成")

        # 批量添加向量 (针对大型数据集)
        batch_size = 1000
        total_added = 0
        start_id = self.total_vectors

        for i in range(0, len(vectors), batch_size):
            end = min(i + batch_size, len(vectors))
            batch_vectors = vectors[i:end]
            batch_metadata = metadata[i:end]

            # 添加向量
            self.index.add(batch_vectors)

            # 保存元数据
            for j, meta in enumerate(batch_metadata):
                self.metadata[str(start_id + i + j)] = meta

            total_added += len(batch_vectors)
            logger.info(
                f"添加批次 {i // batch_size + 1}/{(len(vectors) + batch_size - 1) // batch_size}，数量: {len(batch_vectors)}"
            )

        self.total_vectors += len(vectors)
        logger.info(f"添加 {len(vectors)} 个向量到 FAISS (总数: {self.total_vectors}) ")

        # 定期保存 (每1000个向量或最后一批)
        if len(vectors) >= 1000 or (
            len(vectors) > 0 and i + batch_size >= len(vectors)
        ):
            self.save()

    def search(
        self, query_vector: np.ndarray, top_k: int = 5
    ) -> Tuple[np.ndarray, List[List[Dict]]]:
        """搜索向量"""
        import faiss
        import time

        if not isinstance(query_vector, np.ndarray):
            query_vector = np.array(query_vector, dtype=np.float32)

        # 确保 query_vector 是二维数组
        if query_vector.ndim == 1:
            query_vector = query_vector.reshape(1, -1)

        # 优化搜索参数
        if hasattr(self.index, "hnsw"):
            # HNSW 索引优化
            original_efSearch = self.index.hnsw.efSearch
            self.index.hnsw.efSearch = min(128, top_k * 4)  # 根据 top_k 动态调整
        elif hasattr(self.index, "nprobe"):
            # IVF 索引优化
            original_nprobe = self.index.nprobe
            self.index.nprobe = min(64, top_k * 2)  # 根据 top_k 动态调整

        # 搜索
        start_search = time.time()
        distances, indices = self.index.search(query_vector, top_k)
        search_time = time.time() - start_search
        logger.debug(
            f"FAISS 搜索完成，耗时: {search_time:.4f}s, 返回: {len(indices[0])} 个结果"
        )

        # 恢复原始参数
        if hasattr(self.index, "hnsw"):
            self.index.hnsw.efSearch = original_efSearch
        elif hasattr(self.index, "nprobe"):
            self.index.nprobe = original_nprobe

        # 获取元数据 - 返回嵌套列表结构（每行一个查询的元数据）
        results_metadata = []
        for i in range(len(indices)):
            row_metadata = []
            for idx in indices[i]:
                if idx >= 0:
                    meta = self.metadata.get(str(idx), {})
                    # 确保 meta 是字典类型
                    if isinstance(meta, str):
                        # 如果 meta 是字符串，尝试解析为 JSON
                        try:
                            import json

                            meta = json.loads(meta)
                        except:
                            meta = {}
                    elif not isinstance(meta, dict):
                        # 如果 meta 不是字典，转为空字典
                        meta = {}
                    row_metadata.append(meta)
                else:
                    row_metadata.append({})
            results_metadata.append(row_metadata)

        return distances, results_metadata

    def delete_vectors(self, ids: List[str]):
        """删除向量 (FAISS 不支持直接删除，需要重建索引)"""
        logger.warning("FAISS 不支持直接删除向量，建议使用重建索引方式")
        # TODO: 实现索引重建功能

    def get_status(self) -> VectorStatus:
        """获取状态"""
        return VectorStatus(
            db_type="faiss",
            total_vectors=self.total_vectors,
            dimension=self.dimension,
            status="ready",
        )

    def save(self):
        """保存数据库"""
        import faiss

        self.db_path.parent.mkdir(parents=True, exist_ok=True)

        # 保存索引
        faiss.write_index(self.index, str(self.db_path))

        # 保存元数据
        with open(self.metadata_path, "w", encoding="utf-8") as f:
            json.dump(self.metadata, f, ensure_ascii=False, indent=2)

        logger.info(f"FAISS 索引已保存: {self.db_path}")

    def load(self):
        """加载数据库"""
        import faiss

        if not self.db_path.exists():
            logger.info("FAISS 索引文件不存在，使用空索引")
            self._init_index()
            self.metadata = {}
            self.total_vectors = 0
            return

        try:
            # 加载索引
            self.index = faiss.read_index(str(self.db_path))
            self.total_vectors = self.index.ntotal
            logger.info(f"FAISS 索引文件已加载: {self.db_path}")

            # 检查维度是否匹配
            if hasattr(self.index, "d") and self.index.d != self.dimension:
                logger.error(
                    f"FAISS 索引维度不匹配: 索引维度={self.index.d}, 配置维度={self.dimension}"
                )
                logger.info("重新初始化 FAISS 索引以匹配正确维度")
                self._init_index()
                self.metadata = {}
                self.total_vectors = 0
                return

            # 加载元数据
            if self.metadata_path.exists():
                with open(self.metadata_path, "r", encoding="utf-8") as f:
                    self.metadata = json.load(f)
                logger.info(f"已加载 {len(self.metadata)} 条元数据")
            else:
                logger.warning(f"元数据文件不存在: {self.metadata_path}")
                self.metadata = {}

            # 验证向量数量
            logger.info(
                f"FAISS 索引已加载: {self.total_vectors} 个向量，{len(self.metadata)} 条元数据"
            )

            # 如果索引为0，重新初始化
            if self.total_vectors == 0:
                logger.warning("加载的索引为空，重新初始化")
                self._init_index()

        except Exception as e:
            logger.error(f"加载 FAISS 索引失败: {str(e)}")
            logger.info("重新初始化 FAISS 索引")
            self._init_index()
            self.metadata = {}
            self.total_vectors = 0


class MilvusDatabase(VectorDatabase):
    """Milvus 向量数据库"""

    def __init__(
        self,
        dimension: int,
        host: str = "localhost",
        port: int = 19530,
        collection_name: str = "rag_vectors",
    ):
        super().__init__(dimension)
        self.host = host
        self.port = port
        self.collection_name = collection_name
        self.client = None
        self.collection = None
        self._connect()

    def _connect(self):
        """连接 Milvus"""
        try:
            from pymilvus import MilvusClient

            self.client = MilvusClient(uri=f"http://{self.host}:{self.port}")
            logger.info(f"连接 Milvus: {self.host}:{self.port}")
        except Exception as e:
            logger.error(f"连接 Milvus 失败: {str(e)}")
            raise

    def _ensure_collection(self):
        """确保集合存在"""
        if not self.client.has_collection(self.collection_name):
            self.client.create_collection(
                collection_name=self.collection_name,
                dimension=self.dimension,
                metric_type="COSINE",
            )
            logger.info(f"创建 Milvus 集合: {self.collection_name}")
        self.collection = self.client.get_collection(self.collection_name)

    def add_vectors(self, vectors: np.ndarray, metadata: List[Dict]):
        """添加向量"""
        self._ensure_collection()

        if not isinstance(vectors, np.ndarray):
            vectors = np.array(vectors, dtype=np.float32)

        # 确保 vectors 是二维数组
        if vectors.ndim == 1:
            vectors = vectors.reshape(1, -1)

        # 准备数据
        ids = [
            str(i) for i in range(self.total_vectors, self.total_vectors + len(vectors))
        ]

        # 插入向量
        data = [ids, vectors.tolist(), metadata]
        self.client.insert(self.collection_name, data=data)

        # 刷新
        self.client.flush(self.collection_name)

        self.total_vectors += len(vectors)
        logger.info(f"添加 {len(vectors)} 个向量到 Milvus (总数: {self.total_vectors})")

    def search(
        self, query_vector: np.ndarray, top_k: int = 5
    ) -> Tuple[np.ndarray, List[Dict]]:
        """搜索向量"""
        if not isinstance(query_vector, np.ndarray):
            query_vector = np.array(query_vector, dtype=np.float32)

        # 确保 query_vector 是二维数组
        if query_vector.ndim == 1:
            query_vector = query_vector.reshape(1, -1)

        # 搜索
        results = self.client.search(
            collection_name=self.collection_name,
            data=query_vector.tolist(),
            limit=top_k,
            output_fields=["*"],
        )

        # 提取距离和元数据
        distances = []
        metadata_list = []

        for result in results:
            distances.append([item["distance"] for item in result])
            metadata_list.append([item["entity"] for item in result])

        return np.array(distances), metadata_list

    def delete_vectors(self, ids: List[str]):
        """删除向量"""
        self.client.delete(self.collection_name, ids)
        logger.info(f"从 Milvus 删除 {len(ids)} 个向量")

    def get_status(self) -> VectorStatus:
        """获取状态"""
        try:
            self._ensure_collection()
            num_entities = self.client.get_collection_stats(self.collection_name)[
                "row_count"
            ]
        except:
            num_entities = 0

        return VectorStatus(
            db_type="milvus",
            total_vectors=num_entities,
            dimension=self.dimension,
            status="ready",
        )

    def save(self):
        """保存数据库 (Milvus 自动持久化)"""
        logger.info("Milvus 自动持久化数据")

    def load(self):
        """加载数据库 (Milvus 自动加载)"""
        logger.info("Milvus 自动加载数据")


class VectorDatabaseManager:
    """向量数据库管理器"""

    def __init__(self):
        self.db: Optional[VectorDatabase] = None
        self.config: Optional[VectorDBConfig] = None
        self.secondary_indices: Dict[str, VectorDatabase] = {}  # 多级索引
        self.last_update_time = 0  # 最后更新时间

    def initialize(self, config: VectorDBConfig):
        """初始化向量数据库"""
        self.config = config

        try:
            if config.db_type == VectorDBType.FAISS:
                self.db = FAISSDatabase(
                    config.dimension, config.index_type, config.index_path
                )
            elif config.db_type == VectorDBType.MILVUS:
                self.db = MilvusDatabase(
                    config.dimension,
                    config.host or settings.milvus_host,
                    config.port or settings.milvus_port,
                    config.collection_name or settings.milvus_collection_name,
                )
            else:
                raise ValueError(f"不支持的向量数据库类型: {config.db_type}")

            # 尝试加载数据
            self.db.load()

            # 初始化多级索引
            self._initialize_secondary_indices()

            logger.info(f"向量数据库初始化成功: {config.db_type}")
            return True

        except Exception as e:
            logger.error(f"向量数据库初始化失败: {str(e)}")
            return False

    def _initialize_secondary_indices(self):
        """初始化多级索引"""
        # 示例：创建基于文档类型的二级索引
        if self.config and self.config.db_type == VectorDBType.FAISS:
            # 可以根据需要创建不同类型的二级索引
            # 例如：按文档类型、按时间、按主题等
            pass

    def add_vectors(self, vectors: np.ndarray, metadata: List[Dict]):
        """添加向量"""
        if self.db is None:
            raise ValueError("向量数据库未初始化")

        try:
            # 验证输入
            if vectors is None or len(vectors) == 0:
                logger.warning("尝试添加空向量，跳过操作")
                return

            if metadata is None or len(metadata) != len(vectors):
                logger.warning("向量和元数据长度不匹配，跳过操作")
                return

            # 添加向量到主索引
            self.db.add_vectors(vectors, metadata)

            # 更新最后更新时间
            self.last_update_time = time.time()

            # 可以选择更新二级索引
            # self._update_secondary_indices(vectors, metadata)
        except Exception as e:
            logger.error(f"添加向量失败: {str(e)}")
            # 不抛出异常，避免系统崩溃
            return

    def _apply_filter(self, metadata: Dict, filters: Dict) -> bool:
        """应用过滤器"""
        for key, value in filters.items():
            if key not in metadata:
                return False
            if isinstance(value, list):
                if metadata[key] not in value:
                    return False
            else:
                if metadata[key] != value:
                    return False
        return True

    def get_last_update_time(self) -> float:
        """获取最后更新时间"""
        return self.last_update_time

    def search(
        self, query_vector: np.ndarray, top_k: int = 5, filters: Dict = None
    ) -> Tuple[np.ndarray, List[List[Dict]]]:
        """搜索向量"""
        if self.db is None:
            raise ValueError("向量数据库未初始化")

        try:
            # 基本搜索
            distances, metadata_list = self.db.search(query_vector, top_k)

            # 如果提供了过滤器，可以在搜索结果上应用
            if filters:
                filtered_distances = []
                filtered_metadata = []
                for dist, meta_list in zip(distances, metadata_list):
                    filtered_dist = []
                    filtered_meta = []
                    for d, meta in zip(dist, meta_list):
                        if self._apply_filter(meta, filters):
                            filtered_dist.append(d)
                            filtered_meta.append(meta)
                    if filtered_dist:
                        filtered_distances.append(filtered_dist)
                        filtered_metadata.append(filtered_meta)
                    else:
                        filtered_distances.append([])
                        filtered_metadata.append([])
                return np.array(filtered_distances), filtered_metadata

            return distances, metadata_list
        except Exception as e:
            logger.error(f"搜索向量失败: {str(e)}")
            # 返回空结果，避免系统崩溃
            return np.array([[]]), [[]]

    def get_status(self) -> VectorStatus:
        """获取状态"""
        if self.db is None:
            return VectorStatus(
                db_type="none", total_vectors=0, dimension=0, status="not_initialized"
            )
        return self.db.get_status()

    def save(self):
        """保存数据库"""
        if self.db:
            self.db.save()

    def load(self):
        """加载数据库"""
        if self.db:
            self.db.load()

    def get_all_metadata(self) -> List[Dict[str, Any]]:
        """
        获取所有文档片段的元数据

        Returns:
            文档片段元数据列表
        """
        try:
            if self.db is None:
                return []

            # 对于FAISS数据库，直接返回metadata
            if hasattr(self.db, "metadata") and isinstance(self.db.metadata, dict):
                all_metadata = []
                for key, meta in self.db.metadata.items():
                    # 确保meta是字典类型且包含content字段（过滤系统元信息）
                    if isinstance(meta, dict) and "content" in meta:
                        # 添加chunk_id
                        meta["chunk_id"] = key
                        all_metadata.append(meta)
                logger.info(f"获取到 {len(all_metadata)} 个文档片段元数据")
                return all_metadata
            else:
                logger.warning("当前数据库类型不支持获取所有元数据")
                return []
        except Exception as e:
            logger.error(f"获取所有元数据失败: {str(e)}")
            return []


# 全局向量数据库实例
vector_db_manager = VectorDatabaseManager()
