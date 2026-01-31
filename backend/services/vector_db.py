from typing import List, Optional, Dict, Tuple
import numpy as np
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

    def search(self, query_vector: np.ndarray, top_k: int = 5) -> Tuple[np.ndarray, List[Dict]]:
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

    def __init__(self, dimension: int, index_type: str = "HNSW"):
        super().__init__(dimension)
        self.index_type = index_type
        self.index = None
        self.metadata: Dict[str, Dict] = {}
        self.db_path = Path(settings.vector_db_dir) / "faiss_index"
        self.metadata_path = Path(settings.vector_db_dir) / "faiss_metadata.json"
        self._init_index()

    def _init_index(self):
        """初始化 FAISS 索引"""
        import faiss

        if self.index_type == "HNSW":
            # HNSW 索引 - 高性能
            self.index = faiss.IndexHNSWFlat(self.dimension, 32)
        elif self.index_type == "IVF":
            # IVF 索引 - 倒排文件
            quantizer = faiss.IndexFlatL2(self.dimension)
            self.index = faiss.IndexIVFFlat(quantizer, self.dimension, 100)
        elif self.index_type == "PQ":
            # PQ 索引 - 乘积量化
            self.index = faiss.IndexPQ(self.dimension, 16, 8)
        else:
            # 默认使用 Flat 索引
            self.index = faiss.IndexFlatL2(self.dimension)

        logger.info(f"初始化 FAISS 索引: {self.index_type} (维度: {self.dimension})")

    def add_vectors(self, vectors: np.ndarray, metadata: List[Dict]):
        """添加向量"""
        import faiss

        if not isinstance(vectors, np.ndarray):
            vectors = np.array(vectors, dtype=np.float32)

        # 确保 vectors 是二维数组
        if vectors.ndim == 1:
            vectors = vectors.reshape(1, -1)

        # 训练索引 (如果需要)
        if hasattr(self.index, 'is_trained') and not self.index.is_trained:
            self.index.train(vectors)

        # 添加向量
        start_id = self.total_vectors
        self.index.add(vectors)

        # 保存元数据
        for i, meta in enumerate(metadata):
            self.metadata[str(start_id + i)] = meta

        self.total_vectors += len(vectors)
        logger.info(f"添加 {len(vectors)} 个向量到 FAISS (总数: {self.total_vectors})")
        
        # 立即保存以确保持久化
        self.save()

    def search(self, query_vector: np.ndarray, top_k: int = 5) -> Tuple[np.ndarray, List[List[Dict]]]:
        """搜索向量"""
        import faiss

        if not isinstance(query_vector, np.ndarray):
            query_vector = np.array(query_vector, dtype=np.float32)

        # 确保 query_vector 是二维数组
        if query_vector.ndim == 1:
            query_vector = query_vector.reshape(1, -1)

        # 搜索
        distances, indices = self.index.search(query_vector, top_k)

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
            status="ready"
        )

    def save(self):
        """保存数据库"""
        import faiss

        self.db_path.parent.mkdir(parents=True, exist_ok=True)

        # 保存索引
        faiss.write_index(self.index, str(self.db_path))

        # 保存元数据
        with open(self.metadata_path, 'w', encoding='utf-8') as f:
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
            if hasattr(self.index, 'd') and self.index.d != self.dimension:
                logger.error(f"FAISS 索引维度不匹配: 索引维度={self.index.d}, 配置维度={self.dimension}")
                logger.info("重新初始化 FAISS 索引以匹配正确维度")
                self._init_index()
                self.metadata = {}
                self.total_vectors = 0
                return

            # 加载元数据
            if self.metadata_path.exists():
                with open(self.metadata_path, 'r', encoding='utf-8') as f:
                    self.metadata = json.load(f)
                logger.info(f"已加载 {len(self.metadata)} 条元数据")
            else:
                logger.warning(f"元数据文件不存在: {self.metadata_path}")
                self.metadata = {}

            # 验证向量数量
            logger.info(f"FAISS 索引已加载: {self.total_vectors} 个向量，{len(self.metadata)} 条元数据")
            
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

    def __init__(self, dimension: int, host: str = "localhost", port: int = 19530, collection_name: str = "rag_vectors"):
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
            self.client = MilvusClient(
                uri=f"http://{self.host}:{self.port}"
            )
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
                metric_type="COSINE"
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
        ids = [str(i) for i in range(self.total_vectors, self.total_vectors + len(vectors))]

        # 插入向量
        data = [ids, vectors.tolist(), metadata]
        self.client.insert(self.collection_name, data=data)

        # 刷新
        self.client.flush(self.collection_name)

        self.total_vectors += len(vectors)
        logger.info(f"添加 {len(vectors)} 个向量到 Milvus (总数: {self.total_vectors})")

    def search(self, query_vector: np.ndarray, top_k: int = 5) -> Tuple[np.ndarray, List[Dict]]:
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
            output_fields=["*"]
        )

        # 提取距离和元数据
        distances = []
        metadata_list = []

        for result in results:
            distances.append([item['distance'] for item in result])
            metadata_list.append([item['entity'] for item in result])

        return np.array(distances), metadata_list

    def delete_vectors(self, ids: List[str]):
        """删除向量"""
        self.client.delete(self.collection_name, ids)
        logger.info(f"从 Milvus 删除 {len(ids)} 个向量")

    def get_status(self) -> VectorStatus:
        """获取状态"""
        try:
            self._ensure_collection()
            num_entities = self.client.get_collection_stats(self.collection_name)['row_count']
        except:
            num_entities = 0

        return VectorStatus(
            db_type="milvus",
            total_vectors=num_entities,
            dimension=self.dimension,
            status="ready"
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

    def initialize(self, config: VectorDBConfig):
        """初始化向量数据库"""
        self.config = config

        try:
            if config.db_type == VectorDBType.FAISS:
                self.db = FAISSDatabase(config.dimension, config.index_type)
            elif config.db_type == VectorDBType.MILVUS:
                self.db = MilvusDatabase(
                    config.dimension,
                    config.host or settings.milvus_host,
                    config.port or settings.milvus_port,
                    config.collection_name or settings.milvus_collection_name
                )
            else:
                raise ValueError(f"不支持的向量数据库类型: {config.db_type}")

            # 尝试加载数据
            self.db.load()

            logger.info(f"向量数据库初始化成功: {config.db_type}")
            return True

        except Exception as e:
            logger.error(f"向量数据库初始化失败: {str(e)}")
            return False

    def add_vectors(self, vectors: np.ndarray, metadata: List[Dict]):
        """添加向量"""
        if self.db is None:
            raise ValueError("向量数据库未初始化")
        self.db.add_vectors(vectors, metadata)

    def search(self, query_vector: np.ndarray, top_k: int = 5) -> Tuple[np.ndarray, List[Dict]]:
        """搜索向量"""
        if self.db is None:
            raise ValueError("向量数据库未初始化")
        return self.db.search(query_vector, top_k)

    def get_status(self) -> VectorStatus:
        """获取状态"""
        if self.db is None:
            return VectorStatus(
                db_type="none",
                total_vectors=0,
                dimension=0,
                status="not_initialized"
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


# 全局向量数据库实例
vector_db_manager = VectorDatabaseManager()