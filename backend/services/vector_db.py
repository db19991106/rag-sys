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

        # 检查索引状态
        if self.index is None:
            logger.error("FAISS 索引未初始化")
            return np.array([[]]), [[]]
        
        if self.index.ntotal == 0:
            logger.warning("FAISS 索引为空，没有向量可搜索")
            return np.array([[]]), [[]]
        
        logger.debug(f"FAISS 索引状态: {self.index.ntotal} 个向量")

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
        
        # 调试日志：显示搜索结果
        logger.debug(
            f"FAISS 搜索完成，耗时: {search_time:.4f}s, 返回: {len(indices[0])} 个结果"
        )
        
        # 显示前几个结果的距离和索引
        if len(indices) > 0 and len(indices[0]) > 0:
            for i in range(min(3, len(indices[0]))):
                idx = indices[0][i]
                dist = distances[0][i]
                logger.debug(f"  结果 {i+1}: 索引={idx}, 距离={dist:.4f}")

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
        # 计算数据库文件大小
        db_size = 0
        if self.db_path.exists():
            db_size += self.db_path.stat().st_size
        if self.metadata_path.exists():
            db_size += self.metadata_path.stat().st_size

        return VectorStatus(
            db_type="faiss",
            total_vectors=self.total_vectors,
            dimension=self.dimension,
            status="ready",
            db_size=db_size,
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
            
            # 检测索引和元数据不匹配的情况
            if self.total_vectors > 0 and len(self.metadata) != self.total_vectors:
                logger.warning(
                    f"⚠️ 检测到索引和元数据不匹配: 索引有 {self.total_vectors} 个向量，"
                    f"但元数据只有 {len(self.metadata)} 条。"
                )
                # 不再自动清空索引，而是保留可用数据
                # 只过滤掉没有元数据的搜索结果即可
                if len(self.metadata) < self.total_vectors * 0.9:
                    logger.warning(
                        f"元数据缺失较多，但保留现有数据。"
                        f"检索时会自动过滤无效结果。建议重新执行文档嵌入操作。"
                    )

        except Exception as e:
            logger.error(f"加载 FAISS 索引失败: {str(e)}")
            logger.info("重新初始化 FAISS 索引")
            self._init_index()
            self.metadata = {}
            self.total_vectors = 0

    def clear(self):
        """清空数据库"""
        import os

        try:
            # 删除索引文件
            if self.db_path.exists():
                os.remove(str(self.db_path))
                logger.info(f"已删除 FAISS 索引文件: {self.db_path}")

            # 删除元数据文件
            if self.metadata_path.exists():
                os.remove(str(self.metadata_path))
                logger.info(f"已删除元数据文件: {self.metadata_path}")

            # 重新初始化空索引
            self._init_index()
            self.metadata = {}
            self.total_vectors = 0

            logger.info("FAISS 数据库已清空")
            return True
        except Exception as e:
            logger.error(f"清空 FAISS 数据库失败: {str(e)}")
            return False


class MilvusDatabase(VectorDatabase):
    """Milvus 向量数据库（支持远程服务器和 Milvus Lite 本地模式）"""

    def __init__(
        self,
        dimension: int,
        host: str = "localhost",
        port: int = 19530,
        collection_name: str = "rag_vectors",
        db_path: str = None,  # Milvus Lite 本地文件路径
    ):
        super().__init__(dimension)
        self.host = host
        self.port = port
        self.collection_name = collection_name
        self.db_path = db_path  # 如果提供，则使用 Milvus Lite 模式
        self.client = None
        self.collection = None
        self._connect()

    def _connect(self):
        """连接 Milvus 或 Milvus Lite"""
        try:
            from pymilvus import MilvusClient

            # 关闭旧连接（如果有），避免状态冲突
            if self.client is not None:
                try:
                    if hasattr(self.client, 'close'):
                        self.client.close()
                        logger.info("已关闭旧的 Milvus 连接")
                except Exception as e:
                    logger.warning(f"关闭旧连接时出错: {str(e)}")
                finally:
                    self.client = None

            # Milvus Lite 模式：使用本地文件路径
            if self.db_path:
                # 确保目录存在
                from pathlib import Path
                Path(self.db_path).parent.mkdir(parents=True, exist_ok=True)
                self.client = MilvusClient(uri=self.db_path)
                logger.info(f"连接 Milvus Lite: {self.db_path}")
            else:
                # 远程 Milvus 模式
                self.client = MilvusClient(uri=f"http://{self.host}:{self.port}")
                logger.info(f"连接 Milvus: {self.host}:{self.port}")
        except Exception as e:
            logger.error(f"连接 Milvus 失败: {str(e)}")
            raise

    def _ensure_collection(self):
        """确保集合存在"""
        has_collection = self.client.has_collection(self.collection_name)
        logger.info(f"检查集合是否存在: {self.collection_name}, 结果: {has_collection}")
        
        if not has_collection:
            # 准备 HNSW 索引参数（与 FAISS HNSW 配置一致）
            index_params = self.client.prepare_index_params()
            index_params.add_index(
                field_name="vector",
                index_type="HNSW",
                metric_type="COSINE",
                params={
                    "M": 16,              # 每个节点的最大连接数
                    "efConstruction": 200  # 构建时的搜索宽度
                }
            )

            self.client.create_collection(
                collection_name=self.collection_name,
                dimension=self.dimension,
                metric_type="COSINE",
                enable_dynamic_field=True,  # 启用动态字段支持元数据
                index_params=index_params,
            )
            logger.info(f"创建 Milvus 集合: {self.collection_name} (HNSW索引, M=16, efConstruction=200)")
        # MilvusClient 不需要获取 collection 对象，直接使用 collection_name 即可

    def add_vectors(self, vectors: np.ndarray, metadata: List[Dict]):
        """添加向量"""
        self._ensure_collection()

        if not isinstance(vectors, np.ndarray):
            vectors = np.array(vectors, dtype=np.float32)

        # 确保 vectors 是二维数组
        if vectors.ndim == 1:
            vectors = vectors.reshape(1, -1)

        # 准备数据 - 使用正确的字典格式
        data = []
        for i, (vec, meta) in enumerate(zip(vectors, metadata)):
            # Milvus 要求 id 是 int64 类型
            record = {
                "id": self.total_vectors + i,  # 使用整数 id
                "vector": vec.tolist(),
            }
            # 添加元数据字段（将 chunk_id 保存为单独字段）
            for key, value in meta.items():
                if key != "id" and key != "vector":
                    # 将列表转换为字符串（Milvus 不支持列表类型）
                    if isinstance(value, list):
                        record[key] = str(value)
                    else:
                        record[key] = value
            data.append(record)

        # 插入向量
        self.client.insert(self.collection_name, data=data)

        self.total_vectors += len(vectors)
        logger.info(f"添加 {len(vectors)} 个向量到 Milvus (总数: {self.total_vectors})")

    def search(
        self, query_vector: np.ndarray, top_k: int = 5
    ) -> Tuple[np.ndarray, List[Dict]]:
        """搜索向量"""
        # 确保集合存在
        self._ensure_collection()

        if not isinstance(query_vector, np.ndarray):
            query_vector = np.array(query_vector, dtype=np.float32)

        # 确保 query_vector 是二维数组
        if query_vector.ndim == 1:
            query_vector = query_vector.reshape(1, -1)

        # HNSW 搜索参数：ef 值越大，召回率越高，但搜索越慢
        search_params = {
            "params": {"ef": max(128, top_k * 4)}  # 与 FAISS HNSW efSearch 一致
        }

        # 搜索
        results = self.client.search(
            collection_name=self.collection_name,
            data=query_vector.tolist(),
            limit=top_k,
            output_fields=["*"],
            search_params=search_params,
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

        # 计算数据库文件大小（仅 Milvus Lite 有本地文件）
        db_size = 0
        if self.db_path:
            from pathlib import Path
            db_file = Path(self.db_path)
            if db_file.exists():
                db_size = db_file.stat().st_size

        return VectorStatus(
            db_type="milvus_lite" if self.db_path else "milvus",
            total_vectors=num_entities,
            dimension=self.dimension,
            status="ready",
            db_size=db_size,
        )

    def save(self):
        """保存数据库 (Milvus 自动持久化)"""
        logger.info("Milvus 自动持久化数据")

    def load(self):
        """加载数据库 (Milvus 自动加载)"""
        logger.info("Milvus 自动加载数据")

    def get_all_metadata(self) -> List[Dict[str, Any]]:
        """
        获取所有文档片段的元数据

        Returns:
            文档片段元数据列表
        """
        try:
            self._ensure_collection()
            
            # 使用 query 获取所有数据
            results = self.client.query(
                collection_name=self.collection_name,
                filter="",  # 空过滤器获取所有数据
                output_fields=["*"],
                limit=10000  # 限制最大数量
            )
            
            logger.info(f"Milvus 获取到 {len(results)} 条元数据")
            return results
            
        except Exception as e:
            logger.error(f"Milvus 获取所有元数据失败: {str(e)}")
            return []

    def clear(self):
        """清空数据库
        
        修复：先关闭连接释放文件句柄，再删除文件，最后重新连接。
        这样可以避免 "Channel closed" 和 "internal error" 错误。
        """
        import os
        import gc
        import time

        try:
            # 步骤 1: 先关闭连接，释放文件句柄（关键修复点）
            if self.client is not None:
                try:
                    # 先尝试删除集合（在连接还活着的时候）
                    if hasattr(self.client, 'has_collection') and self.client.has_collection(self.collection_name):
                        self.client.drop_collection(self.collection_name)
                        logger.info(f"已删除 Milvus 集合: {self.collection_name}")
                except Exception as e:
                    logger.warning(f"删除集合时出错（可能已不存在）: {str(e)}")
                
                try:
                    if hasattr(self.client, 'close'):
                        self.client.close()
                        logger.info("已关闭旧的 Milvus 连接")
                except Exception as e:
                    logger.warning(f"关闭旧连接时出错: {str(e)}")
                finally:
                    self.client = None
            
            # 强制垃圾回收，确保资源释放
            gc.collect()
            
            # 给一点时间让操作系统释放文件句柄
            time.sleep(0.1)

            # 步骤 2: 删除文件（连接已关闭，文件句柄已释放）
            if self.db_path:
                db_file = Path(self.db_path)
                
                # 删除数据库文件
                if db_file.exists():
                    try:
                        os.remove(str(db_file))
                        logger.info(f"已删除 Milvus Lite 数据库文件: {self.db_path}")
                    except PermissionError:
                        # 文件可能仍被占用，稍等重试
                        logger.warning("文件被占用，等待后重试...")
                        time.sleep(0.5)
                        gc.collect()
                        os.remove(str(db_file))
                        logger.info(f"重试成功，已删除 Milvus Lite 数据库文件: {self.db_path}")
                
                # 删除锁文件（Milvus Lite 锁文件格式：.<filename>.lock）
                lock_file = db_file.parent / f".{db_file.name}.lock"
                if lock_file.exists():
                    try:
                        os.remove(str(lock_file))
                        logger.info(f"已删除锁文件: {lock_file}")
                    except Exception as e:
                        logger.warning(f"删除锁文件失败: {str(e)}")
            
            # 步骤 3: 重新建立连接
            self._connect()
            logger.info(f"已重新连接 Milvus Lite: {self.db_path}")

            # 重置状态
            self.total_vectors = 0

            logger.info("Milvus 数据库已清空")
            return True
        except Exception as e:
            logger.error(f"清空 Milvus 数据库失败: {str(e)}")
            return False


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
            elif config.db_type == VectorDBType.MILVUS_LITE:
                # Milvus Lite 模式：使用本地文件
                db_path = config.index_path or settings.milvus_lite_db_path
                self.db = MilvusDatabase(
                    config.dimension,
                    db_path=db_path,
                    collection_name=config.collection_name or settings.milvus_collection_name,
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
            # 抛出异常，让调用方知道操作失败
            raise RuntimeError(f"添加向量失败: {str(e)}")

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

            # 对于Milvus数据库，调用其get_all_metadata方法
            if hasattr(self.db, "get_all_metadata") and callable(getattr(self.db, "get_all_metadata")):
                return self.db.get_all_metadata()

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

    def clear(self) -> bool:
        """
        清空向量数据库
        
        Returns:
            是否成功
        """
        import os
        
        try:
            if self.db is None:
                logger.warning("向量数据库未初始化，无需清空")
                return True
            
            # 调用数据库的 clear 方法
            if hasattr(self.db, "clear") and callable(getattr(self.db, "clear")):
                success = self.db.clear()
            else:
                logger.warning("当前数据库类型不支持 clear 方法")
                success = False
            
            # 清空文档元数据文件 (documents.json)
            if self.config and self.config.index_path:
                docs_path = Path(self.config.index_path) / "documents.json"
            else:
                docs_path = Path(settings.vector_db_dir) / "documents.json"
            
            if docs_path.exists():
                os.remove(str(docs_path))
                logger.info(f"已删除文档元数据文件: {docs_path}")
            
            # 清空 BM25 索引文件
            bm25_index_path = Path(settings.vector_db_dir) / "bm25_index.pkl"
            if bm25_index_path.exists():
                os.remove(str(bm25_index_path))
                logger.info(f"已删除 BM25 索引文件: {bm25_index_path}")
            
            # 清空 Milvus Lite 数据库文件（确保文件被删除）
            # 注意：MilvusDatabase.clear() 已经处理了这个，这里作为备份
            milvus_db_path = Path(settings.vector_db_dir) / "milvus_lite.db"
            if milvus_db_path.exists():
                try:
                    os.remove(str(milvus_db_path))
                    logger.info(f"已删除 Milvus Lite 数据库文件: {milvus_db_path}")
                except Exception as e:
                    logger.warning(f"删除 Milvus Lite 数据库文件失败（可能被占用）: {e}")
            
            # 清空 BM25 和元数据的全局缓存
            try:
                from services.retriever import clear_global_caches
                clear_global_caches()
                logger.info("已清空 BM25 和元数据全局缓存")
            except Exception as e:
                logger.warning(f"清空全局缓存时出错: {str(e)}")
            
            # 重置状态
            self.last_update_time = 0
            
            logger.info("向量数据库已完全清空")
            return success
        except Exception as e:
            logger.error(f"清空向量数据库失败: {str(e)}")
            return False


# 全局向量数据库实例
vector_db_manager = VectorDatabaseManager()
