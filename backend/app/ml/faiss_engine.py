"""
SmartLib Kiosk - High-Performance FAISS Vector Search Engine
Layer 3: Hybrid FAISS + pgvector Architecture

Provides sub-millisecond similarity search over tens of thousands of face embeddings.
Acts as an in-memory cache layer over the persistent pgvector database.
"""
import faiss
import numpy as np
import asyncio
import threading
from typing import List, Tuple, Dict, Optional
from loguru import logger
from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy import select
from app.models.face_embedding import FaceEmbedding
from app.models.student import Student, StudentStatus

class FaissEngine:
    """
    In-memory vector database using Meta's FAISS.
    Configured for exact search (IndexFlatIP) since 50k-100k vectors 
    easily fit in RAM and search is sub-millisecond.
    """
    def __init__(self, embedding_dim: int = 512):
        self.embedding_dim = embedding_dim
        # IndexFlatIP uses Inner Product (equivalent to Cosine Similarity if vectors are L2-normalized)
        self.index = faiss.IndexFlatIP(embedding_dim)
        
        # Mapping from FAISS internal integer ID to database string ID
        self.id_map: Dict[int, int] = {}  # faiss_id -> embedding_id
        self.embedding_to_student: Dict[int, str] = {} # embedding_id -> student_id
        self.student_to_embeddings: Dict[str, List[int]] = {} # student_id -> [faiss_id, ...]
        
        self.current_idx = 0
        self.is_ready = False
        self._lock = threading.Lock()
        
    async def sync_from_db(self, db: AsyncSession):
        """
        Load all active embeddings from PostgreSQL to FAISS on startup.
        """
        logger.info("Starting FAISS synchronization from pgvector...")
        try:
            # Fetch all embeddings for active students
            stmt = (
                select(FaceEmbedding.id, FaceEmbedding.embedding, FaceEmbedding.student_id)
                .join(Student, Student.student_id == FaceEmbedding.student_id)
                .where(Student.status == StudentStatus.ACTIVE.value)
            )
            result = await db.execute(stmt)
            rows = result.all()
            
            with self._lock:
                # Reset completely before sync
                self.index = faiss.IndexFlatIP(self.embedding_dim)
                self.id_map.clear()
                self.embedding_to_student.clear()
                self.student_to_embeddings.clear()
                self.current_idx = 0
                self.is_ready = False

                if not rows:
                    logger.warning("No face embeddings found in database. FAISS is empty.")
                    self.is_ready = True
                    return
                    
                vectors = []
                for row in rows:
                    emb_id, emb_data, student_id = row[0], row[1], row[2]
                    
                    # Convert to numpy array safely depending on type returned by pgvector/sqlalchemy
                    if isinstance(emb_data, np.ndarray):
                        vector = emb_data.copy().astype(np.float32)
                    elif isinstance(emb_data, (bytes, memoryview)):
                        vector = np.frombuffer(emb_data, dtype=np.float32).copy()
                    elif isinstance(emb_data, list):
                        vector = np.array(emb_data, dtype=np.float32)
                    else:
                        logger.warning(f"Unknown embedding type {type(emb_data)} for ID {emb_id}")
                        continue
                    
                    if len(vector) != self.embedding_dim:
                        logger.warning(f"Skipping corrupt embedding {emb_id}: dim={len(vector)}")
                        continue
                        
                    norm = np.linalg.norm(vector)
                    if norm < 1e-6:
                        logger.warning(f"Skipping zero vector for embedding {emb_id}")
                        continue
                    
                    # Ensure L2 normalization for Inner Product to act as Cosine Similarity
                    vector = vector / norm
                    vectors.append(vector)
                    
                    # Store mappings
                    self.id_map[self.current_idx] = emb_id
                    self.embedding_to_student[emb_id] = student_id
                    
                    if student_id not in self.student_to_embeddings:
                        self.student_to_embeddings[student_id] = []
                    self.student_to_embeddings[student_id].append(self.current_idx)
                    
                    self.current_idx += 1
                    
                if vectors:
                    # Add to FAISS index
                    vectors_np = np.array(vectors, dtype=np.float32)
                    self.index.add(vectors_np)
                
                self.is_ready = True
                logger.info(f"FAISS sync complete. Loaded {self.current_idx} vectors into RAM.")
            
        except Exception as e:
            logger.error(f"FAISS sync failed: {e}")
            self.is_ready = False

    def add_embedding(self, embedding_id: int, student_id: str, vector: np.ndarray):
        """Add a new embedding dynamically without full sync."""
        if not self.is_ready:
            return
            
        vec = vector.astype(np.float32)
        norm = np.linalg.norm(vec)
        if norm < 1e-6:
            logger.warning(f"Cannot add zero vector for embedding {embedding_id}")
            return
            
        vec = vec / norm
        vec = np.expand_dims(vec, axis=0)
        
        with self._lock:
            self.index.add(vec)
            self.id_map[self.current_idx] = embedding_id
            self.embedding_to_student[embedding_id] = student_id
            
            if student_id not in self.student_to_embeddings:
                self.student_to_embeddings[student_id] = []
            self.student_to_embeddings[student_id].append(self.current_idx)
            
            self.current_idx += 1
            
        logger.debug(f"Added new vector to FAISS. Total: {self.current_idx}")

    def search(self, query_vector: np.ndarray, top_k: int = 3) -> List[Tuple[str, float]]:
        """
        Search for nearest neighbors in sub-millisecond.
        Returns List of (student_id, similarity_score).
        """
        if not self.is_ready or self.current_idx == 0:
            return []
            
        q = query_vector.astype(np.float32)
        norm = np.linalg.norm(q)
        if norm < 1e-6:
            return []
            
        q = q / norm
        q = np.expand_dims(q, axis=0)
        
        # Safe Read
        with self._lock:
            actual_k = min(top_k, self.index.ntotal)
            if actual_k == 0:
                return []
                
            similarities, indices = self.index.search(q, actual_k)
            # Make copies to process outside lock if needed, though dict copying is fast
            sim_list = similarities[0].copy()
            idx_list = indices[0].copy()
            id_map_copy = self.id_map.copy()
            emb_to_student_copy = self.embedding_to_student.copy()
        
        student_best: Dict[str, float] = {}
        for i in range(actual_k):
            idx = int(idx_list[i])
            if idx == -1 or idx not in id_map_copy:
                continue
                
            emb_id = id_map_copy[idx]
            student_id = emb_to_student_copy.get(emb_id)
            if student_id is None:
                continue
                
            score = float(sim_list[i])
            
            # Keep highest score for each student (B14 fix verification)
            if student_id not in student_best or score > student_best[student_id]:
                student_best[student_id] = score
                
        # Return sorted list of tuples
        results = [(sid, score) for sid, score in student_best.items()]
        results.sort(key=lambda x: x[1], reverse=True)
        return results
