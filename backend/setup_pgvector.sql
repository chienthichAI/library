-- 1. Enable pgvector extension
CREATE EXTENSION IF NOT EXISTS vector;

-- 2. Modify column type (Drop and Re-add strategy)
-- Direct casting from bytea (numpy binary) to vector fails (ERROR: 42846).
-- We will drop the old column and add a new one. 
-- WARNING: THIS WILL CLEAR ALL EXISTING FACE EMBEDDINGS. You will need to re-register faces.

ALTER TABLE face_embeddings DROP COLUMN IF EXISTS embedding;
ALTER TABLE face_embeddings ADD COLUMN embedding vector(512);

-- 3. Create index for fast similarity search
-- HNSW index is recommended for performance
CREATE INDEX ON face_embeddings USING hnsw (embedding vector_cosine_ops);

-- 4. Add pgvector embeddings for books (bge-m3 produces 1024-dim vectors)
ALTER TABLE books ADD COLUMN IF NOT EXISTS embedding vector(1024);
CREATE INDEX IF NOT EXISTS books_embedding_hnsw_idx
ON books USING hnsw (embedding vector_cosine_ops);

-- 5. Ensure policy_chunks has embedding + index (bge-m3 produces 1024-dim vectors)
ALTER TABLE policy_chunks ADD COLUMN IF NOT EXISTS embedding vector(1024);
CREATE INDEX IF NOT EXISTS policy_chunks_embedding_hnsw_idx
ON policy_chunks USING hnsw (embedding vector_cosine_ops);
