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
