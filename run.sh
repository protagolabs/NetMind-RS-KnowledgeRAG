export PYTHONPATH=.:$PYTHONPATH:src

# 抑制 protobuf 版本警告
export PYTHONWARNINGS="ignore::UserWarning:google.protobuf.runtime_version"

# python src/knowledge_rag/file_process/step_3_save.py
# python src/knowledge_rag/knowledge_indexing/step_1_get_embedding.py
# python src/knowledge_rag/knowledge_retrieval/doc_matching.py
# python src/app/main_process.py

uv sync 
uv run --env-file .env python src/knowledge_rag/agentic_generation/unit_test.py