export PYTHONPATH=.:$PYTHONPATH:src

# 抑制 protobuf 版本警告
export PYTHONWARNINGS="ignore::UserWarning:google.protobuf.runtime_version"

# uv sync 
uv run --env-file .env python src/knowledge_rag/agentic_generation/unit_test.py
# uv run --env-file .env python src/knowledge_rag/main_process.py


# 新内容入库，检查一下对应的 main 函数，按顺序进行即可。
# uv run --env-file .env python src/knowledge_rag/file_process/step_1_chunk.py
# uv run --env-file .env python src/knowledge_rag/file_process/step_3_save.py
# uv run --env-file .env python src/knowledge_rag/knowledge_indexing/step_1_get_embedding.py
# uv run --env-file .env python src/knowledge_rag/knowledge_indexing/step_2_save_to_db.py
