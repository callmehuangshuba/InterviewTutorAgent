import os
import sys
from functools import lru_cache
from typing import Optional
from langchain_core.embeddings import Embeddings
from langchain_community.embeddings import DashScopeEmbeddings
from langchain_community.chat_models.tongyi import BaseChatModel, ChatTongyi
from utils.config_handler import rag_conf


@lru_cache(maxsize=1)
def get_chat_model() -> Optional[BaseChatModel]:
    try:
        return ChatTongyi(model=rag_conf["chat_model_name"])
    except Exception as e:
        print(f"[ERROR] get_chat_model 初始化失败: {e}", file=sys.stderr)
        raise


@lru_cache(maxsize=1)
def get_embed_model() -> Optional[Embeddings]:
    try:
        model_name = rag_conf["embedding_model_name"]
        api_key = os.environ.get("DASHSCOPE_API_KEY", "")
        print(f"[DEBUG] get_embed_model: model={model_name}, api_key={'已设置' if api_key else '未设置'}", file=sys.stderr)
        embed = DashScopeEmbeddings(model=model_name)
        # 提前测试一次 API 是否可用
        test_result = embed.embed_query("测试")
        print(f"[DEBUG] embed API 测试成功，维度={len(test_result)}", file=sys.stderr)
        return embed
    except Exception as e:
        print(f"[ERROR] get_embed_model 初始化失败: {e}", file=sys.stderr)
        raise
