import os
import sys
from functools import lru_cache
from typing import Optional
from langchain_core.embeddings import Embeddings
from langchain_community.embeddings import DashScopeEmbeddings
from langchain_community.chat_models.tongyi import BaseChatModel, ChatTongyi
from utils.config_handler import rag_conf


def _get_dashscope_key() -> str:
    """从 Streamlit Secrets 或环境变量获取 DashScope API Key"""
    # 尝试 Streamlit Secrets
    try:
        import streamlit as st
        if st.secrets:
            key = st.secrets.get("DASHSCOPE_API_KEY", "")
            if key:
                return key
    except Exception:
        pass
    # 回退到环境变量
    return os.environ.get("DASHSCOPE_API_KEY", "")


def _get_dashscope_base() -> str:
    """获取 DashScope API Base URL"""
    try:
        import streamlit as st
        if st.secrets:
            base = st.secrets.get("DASHSCOPE_API_BASE", "")
            if base:
                return base
    except Exception:
        pass
    return os.environ.get("DASHSCOPE_API_BASE", "https://dashscope.aliyuncs.com/compatible-mode/v1")


@lru_cache(maxsize=1)
def get_chat_model() -> Optional[BaseChatModel]:
    api_key = _get_dashscope_key()
    api_base = _get_dashscope_base()
    print(f"[DEBUG] get_chat_model: key={'已设置' if api_key else '未设置!'}, base={api_base}", file=sys.stderr)
    if not api_key:
        raise ValueError("DASHSCOPE_API_KEY 未设置，请在 Streamlit Secrets 中配置")
    try:
        return ChatTongyi(
            model=rag_conf["chat_model_name"],
            dashscope_api_key=api_key,
            api_base=api_base,
        )
    except Exception as e:
        print(f"[ERROR] get_chat_model 初始化失败: {e}", file=sys.stderr)
        raise


@lru_cache(maxsize=1)
def get_embed_model() -> Optional[Embeddings]:
    api_key = _get_dashscope_key()
    api_base = _get_dashscope_base()
    print(f"[DEBUG] get_embed_model: key={'已设置' if api_key else '未设置!'}, base={api_base}", file=sys.stderr)
    if not api_key:
        raise ValueError("DASHSCOPE_API_KEY 未设置，请在 Streamlit Secrets 中配置")
    try:
        embed = DashScopeEmbeddings(
            model=rag_conf["embedding_model_name"],
            dashscope_api_key=api_key,
        )
        # 提前测试一次 API 是否可用
        test_result = embed.embed_query("测试")
        print(f"[DEBUG] embed API 测试成功，维度={len(test_result)}", file=sys.stderr)
        return embed
    except Exception as e:
        print(f"[ERROR] get_embed_model 初始化失败: {e}", file=sys.stderr)
        raise
