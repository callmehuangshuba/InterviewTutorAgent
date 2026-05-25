import os

# 支持从 Streamlit Secrets 读取 API Key（在 Streamlit Cloud 部署时生效）
try:
    import streamlit as st
    if st.secrets:
        os.environ.setdefault("DASHSCOPE_API_KEY", st.secrets.get("DASHSCOPE_API_KEY", ""))
        os.environ.setdefault("DASHSCOPE_API_BASE", st.secrets.get("DASHSCOPE_API_BASE", "https://dashscope.aliyuncs.com/compatible-mode/v1"))
        os.environ.setdefault("AMAP_API_KEY", st.secrets.get("AMAP_API_KEY", ""))
except Exception:
    pass  # 非 Streamlit 环境（本地开发）忽略，从 .env 读取

from abc import ABC, abstractmethod
from typing import Optional
from langchain_core.embeddings import Embeddings
from langchain_community.chat_models.tongyi import BaseChatModel
from langchain_community.embeddings import DashScopeEmbeddings
from langchain_community.chat_models.tongyi import ChatTongyi
from utils.config_handler import rag_conf


class BaseModelFactory(ABC):
    @abstractmethod
    def generator(self) -> Optional[Embeddings | BaseChatModel]:
        pass


class ChatModelFactory(BaseModelFactory):
    def generator(self) -> Optional[Embeddings | BaseChatModel]:
        return ChatTongyi(model=rag_conf["chat_model_name"])


class EmbeddingsFactory(BaseModelFactory):
    def generator(self) -> Optional[Embeddings | BaseChatModel]:
        return DashScopeEmbeddings(model=rag_conf["embedding_model_name"])


chat_model = ChatModelFactory().generator()
embed_model = EmbeddingsFactory().generator()
