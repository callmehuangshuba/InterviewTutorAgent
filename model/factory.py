import os
from functools import lru_cache
from typing import Optional
from langchain_core.embeddings import Embeddings
from langchain_community.embeddings import DashScopeEmbeddings
from langchain_community.chat_models.tongyi import BaseChatModel, ChatTongyi
from utils.config_handler import rag_conf


@lru_cache(maxsize=1)
def get_chat_model() -> Optional[BaseChatModel]:
    return ChatTongyi(model=rag_conf["chat_model_name"])


@lru_cache(maxsize=1)
def get_embed_model() -> Optional[Embeddings]:
    return DashScopeEmbeddings(model=rag_conf["embedding_model_name"])
