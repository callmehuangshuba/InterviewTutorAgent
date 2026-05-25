import os
import pickle
import hashlib
import glob as glob_module
from langchain_community.vectorstores import FAISS
from langchain_core.documents import Document
from langchain_core.vectorstores import VectorStore
from utils.config_handler import chroma_conf, rag_conf
from model.factory import get_embed_model
from langchain_text_splitters import RecursiveCharacterTextSplitter
from utils.path_tool import get_abs_path
from utils.file_handler import pdf_loader, txt_loader, listdir_with_allowed_type, get_file_md5_hex
from utils.logger_handler import logger


class VectorStoreService:
    def __init__(self):
        self._index_path = get_abs_path(chroma_conf["persist_directory"])
        self.spliter = RecursiveCharacterTextSplitter(
            chunk_size=chroma_conf["chunk_size"],
            chunk_overlap=chroma_conf["chunk_overlap"],
            separators=chroma_conf["separators"],
            length_function=len,
        )
        self.vector_store: VectorStore | None = self._load_index()

    def _load_index(self) -> VectorStore | None:
        faiss_path = os.path.join(self._index_path, "index.faiss")
        if os.path.exists(faiss_path):
            try:
                return FAISS.load_local(
                    self._index_path,
                    get_embed_model(),
                    allow_dangerous_deserialization=True,
                )
            except Exception as e:
                logger.warning(f"加载 FAISS 索引失败，将重新构建：{e}")
        return None

    def _save_index(self):
        if self.vector_store is not None:
            self.vector_store.save_local(self._index_path)
            logger.info("FAISS 索引已保存")

    def get_retriever(self, k: int | None = None):
        target_k = k if isinstance(k, int) and k > 0 else chroma_conf["k"]
        if self.vector_store is None:
            raise RuntimeError("知识库未加载，请先点击「加载/更新知识库」")
        return self.vector_store.as_retriever(search_kwargs={"k": target_k})

    def load_document(self):
        md5_store_path = get_abs_path(chroma_conf["md5_hex_store"])

        def _check_md5(md5: str) -> bool:
            if not os.path.exists(md5_store_path):
                open(md5_store_path, "w", encoding="utf-8").close()
                return False
            with open(md5_store_path, "r", encoding="utf-8") as f:
                return md5 in {line.strip() for line in f}

        def _save_md5(md5: str):
            with open(md5_store_path, "a", encoding="utf-8") as f:
                f.write(md5 + "\n")

        def _get_docs(path: str):
            if path.endswith("txt"):
                return txt_loader(path)
            if path.endswith("pdf"):
                return pdf_loader(path)
            if path.endswith("md"):
                return txt_loader(path)  # markdown 当作纯文本处理
            return []

        allowed_files = listdir_with_allowed_type(
            get_abs_path(chroma_conf["data_path"]),
            tuple(chroma_conf["allow_knowledge_file_type"]),
        )
        # 额外递归扫描子目录中的 .md 文件（面经数据在 interview_exp_md/ 子目录）
        data_root = get_abs_path(chroma_conf["data_path"])
        import os
        for root, dirs, files in os.walk(data_root):
            for fname in files:
                if fname.endswith(tuple(chroma_conf["allow_knowledge_file_type"])):
                    full_path = os.path.join(root, fname)
                    if full_path not in allowed_files:
                        allowed_files = allowed_files + (full_path,)

        all_docs: list[Document] = []

        for path in allowed_files:
            md5_hex = get_file_md5_hex(path)
            if _check_md5(md5_hex):
                logger.info(f"[加载知识库]{path} 已存在，跳过")
                continue

            try:
                docs = _get_docs(path)
                if not docs:
                    logger.warning(f"[加载知识库]{path} 无有效内容，跳过")
                    continue

                split_docs = self.spliter.split_documents(docs)
                if not split_docs:
                    logger.warning(f"[加载知识库]{path} 分片后无有效内容，跳过")
                    continue

                all_docs.extend(split_docs)
                _save_md5(md5_hex)
                logger.info(f"[加载知识库]{path} 准备就绪（{len(split_docs)} 个分片）")
            except Exception as e:
                logger.error(f"[加载知识库]{path} 加载失败：{e}", exc_info=True)

        if not all_docs:
            logger.info("没有新文档需要加载")
            return

        try:
            if self.vector_store is None:
                self.vector_store = FAISS.from_documents(all_docs, get_embed_model())
            else:
                self.vector_store.add_documents(all_docs)
            self._save_index()
            logger.info(f"知识库加载完成，共 {len(all_docs)} 个分片")
        except Exception as e:
            logger.error(f"知识库写入失败：{e}", exc_info=True)


if __name__ == '__main__':
    vs = VectorStoreService()
    vs.load_document()
    retriever = vs.get_retriever()
    for r in retriever.invoke("线程"):
        print(r.page_content)
        print("-" * 20)


