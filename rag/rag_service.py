
"""
总结服务类：用户提问，搜索参考资料，将提问和参考资料提交给模型，让模型总结回复
"""
from langchain_core.documents import Document
from langchain_core.output_parsers import StrOutputParser
from rag.vector_store import VectorStoreService
from rag.rerank_service import RerankService
from utils.prompt_loader import load_rag_prompts
from langchain_core.prompts import PromptTemplate
from model.factory import get_chat_model
from utils.config_handler import rag_conf, chroma_conf


def print_prompt(prompt):
    print("="*20)
    print(prompt.to_string())
    print("="*20)
    return prompt


class RagSummarizeService(object):
    def __init__(self):
        self.vector_store = VectorStoreService()
        self.enable_rerank = bool(rag_conf.get("enable_rerank", False))
        self.recall_k = int(rag_conf.get("rerank_recall_k", 12))
        self.final_k = int(chroma_conf.get("k", 3))
        self._retriever = None
        self.rerank_service = RerankService() if self.enable_rerank else None
        self.prompt_text = load_rag_prompts()
        self.prompt_template = PromptTemplate.from_template(self.prompt_text)
        self._model = None
        self._chain = None

    @property
    def retriever(self):
        if self._retriever is None:
            retriever_k = self.recall_k if self.enable_rerank else self.final_k
            self._retriever = self.vector_store.get_retriever(k=retriever_k)
        return self._retriever

    def _init_chain(self):
        if self._chain is None:
            self._model = get_chat_model()
            self._chain = self.prompt_template | self._model | StrOutputParser()
        return self._chain

    def retriever_docs(self, query: str) -> list[Document]:
        docs = self.retriever.invoke(query)
        if not self.enable_rerank or not self.rerank_service:
            return docs
        return self.rerank_service.rerank(query, docs)

    def rag_summarize(self, query: str) -> str:

        context_docs = self.retriever_docs(query)

        context = ""
        counter = 0
        for doc in context_docs:
            counter += 1
            context += f"【参考资料{counter}】: 参考资料：{doc.page_content} | 参考元数据：{doc.metadata}\n"

        return self._init_chain().invoke(
            {
                "input": query,
                "context": context,
            }
        )


if __name__ == '__main__':
    rag = RagSummarizeService()

    print(rag.rag_summarize("什么是线程？"))
