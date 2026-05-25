def _get_recent_interview_data(company: str, limit: int = 3) -> str:
    """读取该公司最近保存的面经数据（skill 搜索刚保存的文件）"""
    import json
    import os
    if not METADATA_FILE.exists():
        return ""
    try:
        with open(METADATA_FILE, "r", encoding="utf-8") as f:
            metadata = json.load(f)
        records = metadata.get("records", [])
        if company:
            records = [r for r in records if company in r.get("company", "")]
        if not records:
            return ""
        records = records[:limit]

        parts = []
        for r in records:
            md_path = r.get("md_path", "")
            if md_path and os.path.exists(md_path):
                try:
                    with open(md_path, "r", encoding="utf-8") as f:
                        parts.append(f.read())
                except Exception:
                    pass
            if len(parts) >= limit:
                break
        return "\n\n---\n\n".join(parts)
    except Exception:
        return ""



from typing import List, TypedDict
from langgraph.graph import StateGraph, END
from langgraph.checkpoint.memory import MemorySaver
from langchain_core.messages import AIMessage, HumanMessage, SystemMessage
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import PromptTemplate
from langchain_core.runnables import RunnableConfig
from agent.agent_tools import rag_summarize, search_interview_exp, get_local_interview_exp
from agent.interview_search_tools import get_local_interview_exp as get_local_interview_exp_raw
from agent.interview_search_tools import _check_local_exp, _parse_query, METADATA_FILE, INTERVIEW_EXP_DIR, MARKDOWN_DIR, simple_http_search
from model.factory import get_chat_model
from rag.rag_service import RagSummarizeService
from utils.prompt_loader import load_report_prompts, load_system_prompts, load_system_prompts2, load_pm_prompts


# ─────────────────────────────────────────
# State 定义
# ─────────────────────────────────────────
class InterviewState(TypedDict, total=False):
    """面试工作流状态"""
    target_company: str           # 目标公司
    target_position: str          # 目标岗位
    round: int                   # 当前轮次（0=初始化，1+=面试中）
    local_data: str              # 本地面经内容
    local_found: bool            # 本地是否有数据
    is_sufficient: bool          # 数据是否充足（代码判断）
    search_triggered: bool        # 是否已触发在线搜索
    merged_data: str             # 合并后的完整面经数据
    messages: List[dict]          # 对话历史（user/assistant）
    interview_questions: List[str]  # 记录面试问题
    current_input: str            # 当前用户输入
    current_output: str           # 当前 AI 输出


# ─────────────────────────────────────────
# 数据充足性判断阈值
# ─────────────────────────────────────────
SUFFICIENT_RECORD_COUNT = 3      # 最少需要 3 条面经记录
SUFFICIENT_CONTENT_CHARS = 1000  # 正文最少 1000 字
SUFFICIENT_QUESTION_COUNT = 5     # 问题最少 5 道


def _parse_sufficiency(local_result: str) -> dict:
    """
    解析本地数据，评估是否充足。
    返回 {'is_sufficient': bool, 'reason': str, 'stats': dict}
    """
    if not local_result or local_result.strip() == "":
        return {
            "is_sufficient": False,
            "reason": "本地无面经数据",
            "stats": {"record_count": 0, "content_chars": 0, "question_count": 0}
        }

    # 统计记录条数（"## 面经 N：" 出现次数）
    record_count = local_result.count("## 面经 ")

    # 统计正文字数（去掉 markdown 标记后的纯文本）
    import re
    text_only = re.sub(r'[#*|\-\[\]]', '', local_result)
    content_chars = len(text_only.strip())

    # 统计问题数量（"- " 列表项，粗略估算）
    question_count = local_result.count("\n  - ")

    # 综合判断
    is_sufficient = (
        record_count >= SUFFICIENT_RECORD_COUNT
        and content_chars >= SUFFICIENT_CONTENT_CHARS
        and question_count >= SUFFICIENT_QUESTION_COUNT
    )

    reasons = []
    if record_count < SUFFICIENT_RECORD_COUNT:
        reasons.append(f"记录数不足({record_count}<{SUFFICIENT_RECORD_COUNT})")
    if content_chars < SUFFICIENT_CONTENT_CHARS:
        reasons.append(f"内容过少({content_chars}<{SUFFICIENT_CONTENT_CHARS})")
    if question_count < SUFFICIENT_QUESTION_COUNT:
        reasons.append(f"问题不足({question_count}<{SUFFICIENT_QUESTION_COUNT})")
    if is_sufficient:
        reasons.append("数据充足")

    return {
        "is_sufficient": is_sufficient,
        "reason": "; ".join(reasons) if reasons else "未知",
        "stats": {
            "record_count": record_count,
            "content_chars": content_chars,
            "question_count": question_count
        }
    }


# ─────────────────────────────────────────
# 图节点
# ─────────────────────────────────────────

def node_check_local(state: InterviewState) -> InterviewState:
    """
    节点1：检查本地面经是否存在
    """
    import time
    t0 = time.time()
    company = state.get("target_company", "")
    print(f"[node_check_local] company={company}, t=0ms")
    check_result = _check_local_exp(company or state.get("current_input", ""))
    print(f"[node_check_local] done, local_found={bool(check_result)}, t={int((time.time()-t0)*1000)}ms")

    state["local_found"] = bool(check_result)
    return state


def node_fetch_local(state: InterviewState) -> InterviewState:
    """
    节点2：获取本地面经完整内容
    """
    import time
    t0 = time.time()
    company = state.get("target_company", "")
    topic = state.get("target_position", "")
    print(f"[node_fetch_local] company={company}, topic={topic}, t=0ms")

    try:
        local_data = get_local_interview_exp_raw(company=company, topic=topic)
        print(f"[node_fetch_local] get_local_interview_exp done, len={len(local_data)}, t={int((time.time()-t0)*1000)}ms")
        # 如果返回的是"暂无"提示，转为空
        if "暂无" in local_data or "没有" in local_data:
            state["local_data"] = ""
            state["local_found"] = False
            print(f"[node_fetch_local] no data found")
        else:
            state["local_data"] = local_data
            state["local_found"] = True
            print(f"[node_fetch_local] data found, chars={len(local_data)}")
    except Exception as e:
        print(f"[node_fetch_local] EXCEPTION: {e}")
        state["local_data"] = ""
        state["local_found"] = False

    return state


def node_eval_sufficient(state: InterviewState) -> InterviewState:
    """
    节点3：评估本地数据是否充足
    """
    result = _parse_sufficiency(state.get("local_data", ""))
    state["is_sufficient"] = result["is_sufficient"]
    return state


def node_search_online(state: InterviewState) -> InterviewState:
    """
    节点4：在线抓取面经补充数据
    """
    company = state.get("target_company", "")
    position = state.get("target_position", "")
    query = f"{company} {position}".strip()

    try:
        online_data = search_interview_exp.invoke({
            "query": query,
            "max_results": 5
        })
        # 合并到 local_data 后面
        existing = state.get("local_data", "")
        if existing:
            state["local_data"] = existing + "\n\n---\n\n" + str(online_data)
        else:
            state["local_data"] = str(online_data)
        state["search_triggered"] = True
    except Exception:
        # 在线搜索失败，保留本地数据
        state["search_triggered"] = False

    return state


def node_merge_data(state: InterviewState) -> InterviewState:
    """
    节点5：合并数据并添加引导语，供后续节点使用
    """
    data = state.get("local_data", "")
    company = state.get("target_company", "未知公司")
    position = state.get("target_position", "未知岗位")

    if data:
        merged = (
            f"【面试背景】目标公司：{company}，目标岗位：{position}。\n"
            f"以下是该公司的真实面经内容，包含真实面试问题、高频考点和正文摘要，"
            f"请优先从这些真实问题中出题。\n\n"
            f"{data}"
        )
    else:
        merged = (
            f"【面试背景】目标公司：{company}，目标岗位：{position}。\n"
            f"本地和在线均未找到面经数据，请基于你的专业知识进行常规面试。"
        )

    state["merged_data"] = merged
    return state


def node_interview(state: InterviewState) -> InterviewState:
    """
    节点6：LLM 出题/回复（核心节点）
    """
    import time
    t0 = time.time()
    merged_data = state.get("merged_data", "")
    messages = state.get("messages", [])
    current_input = state.get("current_input", "")
    round_ = state.get("round", 0)
    print(f"[node_interview] start, round={round_}, merged_len={len(merged_data)}, t=0ms")

    # 构建系统 prompt（产品经理使用专用 prompt）
    position = state.get("target_position", "")
    if position == "产品经理":
        system_prompt = load_pm_prompts()
    else:
        system_prompt = load_system_prompts()

    # 构建 LangChain 消息
    langchain_msgs = [SystemMessage(content=system_prompt)]

    # 第一轮注入面经数据
    if round_ == 0 and merged_data:
        langchain_msgs.append(
            SystemMessage(content=f"【面试准备】\n{merged_data}")
        )
        langchain_msgs.append(HumanMessage(content=current_input))
    else:
        for msg in messages:
            role = msg.get("role", "")
            content = msg.get("content", "")
            if role == "user":
                langchain_msgs.append(HumanMessage(content=content))
            elif role == "assistant":
                langchain_msgs.append(AIMessage(content=content))
        langchain_msgs.append(HumanMessage(content=current_input))

    print(f"[node_interview] calling chat_model, msg_count={len(langchain_msgs)}, t={int((time.time()-t0)*1000)}ms")

    # 调用 LLM
    try:
        response = get_chat_model().invoke(langchain_msgs)
        print(f"[node_interview] chat_model returned, t={int((time.time()-t0)*1000)}ms")
        output = response.content if hasattr(response, "content") else str(response)
    except Exception as e:
        print(f"[node_interview] LLM EXCEPTION: {e}")
        output = "抱歉，面试官暂时无法回复，请稍后重试。"

    # 记录问题
    questions = state.get("interview_questions", [])
    if "?" in output or "？" in output:
        questions.append(output)
    state["interview_questions"] = questions
    state["current_output"] = output

    # 更新对话历史
    messages = state.get("messages", [])
    messages.append({"role": "user", "content": current_input})
    messages.append({"role": "assistant", "content": output})
    state["messages"] = messages
    state["round"] = round_ + 1

    return state


# ─────────────────────────────────────────
# 构建图
# ─────────────────────────────────────────

def _build_interview_graph():
    """
    构建面试工作流图
    """

    def should_search(state: InterviewState) -> str:
        """边路由：本地数据是否充足"""
        if not state.get("local_found"):
            # 本地没有数据，直接去搜索
            return "search_online"
        if not state.get("is_sufficient"):
            # 本地数据不足，去搜索补充
            return "search_online"
        # 数据充足
        return "merge_data"

    # 建图
    builder = StateGraph(InterviewState)

    # 注册节点
    builder.add_node("check_local", node_check_local)
    builder.add_node("fetch_local", node_fetch_local)
    builder.add_node("eval_sufficient", node_eval_sufficient)
    builder.add_node("search_online", node_search_online)
    builder.add_node("merge_data", node_merge_data)
    builder.add_node("interview", node_interview)

    # 设置入口
    builder.set_entry_point("fetch_local")

    # 边
    builder.add_edge("fetch_local", "eval_sufficient")

    # 条件边：根据评估结果决定是否搜索
    builder.add_conditional_edges(
        "eval_sufficient",
        should_search,
        {
            "search_online": "search_online",
            "merge_data": "merge_data",
        }
    )

    # 搜索完后合并数据
    builder.add_edge("search_online", "merge_data")

    # 合并完进入面试
    builder.add_edge("merge_data", "interview")

    # 结束
    builder.add_edge("interview", END)

    # 编译（带内存 checkpoint，支持多轮对话状态保持）
    checkpointer = MemorySaver()
    graph = builder.compile(checkpointer=checkpointer)
    return graph


# 编译后的图（单例）
_interview_graph = _build_interview_graph()


# ─────────────────────────────────────────
# InterviewAssistantService
# ─────────────────────────────────────────

class InterviewAssistantService:
    def __init__(self):
        self.rag_service = RagSummarizeService()
        self.tools = [rag_summarize, search_interview_exp, get_local_interview_exp]
        self.report_chain = self._build_report_chain()

    @staticmethod
    def _build_report_chain():
        report_prompt = PromptTemplate.from_template(load_report_prompts())
        return report_prompt | get_chat_model() | StrOutputParser()

    def interview_chat(self, user_input: str, history: List[dict],
                       target_company: str = "", target_position: str = "",
                       thread_id: str = "default") -> str:
        """
        模拟面试主入口（使用 StateGraph 工作流）

        上下文一致性保证策略：
        1. 通过 MemorySaver checkpointer 恢复历史状态
        2. 比较当前目标公司/岗位 vs 上轮状态中的目标
           - 如果目标未变：后续轮次跳过数据检索，直接进入面试节点（复用已注入的面经数据）
           - 如果目标已变：强制重新走完整工作流（重新检索、评估、搜索），历史对话保留但出题依据更新
        3. 历史对话（messages）始终保留，不因目标切换而清空
        """
        from langgraph.checkpoint.memory import MemorySaver

        is_first_round = len(history) == 0
        current_company = target_company or ""
        current_position = target_position or ""

        # ── 读取 checkpointer 中的历史状态 ──
        checkpoint_config = {"configurable": {"thread_id": thread_id}}
        prev_state: InterviewState | None = None
        try:
            prev_state = _interview_graph.get_state(checkpoint_config).values
        except Exception:
            prev_state = None

        # ── 目标变更检测：当前 vs 上轮状态 ──
        prev_company = (prev_state or {}).get("target_company", "")
        prev_position = (prev_state or {}).get("target_position", "")
        target_changed = (
            (current_company and current_company != prev_company)
            or (current_position and current_position != prev_position)
        )

        # 初始化状态
        state: InterviewState = {
            "target_company": current_company,
            "target_position": current_position,
            "round": 0,
            "local_data": "",
            "local_found": False,
            "is_sufficient": False,
            "search_triggered": False,
            "merged_data": "",
            "messages": list(history),
            "interview_questions": [],
            "current_input": user_input,
            "current_output": "",
        }

        # ── 分支逻辑 ──
        if is_first_round or target_changed:
            # 情况A：第一轮 或 目标已变更 → 走完整工作流
            if target_changed and not is_first_round:
                # 目标切换：保留对话历史，但清空面经数据，重新检索
                state["messages"] = list(history)
                state["merged_data"] = ""
                # 日志改用 print，避免引入额外依赖
                print(
                    f"[interview_chat] 目标变更检测："
                    f"公司 {prev_company}→{current_company}, "
                    f"岗位 {prev_position}→{current_position}，重新检索面经"
                )

            # 完整工作流：fetch → eval → [search] → merge → interview
            result = _interview_graph.invoke(
                state,
                config=RunnableConfig(configurable={"thread_id": thread_id})
            )
        else:
            # 情况B：非第一轮且目标未变 → 跳过数据检索，直接面试
            state["round"] = len(history) // 2
            state["merged_data"] = (prev_state or {}).get("merged_data", "")
            state["interview_questions"] = (prev_state or {}).get("interview_questions", [])
            result = _interview_graph.invoke(
                state,
                config=RunnableConfig(configurable={"thread_id": thread_id})
            )

        return result.get("current_output", "")

    def qa_chat(self, user_input: str, history: List[dict]) -> str:
        """
        问答模式：智能路由 + skill 能力增强

        路由策略：
          1. 解析用户问题中是否包含公司/岗位关键词
          2. 有公司关键词 → 先查本地面经文件
             → 有数据：直接用 RAG 回答
             → 无数据：触发 skill search_interview_exp 实时搜索抓取
               → skill 搜索后：读取新保存的 .md 文件作为上下文，传给 LLM 回答
               （不走 RAG，因为 FAISS 索引不含刚保存的新文件）
          3. 无公司关键词 → 纯 RAG 回答（从 FAISS 向量库检索）
          4. 向量库未加载 → 降级为纯 LLM 回答
        """
        # ── 步骤1：解析问题中的公司/岗位关键词 ──
        parsed = _parse_query(user_input)
        company = parsed.get("company", "")

        # ── 步骤2a：有公司关键词，优先查本地面经 ──
        if company:
            local_data = get_local_interview_exp_raw(company=company)
            if local_data and "暂无" not in local_data and "没有" not in local_data:
                # 先让 LLM 判断本地数据是否真正回答了用户问题
                system_prompt = load_system_prompts2()
                check_messages = [SystemMessage(content=system_prompt)]
                check_messages.append(HumanMessage(
                    content=f"【本地面经知识库】\n{local_data}\n\n"
                            f"用户问题：{user_input}\n\n"
                            f"请判断：以上面经内容是否能回答用户的问题？"
                            f"如果能，请直接基于面经内容给出详细回答。"
                            f"如果不能（即面经内容不相关或缺少关键信息），请只回复\"SKILL_SEARCH\"，不要回复其他内容。"
                ))
                try:
                    check_resp = get_chat_model().invoke(check_messages)
                    check_text = check_resp.content if hasattr(check_resp, "content") else str(check_resp)
                except Exception:
                    check_text = ""

                if "SKILL_SEARCH" not in check_text:
                    # 本地数据够用，直接返回
                    return check_text
                # 本地数据不够用，继续触发 skill 搜索

            # ── 步骤2b：本地没有 或 数据不充分，触发 skill 实时搜索 ──
            # 走 skill 的完整流程：HTTP 搜索 → Playwright 抓取 → NLP 分析 → 保存
            skill_succeeded = False
            try:
                search_result = search_interview_exp.invoke({
                    "query": f"{company} {parsed.get('position', '')}".strip(),
                    "max_results": 5
                })
                skill_succeeded = True
            except Exception:
                search_result = ""

            # ── 步骤2c：skill 搜索结果直接作为上下文传给 LLM ──
            # 读取 skill 刚保存的面经文件（rag_summarize 用的是旧 FAISS 索引，不含新数据）
            recent_data = _get_recent_interview_data(company, limit=3)
            if skill_succeeded and recent_data:
                context = (
                    f"【以下是关于「{company}」的最新面经数据】\n"
                    f"{recent_data}\n\n"
                    f"请基于以上真实面经内容回答用户问题，如果面经中有相关问题请引用具体内容。"
                )
                system_prompt = load_system_prompts2()
                qa_messages = [SystemMessage(content=system_prompt)]
                if history:
                    for msg in history:
                        role = msg.get("role", "")
                        content = msg.get("content", "")
                        if role == "user":
                            qa_messages.append(HumanMessage(content=content))
                        elif role == "assistant":
                            qa_messages.append(AIMessage(content=content))
                qa_messages.append(HumanMessage(content=f"上下文：\n{context}\n\n用户问题：{user_input}"))
                try:
                    response = get_chat_model().invoke(qa_messages)
                    return response.content if hasattr(response, "content") else str(response)
                except Exception:
                    pass
            elif skill_succeeded and not recent_data:
                # skill 搜索成功但没保存数据，尝试轻量 HTTP 搜索兜底
                keyword = f"{company} {parsed.get('position', '')}".strip()
                try:
                    http_result = simple_http_search(keyword, max_results=5)
                    if "未找到" not in http_result and "出错" not in http_result:
                        system_prompt = load_system_prompts2()
                        qa_messages = [SystemMessage(content=system_prompt)]
                        if history:
                            for msg in history:
                                if msg.get("role") == "user":
                                    qa_messages.append(HumanMessage(content=msg.get("content", "")))
                                elif msg.get("role") == "assistant":
                                    qa_messages.append(AIMessage(content=msg.get("content", "")))
                        qa_messages.append(HumanMessage(
                            content=f"【牛客网搜索结果】\n{http_result}\n\n请基于以上面经内容回答用户问题。"
                        ))
                        response = get_chat_model().invoke(qa_messages)
                        return response.content if hasattr(response, "content") else str(response)
                except Exception:
                    pass

        # ── 步骤3：RAG 回答（从 FAISS 向量库检索） ──
        try:
            return self.rag_service.rag_summarize(user_input)
        except RuntimeError:
            pass

        # ── 步骤4：降级为纯 LLM 回答 ──
        system_prompt = load_system_prompts2()
        messages = [SystemMessage(content=system_prompt)]
        for msg in history:
            if msg.get("role") == "user":
                messages.append(HumanMessage(content=msg.get("content", "")))
            elif msg.get("role") == "assistant":
                messages.append(AIMessage(content=msg.get("content", "")))
        messages.append(HumanMessage(content=user_input))
        try:
            response = get_chat_model().invoke(messages)
            return response.content if hasattr(response, "content") else str(response)
        except Exception:
            return "抱歉，知识库尚未加载，请先点击左侧「加载/更新知识库」，然后重新提问。"

    def generate_report(self, interview_history: List[dict], interview_questions: List[str]) -> str:
        full_log = []
        for message in interview_history:
            role = "候选人" if message["role"] == "user" else "面试官"
            full_log.append(f"{role}：{message['content']}")

        questions_text = "\n".join([f"{idx + 1}. {q}" for idx, q in enumerate(interview_questions)])
        question_query = "；".join(interview_questions) if interview_questions else "本次面试问题"

        # 尝试从知识库检索参考资料，知识库未加载时跳过
        references = []
        try:
            docs = self.rag_service.retriever_docs(question_query)
            for idx, doc in enumerate(docs, start=1):
                references.append(f"【参考资料{idx}】{doc.page_content}")
        except RuntimeError:
            references.append("（知识库未加载，无法提供参考资料）")

        interview_log = (
            f"【本次面试问题】\n{questions_text}\n\n"
            f"【完整对话记录】\n{chr(10).join(full_log)}\n\n"
            f"【知识库参考】\n{chr(10).join(references)}"
        )

        return self.report_chain.invoke({"interview_log": interview_log})
