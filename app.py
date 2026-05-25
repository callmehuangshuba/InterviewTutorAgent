import os
import uuid
import re
import streamlit as st

from agent.interview_assistant_service import InterviewAssistantService
from agent.agent_tools import get_city, get_weather
from rag.vector_store import VectorStoreService
from utils.user_history_store import load_user_state, save_user_state

# 浏览器标题
st.set_page_config(page_title="基于RAG与Agent的多模态面试辅导助手(扁平版)", page_icon="💼", layout="wide")

# ── 密码保护 ─────────────────────────────────────────
def _check_password():
    """检查是否设置了密码保护，并验证登录"""
    try:
        app_password = st.secrets.get("APP_PASSWORD", "")
    except Exception:
        app_password = ""
    if not app_password:
        return True  # 未设置密码，跳过验证

    if st.session_state.get("password_verified", False):
        return True  # 已验证过

    st.markdown("🔒 **请先登录**")
    with st.form("login_form", clear_on_submit=True):
        pwd = st.text_input("请输入访问密码", type="password")
        submitted = st.form_submit_button("进入")
        if submitted:
            if pwd == app_password:
                st.session_state.password_verified = True
                st.rerun()
            else:
                st.error("密码错误，请重试")
    return False

if not _check_password():
    st.stop()
# 一级标题
st.title("💼 基于RAG与Agent的多模态面试辅导助手")

#用户id
if "current_user_id" not in st.session_state:
    st.session_state.current_user_id = f"guest_{uuid.uuid4().hex[:8]}"
#模拟面试 历史对话
if "interview_history" not in st.session_state:
    st.session_state.interview_history = []
# 问答历史
if "qa_history" not in st.session_state:
    st.session_state.qa_history = []
# 面试问题列表
if "interview_questions" not in st.session_state:
    st.session_state.interview_questions = []
# 面试状态标志
if "interview_started" not in st.session_state:
    st.session_state.interview_started = False
if "interview_finished" not in st.session_state:
    st.session_state.interview_finished = False
#  面试报告
if "interview_report" not in st.session_state:
    st.session_state.interview_report = ""
# 目标公司和岗位
if "target_company" not in st.session_state:
    st.session_state.target_company = ""
if "target_position" not in st.session_state:
    st.session_state.target_position = ""

# 将用户的当前会话状态保存到持久化存储中   utils/user_history_store.py
def persist_state():
    save_user_state(
        st.session_state.current_user_id,
        {
            "interview_history": st.session_state.interview_history,
            "qa_history": st.session_state.qa_history,
            "interview_questions": st.session_state.interview_questions,
            "interview_started": st.session_state.interview_started,
            "interview_finished": st.session_state.interview_finished,
            "interview_report": st.session_state.interview_report,
            "target_company": st.session_state.target_company,
            "target_position": st.session_state.target_position,
        },
    )

# 在Streamlit界面中渲染聊天记录
def render_chat_history(messages):
    for m in messages:
        with st.chat_message(m["role"]):
            st.markdown(m["content"])


def auto_scroll_bottom():
    """注入 JS 让聊天容器自动滚动到底部"""
    st.markdown(
        """
        <script>
        const scrollToBottom = () => {
            const containers = window.parent.document.querySelectorAll('[data-testid="stChatMessageContainer"]');
            if (containers.length > 0) {
                containers[containers.length - 1].scrollIntoView({ behavior: "smooth", block: "end" });
            }
        };
        setTimeout(scrollToBottom, 100);
        </script>
        """,
        unsafe_allow_html=True,
    )


#根据天气信息生成面试穿衣和出行建议
def generate_life_advice(weather_text: str) -> tuple[str, str]:
    if not weather_text:
        return "穿衣建议：暂无", "出行提醒：暂无"

    clothing = "穿衣建议：常规穿搭即可。"
    travel = "出行提醒：保持出行节奏，注意补水。"

    temp_match = re.search(r"气温\s*([\-]?\d+)", weather_text)
    temp = int(temp_match.group(1)) if temp_match else None
    lower_weather = weather_text.lower()

    if temp is not None:
        if temp <= 5:
            clothing = "面试穿衣建议：天气偏冷，建议厚外套/羽绒服+衬衫+加绒西装裤，整洁又干练，让面试官眼前一新！"
        elif temp <= 15:
            clothing = "面试穿衣建议：建议风衣+衬衫+厚西装裤，保暖不臃肿，你就是面试场上最靓的崽！"
        elif temp <= 26:
            clothing = "面试穿衣建议：温度舒适，建议穿白衬衫+西装裤，更得体哦~"
        else:
            clothing = "面试穿衣建议：天气较热，建议纯色短袖衬衫+垂感/直筒西装裤，显出你的重视！"

    if any(k in lower_weather for k in ["雨", "雷", "阵雨", "暴雨"]):
        travel = "出行提醒：可能降雨，建议带伞，注意路滑和交通安全。"
    elif any(k in lower_weather for k in ["雪", "冰"]):
        travel = "出行提醒：可能有雨雪结冰，建议减速慢行，注意防滑。"
    elif any(k in lower_weather for k in ["雾", "霾"]):
        travel = "出行提醒：能见度或空气质量一般，建议佩戴口罩并减少久留户外。"
    elif any(k in lower_weather for k in ["大风", "风"]):
        travel = "出行提醒：风力较大，注意高空坠物，骑行请减速。"

    return clothing, travel


def get_sidebar_weather_info() -> tuple[str, str]:
    try:
        city = get_city.invoke({})
    except Exception:
        city = "未知城市"

    try:
        weather_text = get_weather.invoke({"city": city})
    except Exception:
        weather_text = "天气获取失败，请稍后重试。"
    return str(city), str(weather_text)


# 首次进入加载用户数据（只在当前用户未显式切换时执行）
if "user_state_loaded" not in st.session_state:
    loaded = load_user_state(st.session_state.current_user_id)
    st.session_state.interview_history = loaded.get("interview_history", [])
    st.session_state.qa_history = loaded.get("qa_history", [])
    st.session_state.interview_questions = loaded.get("interview_questions", [])
    st.session_state.interview_started = loaded.get("interview_started", False)
    st.session_state.interview_finished = loaded.get("interview_finished", False)
    st.session_state.interview_report = loaded.get("interview_report", "")
    st.session_state.target_company = loaded.get("target_company", "")
    st.session_state.target_position = loaded.get("target_position", "")
    st.session_state.user_state_loaded = True
    os.environ["CURRENT_USER_ID"] = st.session_state.current_user_id


# 启动诊断：检查关键配置是否正常
import os
diagnostic_errors = []
data_dir = os.path.join(os.path.dirname(__file__), "data")
for fname in ["hr_questions.txt", "tech_knowledge.txt"]:
    fpath = os.path.join(data_dir, fname)
    if not os.path.exists(fpath):
        diagnostic_errors.append(f"缺少文件: {fname}")
api_key = os.environ.get("DASHSCOPE_API_KEY", "")
if not api_key:
    diagnostic_errors.append("缺少 DASHSCOPE_API_KEY，请检查 Streamlit Secrets 中是否配置了 DASHSCOPE_API_KEY")
if diagnostic_errors:
    for err in diagnostic_errors:
        st.error(f"[启动检查] {err}")
    st.stop()

# 延迟初始化 service，避免在 app 加载时就调用 API
service = None

def get_service():
    global service
    if service is None:
        service = InterviewAssistantService()
    return service

st.sidebar.header("用户管理")
user_id_input = st.sidebar.text_input("用户 ID", value=st.session_state.current_user_id)
if st.sidebar.button("切换/加载用户", use_container_width=True):
    target_user_id = user_id_input.strip() or "guest"
    loaded = load_user_state(target_user_id)
    st.session_state.current_user_id = target_user_id
    st.session_state.interview_history = loaded.get("interview_history", [])
    st.session_state.qa_history = loaded.get("qa_history", [])
    st.session_state.interview_questions = loaded.get("interview_questions", [])
    st.session_state.interview_started = loaded.get("interview_started", False)
    st.session_state.interview_finished = loaded.get("interview_finished", False)
    st.session_state.interview_report = loaded.get("interview_report", "")
    st.session_state.target_company = loaded.get("target_company", "")
    st.session_state.target_position = loaded.get("target_position", "")
    os.environ["CURRENT_USER_ID"] = target_user_id
    st.sidebar.success(f"已加载用户：{target_user_id}")
    st.rerun()

st.sidebar.header("知识库管理")
st.sidebar.write("首次使用建议先加载知识库。")
if st.sidebar.button("加载/更新知识库", use_container_width=True):
    with st.sidebar:
        with st.spinner("正在加载知识库（首次可能需要10-30秒）..."):
            try:
                vs = VectorStoreService()
                vs.load_document()
                st.session_state.vector_store = vs
                st.success("知识库加载完成！可以开始使用了。")
            except Exception as e:
                st.error(f"加载失败：{e}")

st.sidebar.header("天气与出行建议")
city_name, weather_text = get_sidebar_weather_info()
dress_advice, travel_advice = generate_life_advice(weather_text)
st.sidebar.caption(f"当前城市：{city_name}")
st.sidebar.caption(f"实时天气：{weather_text}")
st.sidebar.caption(dress_advice)
st.sidebar.caption(travel_advice)


st.sidebar.header("模式切换")
mode = st.sidebar.radio("请选择模式", ["问答模式", "模拟面试"])

if mode == "问答模式":
    st.subheader("问答模式")
    st.caption("用户提问，模型结合知识库与自身知识进行回答。")

    if st.button("清空问答历史", use_container_width=False):
        st.session_state.qa_history = []
        persist_state()
        st.rerun()

    render_chat_history(st.session_state.qa_history)
    auto_scroll_bottom()

    question = st.chat_input("请输入你想问的问题...")
    if question:
        st.session_state.qa_history.append({"role": "user", "content": question})
        # 立即显示用户消息
        st.chat_message("user").markdown(question)
        # spinner 单独显示在下面
        thinking_ph = st.empty()
        with thinking_ph:
            with st.spinner("🤔 AI 正在思考中，请稍候..."):
                answer = get_service().qa_chat(question, st.session_state.qa_history)
        thinking_ph.empty()
        if not (answer or "").strip():
            answer = "抱歉，我这次没有成功生成回答。请重试一次，或先点击左侧「加载/更新知识库」后再提问。"
        st.session_state.qa_history.append({"role": "assistant", "content": answer})
        persist_state()
        st.rerun()
else:
    st.subheader("模拟面试模式")
    st.caption("模型将基于知识库和真实面经模拟面试官提问。")

    # 目标公司与岗位选择
    with st.expander("🎯 设置面试目标（可选）", expanded=True):
        col_c, col_p = st.columns(2)

        # 预设公司列表 + 自定义选项
        preset_companies = [
            "", "字节跳动", "腾讯", "阿里巴巴", "美团", "快手", "百度",
            "拼多多", "京东", "网易", "华为", "小米", "滴滴", "滴滴出行",
            "哔哩哔哩", "小红书", "米哈游", "蚂蚁集团", "商汤科技", "旷视科技",
            "地平线机器人", "SHEIN", "携程", "雪球", "富途证券", "老虎证券",
            "OPPO", "VIVO", "大疆", "蔚来", "理想汽车", "小鹏汽车",
        ]
        is_custom_company = st.session_state.target_company not in preset_companies and st.session_state.target_company != ""

        # 预设岗位列表 + 自定义选项
        preset_positions = [
            "", "后端", "前端", "算法", "测试", "运维", "客户端", "数据",
            "基础架构", "LLM应用开发", "Agent/AI应用开发", "AI Infra",
            "大模型工程", "RAG开发", "NLP算法", "CV算法", "推荐算法",
            "搜索算法", "服务端", "全栈", "平台开发", "安全", "DBA",
            "游戏客户端", "游戏服务端", "Unity开发", "UE开发",
            "产品经理",
        ]
        is_custom_position = st.session_state.target_position not in preset_positions and st.session_state.target_position != ""

        with col_c:
            company_options = preset_companies + (["自定义"] if is_custom_company else ["自定义"])
            default_idx = company_options.index(st.session_state.target_company) if st.session_state.target_company in company_options else 0
            sel_c = st.selectbox("目标公司", company_options, index=default_idx)
            if sel_c == "自定义":
                custom_c = st.text_input("请输入自定义公司名称", value=st.session_state.target_company if is_custom_company else "", placeholder="例如：创业公司名称")
                st.session_state.target_company = custom_c.strip()
            else:
                st.session_state.target_company = sel_c

        with col_p:
            position_options = preset_positions + (["自定义"] if is_custom_position else ["自定义"])
            default_idx_p = position_options.index(st.session_state.target_position) if st.session_state.target_position in position_options else 0
            sel_p = st.selectbox("目标岗位", position_options, index=default_idx_p)
            if sel_p == "自定义":
                custom_p = st.text_input("请输入自定义岗位名称", value=st.session_state.target_position if is_custom_position else "", placeholder="例如：Agent开发工程师")
                st.session_state.target_position = custom_p.strip()
            else:
                st.session_state.target_position = sel_p

        if st.session_state.target_company or st.session_state.target_position:
            target_tip = []
            if st.session_state.target_company:
                target_tip.append(f"公司：{st.session_state.target_company}")
            if st.session_state.target_position:
                target_tip.append(f"岗位：{st.session_state.target_position}")
            st.success(" | ".join(target_tip) + " — 面试官将优先参考该公司的真实面经出题")

    col1, col2, col3 = st.columns([1, 1, 2])
    with col1:
        if st.button("开始/重置面试", use_container_width=True):
            st.session_state.interview_history = []
            st.session_state.interview_questions = []
            st.session_state.interview_report = ""
            st.session_state.interview_started = True
            st.session_state.interview_finished = False
            persist_state()

            with st.spinner("🤖 面试官正在准备中..."):
                first_question = get_service().interview_chat(
                    "请开始本次面试，先简单寒暄并提出第一个问题。",
                    st.session_state.interview_history,
                    target_company=st.session_state.target_company,
                    target_position=st.session_state.target_position,
                    thread_id=st.session_state.current_user_id,
                )

            st.session_state.interview_history.append({"role": "assistant", "content": first_question})
            if "?" in first_question or "？" in first_question:
                st.session_state.interview_questions.append(first_question)
            persist_state()
            st.rerun()

    with col2:
        if st.button("结束本次面试", use_container_width=True):
            st.session_state.interview_finished = True
            persist_state()
            st.rerun()

    with col3:
        st.write("当前状态：", "已结束" if st.session_state.interview_finished else "进行中")
    #显示完整的面试对话记录
    render_chat_history(st.session_state.interview_history)
    auto_scroll_bottom()

    if st.session_state.interview_started and not st.session_state.interview_finished:
        user_reply = st.chat_input("请输入你的回答...")
        if user_reply:
            st.session_state.interview_history.append({"role": "user", "content": user_reply})
            # 立即显示用户消息
            st.chat_message("user").markdown(user_reply)
            # spinner 单独显示在下面
            thinking_ph = st.empty()
            with thinking_ph:
                with st.spinner("🤔 面试官正在分析你的回答并准备下一个问题，请稍候..."):
                    interviewer_reply = get_service().interview_chat(
                        user_reply,
                        st.session_state.interview_history,
                        target_company=st.session_state.target_company,
                        target_position=st.session_state.target_position,
                        thread_id=st.session_state.current_user_id,
                    )
            thinking_ph.empty()
            st.session_state.interview_history.append({"role": "assistant", "content": interviewer_reply})
            if "?" in interviewer_reply or "？" in interviewer_reply:
                st.session_state.interview_questions.append(interviewer_reply)
            persist_state()
            st.rerun()

    if st.session_state.interview_finished:
        want_report = st.checkbox("我希望生成本次面试报告", value=False)
        if want_report and st.button("生成面试报告", use_container_width=True):
            with st.spinner("正在生成报告..."):
                st.session_state.interview_report = get_service().generate_report(
                    st.session_state.interview_history,
                    st.session_state.interview_questions,
                )
            persist_state()
            st.rerun()

        if st.session_state.interview_report:
            st.markdown("### 面试报告")
            st.markdown(st.session_state.interview_report)
