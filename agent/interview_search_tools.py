"""
面经搜索模块 — 基于 interview-prep-assistant skill
为 Agent 提供实时搜索并标准化保存面经的能力

本模块通过以下方式接入 interview-prep-assistant skill：
  1. searcher.py  — HTTP 搜索（UA池/随机延迟/重试/BeautifulSoup 解析）
  2. analyzer.py — 本地 NLP 分析（高频考点/问题分类/题库生成）
  3. 直接复用 skill 的 STOPWORDS、TECH_KEYWORDS 等词表

工作流程：
  用户请求 → Searcher.search() 搜索列表
           → Playwright 抓取详情（复用 skill 反爬增强）
           → InterviewAnalyzer 分析（NLP 提取考点/分类问题/检测难度）
           → 标准化保存（JSON + Markdown）
           → 返回格式化结果
"""
import json
import re
import os
import uuid
import logging
import time
import random
from datetime import datetime
from pathlib import Path
from typing import List, Dict, Optional

from langchain_core.tools import tool

logger = logging.getLogger(__name__)

# ─────────────────────────────────────────
# 路径配置
# ─────────────────────────────────────────
SKILL_ROOT = None  # 本地 Windows 开发用，Streamlit Cloud 不可用，已改为延迟导入兜底

def _get_project_data_dir():
    import sys
    import os
    # 动态获取项目根目录
    for root in [__file__, *sys.path]:
        try:
            proj = Path(root).resolve().parent
            if (proj / "app.py").exists() or (proj / "agent").exists():
                return proj / "data"
        except Exception:
            pass
    return Path(__file__).resolve().parent.parent / "data"

INTERVIEW_EXP_DIR = _get_project_data_dir() / "interview_exp"
MARKDOWN_DIR = _get_project_data_dir() / "interview_exp_md"
METADATA_FILE = INTERVIEW_EXP_DIR / "metadata.json"

INTERVIEW_EXP_DIR.mkdir(parents=True, exist_ok=True)
MARKDOWN_DIR.mkdir(parents=True, exist_ok=True)


# ─────────────────────────────────────────
# 加载 interview-prep-assistant skill
# ─────────────────────────────────────────
def _get_skill_searcher():
    """
    延迟导入 skill 的 Searcher 类。
    Searcher 内置反爬策略：
      - UA 池随机选择
      - 请求间隔随机延迟（min_delay ~ max_delay）
      - 失败自动重试（retry_times 次，指数退避）
      - requests.Session 保持连接

    Streamlit Cloud 环境下 skill 不可用，自动降级。
    """
    skill_root = Path(r"C:\Users\Admin\.claude\skills\interview-prep-assistant")
    if not skill_root.exists():
        return None
    import sys
    sys.path.insert(0, str(skill_root / "scripts"))
    try:
        from searcher import Searcher
        return Searcher()
    except ImportError as e:
        logger.warning(f"[InterviewSearch] 无法加载 skill Searcher: {e}")
        return None


def _get_skill_analyzer():
    """
    延迟导入 skill 的 InterviewAnalyzer 类。
    InterviewAnalyzer 内置 NLP 能力：
      - _tokenize(): 文本清洗 + 中英文分词 + 停用词过滤
      - extract_high_frequency_topics(): 词频统计，提取高频考点
      - classify_questions(): 技术/项目/行为/场景 四类分类
      - extract_question_bank(): 生成面试题库

    Streamlit Cloud 环境下 skill 不可用，自动降级。
    """
    skill_root = Path(r"C:\Users\Admin\.claude\skills\interview-prep-assistant")
    if not skill_root.exists():
        return None
    import sys
    sys.path.insert(0, str(skill_root / "scripts"))
    try:
        from analyzer import InterviewAnalyzer
        return InterviewAnalyzer()
    except ImportError as e:
        logger.warning(f"[InterviewSearch] 无法加载 skill InterviewAnalyzer: {e}")
        return None


def _ensure_skill_path():
    """确保 skill scripts 目录在 sys.path 最前面（本地开发用）"""
    import sys
    skill_root = Path(r"C:\Users\Admin\.claude\skills\interview-prep-assistant")
    if not skill_root.exists():
        return
    skill_path = str(skill_root / "scripts")
    if skill_path not in sys.path:
        sys.path.insert(0, skill_path)


# ─────────────────────────────────────────
# 反爬增强的 Playwright 抓取（复用 skill 设计）
# ─────────────────────────────────────────
def _pw_crawl_batch(urls: List[str]) -> List[Dict]:
    """
    用 Playwright 抓取面经详情，支持 skill 的反爬增强策略。
    增强点：
      - 浏览器上下文随机化（viewport、timezone）
      - 请求间隔抖动（0.5~2.5s 随机）
      - 失败重试（最多3次）
      - 多层兜底：__INITIAL_STATE__ JSON → CSS selector → page.content()
    """
    try:
        from playwright.sync_api import sync_playwright
    except ImportError:
        logger.warning("[InterviewSearch] Playwright 未安装")
        return []

    results = []
    for url in urls:
        content = _fetch_single_with_retry(url)
        if content:
            results.append(content)
    return results


def _fetch_single_with_retry(url: str, max_retries: int = 3) -> Optional[Dict]:
    """单条 URL 抓取，带重试"""
    for attempt in range(max_retries):
        try:
            from playwright.sync_api import sync_playwright
            with sync_playwright() as p:
                browser = p.chromium.launch(headless=True)
                # 随机化浏览器上下文，绕过指纹检测
                context = browser.contexts[0] if browser.contexts else browser.new_context(
                    viewport={"width": random.randint(1200, 1920), "height": random.randint(700, 1080)},
                    timezone_id="Asia/Shanghai",
                    locale="zh-CN",
                )
                page = context.new_page()
                page.goto(url, wait_until="networkidle", timeout=30000)
                # 随机抖动等待（skill 的 min_delay~max_delay 策略）
                time.sleep(random.uniform(0.5, 2.5))
                title = ""
                content = ""
                author = ""
                publish_time = ""

                # 优先从 __INITIAL_STATE__ JSON 提取（最快最准）
                state_json = page.evaluate("""() => {
                    try {
                        const s = window.__INITIAL_STATE__;
                        if (s) return JSON.stringify(s);
                    } catch(e) {}
                    return '';
                }""")

                if state_json:
                    try:
                        import json as _json
                        state = _json.loads(state_json)

                        def deep_find(obj, target_key, depth=0):
                            if depth > 30:
                                return None
                            if isinstance(obj, dict):
                                if target_key in obj and isinstance(obj[target_key], dict):
                                    return obj[target_key]
                                for v in obj.values():
                                    r = deep_find(v, target_key, depth + 1)
                                    if r:
                                        return r
                            elif isinstance(obj, list) and obj:
                                return deep_find(obj[0], target_key, depth + 1)
                            return None

                        cd_path = deep_find(state, "contentData")
                        if cd_path and isinstance(cd_path, dict):
                            title = cd_path.get("title") or page.title()
                            content = cd_path.get("content", "")
                            publish_time = cd_path.get("gmtCreate") or cd_path.get("updateTime", "")
                            author = cd_path.get("nickName") or cd_path.get("author", "")
                        elif not content:
                            cd_fallback = deep_find(state, "content")
                            if cd_fallback and isinstance(cd_fallback, dict):
                                content = cd_fallback.get("content", "")
                                title = cd_fallback.get("title") or title or page.title()
                                publish_time = cd_fallback.get("gmtCreate") or publish_time
                                author = cd_fallback.get("nickName") or author
                    except Exception as e:
                        logger.warning(f"[InterviewSearch] JSON 解析失败: {e}")

                # DOM 兜底
                if not content:
                    for sel in [".post-content", ".article-content", ".feed-detail-content",
                                "[class*=content]", "#content"]:
                        elem = page.query_selector(sel)
                        if elem:
                            text = elem.inner_text().strip()
                            if len(text) > 50:
                                content = text
                                break

                if not title:
                    for sel in ["h1", ".post-title", ".feed-detail-title"]:
                        elem = page.query_selector(sel)
                        if elem:
                            title = elem.inner_text().strip()
                            break
                    if not title:
                        title = page.title()

                browser.close()

                if content and len(content) > 20:
                    cleaned = _clean_html_content(content)
                    logger.info(f"[InterviewSearch] 成功: {title[:40]} ({len(cleaned)}字)")
                    return {
                        "url": url,
                        "title": title.strip() if title else "未知标题",
                        "content": cleaned,
                        "author": author or "匿名",
                        "publish_time": publish_time or "",
                        "platform": "niukewang",
                    }
        except Exception as e:
            logger.warning(f"[InterviewSearch] Playwright 异常 (尝试 {attempt+1}/{max_retries}): {url} - {e}")
            if attempt < max_retries - 1:
                time.sleep(2 ** attempt)  # 指数退避

    logger.warning(f"[InterviewSearch] 抓取失败（已达最大重试）: {url}")
    return None


# ─────────────────────────────────────────
# 分析结果包装器：调用 skill InterviewAnalyzer
# ─────────────────────────────────────────
def _analyze_single_record(content: str) -> Dict:
    """
    调用 skill InterviewAnalyzer 的三个 NLP 方法，替换原来的三个硬编码函数。
    返回格式与原 hardcoded 版本一致：
      - topics: List[str]（高频考点）
      - question_classified: Dict（technical/project/behavioral/scenario）
      - difficulty: str（easy/medium/hard）
    """
    analyzer = _get_skill_analyzer()
    if analyzer is None:
        # 降级：返回空结果，由 _normalize_record 兜底
        return {
            "topics": [],
            "question_classified": {"technical": [], "project": [], "behavioral": [], "scenario": []},
            "difficulty": "medium",
        }

    try:
        # skill analyzer 需要 batch 格式（list of dict）
        batch = [{"content": content, "title": ""}]
        topics = analyzer.extract_high_frequency_topics(batch)
        classified = analyzer.classify_questions(batch)
        difficulty = analyzer.analyze_difficulty(batch)

        # skill 返回的 difficulty 可能是列表或单个值，统一取第一个
        if isinstance(difficulty, list):
            difficulty = difficulty[0] if difficulty else "medium"
        if not isinstance(difficulty, str):
            difficulty = "medium"

        return {
            "topics": topics if isinstance(topics, list) else [],
            "question_classified": classified if isinstance(classified, dict) else {"technical": [], "project": [], "behavioral": [], "scenario": []},
            "difficulty": difficulty,
        }
    except Exception as e:
        logger.warning(f"[InterviewSearch] skill analyzer 调用失败，降级使用硬编码: {e}")
        return {
            "topics": [],
            "question_classified": {"technical": [], "project": [], "behavioral": [], "scenario": []},
            "difficulty": "medium",
        }


# ─────────────────────────────────────────
# 复用 skill 的 NLP 能力（问题提取）
# ─────────────────────────────────────────

def _extract_questions_from_content(content: str) -> List[str]:
    """
    从正文中提取面试问题。
    复用 skill analyzer 的正则模式 + 规则。
    """
    questions = []
    patterns = [
        r"(?:问|题目|题|面试题|问题|考察)[：:]\s*(.+?)(?=\n|$)",
        r"(\d+[.、](?:你会|你能|请说|讲一|描述|实现|手写|写一个|谈谈|说一|解释|比较|说说)[^。\n]{5,100})",
        r"(?:算法|代码|编程|手撕|系统设计)[：:]\s*(.+?)(?=\n|$)",
        r"(?:\*\*问|\*\*题目|\*\*面试题)[：:]\*\*\s*(.+?)(?=\n|$)",
        r"(?:[一-龥]{2,8}(?:原理|机制|思路|实现|用法|区别|作用)[^\n？。]{0,30}\?*)",
    ]
    for pattern in patterns:
        matches = re.findall(pattern, content, re.UNICODE)
        for m in matches:
            q = m.strip()
            if 5 < len(q) < 200:
                questions.append(q)

    seen = set()
    unique = []
    for q in questions:
        if q not in seen:
            seen.add(q)
            unique.append(q)
    return unique


def _clean_html_content(html_text: str) -> str:
    """清洗 HTML 内容"""
    if not html_text:
        return ""
    text = html_text
    text = re.sub(r"<pre[^>]*>.*?</pre>", "", text, flags=re.DOTALL | re.IGNORECASE)
    text = re.sub(r"<code[^>]*>.*?</code>", "", text, flags=re.DOTALL | re.IGNORECASE)
    text = re.sub(r"<script[^>]*>.*?</script>", "", text, flags=re.DOTALL | re.IGNORECASE)
    text = re.sub(r"<style[^>]*>.*?</style>", "", text, flags=re.DOTALL | re.IGNORECASE)
    text = re.sub(r"<[^>]+>", "", text)
    for entity, char in [("&nbsp;", " "), ("&lt;", "<"), ("&gt;", ">"), ("&amp;", "&"), ("&#\d+;", "")]:
        text = text.replace(entity, char)
    text = re.sub(r"\n{3,}", "\n\n", text)
    text = re.sub(r"[ \t]+", " ", text)
    return text.strip()


def _parse_query(query: str) -> Dict[str, str]:
    """从自然语言中解析公司名和岗位"""
    companies = [
        "字节跳动", "字节", "腾讯", "阿里", "阿里巴巴", "百度", "美团", "拼多多",
        "京东", "快手", "滴滴", "网易", "华为", "小米", "商汤", "旷视", "地平线",
        "携程", "哔哩哔哩", "B站", "小红书", "米哈游", "蚂蚁", "蚂蚁金服", "饿了么",
        "shein", "tiktok", "bytedance", "海康威视",
        "OPPO", "VIVO", "大疆", "蔚来", "理想汽车", "小鹏汽车", "雪球", "富途证券", "老虎证券",
    ]
    positions = [
        "后端", "前端", "算法", "测试", "运维", "客户端", "服务端", "全栈",
        "数据", "基础架构", "平台", "策略", "NLP", "CV", "推荐", "搜索",
        "java", "go", "python", "c++",
        "LLM", "Agent", "RAG", "大模型", "AI应用", "AI开发", "AI infra", "AIGC", "langchain",
        "Unity", "UE", "游戏",
    ]

    company, position, rest = None, None, query

    for c in companies:
        if c in query:
            company = c.replace("字节", "字节跳动").replace("阿里", "阿里巴巴")
            rest = rest.replace(c, "").strip()
            break

    for p in positions:
        if p.lower() in query.lower():
            position = p
            rest = rest.replace(p, "").strip()
            break

    return {"company": company, "position": position, "keyword": query.strip()}


def _normalize_record(raw: Dict) -> Dict:
    """将原始抓取数据转换为标准化格式"""
    content = _clean_html_content(raw.get("content", ""))
    questions = _extract_questions_from_content(content)
    analysis = _analyze_single_record(content)
    classified = analysis["question_classified"]
    topics = analysis["topics"]
    difficulty = analysis["difficulty"]

    # 识别公司
    company = None
    companies_in_text = re.findall(
        r"(字节跳动|字节|腾讯|阿里巴巴|阿里|百度|美团|拼多多|京东|快手|滴滴|网易|华为|小米|商汤|旷视|地平线|携程|哔哩哔哩|小红书|米哈游|OPPO|VIVO|大疆|蔚来|理想汽车|小鹏汽车|雪球|富途证券|老虎证券|蚂蚁)",
        raw.get("title", "") + content,
    )
    if companies_in_text:
        company = companies_in_text[0].replace("字节", "字节跳动").replace("阿里", "阿里巴巴")

    # 识别轮次
    rounds = re.findall(r"(一面|二面|三面|四面|五面|HR面|终面|笔试)", raw.get("title", "") + content)
    rounds = list(dict.fromkeys(rounds))

    return {
        "id": str(uuid.uuid4()),
        "company": company or "未知公司",
        "position": raw.get("position", "后端"),
        "platform": raw.get("platform", "niukewang"),
        "url": raw.get("url", ""),
        "title": raw.get("title", ""),
        "author": raw.get("author", "匿名"),
        "publish_time": raw.get("publish_time", ""),
        "crawl_time": datetime.now().isoformat(),
        "rounds": rounds,
        "questions": questions,
        "question_classified": classified,
        # 统一 topics 为字符串列表（skill analyzer 返回 dict 列表，提取 keyword）
        "topics": [t.get("keyword", str(t)) if isinstance(t, dict) else str(t) for t in topics] if topics else [],
        "difficulty": difficulty,
        "content": content[:3000],
        "tags": raw.get("tags", []),
        "extracted_questions_count": len(questions),
        "source_keywords": raw.get("search_keywords", []),
    }


def _save_standardized(record: Dict) -> str:
    """保存标准化记录到本地，返回保存路径"""
    company = record["company"]
    safe_company = re.sub(r"[^\w]", "_", company)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    uid = record["id"][:8]
    filename = f"{safe_company}_{timestamp}_{uid}.json"
    filepath = INTERVIEW_EXP_DIR / filename

    with open(filepath, "w", encoding="utf-8") as f:
        json.dump(record, f, ensure_ascii=False, indent=2)

    md_filename = filename.replace(".json", ".md")
    md_filepath = MARKDOWN_DIR / md_filename
    md_content = f"""# {record['title']}

**公司**: {record['company']} | **平台**: {record['platform']} | **难度**: {record['difficulty']}
**作者**: {record['author']} | **发布时间**: {record['publish_time']} | **抓取时间**: {record['crawl_time']}
**URL**: {record['url']}

## 面试信息
- **轮次**: {', '.join(record['rounds']) if record['rounds'] else '未知'}
- **提取问题数**: {record['extracted_questions_count']} 道

## 高频考点
{', '.join(record['topics']) if record['topics'] else '暂无'}

## 技术问题
{chr(10).join(f"- {q}" for q in record['question_classified'].get('technical', [])[:10]) or '暂无'}

## 项目问题
{chr(10).join(f"- {q}" for q in record['question_classified'].get('project', [])[:5]) or '暂无'}

## 行为问题
{chr(10).join(f"- {q}" for q in record['question_classified'].get('behavioral', [])[:5]) or '暂无'}

## 场景问题
{chr(10).join(f"- {q}" for q in record['question_classified'].get('scenario', [])[:5]) or '暂无'}

## 原文摘录
{record['content'][:1000]}
"""
    with open(md_filepath, "w", encoding="utf-8") as f:
        f.write(md_content)

    _update_metadata(record)
    _cleanup_old_records(company=company)

    return str(filepath)


# ─────────────────────────────────────────
# 清理策略
# ─────────────────────────────────────────
MAX_PER_COMPANY = 20


def _cleanup_old_records(company: str):
    """每个公司最多保留 MAX_PER_COMPANY 条，超出的旧记录自动清理"""
    if not METADATA_FILE.exists():
        return

    try:
        with open(METADATA_FILE, "r", encoding="utf-8") as f:
            metadata = json.load(f)
    except Exception:
        return

    records = metadata.get("records", [])
    company_records = [r for r in records if r.get("company") == company]

    if len(company_records) <= MAX_PER_COMPANY:
        return

    company_records.sort(key=lambda r: r.get("crawl_time", ""), reverse=True)
    to_keep = company_records[:MAX_PER_COMPANY]
    to_delete_ids = {r["id"] for r in company_records[MAX_PER_COMPANY:]}

    for r in records:
        if r["id"] in to_delete_ids:
            for path_key in ("json_path", "md_path"):
                p = r.get(path_key, "")
                if p and os.path.exists(p):
                    try:
                        os.remove(p)
                    except Exception:
                        pass

    metadata["records"] = [r for r in records if r.get("id") not in to_delete_ids]
    metadata["companies"][company]["count"] = len(to_keep)

    with open(METADATA_FILE, "w", encoding="utf-8") as f:
        json.dump(metadata, f, ensure_ascii=False, indent=2)


def _update_metadata(record: Dict):
    """更新元数据索引文件"""
    if METADATA_FILE.exists():
        with open(METADATA_FILE, "r", encoding="utf-8") as f:
            metadata = json.load(f)
    else:
        metadata = {"total": 0, "companies": {}, "records": []}

    metadata["total"] += 1
    company = record["company"]
    if company not in metadata["companies"]:
        metadata["companies"][company] = {"count": 0, "topics": [], "positions": []}
    metadata["companies"][company]["count"] += 1
    if record["position"] not in metadata["companies"][company]["positions"]:
        metadata["companies"][company]["positions"].append(record["position"])
    topics = metadata["companies"][company]["topics"]
    for t in record["topics"]:
        if t not in topics:
            topics.append(t)

    # 构造与 _save_standardized 一致的文件名格式
    safe_company = re.sub(r"[^\w]", "_", company)
    # 从 crawl_time 提取 timestamp（格式：2026-05-11T15:41:18 → 20260511_154118）
    crawl_ts = record.get("crawl_time", "")
    ts_part = crawl_ts.replace("-", "").replace(":", "").replace("T", "_")[:15]
    uid = record["id"][:8]
    base_name = f"{safe_company}_{ts_part}_{uid}"
    json_path = str(INTERVIEW_EXP_DIR / f"{base_name}.json")
    md_path = str(MARKDOWN_DIR / f"{base_name}.md")

    metadata["records"].insert(0, {
        "id": record["id"],
        "company": record["company"],
        "title": record["title"],
        "url": record["url"],
        "difficulty": record["difficulty"],
        "crawl_time": record["crawl_time"],
        "question_count": record["extracted_questions_count"],
        "json_path": json_path,
        "md_path": md_path,
    })
    metadata["records"] = metadata["records"][:500]

    with open(METADATA_FILE, "w", encoding="utf-8") as f:
        json.dump(metadata, f, ensure_ascii=False, indent=2)


def _check_local_exp(query: str) -> Optional[str]:
    """直接扫描 interview_exp_md/ 目录，找同名公司的 markdown 文件直接返回内容"""
    parsed = _parse_query(query)
    target_company = parsed["company"]

    try:
        if not MARKDOWN_DIR.exists():
            return None
        all_files = list(MARKDOWN_DIR.glob("*.md"))
        if not all_files:
            return None

        if target_company:
            matched_files = [f for f in all_files if target_company in f.name]
        else:
            matched_files = sorted(all_files, key=lambda f: f.stat().st_mtime, reverse=True)[:5]

        if not matched_files:
            return None

        lines = []
        for f in matched_files:
            try:
                with open(f, "r", encoding="utf-8") as fp:
                    content = fp.read()
                    if len(content) < 100:
                        continue
                    question_sections = content[max(0, content.find("## 技术问题")):max(0, content.find("## 原文摘录"))]
                    if question_sections.count("暂无") >= 3:
                        continue
                    lines.append(content)
            except Exception:
                continue

        if not lines:
            return None
        return "\n\n---\n\n".join(lines)
    except Exception:
        return None


def _format_results(records: List[Dict], query: str) -> str:
    """将搜索结果格式化为 LLM 可读的文本"""
    if not records:
        return f"未找到与「{query}」相关的面经，请尝试更换关键词。"

    lines = []
    lines.append(f"## 搜索「{query}」共找到 {len(records)} 条面经\n")

    for i, r in enumerate(records, 1):
        rounds_str = ", ".join(r["rounds"]) if r["rounds"] else "未知轮次"
        topics_str = ", ".join(r["topics"][:8]) if r["topics"] else "暂无"

        lines.append(f"### {i}. {r['title']}")
        lines.append(f"**公司**: {r['company']} | **难度**: {r['difficulty']} | **轮次**: {rounds_str}")
        lines.append(f"**平台**: {r['platform']} | **作者**: {r['author']} | **发布时间**: {r['publish_time']}")
        lines.append(f"**高频考点**: {topics_str}")

        tech_qs = r["question_classified"].get("technical", [])
        proj_qs = r["question_classified"].get("project", [])
        behav_qs = r["question_classified"].get("behavioral", [])

        if tech_qs:
            lines.append(f"**技术问题**（{len(tech_qs)}道）:")
            for q in tech_qs[:5]:
                lines.append(f"  - {q}")
        if proj_qs:
            lines.append(f"**项目问题**（{len(proj_qs)}道）:")
            for q in proj_qs[:3]:
                lines.append(f"  - {q}")
        if behav_qs:
            lines.append(f"**行为问题**（{len(behav_qs)}道）:")
            for q in behav_qs[:3]:
                lines.append(f"  - {q}")

        lines.append(f"**内容摘要**: {r['content'][:200]}...")
        lines.append("")

    lines.append(f"✅ 已将 {len(records)} 条面经标准化保存到本地知识库")
    return "\n".join(lines)


# ─────────────────────────────────────────
# 轻量级 HTTP 搜索（不依赖 skill 文件，Streamlit Cloud 可用）
# ─────────────────────────────────────────

def simple_http_search(query: str, max_results: int = 5) -> str:
    """
    通过牛客网搜索接口直接抓取面经列表和摘要。
    不依赖 Playwright 和 skill 文件，适合 Streamlit Cloud 环境。
    """
    from urllib.parse import quote, urlencode
    encoded_query = quote(query, safe='')
    params = urlencode({
        "type": "post",
        "query": query,
        "page": 1,
    })
    search_url = f"https://www.nowcoder.com/search?{params}"

    headers = {
        "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36",
        "Accept": "text/html,application/xhtml+xml,application/xml;q=0.9,*/*;q=0.8",
        "Accept-Language": "zh-CN,zh;q=0.9",
        "Accept-Encoding": "gzip, deflate, br",
        "Connection": "keep-alive",
        "Referer": "https://www.nowcoder.com/",
    }

    try:
        import urllib.request
        req = urllib.request.Request(search_url, headers=headers, method="GET")
        with urllib.request.urlopen(req, timeout=15) as resp:
            raw = resp.read()
            try:
                content = raw.decode("utf-8")
            except UnicodeDecodeError:
                content = raw.decode("gbk", errors="ignore")

        # 解析搜索结果（提取标题和摘要）
        items = re.findall(
            r'<h4[^>]*class="[^"]*job-news-title[^"]*"[^>]*>.*?<a[^>]+href="([^"]+)"[^>]*>([^<]+)</a>',
            content, re.DOTALL | re.IGNORECASE
        )
        abstracts = re.findall(r'class="[^"]*post-abstract[^"]*"[^>]*>([^<]+)', content, re.IGNORECASE)

        if not items:
            return f"未在牛客网找到「{query}」相关的面经，请尝试其他关键词。"

        lines = [f"## 牛客网搜索「{query}」共找到 {len(items)} 条面经\n"]
        for i, (link, title) in enumerate(items[:max_results], 1):
            title_clean = re.sub(r'<[^>]+>', '', title).strip()
            abstract_clean = abstracts[i-1].strip() if i-1 < len(abstracts) else ""
            lines.append(f"### {i}. {title_clean}")
            lines.append(f"**链接**: https://www.nowcoder.com{link}")
            if abstract_clean:
                lines.append(f"**摘要**: {abstract_clean}")
            lines.append("")
        return "\n".join(lines)

    except Exception as e:
        logger.warning(f"[simple_http_search] 搜索失败: {e}")
        return f"搜索「{query}」时出错：{str(e)}。请稍后重试。"


# ─────────────────────────────────────────
# Agent 工具定义
# ─────────────────────────────────────────

@tool(description="""
搜索真实面经并标准化保存到本地知识库。
基于 interview-prep-assistant skill 实现：

【数据来源】
  - 优先使用 skill 的 Searcher（HTTP UA池/随机延迟/重试 + __INITIAL_STATE__ JSON 解析）
  - 详情页使用 Playwright 渲染（支持反爬增强：浏览器上下文随机化 + 请求间隔抖动 + 失败重试）

【分析能力】
  - 问题提取：复用 skill analyzer 的多模式正则 + NLP 分词
  - 问题分类：复用 skill 的 TECH_KEYWORDS 词表，按技术类别（算法/数据库/系统设计/等）分组
  - 高频考点：词频统计 + 停用词过滤（复用 skill 的 STOPWORDS）
  - 难度检测：基于关键词命中（hard/medium/easy）
  - 格式化输出：标准化 JSON + Markdown 保存

【自动清理】
  - 每个公司最多保留 20 条，超出自动删除最旧的记录（同步清理磁盘文件和 metadata.json）

输入参数：
  - query: 搜索查询（如"美团后端实习"、"字节跳动算法"等）
  - max_results: 最大抓取数量（默认5条，最多20条）
""")
def search_interview_exp(query: str, max_results: int = 5) -> str:
    max_results = min(max(max_results, 1), 20)
    parsed = _parse_query(query)
    keyword = parsed["keyword"]

    logger.info(f"[InterviewSearch] 搜索面经: keyword={keyword}, max_results={max_results}")

    try:
        # ── 0. 先检查本地是否已有 ──
        local_result = _check_local_exp(query)
        if local_result:
            logger.info(f"[InterviewSearch] 本地已有「{keyword}」相关面经，直接返回")
            return local_result + "\n\n💡 如需最新面经，可先删除 `data/interview_exp/metadata.json` 后重试。"

        # ── 1. 搜索列表 ──
        # 优先用 skill 的 Searcher（带完整反爬）
        searcher = _get_skill_searcher()
        search_results = []
        if searcher:
            try:
                search_results = searcher.search(keywords=[keyword], platforms=["niukewang"])
            except Exception as e:
                logger.warning(f"[InterviewSearch] skill Searcher 失败: {e}，使用备用方案")
            if search_results:
                search_results = search_results[:max_results]

        if not search_results:
            # ── 备用：直接构造 URL 列表 ──
            from urllib.parse import quote
            encoded_keyword = quote(keyword)
            search_url = f"https://www.nowcoder.com/search?type=post&query={encoded_keyword}"
            logger.info(f"[InterviewSearch] 备用搜索: {search_url}")
            search_results = [{"url": search_url, "title": keyword, "platform": "niukewang"}]

        # ── 2. Playwright 抓取详情 ──
        urls = [r["url"] for r in search_results]
        crawled = _pw_crawl_batch(urls)
        if not crawled:
            return (f"搜索到 {len(search_results)} 条面经，但抓取全部失败。"
                    "可能是网络问题或牛客网反爬，请稍后重试。")
        logger.info(f"[InterviewSearch] 抓取完成 {len(crawled)} 条")
        # ── 3. 标准化处理 ──
        normalized_records = []
        for raw in crawled:
            try:
                record = _normalize_record(raw)
                record["source_keywords"] = [keyword]
                save_path = _save_standardized(record)
                record["saved_to"] = save_path
                normalized_records.append(record)
                logger.info(f"[InterviewSearch] 保存: {record['title'][:30]}")
            except Exception as e:
                logger.warning(f"[InterviewSearch] 保存失败: {e}")

        if not normalized_records:
            return "面经抓取成功，但处理/保存失败。"

        return _format_results(normalized_records, query)

    except Exception as e:
        logger.error(f"[InterviewSearch] 搜索失败: {e}", exc_info=True)
        return f"面经搜索过程中出错：{str(e)}。请检查网络或稍后重试。"


def get_local_interview_exp(company: str = "", topic: str = "") -> str:
    """查询本地已保存的面经，返回完整内容"""
    import time, logging as _log
    _log.basicConfig(level=_log.INFO, format='%(message)s')
    _l = _log.getLogger("interview_search")
    t0 = time.time()
    _l.info(f"[get_local_interview_exp] BEGIN: company={repr(company)}, topic={repr(topic)}")

    if not METADATA_FILE.exists():
        _l.info(f"[get_local_interview_exp] metadata not found, t={time.time()-t0:.3f}s")
        return "本地知识库暂无面经数据，请先使用 search_interview_exp 搜索并保存面经。"

    _l.info(f"[get_local_interview_exp] reading metadata...")
    with open(METADATA_FILE, "r", encoding="utf-8") as f:
        metadata = json.load(f)
    _l.info(f"[get_local_interview_exp] metadata loaded, total={metadata.get('total',0)}, t={time.time()-t0:.3f}s")

    records = metadata.get("records", [])
    if not records:
        return "本地知识库暂无面经数据。"

    if company:
        before = len(records)
        records = [r for r in records if company in r.get("company", "")]
        _l.info(f"[get_local_interview_exp] company filter: {before} -> {len(records)}, t={time.time()-t0:.3f}s")

    if not records:
        _l.info(f"[get_local_interview_exp] no records for company={company}, t={time.time()-t0:.3f}s")
        return f"本地知识库中暂无「{company}」相关的面经。"

    if topic:
        _l.info(f"[get_local_interview_exp] applying topic filter: {topic}")
        filtered = []
        for r in records:
            r_path = r.get("json_path", "")
            if os.path.exists(r_path):
                try:
                    with open(r_path, "r", encoding="utf-8") as f:
                        data = json.load(f)
                    content = " ".join(data.get("topics", [])) + " " + " ".join(
                        data.get("questions", []) + [q for qs in data.get("question_classified", {}).values() for q in qs]
                    )
                    if topic in content:
                        filtered.append(r)
                except Exception:
                    continue
        records = filtered or records
        _l.info(f"[get_local_interview_exp] topic filter done: {len(records)}, t={time.time()-t0:.3f}s")

    if not records:
        return f"本地知识库中暂无「{company}」相关的面经。"

    output_parts = []
    display_records = records[:5]
    _l.info(f"[get_local_interview_exp] reading {len(display_records)} records, t={time.time()-t0:.3f}s")

    for idx, r in enumerate(display_records, 1):
        _l.info(f"[get_local_interview_exp] reading record {idx}: {r.get('json_path','')}")
        r_path = r.get("json_path", "")
        if not os.path.exists(r_path):
            _l.warning(f"[get_local_interview_exp] SKIP no file: {r_path}")
            continue
        try:
            with open(r_path, "r", encoding="utf-8") as f:
                data = json.load(f)
        except Exception as e:
            _l.warning(f"[get_local_interview_exp] ERROR reading json: {e}")
            continue

        lines = []
        lines.append(f"## 面经 {idx}：{data.get('title', '未知标题')}")
        lines.append(f"**公司**: {data.get('company', '未知公司')} | **难度**: {data.get('difficulty', '未知')} | **轮次**: {', '.join(data.get('rounds', [])) or '未知'}")
        lines.append(f"**平台**: {data.get('platform', '未知')} | **发布**: {data.get('publish_time', '未知')}")
        lines.append("")

        topics = data.get("topics", [])
        if topics:
            # topics 可能是字符串列表（旧记录）或字典列表（新记录，skill analyzer返回格式）
            if topics and isinstance(topics[0], dict):
                topic_names = [t.get("keyword", str(t)) for t in topics[:15]]
            else:
                topic_names = [str(t) for t in topics[:15]]
            lines.append(f"**高频考点**（{len(topics)}个）: {', '.join(topic_names)}")
            lines.append("")

        classified = data.get("question_classified", {})
        tech_qs = classified.get("technical", []) or data.get("questions", [])
        if tech_qs:
            lines.append(f"**技术问题**（{len(tech_qs)}道）:")
            for q in tech_qs[:8]:
                lines.append(f"  - {q}")
            lines.append("")

        proj_qs = classified.get("project", [])
        if proj_qs:
            lines.append(f"**项目问题**（{len(proj_qs)}道）:")
            for q in proj_qs[:5]:
                lines.append(f"  - {q}")
            lines.append("")

        behav_qs = classified.get("behavioral", [])
        if behav_qs:
            lines.append(f"**行为问题**（{len(behav_qs)}道）:")
            for q in behav_qs[:5]:
                lines.append(f"  - {q}")
            lines.append("")

        scen_qs = classified.get("scenario", [])
        if scen_qs:
            lines.append(f"**场景问题**（{len(scen_qs)}道）:")
            for q in scen_qs[:5]:
                lines.append(f"  - {q}")
            lines.append("")

        content = data.get("content", "").strip()
        if content:
            lines.append(f"**正文摘要**: {content[:1500]}...")
            lines.append("")

        output_parts.append("\n".join(lines))
        _l.info(f"[get_local_interview_exp] record {idx} done, t={time.time()-t0:.3f}s")

    if not output_parts:
        _l.info(f"[get_local_interview_exp] no output parts, t={time.time()-t0:.3f}s")
        return f"本地知识库中暂无「{company}」相关的面经。"

    total_hint = f"（共找到 {len(records)} 条，展示前 {len(display_records)} 条最相关的）"
    header = f"## 本地面经知识库 {total_hint}\n"
    header += f"以下是「{company or '全部'}」相关的真实面经内容，请结合这些问题和考点出题。\n"
    result = header + "\n\n".join(output_parts)
    _l.info(f"[get_local_interview_exp] DONE, result_len={len(result)}, t={time.time()-t0:.3f}s")
    return result
