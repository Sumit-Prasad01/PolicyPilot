import os
from pathlib import Path
from typing import Literal

from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_tavily import TavilySearch
from langgraph.graph import END, START, StateGraph

from app.core.config import get_settings
from app.core.logger import get_logger
from app.rag.state import AgentState, EvidenceGrade, RouteDecision
from app.rag.vector_store import get_retriever

logger = get_logger(__name__)
settings = get_settings()


_llm = None
_web_search = None


def llm():
    global _llm
    if _llm is None:
        if not settings.gemini_api_key:
            raise RuntimeError("GEMINI_API_KEY is missing")
        _llm = ChatGoogleGenerativeAI(
            model=settings.gemini_model,
            temperature=0,
            api_key=settings.gemini_api_key,
            retries=1,
            request_timeout=30,
        )

    return _llm


def web_search_tool():
    global _web_search
    if _web_search is None:
        if not settings.tavily_api_key:
            raise RuntimeError("TAVILY_API_KEY is missing")

        # TavilySearch reads its key from the environment, not from a constructor arg.
        os.environ["TAVILY_API_KEY"] = settings.tavily_api_key

        _web_search = TavilySearch(
            max_results=5,
            topic="general",
            include_answer=True,
            include_raw_content=False,
        )

    return _web_search


def add_trace(state: AgentState, message: str):
    return [*state.get("trace", []), message]


def message_text(message) -> str:
    """Normalize Gemini/LangChain message content to plain text.

    langchain-google-genai 4.x returns a list of content blocks by default
    (for example ``[{"type": "text", "text": "OK"}]``), while older versions
    return a plain string.
    """
    content = getattr(message, "content", message)

    if isinstance(content, str):
        return content.strip()

    if isinstance(content, dict):
        text = content.get("text")
        return text.strip() if isinstance(text, str) else str(content).strip()

    if isinstance(content, list):
        parts = []
        for block in content:
            if isinstance(block, str):
                parts.append(block)
                continue

            if isinstance(block, dict):
                if block.get("type") in {"thinking", "reasoning"}:
                    continue
                text = block.get("text")
            else:
                if getattr(block, "type", None) in {"thinking", "reasoning"}:
                    continue
                text = getattr(block, "text", None)

            if isinstance(text, str):
                parts.append(text)

        return "".join(parts).strip()

    return str(content).strip()


def route_question(state: AgentState):
    try:
        router = llm().with_structured_output(RouteDecision)
        decision = router.invoke(f"""
            You route messages for an enterprise HR policy and employee support assistant.
            Use kb for questions about company HR policies, leave, holidays, benefits, payroll,
            remote work, attendance, onboarding, performance, expenses, travel, conduct, or employee support.
            Use direct only for greetings, thanks, or casual chat that needs no company knowledge.
            Question: {state['question']}
            Return valid JSON like {{"route":"kb"}}.
        """)
        route = decision.route
    except Exception:
        logger.exception("Route classification failed; defaulting to KB retrieval")
        route = "kb"

    return {
        "source_used": route,
        "trace": add_trace(state, f"Route -> {route.upper()}"),
    }


def route_after_router(state: AgentState) -> Literal["retrieve_kb", "direct_answer"]:
    return "retrieve_kb" if state["source_used"] == "kb" else "direct_answer"


def retrieve_kb(state: AgentState):
    try:
        docs = get_retriever().invoke(state["current_query"])
        message = f"Private KB retrieval -> {len(docs)} chunks"
    except Exception:
        logger.exception("Private KB retrieval failed")
        docs = []
        message = "Private KB retrieval failed -> continuing without KB evidence"

    return {
        "kb_docs": docs,
        "trace": add_trace(state, message),
    }


def grade_kb(state: AgentState):
    docs = state.get("kb_docs") or []
    if not docs:
        return {
            "kb_grade": "weak",
            "trace": add_trace(state, "KB evidence grade -> WEAK (no chunks)"),
        }

    context = "\n\n".join(
        f"Source: {d.metadata.get('source', 'unknown')}\n{d.page_content}" for d in docs
    )
    try:
        grader = llm().with_structured_output(EvidenceGrade)
        grade = grader.invoke(f"""
            You grade evidence for an enterprise HR policy and employee support assistant.
            Question: {state['question']}
            Private company HR KB evidence:\n{context}
            Return good only if the evidence is sufficient to answer confidently and specifically.
            Otherwise return weak. JSON: {{"grade":"good"}} or {{"grade":"weak"}}.
        """)
        value = grade.grade
    except Exception:
        logger.exception("KB evidence grading failed")
        value = "weak"

    return {
        "kb_grade": value,
        "trace": add_trace(state, f"KB evidence grade -> {value.upper()}"),
    }


def after_kb(state: AgentState) -> Literal["generate_from_kb", "search_web"]:
    return "generate_from_kb" if state.get("kb_grade") == "good" else "search_web"


def search_web(state: AgentState):
    try:
        result = web_search_tool().invoke({"query": state["current_query"]})
    except Exception:
        logger.exception("Web search failed")
        return {
            "web_results": "",
            "citations": [],
            "source_used": "web_search",
            "trace": add_trace(state, "Web fallback -> Tavily search failed"),
        }

    lines, citations = [], []
    seen_urls = set()

    if isinstance(result, dict):
        if result.get("answer"):
            lines.append("Search answer: " + str(result["answer"]))

        for item in result.get("results", []) or []:
            if not isinstance(item, dict):
                continue

            title = item.get("title", "") or ""
            url = item.get("url", "") or ""
            content = item.get("content", "") or ""
            lines.append(f"Title: {title}\nURL: {url}\nContent: {content}")

            if url and url not in seen_urls:
                seen_urls.add(url)
                citations.append({"title": title or url, "url": url, "type": "web"})
    else:
        lines.append(str(result))

    if not lines:
        return {
            "web_results": "",
            "citations": [],
            "source_used": "web_search",
            "trace": add_trace(state, "Web fallback -> no results"),
        }

    return {
        "web_results": "\n\n".join(lines),
        "citations": citations,
        "source_used": "web_search",
        "trace": add_trace(state, "Web fallback -> Tavily search"),
    }


def grade_web(state: AgentState):
    evidence = (state.get("web_results") or "").strip()
    if not evidence:
        return {
            "web_grade": "weak",
            "trace": add_trace(state, "Web evidence grade -> WEAK (no results)"),
        }

    try:
        grader = llm().with_structured_output(EvidenceGrade)
        grade = grader.invoke(f"""
            Question: {state['question']}
            Web evidence:\n{evidence}
            Return good if the evidence is sufficient and directly relevant; otherwise weak.
            Return valid JSON like {{"grade":"good"}}.
        """)
        value = grade.grade
    except Exception:
        logger.exception("Web evidence grading failed")
        value = "weak"

    return {
        "web_grade": value,
        "trace": add_trace(state, f"Web evidence grade -> {value.upper()}"),
    }


def after_web(state: AgentState) -> Literal["generate_from_web", "rewrite_query", "insufficient"]:
    if state.get("web_grade") == "good":
        return "generate_from_web"
    if state.get("retry_count", 0) < settings.max_retries:
        return "rewrite_query"

    return "insufficient"


def rewrite_query(state: AgentState):
    try:
        rewritten = message_text(llm().invoke(f"""
            Rewrite this HR/employee-support question for better private knowledge retrieval and public web search.
            Preserve intent, add useful HR/policy keywords, do not answer, return only the query.
            Question: {state['question']}
        """))
    except Exception:
        logger.exception("Query rewrite failed")
        rewritten = state["current_query"]

    if not rewritten:
        rewritten = state["current_query"]

    return {
        "current_query": rewritten,
        "retry_count": state.get("retry_count", 0) + 1,
        "trace": add_trace(state, f"Query rewrite -> {rewritten}"),
    }


def generate_from_kb(state: AgentState):
    docs = state.get("kb_docs") or []
    context = "\n\n".join(
        f"[Source: {d.metadata.get('source', 'unknown')}]\n{d.page_content}" for d in docs
    )
    answer = message_text(llm().invoke(f"""
        You are an enterprise HR policy and employee support copilot. Answer ONLY from the private company HR KB below.
        Be concise, practical, respectful, and policy-grounded. If steps are present, present them clearly.
        Do not invent policy details. Mention that the answer is based on the company's private knowledge base.
        Question: {state['question']}\n\nPrivate KB:\n{context}
    """))

    citations = []
    seen = set()
    for d in docs:
        src = d.metadata.get("source") or "Private KB"
        if src not in seen:
            seen.add(src)
            citations.append({"title": Path(src).name, "url": "", "type": "private_kb"})

    return {
        "answer": answer,
        "source_used": "private_kb",
        "citations": citations,
        "trace": add_trace(state, "Answer generation -> PRIVATE KB"),
    }


def generate_from_web(state: AgentState):
    answer = message_text(llm().invoke(f"""
        You are an enterprise HR policy and employee support copilot. The private company HR KB was insufficient.
        Answer ONLY from the web evidence below. Clearly say this is external public information and may require HR validation before being treated as company policy or employment guidance.
        Question: {state['question']}\n\nWeb evidence:\n{state['web_results']}
    """))

    return {
        "answer": answer,
        "source_used": "web_search",
        "citations": state.get("citations", []),
        "trace": add_trace(state, "Answer generation -> WEB SEARCH"),
    }


def direct_answer(state: AgentState):
    answer = message_text(llm().invoke(f"Respond briefly and naturally to: {state['question']}"))
    return {
        "answer": answer,
        "source_used": "direct",
        "trace": add_trace(state, "Direct response -> no retrieval"),
    }


def insufficient(state: AgentState):
    return {
        "answer": "I couldn't find enough reliable evidence in the company HR knowledge base or external search to answer confidently. Please contact the HR team or provide more details.",
        "source_used": "insufficient_evidence",
        "trace": add_trace(state, "Stopped -> insufficient reliable evidence"),
    }


def build_graph():
    graph = StateGraph(AgentState)
    for name, fn in {
        "route_question": route_question,
        "retrieve_kb": retrieve_kb,
        "grade_kb": grade_kb,
        "search_web": search_web,
        "grade_web": grade_web,
        "rewrite_query": rewrite_query,
        "generate_from_kb": generate_from_kb,
        "generate_from_web": generate_from_web,
        "direct_answer": direct_answer,
        "insufficient": insufficient,
    }.items():
        graph.add_node(name, fn)

    graph.add_edge(START, "route_question")
    graph.add_conditional_edges("route_question", route_after_router, {
        "retrieve_kb": "retrieve_kb", "direct_answer": "direct_answer",
    })
    graph.add_edge("retrieve_kb", "grade_kb")
    graph.add_conditional_edges("grade_kb", after_kb, {
        "generate_from_kb": "generate_from_kb", "search_web": "search_web",
    })
    graph.add_edge("search_web", "grade_web")
    graph.add_conditional_edges("grade_web", after_web, {
        "generate_from_web": "generate_from_web",
        "rewrite_query": "rewrite_query",
        "insufficient": "insufficient",
    })
    graph.add_edge("rewrite_query", "retrieve_kb")
    graph.add_edge("generate_from_kb", END)
    graph.add_edge("generate_from_web", END)
    graph.add_edge("direct_answer", END)
    graph.add_edge("insufficient", END)
    return graph.compile()


agent_graph = build_graph()


def ask(question: str):
    initial: AgentState = {
        "question": question,
        "current_query": question,
        "kb_docs": [],
        "web_results": "",
        "kb_grade": "",
        "web_grade": "",
        "answer": "",
        "source_used": "",
        "retry_count": 0,
        "trace": [],
        "citations": [],
    }
    return agent_graph.invoke(initial)
