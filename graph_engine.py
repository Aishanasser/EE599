"""graph_engine.py LangGraph orchestration layer for the CV <-> Job Description <-> Question Generation pipeline."""

from typing import Annotated, TypedDict, Optional

from langchain_core.messages import HumanMessage, SystemMessage
from langchain_core.tools import tool
from langgraph.graph import StateGraph, START, END
from langgraph.graph.message import add_messages
from langgraph.prebuilt import ToolNode

from ai_engine import (
    extract_skills,
    extract_jd_requirements,
    compute_skill_gap,
    generate_questions,
    evaluate_answer,
    generate_followup,
    generate_off_plan_question,
    _build_skill_vocabulary,
    _is_soft_skill_target,
    agent_client,
    AGENT_SYSTEM_PROMPT,
    MAX_PROBE_DEPTH,
    MAX_OFF_PLAN_QUESTIONS,
    ANSWER_WEAK_THRESHOLD,
    ANSWER_STRONG_THRESHOLD,
)

MAX_QUESTION_GEN_ATTEMPTS = 2


class PipelineState(TypedDict, total=False):
    cv_text: str
    jd_text: str
    cv_skills: dict
    jd_skills: dict
    skill_gap: dict
    questions: dict
    question_gen_attempts: int
    error: Optional[str]


def _extract_cv_node(state: PipelineState) -> dict:
    if state.get("cv_skills"):
        return {}
    result = extract_skills(state["cv_text"], document_type="CV")
    if "error" in result:
        return {"error": f"CV extraction failed: {result['error']}"}
    return {"cv_skills": result}


def _extract_jd_node(state: PipelineState) -> dict:
    result = extract_jd_requirements(state["jd_text"])
    if "error" in result:
        return {"error": f"JD extraction failed: {result['error']}"}
    return {"jd_skills": result}


def _compute_gap_node(state: PipelineState) -> dict:
    if state.get("error"):
        return {}
    if not state.get("cv_skills") or not state.get("jd_skills"):
        return {"error": "Missing cv_skills or jd_skills before gap computation."}
    gap = compute_skill_gap(state["cv_skills"], state["jd_skills"])
    return {"skill_gap": gap}


def _generate_questions_node(state: PipelineState) -> dict:
    if state.get("error"):
        return {}
    attempts = state.get("question_gen_attempts", 0) + 1
    result = generate_questions(state["jd_skills"], state["cv_skills"], state["skill_gap"])
    if "error" in result:
        return {"error": f"Question generation failed: {result['error']}", "question_gen_attempts": attempts}
    return {"questions": result, "question_gen_attempts": attempts}


def _gate_router(state: PipelineState) -> str:
    """Conditional edge: if any generated question fails the Answerability Score gate and we haven't exhausted retrie"""
    if state.get("error"):
        return "end"
    questions = state.get("questions", {}).get("questions", [])
    all_pass = all(q.get("passes_gate") for q in questions) if questions else False
    if all_pass or state.get("question_gen_attempts", 0) >= MAX_QUESTION_GEN_ATTEMPTS:
        return "end"
    return "retry"


def _build_graph():
    graph = StateGraph(PipelineState)

    graph.add_node("extract_cv", _extract_cv_node)
    graph.add_node("extract_jd", _extract_jd_node)
    graph.add_node("compute_gap", _compute_gap_node)
    graph.add_node("generate_questions", _generate_questions_node)

    graph.add_edge(START, "extract_cv")
    graph.add_edge(START, "extract_jd")

    graph.add_edge("extract_cv", "compute_gap")
    graph.add_edge("extract_jd", "compute_gap")
    graph.add_edge("compute_gap", "generate_questions")

    graph.add_conditional_edges(
        "generate_questions",
        _gate_router,
        {"retry": "generate_questions", "end": END},
    )

    return graph.compile()


_COMPILED_GRAPH = _build_graph()


def run_cv_jd_pipeline(jd_text: str, cv_text: str = "", cv_skills: Optional[dict] = None) -> dict:
    """Runs the full CV <-> Job Description <-> Question Generation graph: extract CV skills (or reuse already-extrac"""
    initial_state: PipelineState = {"cv_text": cv_text, "jd_text": jd_text}
    if cv_skills:
        initial_state["cv_skills"] = cv_skills
    final_state = _COMPILED_GRAPH.invoke(initial_state)
    return dict(final_state)


class AnswerCycleState(TypedDict, total=False):
    messages: Annotated[list, add_messages]
    evaluation: dict
    error: Optional[str]


def _build_agent_tools(ctx: dict) -> list:
    """Tools are built per-invocation as closures over `ctx`, so they can read the live interview state and record th"""

    @tool
    def get_interview_state() -> dict:
        """Read the current interview state: which skills are still queued, which have been covered, how many questions h"""
        return {
            "remaining_skills": ctx["remaining_skills"],
            "covered_skills": ctx["covered_skills"],
            "questions_asked": ctx["asked_count"],
            "questions_budget_left": ctx["budget_left"],
            "probe_depth_on_current_skill": ctx["probe_depth"],
            "max_probe_depth": MAX_PROBE_DEPTH,
            "off_plan_questions_used": ctx["off_plan_used"],
            "max_off_plan_questions": MAX_OFF_PLAN_QUESTIONS,
        }

    @tool
    def move_to_another_skill(reason: str) -> dict:
        """Skip the remaining planned questions on the CURRENT skill and move to the next skill."""
        ctx["decision"] = {"route": "skip_skill", "reason": reason}
        return {"status": "will skip remaining questions on "
                          f"'{ctx['current_skill']}'"}

    @tool
    def probe_deeper(aspect: str) -> dict:
        """Ask a deeper follow-up on the SAME skill."""
        ctx["decision"] = {"route": "probe", "reason": aspect}
        return {"status": f"will generate a follow-up on '{ctx['current_skill']}'",
                "aspect": aspect}

    @tool
    def ask_about_skill(skill_name: str, reason: str) -> dict:
        """Ask about a skill the candidate raised themselves that was not in the planned set."""
        ctx["decision"] = {"route": "ask_about_skill",
                           "skill": skill_name, "reason": reason}
        return {"status": f"will ask about '{skill_name}' (off-plan)"}

    @tool
    def continue_as_planned() -> dict:
        """Move to the next question in the plan without changing anything."""
        ctx["decision"] = {"route": "next", "reason": "skill adequately covered"}
        return {"status": "will continue with the planned questions"}

    return [get_interview_state, move_to_another_skill, probe_deeper,
            ask_about_skill, continue_as_planned]


def _fallback_route(evaluation: dict, probe_depth: int) -> dict:
    """Deterministic safety net."""
    score = evaluation.get("final_score", 0.0)
    if score < ANSWER_WEAK_THRESHOLD:
        return {"route": "skip_skill",
                "reason": f"[fallback] weak answer (score {score:.2f})"}
    if score > ANSWER_STRONG_THRESHOLD and probe_depth < MAX_PROBE_DEPTH:
        return {"route": "probe",
                "reason": f"[fallback] strong answer (score {score:.2f})"}
    return {"route": "next", "reason": f"[fallback] score {score:.2f}"}


def _build_answer_cycle_graph(tools: list):
    agent_model = agent_client.bind_tools(tools)

    def _agent_node(state: AnswerCycleState) -> dict:
        return {"messages": [agent_model.invoke(state["messages"])]}

    def _should_continue(state: AnswerCycleState) -> str:
        last = state["messages"][-1]
        return "tools" if getattr(last, "tool_calls", None) else "end"

    graph = StateGraph(AnswerCycleState)
    graph.add_node("agent", _agent_node)
    graph.add_node("tools", ToolNode(tools))

    graph.add_edge(START, "agent")
    graph.add_conditional_edges("agent", _should_continue,
                                {"tools": "tools", "end": END})
    graph.add_edge("tools", "agent")
    return graph.compile()


def run_answer_cycle(question: dict, answer: str, cv_skills: dict,
                     jd_skills: dict, interview_state: dict) -> dict:
    """One adaptive interview step: evaluate the candidate's answer, then let the agent decide what to ask next."""
    targets_skill = question.get("targets_skill", "")
    vocabulary = _build_skill_vocabulary(jd_skills, cv_skills)
    is_soft = _is_soft_skill_target(targets_skill, jd_skills, cv_skills)

    evaluation = evaluate_answer(question.get("question", ""), answer,
                                 targets_skill, vocabulary, is_soft_skill=is_soft)
    if "error" in evaluation:
        return {"error": f"Answer evaluation failed: {evaluation['error']}"}

    ctx = dict(interview_state)
    ctx["current_skill"] = targets_skill
    ctx["decision"] = None

    tools = _build_agent_tools(ctx)
    graph = _build_answer_cycle_graph(tools)

    if evaluation.get("is_soft_skill"):
        star = evaluation.get("star", {})
        criteria = (
            f"  [behavioural question — graded on STAR, not technical accuracy]\n"
            f"  situation         = {star.get('situation')}\n"
            f"  task              = {star.get('task')}\n"
            f"  action            = {star.get('action')}\n"
            f"  result            = {star.get('result')}\n"
            f"  star_completeness = {evaluation.get('star_completeness')}\n"
            f"  relevance         = {evaluation['relevance']}\n"
            f"  depth             = {evaluation['depth']}\n"
        )
    else:
        criteria = (
            f"  technical_accuracy= {evaluation['technical_accuracy']}\n"
            f"  relevance         = {evaluation['relevance']}\n"
            f"  depth             = {evaluation['depth']}\n"
        )

    briefing = (
        f"Skill just probed: {targets_skill}\n"
        f"Question asked: {question.get('question', '')}\n"
        f"Candidate's answer: {answer}\n\n"
        f"Evaluation:\n"
        f"  final_score       = {evaluation['final_score']}\n"
        f"{criteria}"
        f"  filler_words      = {evaluation['filler_count']}\n\n"
        f"Interview so far: {ctx['asked_count']} asked, "
        f"{ctx['budget_left']} left, probe depth on this skill "
        f"{ctx['probe_depth']}/{MAX_PROBE_DEPTH}, "
        f"off-plan used {ctx['off_plan_used']}/{MAX_OFF_PLAN_QUESTIONS}.\n"
        f"Skills still queued: {ctx['remaining_skills']}\n\n"
        f"Decide the next step."
    )

    try:
        final_state = graph.invoke({"messages": [
            SystemMessage(content=AGENT_SYSTEM_PROMPT),
            HumanMessage(content=briefing),
        ]})
    except Exception as e:
        return {"error": f"Agent step failed: {e}", "evaluation": evaluation}

    thought = " ".join(
        m.content for m in final_state["messages"]
        if getattr(m, "type", "") == "ai" and isinstance(m.content, str) and m.content.strip()
    ).strip()

    tool_calls = [tc["name"] for m in final_state["messages"]
                  for tc in (getattr(m, "tool_calls", None) or [])]

    decision = ctx["decision"]
    used_fallback = decision is None
    if used_fallback:
        decision = _fallback_route(evaluation, ctx["probe_depth"])

    new_question = None
    if decision["route"] == "probe":
        new_question = generate_followup(
            question.get("question", ""), answer, targets_skill,
            vocabulary, is_soft, star=evaluation.get("star"))
        if new_question is None:
            decision = {"route": "next",
                        "reason": "follow-up failed the Answerability gate"}
    elif decision["route"] == "ask_about_skill":
        skill_name = decision.get("skill", "")
        new_question = generate_off_plan_question(
            skill_name, answer, vocabulary,
            _is_soft_skill_target(skill_name, jd_skills, cv_skills))
        if new_question is None:
            decision = {"route": "next",
                        "reason": "off-plan question failed the Answerability gate"}

    return {
        "evaluation": evaluation,
        "route": decision["route"],
        "reason": decision.get("reason", ""),
        "thought": thought,
        "new_question": new_question,
        "used_fallback": used_fallback,
        "tool_calls": tool_calls,
    }


if __name__ == "__main__":
    import json

    sample_cv = "Experienced in Python, Docker, and Git. Strong communication skills."
    sample_jd = "Requirements: Python, Kubernetes, AWS. Excellent communication skills."

    print("Running CV/JD/Question-Generation pipeline via LangGraph...\n")
    result = run_cv_jd_pipeline(jd_text=sample_jd, cv_text=sample_cv)
    print(json.dumps(result, indent=2, ensure_ascii=False))
