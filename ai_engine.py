
import os
import re
import json
from dotenv import load_dotenv
from langchain_openai import ChatOpenAI
from langchain_core.messages import SystemMessage, HumanMessage

load_dotenv()

FIREWORKS_API_KEY = os.getenv("FIREWORKS_API_KEY")
FIREWORKS_BASE_URL = "https://api.fireworks.ai/inference/v1"
MODEL_NAME = "accounts/fireworks/models/deepseek-v4-pro"

if not FIREWORKS_API_KEY:
    raise RuntimeError(
        "FIREWORKS_API_KEY environment variable is not set. "
        "Add it to your .env file or export it before running."
    )

client = ChatOpenAI(
    model=MODEL_NAME,
    api_key=FIREWORKS_API_KEY,
    base_url=FIREWORKS_BASE_URL,
    temperature=0.0,
    max_tokens=16384,
    model_kwargs={"response_format": {"type": "json_object"}},
)

agent_client = ChatOpenAI(
    model=MODEL_NAME,
    api_key=FIREWORKS_API_KEY,
    base_url=FIREWORKS_BASE_URL,
    temperature=0.0,
    max_tokens=16384,
)

CV_SKILL_EXTRACTION_PROMPT = """You are an expert AI system for resume skill extraction.

Your task is to analyze the provided CV and identify all skills mentioned.
The text may be in English, Arabic, or a mix of both.

Before answering, scan the CV section by section (Personal Skills, Experience,
Training/Certificates, Honors/Achievements, Activities/Memberships, etc.) —
a skill can appear in any section, not just one literally named "Skills". Do
not skip any section. However, scanning every section does NOT mean every
sentence contains a skill: a job duty, responsibility, task, or achievement
described in prose (e.g., "documented procedures", "asset & account
management", "provided IT support for 1,000+ staff", "secured sponsorships")
is NOT itself a skill — only extract something from a narrative sentence if
it names a concrete, specific tool, technology, programming language,
framework, methodology, or technique (e.g., "Docker", "Deep Learning",
"Python"). Do not turn a whole task/responsibility description into a skill
entry just because it appeared while scanning.

Rules:

1. Extract only skills — concrete named tools, technologies, programming
   languages, frameworks, methodologies, techniques, or competencies. Not
   general job duties, responsibilities, or achievements described in prose.
2. Keep multi-word skills together. If a longer skill phrase contains a shorter
   skill inside it (e.g., "Microsoft SQL Server" contains "SQL Server" and "SQL"),
   extract only the longest/complete form — do not also list the shorter
   sub-phrase separately.
3. Separate Technical Skills and Soft Skills.
4. Languages (e.g., Arabic, English, Turkish) always go in their own
   "languages" list — never in technical_skills or soft_skills. Extract each
   language exactly as it appears including any proficiency qualifier, per
   rule 7 (e.g., if the CV says "Arabic (Native)" or "English — Advanced (C1)",
   extract that full string, not just "Arabic" or "English" alone).
5. Ignore names, companies, universities, projects, and job titles.
6. Remove duplicates.
7. Preserve the original wording exactly as it appears in the text. Never
   paraphrase, summarize, or convert a descriptive sentence into a generic
   label — extract the exact phrase as written, word for word, including any
   qualifiers (e.g., "Proven ability to work under pressure" must stay exactly
   that, not "Ability to work under pressure"; "Social person with lots of
   connections" must stay exactly that, not "Networking"). The only exception
   is a trailing sentence-ending punctuation mark (e.g. a final "." or ".."),
   which must be dropped — everything else stays untouched.
8. A certificate, training course, or award mentioning a specific skill (e.g.,
   "First aid certificate from the World First Aid Organization", "Email
   etiquette course with [name]") is a valid skill source — extract the named
   skill from it (e.g., "First aid", "Email etiquette"), ignoring the issuing
   organization and any person's name per rule 5.
9. Return only JSON — no preamble, no explanation, no markdown code fences.

Output format:

{
  "technical_skills": [],
  "soft_skills": [],
  "languages": []
}
"""

JD_REQUIREMENT_EXTRACTION_PROMPT = """You are an expert AI system for extracting required
skills from a company's Job Description (JD).

Your task is to analyze the provided Job Description and identify all skills
it asks the candidate to have — whether listed under "Requirements",
"Qualifications", "Nice to have", "Preferred", "Responsibilities", or written
in plain prose anywhere in the text. The text may be in English, Arabic, or a
mix of both.

Rules:

1. Extract only skills — concrete named tools, technologies, programming
   languages, frameworks, methodologies, techniques, or competencies. Not
   generic phrases like "team player mindset" unless they name an actual
   competency (e.g., "communication skills" is fine; "fast-paced
   environment" is not a skill).
2. Keep multi-word skills together. If a longer skill phrase contains a shorter
   skill inside it (e.g., "Microsoft SQL Server" contains "SQL Server" and "SQL"),
   extract only the longest/complete form — do not also list the shorter
   sub-phrase separately.
3. Separate Technical Skills and Soft Skills.
4. Languages (e.g., Arabic, English, Turkish) always go in their own
   "languages" list — never in technical_skills or soft_skills. Extract each
   language exactly as it appears including any proficiency qualifier (e.g.,
   "Fluent in English" -> "Fluent in English", not just "English").
5. Ignore the company name, job title, location, salary, and benefits.
6. Remove duplicates.
7. Preserve the original wording exactly as it appears in the text — do not
   paraphrase or generalize a requirement into a shorter label. The only
   exception is a trailing sentence-ending punctuation mark, which must be
   dropped.
8. Treat "required" and "nice to have"/"preferred" skills the same way —
   extract both into the same lists (no separate priority tier).
9. Return only JSON — no preamble, no explanation, no markdown code fences.

Output format:

{
  "technical_skills": [],
  "soft_skills": [],
  "languages": []
}
"""

def _call_llm(system_prompt: str, user_prompt: str) -> str:
    response = client.invoke([
        SystemMessage(content=system_prompt),
        HumanMessage(content=user_prompt),
    ])
    return response.content


_FENCE_RE = re.compile(r"^```(?:json)?\s*|\s*```$", re.IGNORECASE | re.MULTILINE)


def _strip_code_fences(raw_output: str) -> str:
    return _FENCE_RE.sub("", raw_output.strip()).strip()


def _validate_schema(parsed: dict) -> str | None:
    """Return an error message if the parsed JSON doesn't match the expected skill-extraction schema, or None if it's"""
    if not isinstance(parsed, dict):
        return "Top-level JSON must be an object."
    for key in ("technical_skills", "soft_skills", "languages"):
        values = parsed.get(key)
        if not isinstance(values, list):
            return f"Missing or invalid '{key}' list."
        if not all(isinstance(v, str) for v in values):
            return f"'{key}' must contain only strings."
    return None


def _call_llm_json(system_prompt: str, user_prompt: str) -> dict:
    """Call the LLM and parse its response as JSON (stripping code fences if needed)."""
    try:
        raw_output = _call_llm(system_prompt, user_prompt)
    except Exception as e:
        return {"error": f"LLM call failed: {str(e)}"}

    try:
        return json.loads(raw_output)
    except json.JSONDecodeError:
        cleaned = _strip_code_fences(raw_output)
        try:
            return json.loads(cleaned)
        except json.JSONDecodeError:
            return {"error": "Model did not return valid JSON.", "raw_output": raw_output}


def _run_extraction(system_prompt: str, user_prompt: str) -> dict:
    """Shared LLM-call + JSON-parse + schema-validation pipeline used by both extract_skills() and extract_jd_require"""
    parsed = _call_llm_json(system_prompt, user_prompt)
    if "error" in parsed:
        return parsed

    schema_error = _validate_schema(parsed)
    if schema_error:
        return {"error": f"Model output failed schema validation: {schema_error}", "raw_output": parsed}

    return parsed


def extract_skills(document_text: str, document_type: str = "CV") -> dict:
    """Phase 1: Extract the skills a candidate has, from their CV text."""
    if not document_text or not document_text.strip():
        return {"error": "Empty document_text provided."}

    user_prompt = f"Text:\n{document_text}"
    return _run_extraction(CV_SKILL_EXTRACTION_PROMPT, user_prompt)


def extract_jd_requirements(jd_text: str) -> dict:
    """Phase 2: Extract the skills a Job Description requires from the candidate."""
    if not jd_text or not jd_text.strip():
        return {"error": "Empty jd_text provided."}

    user_prompt = f"Text:\n{jd_text}"
    return _run_extraction(JD_REQUIREMENT_EXTRACTION_PROMPT, user_prompt)


_SEMANTIC_MODEL = None
_GAP_SIMILARITY_THRESHOLD = 0.6


def _get_semantic_model():
    """Lazy-load the sentence-transformers model (only needed for skill-gap matching)."""
    global _SEMANTIC_MODEL
    if _SEMANTIC_MODEL is None:
        from sentence_transformers import SentenceTransformer
        _SEMANTIC_MODEL = SentenceTransformer("all-MiniLM-L6-v2")
    return _SEMANTIC_MODEL


def compute_skill_gap(candidate_result: dict, jd_result: dict) -> dict:
    """Compare a candidate's extracted skills (from extract_skills() on a CV) against a job description's required sk"""
    model = _get_semantic_model()
    gap = {}

    for category, gap_key in [
        ("technical_skills", "missing_technical_skills"),
        ("soft_skills", "missing_soft_skills"),
        ("languages", "missing_languages"),
    ]:
        jd_skills = jd_result.get(category, [])
        candidate_skills = candidate_result.get(category, [])

        if not jd_skills:
            gap[gap_key] = []
            continue
        if not candidate_skills:
            gap[gap_key] = list(jd_skills)
            continue

        jd_embeddings = model.encode(jd_skills, convert_to_tensor=True)
        candidate_embeddings = model.encode(candidate_skills, convert_to_tensor=True)

        from sentence_transformers import util
        similarity_matrix = util.cos_sim(jd_embeddings, candidate_embeddings)

        missing = []
        for i, jd_skill in enumerate(jd_skills):
            best_match_score = similarity_matrix[i].max().item()
            if best_match_score < _GAP_SIMILARITY_THRESHOLD:
                missing.append(jd_skill)
        gap[gap_key] = missing

    return gap


QUESTION_GENERATION_PROMPT = """You are an expert technical interviewer designing
questions for a mock interview.

You are given three things:
1. The skills the job (JD) requires.
2. The skills the candidate's CV shows they already have.
3. The skill gap — JD-required skills the candidate does NOT appear to have.

Your task: generate exactly {num_questions} interview questions, split the way
a real interviewer splits them — part on what the candidate is missing, part
on what they already claim:

- **Exactly {gap_count} questions on GAP skills** (skills the role requires
  that the candidate does not appear to have). Pick the gap skills that are
  most central to the role — the ones a candidate genuinely could not do the
  job without — not peripheral "nice to have" items. These questions test
  transferable experience, awareness of the topic, and how the candidate
  reasons about something they have not used in production.
- **Exactly {existing_count} questions on EXISTING skills** (skills the CV
  already claims). These matter just as much: they are where the candidate's
  real depth is measured. An interview that only probes weaknesses reveals
  nothing about what the person is actually good at.

If one of the two lists holds fewer distinct skills than its quota, take the
shortfall from the other list rather than repeating a skill.

TWO KINDS OF SKILL NEED TWO KINDS OF QUESTION:

- **Technical skills** — ask about a concrete task, mechanism, trade-off or
  scenario involving that technology.

- **Soft / behavioural skills** (communication, teamwork, leadership, time
  management, adaptability, working under pressure) — these CANNOT be tested
  with a knowledge question. "What makes communication effective?" measures
  vocabulary, not behaviour; any candidate can recite the right words.
  Use the **STAR framework** that real HR interviewers use, and ask, inside a
  single spoken question, for all four parts:
    S — a SPECIFIC past situation ("a time when...", "a situation where...")
    T — the candidate's own task or role in it
    A — the actions THEY personally took
    R — the result or outcome
  Anchor it in the past and in one concrete episode. Never hypothetical
  ("what would you do if..."), never general ("how do you usually handle
  conflict?") — a hypothetical answer is an opinion, and only a real episode
  is evidence.
  Example shape: "Describe a specific situation where you had to resolve a
  disagreement inside your team — what was your role, what did you personally
  do, and how did it turn out?"

Rules:

1. Each question must be self-contained and answerable in a spoken interview
   — no "see attached", no multi-part essay prompts.
2. Each question MUST explicitly name, inside its own text, the skill given
   in its "targets_skill" field. A question about RTOS that never writes
   "RTOS" is invalid.
3. Stay focused: one question probes one skill. Do not pad a question with
   unrelated skill names just to make it look technical.
4. Phrase each question with a clear interrogative ("What/How/Why/Which...")
   or a directive verb ("Describe/Explain/Compare/Walk me through"), and ask
   about a concrete task or scenario rather than a vague invitation to talk
   (avoid "Tell me about X").
5. Do not repeat the same targets_skill across questions unless there are
   fewer distinct skills available than {num_questions}.
6. Return only JSON — no preamble, no explanation, no markdown code fences.

Output format:

{{
  "questions": [
    {{
      "question": "string",
      "targets_skill": "string (must also appear inside the question text)"
    }}
  ]
}}
"""


def _validate_questions_schema(parsed: dict) -> str | None:
    if not isinstance(parsed, dict):
        return "Top-level JSON must be an object."
    questions = parsed.get("questions")
    if not isinstance(questions, list) or not questions:
        return "Missing or empty 'questions' list."
    for i, q in enumerate(questions):
        if not isinstance(q, dict) or not {"question", "targets_skill"}.issubset(q):
            return f"Question at index {i} is missing 'question' or 'targets_skill'."
        if not isinstance(q["question"], str) or not isinstance(q["targets_skill"], str):
            return f"Question at index {i} has non-string 'question'/'targets_skill'."
    return None


_AS_WEIGHTS = {"content_entities": 0.5, "context_clarity": 0.3, "task_specificity": 0.2}
_ANSWERABILITY_GATE = 0.7

_INTERROGATIVES = ("what", "how", "why", "which", "when", "where", "who")
_DIRECTIVE_VERBS = ("describe", "explain", "compare", "walk me through",
                    "walk us through", "outline", "discuss")
_TASK_VERBS = ("implement", "design", "debug", "optimize", "optimise", "deploy",
               "configure", "build", "troubleshoot", "integrate", "test",
               "handle", "resolve", "identify", "measure", "profile",
               "refactor", "migrate", "scale", "secure", "validate",
               "diagnose", "write", "structure", "ensure", "choose", "select",
               "manage", "process", "approach", "apply", "maintain", "monitor",
               "automate", "containerize", "containerise", "analyze", "analyse",
               "evaluate", "reduce", "improve", "extend", "connect", "trace",
               "verify", "benchmark", "tune", "prevent", "set up")

_SPECIFICITY_PHRASES = (
    "specific example", "concrete example", "for example", "a time when",
    "a time you", "a situation where", "scenario", "difference between",
    "differences between", "compare", "versus", " vs ", "trade-off",
    "tradeoff", "what steps", "which steps", "how would you", "how do you",
    "how have you", "walk me through", "walk us through", "step by step",
)

_TARGET_STOPWORDS = {
    "and", "or", "the", "of", "in", "with", "for", "to", "on", "at", "by",
    "using", "skills", "skill", "experience", "knowledge", "understanding",
    "familiarity", "strong", "excellent", "proven", "ability", "proficiency",
    "proficient", "basic", "advanced", "solid", "good", "fluent", "native",
}

_MAX_CLEAR_QUESTION_WORDS = 60
_MIN_DETAILED_QUESTION_WORDS = 12


def _word_boundary_search(needle: str, haystack_lower: str):
    """Search for `needle` as a standalone token, so that e.g."""
    pattern = r"(?<![a-z0-9])" + re.escape(needle) + r"(?![a-z0-9])"
    return re.search(pattern, haystack_lower)


def _verb_search(verb: str, haystack_lower: str):
    """Match a task verb including its common inflections, so that "deploy" also matches "deployed"/"deploying" and"""
    stem = verb[:-1] if verb.endswith("e") else verb
    pattern = r"(?<![a-z0-9])" + re.escape(stem) + r"(?:e|es|ed|ing|s)?(?![a-z0-9])"
    return re.search(pattern, haystack_lower)


def _build_skill_vocabulary(jd_result: dict, cv_result: dict) -> set:
    vocab = set()
    for source in (jd_result, cv_result):
        for key in ("technical_skills", "soft_skills", "languages"):
            for skill in source.get(key, []):
                if isinstance(skill, str) and skill.strip():
                    vocab.add(skill.strip())
    return vocab


def _count_named_entities(question_text: str, vocabulary: set) -> int:
    """Count DISTINCT known skills named in the question."""
    text = question_text.lower()
    count = 0
    for skill in sorted(vocabulary, key=len, reverse=True):
        match = _word_boundary_search(skill.lower(), text)
        if match:
            count += 1
            text = text[:match.start()] + " " * (match.end() - match.start()) + text[match.end():]
    return count


def _target_is_named(question_text: str, targets_skill: str) -> bool:
    """Does the question actually name the skill it claims to target?"""
    text = question_text.lower()

    candidates = [targets_skill] + re.split(r"[/,()]", targets_skill)
    for candidate in candidates:
        candidate = candidate.strip().lower()
        if candidate and _word_boundary_search(candidate, text):
            return True

    for word in re.split(r"[\s/,()]+", targets_skill.lower()):
        word = word.strip()
        if len(word) >= 3 and word not in _TARGET_STOPWORDS and _word_boundary_search(word, text):
            return True
    return False


def _score_content_entities(question_text: str, targets_skill: str, vocabulary: set) -> float:
    if not _target_is_named(question_text, targets_skill):
        return 0.0
    extra_entities = max(0, _count_named_entities(question_text, vocabulary) - 1)
    return min(1.0, 0.6 + 0.2 * extra_entities)


def _score_context_clarity(question_text: str) -> float:
    text = question_text.lower()
    score = 0.4
    if any(_word_boundary_search(w, text) for w in _INTERROGATIVES) or \
       any(v in text for v in _DIRECTIVE_VERBS):
        score += 0.3
    if question_text.strip().endswith("?"):
        score += 0.3
    if len(question_text.split()) > _MAX_CLEAR_QUESTION_WORDS:
        score -= 0.3
    return max(0.0, min(1.0, score))


def _score_task_specificity(question_text: str) -> float:
    text = question_text.lower()
    has_task_verb = any(_verb_search(v, text) for v in _TASK_VERBS)
    has_specific_ask = any(p in text for p in _SPECIFICITY_PHRASES)
    if not has_task_verb and not has_specific_ask:
        return 0.0
    return 1.0 if len(question_text.split()) >= _MIN_DETAILED_QUESTION_WORDS else 0.5


_STAR_CUES = {
    "situation": ("a time when", "a time you", "a situation where",
                  "a situation in which", "an occasion when", "an instance where",
                  "describe a time", "tell me about a time", "a specific example",
                  "a specific situation", "an example of when", "when you had to",
                  "when you were"),
    "task": ("your role", "your responsibility", "your task", "you were responsible",
             "what were you", "you had to", "expected of you", "your part"),
    "action": ("what did you do", "what you did", "what steps", "how did you handle",
               "how did you approach", "how did you respond", "how did you deal",
               "what actions", "actions you took", "you personally do"),
    "result": ("outcome", "result", "what happened", "turn out", "turned out",
               "end up", "ended up", "how did it end", "how did that end",
               "impact", "in the end", "what came of"),
}


def _score_star_elicitation(question_text: str,
                            situation_established: bool = False) -> float:
    """For a behavioural question this plays the role entity-counting plays for a technical one: it measures whether"""
    text = question_text.lower()
    present = {c: any(cue in text for cue in cues) for c, cues in _STAR_CUES.items()}
    if situation_established:
        return 1.0 if any(present.values()) else 0.0
    if not present["situation"]:
        return 0.0
    return round(sum(present.values()) / len(_STAR_CUES), 3)


def _collect_labels(jd_result: dict, cv_result: dict, category: str) -> list:
    return [s for src in (jd_result, cv_result)
            for s in src.get(category, []) if isinstance(s, str) and s.strip()]


def _best_label_overlap(target: str, labels: list) -> tuple:
    """How strongly `target` matches any label in `labels`, as the comparable pair (exact_match, longest_matching_lab"""
    exact = 0
    longest = 0
    for label in labels:
        candidate = label.strip().lower()
        if candidate == target:
            exact = 1
            longest = max(longest, len(candidate))
        elif (_word_boundary_search(candidate, target)
                or _word_boundary_search(target, candidate)):
            longest = max(longest, len(candidate))
    return (exact, longest if exact == 0 else len(target))


def _is_soft_skill_target(targets_skill: str, jd_result: dict, cv_result: dict) -> bool:
    """Decide whether a question's target is a behavioural skill."""
    target = targets_skill.strip().lower()
    if not target:
        return False

    soft_match = _best_label_overlap(target, _collect_labels(jd_result, cv_result, "soft_skills"))
    tech_match = _best_label_overlap(target, _collect_labels(jd_result, cv_result, "technical_skills"))
    return soft_match > tech_match


def _derive_is_gap_skill(targets_skill: str, gap_skills: list) -> bool:
    """Derive the gap flag from the *computed* skill gap rather than trusting the generating model's own claim, using"""
    if not gap_skills:
        return False
    target_norm = targets_skill.strip().lower()
    if any(g.strip().lower() == target_norm for g in gap_skills):
        return True
    model = _get_semantic_model()
    from sentence_transformers import util
    target_embedding = model.encode([targets_skill], convert_to_tensor=True)
    gap_embeddings = model.encode(gap_skills, convert_to_tensor=True)
    return util.cos_sim(target_embedding, gap_embeddings).max().item() >= _GAP_SIMILARITY_THRESHOLD


def _compute_answerability(question_text: str, targets_skill: str, vocabulary: set,
                           is_soft_skill: bool,
                           situation_established: bool = False) -> dict:
    clarity = _score_context_clarity(question_text)
    specificity = _score_task_specificity(question_text)

    if is_soft_skill:
        entities = None
        star = _score_star_elicitation(question_text, situation_established)
        score = (_AS_WEIGHTS["content_entities"] * star
                 + _AS_WEIGHTS["context_clarity"] * clarity
                 + _AS_WEIGHTS["task_specificity"] * specificity)
    else:
        star = None
        entities = _score_content_entities(question_text, targets_skill, vocabulary)
        score = (_AS_WEIGHTS["content_entities"] * entities
                 + _AS_WEIGHTS["context_clarity"] * clarity
                 + _AS_WEIGHTS["task_specificity"] * specificity)

    return {
        "content_entities": entities,
        "star_elicitation": star,
        "context_clarity": round(clarity, 3),
        "task_specificity": round(specificity, 3),
        "answerability_score": round(score, 3),
        "passes_gate": score >= _ANSWERABILITY_GATE,
    }


_GAP_QUESTION_RATIO = 0.6


def generate_questions(jd_result: dict, cv_result: dict, skill_gap: dict, num_questions: int = 10) -> dict:
    """Phase 2: Generate strategic interview questions balanced between the candidate's skill gap and the skills they"""
    gap_count = round(num_questions * _GAP_QUESTION_RATIO)
    existing_count = num_questions - gap_count

    user_prompt = (
        f"JD required skills: {json.dumps(jd_result, ensure_ascii=False)}\n\n"
        f"Candidate skills: {json.dumps(cv_result, ensure_ascii=False)}\n\n"
        f"Skill gap (JD requires, candidate lacks): {json.dumps(skill_gap, ensure_ascii=False)}"
    )
    system_prompt = QUESTION_GENERATION_PROMPT.format(
        num_questions=num_questions,
        gap_count=gap_count,
        existing_count=existing_count,
    )

    parsed = _call_llm_json(system_prompt, user_prompt)
    if "error" in parsed:
        return parsed

    schema_error = _validate_questions_schema(parsed)
    if schema_error:
        return {"error": f"Model output failed schema validation: {schema_error}", "raw_output": parsed}

    vocabulary = _build_skill_vocabulary(jd_result, cv_result)
    gap_skills = (skill_gap.get("missing_technical_skills", [])
                  + skill_gap.get("missing_soft_skills", [])
                  + skill_gap.get("missing_languages", []))

    for q in parsed["questions"]:
        is_soft = _is_soft_skill_target(q["targets_skill"], jd_result, cv_result)
        q["is_soft_skill"] = is_soft
        q["is_gap_skill"] = _derive_is_gap_skill(q["targets_skill"], gap_skills)
        q.update(_compute_answerability(q["question"], q["targets_skill"], vocabulary, is_soft))

    return parsed


ANSWER_EVALUATION_PROMPT = """You are a strict technical interviewer reviewing a
candidate's answer during a job interview.

You are given an interview question, the skill it is meant to probe, and the
candidate's spoken answer (transcribed).

Judge the three criteria below. They are INDEPENDENT of one another — score
each one on its own terms and do not let a low score on one drag down another.

In particular: a SHORT answer is not automatically a WRONG answer. Brevity is
measured by "depth" alone. If a candidate states something true in one
sentence, "technical_accuracy" is high even though "depth" is low.

For EACH criterion, first state briefly what you observed, then assign a score
between 0.0 and 1.0:

1. "technical_accuracy" — Are the claims the candidate made TRUE?
   Judge only correctness, never how much was said.
   1.0 = everything stated is correct (even if only one sentence was stated)
   0.5 = mostly correct with an imprecision
   0.0 = contains a clear technical error, or is factually wrong
   Example: "Kubernetes manages containers and restarts them if they fail" is
   brief but TRUE → technical_accuracy = 1.0 (and depth would be low).

2. "relevance" — Does the answer address the question that was asked?
   Judge only topic match, never completeness.
   1.0 = it is about what was asked   0.5 = partly, or drifts to a near topic
   0.0 = it is about something else entirely
   Example: a brief answer that engages the right topic is still relevant.

3. "depth" — How far below the surface does it go?
   THIS is where brevity and superficiality are penalised.
   1.0 = explains mechanisms, trade-offs, failure modes, or real experience
   0.5 = correct but textbook-level, no mechanism explained
   0.0 = a bare assertion with nothing behind it

Do not reward confidence or polished phrasing on their own: a fluent answer
that is factually wrong must still score 0.0 on technical_accuracy.

Also write "feedback": two sentences maximum, addressed to the candidate,
naming one concrete thing to improve. Be specific — not "add more detail" but
what detail was missing.

Return only JSON — no preamble, no explanation, no markdown code fences.

Output format:

{{
  "observations": "string (brief, what you noticed)",
  "technical_accuracy": 0.0,
  "relevance": 0.0,
  "depth": 0.0,
  "feedback": "string"
}}
"""


SOFT_ANSWER_EVALUATION_PROMPT = """You are an experienced interviewer reviewing a
candidate's answer to a BEHAVIOURAL interview question.

Behavioural answers are assessed with the STAR framework: a usable answer
describes a specific Situation, the candidate's own Task or role in it, the
Actions they personally took, and the Result.

There is NO factually correct answer here. Do NOT judge technical correctness.
Judge only whether the candidate supplied evidence of the behaviour.

For each STAR component report:
  1.0 = supplied concretely and specifically
  0.5 = alluded to, but vague or generic
  0.0 = absent

Grade the four independently:

1. "situation" — is there ONE specific, real past episode? A general habit or
   a hypothetical is NOT a situation.
   "Last term our team's deployment broke two days before the demo" = 1.0
   "I always make sure to communicate clearly with my team" = 0.0 (a claim
   about themselves, not an episode)

2. "task" — is the candidate's OWN role or responsibility in that episode
   clear? What were they specifically accountable for?

3. "action" — what did THEY personally do? Look for first-person, concrete
   steps. Be careful with answers written entirely in "we": if the candidate's
   own contribution never becomes visible, this is at most 0.5.

4. "result" — what was the outcome? Any stated consequence counts, and a
   negative or mixed outcome counts fully — a quantified figure is NOT
   required.

Then two further criteria, judged independently of the four above:

"relevance" — does the episode actually demonstrate the skill being probed, or
is it a story about something else? 1.0 = it demonstrates it directly.

"depth" — is there reflection: trade-offs weighed, what they learned, what
they would do differently? 1.0 = genuine reflection. 0.0 = a flat retelling.

Also write "feedback": two sentences maximum, addressed to the candidate,
naming the single weakest STAR component and exactly what was missing from it
— not "give more detail" but which detail.

Return only JSON — no preamble, no explanation, no markdown code fences.

Output format:

{{
  "observations": "string (brief, what you noticed)",
  "situation": 0.0,
  "task": 0.0,
  "action": 0.0,
  "result": 0.0,
  "relevance": 0.0,
  "depth": 0.0,
  "feedback": "string"
}}
"""

_ANSWER_WEIGHTS = {
    "technical_accuracy": 0.35,
    "relevance": 0.25,
    "depth": 0.20,
    "technical_density": 0.10,
    "substance": 0.10,
}

_SOFT_ANSWER_WEIGHTS = {
    "star_completeness": 0.45,
    "relevance": 0.25,
    "depth": 0.20,
    "substance": 0.10,
}

_STAR_COMPONENT_WEIGHTS = {
    "situation": 0.30,
    "task": 0.20,
    "action": 0.30,
    "result": 0.20,
}

_FILLER_MAX_PENALTY = 0.10

_FILLER_PATTERNS = (
    "um", "umm", "uh", "uhh", "erm", "hmm", "mmm",
    "you know", "i mean", "sort of", "kind of",
)

ANSWER_WEAK_THRESHOLD = 0.4
ANSWER_STRONG_THRESHOLD = 0.8

_SUBSTANCE_BANDS = ((10, 0.3), (30, 0.7))


def _score_substance(answer_text: str) -> float:
    """Is there actually an answer here?"""
    words = len(answer_text.split())
    if words == 0:
        return 0.0
    for limit, score in _SUBSTANCE_BANDS:
        if words < limit:
            return score
    return 1.0


def _count_filler_words(answer_text: str) -> int:
    text = answer_text.lower()
    total = 0
    for filler in _FILLER_PATTERNS:
        if " " in filler:
            total += text.count(filler)
        else:
            total += len(re.findall(
                r"(?<![a-z0-9])" + re.escape(filler) + r"(?![a-z0-9])", text))
    return total


def _score_filler_penalty(answer_text: str) -> tuple[int, float]:
    """Returns (filler_count, penalty)."""
    words = len(answer_text.split())
    count = _count_filler_words(answer_text)
    if words == 0 or count == 0:
        return count, 0.0
    ratio = count / words
    penalty = min(_FILLER_MAX_PENALTY, (ratio / 0.10) * _FILLER_MAX_PENALTY)
    return count, round(penalty, 3)


def _score_technical_density(answer_text: str, vocabulary: set) -> float:
    """How many known skills/technologies the answer actually names."""
    named = _count_named_entities(answer_text, vocabulary)
    if named == 0:
        return 0.0
    if named == 1:
        return 0.5
    return 1.0


def _score_star_completeness(parsed: dict) -> float:
    """Combine the four model-judged STAR components into one figure."""
    return round(sum(_STAR_COMPONENT_WEIGHTS[c] * parsed[c]
                     for c in _STAR_COMPONENT_WEIGHTS), 3)


def _evaluate_soft_answer(question: str, answer: str, targets_skill: str,
                          substance: float, filler_count: int,
                          filler_penalty: float) -> dict:
    """STAR-based evaluation for a behavioural question."""
    user_prompt = (
        f"Behavioural skill being probed: {targets_skill}\n\n"
        f"Interview question:\n{question}\n\n"
        f"Candidate's answer:\n{answer}"
    )
    parsed = _call_llm_json(SOFT_ANSWER_EVALUATION_PROMPT, user_prompt)
    if "error" in parsed:
        return parsed

    required = ("situation", "task", "action", "result", "relevance", "depth")
    for field in required:
        if not isinstance(parsed.get(field), (int, float)):
            return {"error": f"Soft-skill evaluation returned a non-numeric '{field}'.",
                    "raw_output": parsed}

    star_completeness = _score_star_completeness(parsed)
    score = (
        _SOFT_ANSWER_WEIGHTS["star_completeness"] * star_completeness
        + _SOFT_ANSWER_WEIGHTS["relevance"] * parsed["relevance"]
        + _SOFT_ANSWER_WEIGHTS["depth"] * parsed["depth"]
        + _SOFT_ANSWER_WEIGHTS["substance"] * substance
    ) - filler_penalty

    return {
        "final_score": round(max(0.0, min(1.0, score)), 3),
        "is_soft_skill": True,
        "substance": substance,
        "skill_addressed": None,
        "technical_density": None,
        "filler_count": filler_count,
        "filler_penalty": filler_penalty,
        "star": {c: parsed[c] for c in _STAR_COMPONENT_WEIGHTS},
        "star_completeness": star_completeness,
        "technical_accuracy": None,
        "relevance": parsed["relevance"],
        "depth": parsed["depth"],
        "feedback": parsed.get("feedback", ""),
        "llm_called": True,
    }


def evaluate_answer(question: str, answer: str, targets_skill: str,
                    vocabulary: set, is_soft_skill: bool = False) -> dict:
    """Phase 3: Evaluate a candidate's answer to one interview question."""
    answer = (answer or "").strip()

    substance = _score_substance(answer)
    filler_count, filler_penalty = _score_filler_penalty(answer)

    skill_addressed = (None if is_soft_skill
                       else (_target_is_named(answer, targets_skill) if answer else False))

    if substance == 0.0 or skill_addressed is False:
        return {
            "final_score": 0.0,
            "is_soft_skill": is_soft_skill,
            "substance": substance,
            "skill_addressed": skill_addressed,
            "technical_density": 0.0,
            "filler_count": filler_count,
            "filler_penalty": 0.0,
            "technical_accuracy": None,
            "relevance": None,
            "depth": None,
            "feedback": ("No substantive answer was given for this skill — the "
                         "answer does not engage with the topic asked about."),
            "llm_called": False,
        }

    if is_soft_skill:
        return _evaluate_soft_answer(question, answer, targets_skill,
                                     substance, filler_count, filler_penalty)

    technical_density = _score_technical_density(answer, vocabulary)

    user_prompt = (
        f"Skill being probed: {targets_skill}\n\n"
        f"Interview question:\n{question}\n\n"
        f"Candidate's answer:\n{answer}"
    )
    parsed = _call_llm_json(ANSWER_EVALUATION_PROMPT, user_prompt)
    if "error" in parsed:
        return parsed

    for field in ("technical_accuracy", "relevance", "depth"):
        if not isinstance(parsed.get(field), (int, float)):
            return {"error": f"Answer evaluation returned a non-numeric '{field}'.",
                    "raw_output": parsed}

    score = (
        _ANSWER_WEIGHTS["technical_accuracy"] * parsed["technical_accuracy"]
        + _ANSWER_WEIGHTS["relevance"] * parsed["relevance"]
        + _ANSWER_WEIGHTS["depth"] * parsed["depth"]
        + _ANSWER_WEIGHTS["technical_density"] * technical_density
        + _ANSWER_WEIGHTS["substance"] * substance
    ) - filler_penalty

    return {
        "final_score": round(max(0.0, min(1.0, score)), 3),
        "is_soft_skill": False,
        "substance": substance,
        "skill_addressed": True,
        "technical_density": technical_density,
        "filler_count": filler_count,
        "filler_penalty": filler_penalty,
        "technical_accuracy": parsed["technical_accuracy"],
        "relevance": parsed["relevance"],
        "depth": parsed["depth"],
        "feedback": parsed.get("feedback", ""),
        "llm_called": True,
    }


FOLLOWUP_QUESTION_PROMPT = """You are a technical interviewer. The candidate just
gave a strong answer about a skill, and you want to find the limit of what they
actually know.

Write ONE follow-up question that goes deeper.

Rules:

1. Build on what they actually said — quote or reference a specific term from
   their answer, so the question could not have been written in advance.
2. Ask about something they did NOT cover: a failure mode, an edge case, a
   trade-off, or the mechanism behind what they described.
3. Do not re-ask what they already answered.
4. The question MUST name the skill being probed in its own text.
5. Phrase it with a clear interrogative ("What/How/Why/Which") or a directive
   verb ("Describe/Explain/Compare"), ask for something concrete, and avoid
   vague openers like "Tell me about".
6. Return only JSON — no preamble, no markdown code fences.

Output format:

{{
  "question": "string (must name the skill, and reference their answer)",
  "targets_skill": "{skill}"
}}
"""

OFF_PLAN_QUESTION_PROMPT = """You are a technical interviewer. While answering a
different question, the candidate mentioned a skill that is relevant to this
role but was not part of the planned question set. You want to follow that
thread, as a real interviewer would.

Write ONE question about that skill.

Rules:

1. The question MUST name the skill "{skill}" in its own text.
2. Ask about a concrete task, scenario, or comparison — not a definition.
3. Phrase it with a clear interrogative ("What/How/Why/Which") or a directive
   verb ("Describe/Explain/Compare"), and avoid vague openers like
   "Tell me about".
4. Keep it self-contained and answerable in a spoken interview.
5. Return only JSON — no preamble, no markdown code fences.

Output format:

{{
  "question": "string (must name {skill})",
  "targets_skill": "{skill}"
}}
"""


SOFT_FOLLOWUP_QUESTION_PROMPT = """You are an interviewer. The candidate has just
described a real situation from their past, but their account is incomplete.

Here is how complete each part of their STAR answer was
(1.0 = given concretely, 0.5 = vague, 0.0 = absent):

  Situation = {situation}
  Task      = {task}
  Action    = {action}
  Result    = {result}

Write ONE follow-up question that fills in the WEAKEST part.

Rules:

1. Reference something specific the candidate actually said — a name, a
   decision, a deadline — so the question could not have been written before
   hearing their answer.
2. Ask for the missing part directly, and use its wording:
   - Situation weakest → ask for one concrete occasion, using the words
     "a specific situation where" or "a time when".
   - Task weakest    → ask what THEIR OWN role or responsibility was, using
     the words "your role" or "your responsibility".
   - Action weakest  → ask what they personally did, using the words
     "what did you do". This is the right choice when they told the story
     entirely in "we" and their own contribution never became visible.
   - Result weakest  → ask about the outcome, using the words
     "how did it turn out" or "what happened".
3. Keep those exact cue phrases. They are what makes the question answerable
   rather than an invitation to generalise.
4. Do NOT ask about technical mechanisms, failure modes, edge cases or
   trade-offs. This is a behavioural question, not a technical one.
5. One question, answerable out loud, ending in "?".
6. Return only JSON — no preamble, no markdown code fences.

Output format:

{{
  "question": "string",
  "targets_skill": "{skill}"
}}
"""

SOFT_OFF_PLAN_QUESTION_PROMPT = """You are an interviewer. While answering a
different question, the candidate revealed something about the behavioural
skill "{skill}", which was not part of the planned question set. You want to
follow that thread, as a real interviewer would.

Write ONE behavioural question about "{skill}", using the STAR framework.

Rules:

1. Anchor it in ONE specific past episode. Use the words "a specific situation
   where" or "a time when" — never "what would you do if" and never "how do
   you usually".
2. In the same question, also ask for their own role, what they personally
   did, and how it turned out.
3. Reference what they just said, so the question is clearly a response to
   them rather than a generic prompt.
4. One question, answerable out loud, ending in "?".
5. Return only JSON — no preamble, no markdown code fences.

Output format:

{{
  "question": "string",
  "targets_skill": "{skill}"
}}
"""

MAX_PROBE_DEPTH = 2
MAX_OFF_PLAN_QUESTIONS = 2
MAX_TOTAL_QUESTIONS = 12
_MAX_QUESTION_GEN_RETRIES = 2


def _generate_gated_question(system_prompt: str, user_prompt: str, skill: str,
                             vocabulary: set, is_soft_skill: bool,
                             situation_established: bool = False) -> dict:
    """Generate one question and hold it to the same Answerability gate used for the planned set."""
    for _ in range(_MAX_QUESTION_GEN_RETRIES):
        parsed = _call_llm_json(system_prompt, user_prompt)
        if "error" in parsed:
            continue
        question_text = parsed.get("question")
        if not isinstance(question_text, str) or not question_text.strip():
            continue

        parsed["targets_skill"] = parsed.get("targets_skill") or skill
        parsed["is_soft_skill"] = is_soft_skill
        parsed.update(_compute_answerability(
            question_text, parsed["targets_skill"], vocabulary, is_soft_skill,
            situation_established))
        parsed["is_gap_skill"] = False
        if parsed["passes_gate"]:
            return parsed
    return None


def generate_followup(original_question: str, answer: str, targets_skill: str,
                      vocabulary: set, is_soft_skill: bool = False,
                      star: dict | None = None) -> dict:
    """Generate a deeper follow-up grounded in what the candidate actually said."""
    user_prompt = (
        f"Skill being probed: {targets_skill}\n\n"
        f"Question already asked:\n{original_question}\n\n"
        f"Candidate's answer:\n{answer}"
    )

    if is_soft_skill:
        star = star or {}
        system_prompt = SOFT_FOLLOWUP_QUESTION_PROMPT.format(
            skill=targets_skill,
            situation=star.get("situation", 0.0),
            task=star.get("task", 0.0),
            action=star.get("action", 0.0),
            result=star.get("result", 0.0),
        )
        return _generate_gated_question(system_prompt, user_prompt, targets_skill,
                                        vocabulary, True, situation_established=True)

    return _generate_gated_question(
        FOLLOWUP_QUESTION_PROMPT.format(skill=targets_skill),
        user_prompt, targets_skill, vocabulary, False)


def generate_off_plan_question(skill_name: str, answer_context: str,
                               vocabulary: set, is_soft_skill: bool = False) -> dict:
    """Generate a question about a skill the candidate raised themselves, outside the planned set."""
    user_prompt = (
        f"Skill to ask about: {skill_name}\n\n"
        f"The candidate mentioned it while saying:\n{answer_context}"
    )
    prompt = (SOFT_OFF_PLAN_QUESTION_PROMPT if is_soft_skill
              else OFF_PLAN_QUESTION_PROMPT)
    return _generate_gated_question(
        prompt.format(skill=skill_name),
        user_prompt, skill_name, vocabulary, is_soft_skill)


AGENT_SYSTEM_PROMPT = """You are a technical interviewer running an adaptive
interview. After each answer you receive a numeric evaluation, and you decide
what to do next — exactly as an experienced interviewer would.

Reason step by step before acting:
  1. What do the evaluation components say? Accuracy, relevance and depth are
     SEPARATE signals — read them individually, not just the final score.
  2. Do you need more information about the interview state before deciding?
  3. Which tool serves the goal?

How to read the components:

• LOW accuracy + LOW depth    → they do not know this topic → move on.
• HIGH accuracy + LOW depth   → they know it but did not explain → probe deeper.
• HIGH accuracy + HIGH depth  → topic is covered → continue to the next skill.
• LOW accuracy + HIGH depth   → they talked at length but were WRONG. This is
  worse than a short correct answer, not better. Move on.

Also: if the candidate mentioned a named skill that is relevant to the role but
was not the subject of the question, you may follow that thread with
ask_about_skill — a real interviewer notices such openings.

Constraints:
- Do not move on if no other skills remain — check the state first.
- Do not probe the same skill more than twice.
- Always give a concrete reason, citing the numbers you based it on.

Call exactly one decision tool (move_to_another_skill, probe_deeper,
ask_about_skill, or continue_as_planned) once you have decided. You may call
get_interview_state first if you need it.
"""


if __name__ == "__main__":
    sample_cv_text = """
    Aisha is a final-year Electrical and Electronic Engineering student specializing
    in Automatic Control. Experience with Arduino, ESP32, and MPU6050 sensors for
    embedded IoT projects. Built a Smart Car Black Box System using AWS-free local
    MySQL storage, OLED display integration, and Blynk for cloud connectivity.
    Familiar with Python, MATLAB, and C for control systems coursework. Has worked
    with ROS 2 (Humble) and Turtlesim for robotics simulation. Member of IEEE and
    the Lybotics Wizards Robotics Team. Strong communication and teaching skills,
    having taught Arduino and computer science at a school level.
    """

    print("Running Phase 1 skill extraction test...\n")
    result = extract_skills(sample_cv_text, document_type="CV")
    print(json.dumps(result, indent=2, ensure_ascii=False))
