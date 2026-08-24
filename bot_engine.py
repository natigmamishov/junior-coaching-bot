"""
Junior Coaching — Conversation Engine V9

Core logic:
Understand
→ Extract
→ Resolve entities / corrections
→ Re-evaluate situation
→ Check business rules
→ Decide response strategy
→ Generate concise answer
→ Decide next step
→ Persist state

V9 priorities:
1. Primary intent first
2. Minimal-friction lead flow
3. Consultative discovery
4. Contextual reasoning
5. Business-rule gating
6. State recall
7. Multi-child entity handling
8. No guessed names
9. No diagnosis
10. Short WhatsApp / Instagram-style responses
"""

import os
import re
import json
import sqlite3

from datetime import datetime
from typing import Optional, List, Dict, Any
from zoneinfo import ZoneInfo

import httpx
import numpy as np

from dotenv import load_dotenv
from openai import OpenAI

from sklearn.pipeline import FeatureUnion
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity


# =========================================================
# 0. PATHS
# =========================================================

BASE_DIR = os.path.dirname(
    os.path.abspath(__file__)
)

DATASET_PATH = os.path.join(
    BASE_DIR,
    "Junior_Coaching_sesli_AI_FAQ.txt",
)

DB_PATH = os.path.join(
    BASE_DIR,
    "junior_coaching.db",
)


# =========================================================
# 1. OPENAI
# =========================================================

load_dotenv(
    os.path.join(
        BASE_DIR,
        ".env",
    )
)

OPENAI_API_KEY = os.getenv(
    "OPENAI_API_KEY"
)

client: Optional[OpenAI] = None

if OPENAI_API_KEY:

    http_client = httpx.Client(
        verify=False,
        timeout=60,
    )

    client = OpenAI(
        api_key=OPENAI_API_KEY,
        http_client=http_client,
    )


# =========================================================
# 2. NORMALIZATION
# =========================================================

AZ_TRANSLATION = str.maketrans({
    "ə": "e",
    "Ə": "e",
    "ı": "i",
    "İ": "i",
    "ö": "o",
    "Ö": "o",
    "ü": "u",
    "Ü": "u",
    "ş": "s",
    "Ş": "s",
    "ç": "c",
    "Ç": "c",
    "ğ": "g",
    "Ğ": "g",
})


def normalize_text(text: str) -> str:

    return re.sub(
        r"\s+",
        " ",
        str(text).strip().lower(),
    )


def normalize_for_search(text: str) -> str:

    text = normalize_text(text)
    text = text.translate(AZ_TRANSLATION)

    text = re.sub(
        r"[^\w\s?]",
        " ",
        text,
    )

    return re.sub(
        r"\s+",
        " ",
        text,
    ).strip()


# =========================================================
# 3. PHONE
# =========================================================

def normalize_phone(text: str) -> Optional[str]:

    digits = re.sub(
        r"\D",
        "",
        str(text),
    )

    if (
        len(digits) == 10
        and digits.startswith("0")
    ):
        return digits

    if (
        len(digits) == 12
        and digits.startswith("994")
    ):
        return digits

    return None


# =========================================================
# 4. AGE
# =========================================================

def extract_contextual_ages(
    text: str,
) -> List[int]:

    value = normalize_for_search(
        text
    )

    ages = []

    patterns = [
        r"\b(\d{1,2})\s*yas\b",
        r"\b(\d{1,2})\s*yasi\b",
        r"\b(\d{1,2})\s*yasli\b",
        r"\b(\d{1,2})\s*yasinda\b",
    ]

    for pattern in patterns:

        for item in re.findall(
            pattern,
            value,
        ):

            age = int(item)

            if (
                5 <= age <= 25
                and age not in ages
            ):
                ages.append(age)

    return ages


def extract_simple_age(
    text: str,
) -> Optional[int]:

    if normalize_phone(text):
        return None

    value = normalize_for_search(
        text
    )

    if re.search(
        r"\b\d{1,2}\s*[:.-]\s*\d{1,2}\b",
        value,
    ):
        return None

    numbers = re.findall(
        r"\b\d{1,2}\b",
        value,
    )

    if len(numbers) != 1:
        return None

    age = int(numbers[0])

    if 5 <= age <= 25:
        return age

    return None


# =========================================================
# 5. GREETING
# =========================================================

def is_greeting(
    text: str,
) -> bool:

    value = normalize_for_search(
        text
    )

    return value in {
        "salam",
        "slm",
        "salamlar",
        "hello",
        "hi",
        "hey",
        "salam necesiz",
        "salam necesiniz",
        "salam aleykum",
        "aleykum salam",
    }


# =========================================================
# 6. NAME SAFETY
# =========================================================

NAME_BLOCKLIST = {
    "men",
    "mene",
    "bu",
    "nomre",
    "nomreyle",
    "nomreile",
    "elaqe",
    "saxlayin",
    "anasi",
    "anasiyam",
    "atasi",
    "atasiyam",
    "oglum",
    "qizim",
    "usaq",
    "usaqdir",
    "ovlad",
    "ovladim",
    "valideyn",
    "valideynem",
    "salam",
    "adim",
    "adi",
    "deyirem",
    "dedim",
    "maraqlaniram",
    "proqram",
}


def clean_name(
    value: str,
) -> Optional[str]:

    if not value:
        return None

    value = str(value).strip()

    value = re.sub(
        r"(?i)\b(xanım|xanim|bəy|bey|müəllim|muellim)\b",
        "",
        value,
    )

    value = re.sub(
        r"[.,!?+():;]",
        " ",
        value,
    )

    value = re.sub(
        r"\s+",
        " ",
        value,
    ).strip()

    if not value:
        return None

    words = value.split()

    if len(words) > 2:
        return None

    cleaned = []

    for word in words:

        normalized = normalize_for_search(
            word
        )

        if normalized in NAME_BLOCKLIST:
            return None

        if not re.fullmatch(
            r"[A-Za-zƏəÖöÜüĞğÇçŞşİı\-]+",
            word,
        ):
            return None

        cleaned.append(word)

    result = " ".join(cleaned).title()

    if len(result) < 2:
        return None

    return result


def deterministic_parent_name_extract(
    text: str,
) -> Optional[str]:

    patterns = [
        r"(?i)\bmənim\s+adım\s+([A-Za-zƏəÖöÜüĞğÇçŞşİı\-]+)",
        r"(?i)\bmenim\s+adim\s+([A-Za-zƏəÖöÜüĞğÇçŞşİı\-]+)",
        r"(?i)\badım\s+([A-Za-zƏəÖöÜüĞğÇçŞşİı\-]+)",
        r"(?i)\badim\s+([A-Za-zƏəÖöÜüĞğÇçŞşİı\-]+)",
    ]

    for pattern in patterns:

        match = re.search(
            pattern,
            text,
        )

        if match:

            return clean_name(
                match.group(1)
            )

    return None


# =========================================================
# 7. AZ SUFFIX
# =========================================================

def get_last_vowel(
    word: str,
) -> Optional[str]:

    vowels = "aıoueəiöü"

    for char in reversed(
        word.lower()
    ):

        if char in vowels:
            return char

    return None


def get_genitive_suffix(
    word: str,
) -> str:

    vowel = get_last_vowel(
        word
    )

    if vowel in ["a", "ı"]:
        return "ın"

    if vowel in ["e", "ə", "i"]:
        return "in"

    if vowel in ["o", "u"]:
        return "un"

    if vowel in ["ö", "ü"]:
        return "ün"

    return "ın"


def child_genitive(
    name: str,
) -> str:

    if not name:
        return "övladınızın"

    suffix = get_genitive_suffix(
        name
    )

    if name[-1].lower() in "aıoueəiöü":
        return name + "n" + suffix

    return name + suffix


# =========================================================
# 8. CHILD / CONSULTATION STATE
# =========================================================

def create_empty_consultation() -> dict:

    return {
        "context": None,
        "duration": None,
        "impact": None,
        "future_concern": None,
        "desired_outcome": None,
    }


def create_empty_reasoning_state() -> dict:

    return {
        "current_hypothesis": [],
        "supporting_evidence": [],
        "contradicting_evidence": [],
        "confidence": None,
    }


def create_empty_child() -> dict:

    return {
        "name": None,
        "age": None,

        "raw_concern": None,
        "main_concern": None,
        "interpreted_needs": [],

        "consultation": create_empty_consultation(),
        "reasoning_state": create_empty_reasoning_state(),

        "needs_concern_followup": False,
        "concern_duration": None,
        "concern_onset": None,
    }


def create_empty_lead(
    source: str = "CLI",
) -> dict:

    return {
        "parent_name": None,
        "parent_title": None,

        "child_name": None,
        "child_age": None,
        "main_concern": None,

        "needs_concern_followup": False,
        "concern_duration": None,
        "concern_onset": None,

        "children": [
            create_empty_child()
        ],

        "declared_child_count": 1,
        "multiple_children": False,
        "active_child_index": 0,

        "phone": None,
        "preferred_call_time": None,

        "source": source,
        "status": "NEW",

        "_greeted": False,
        "_flow_started": False,

        "_last_intent": None,
        "_last_primary_intent": None,
        "_last_confidence": None,
        "_last_faq_score": None,

        "_consultative_turns": 0,
    }


def ensure_lead_structure(
    lead: dict,
):

    if "children" not in lead:
        lead["children"] = [
            create_empty_child()
        ]

    if not lead["children"]:
        lead["children"] = [
            create_empty_child()
        ]

    for child in lead["children"]:

        child.setdefault(
            "name",
            None,
        )

        child.setdefault(
            "age",
            None,
        )

        child.setdefault(
            "raw_concern",
            child.get(
                "main_concern"
            ),
        )

        child.setdefault(
            "main_concern",
            None,
        )

        child.setdefault(
            "interpreted_needs",
            [],
        )

        child.setdefault(
            "consultation",
            create_empty_consultation(),
        )

        child.setdefault(
            "reasoning_state",
            create_empty_reasoning_state(),
        )

        child.setdefault(
            "needs_concern_followup",
            False,
        )

        child.setdefault(
            "concern_duration",
            None,
        )

        child.setdefault(
            "concern_onset",
            None,
        )

    lead.setdefault(
        "declared_child_count",
        1,
    )

    lead.setdefault(
        "multiple_children",
        False,
    )

    lead.setdefault(
        "active_child_index",
        0,
    )

    lead.setdefault(
        "_greeted",
        False,
    )

    lead.setdefault(
        "_flow_started",
        False,
    )

    lead.setdefault(
        "_consultative_turns",
        0,
    )


def get_active_child(
    lead: dict,
) -> dict:

    ensure_lead_structure(
        lead
    )

    index = lead.get(
        "active_child_index",
        0,
    )

    if (
        index < 0
        or index >= len(
            lead["children"]
        )
    ):
        index = 0
        lead["active_child_index"] = 0

    return lead[
        "children"
    ][index]


def sync_flat_fields(
    lead: dict,
):

    ensure_lead_structure(
        lead
    )

    first = lead["children"][0]

    lead["child_name"] = first.get(
        "name"
    )

    lead["child_age"] = first.get(
        "age"
    )

    lead["main_concern"] = first.get(
        "main_concern"
    )

    lead[
        "needs_concern_followup"
    ] = first.get(
        "needs_concern_followup",
        False,
    )

    lead[
        "concern_duration"
    ] = first.get(
        "concern_duration"
    )

    lead[
        "concern_onset"
    ] = first.get(
        "concern_onset"
    )


# =========================================================
# 9. CHILD COUNT
# =========================================================

def detect_explicit_child_count(
    text: str,
) -> Optional[int]:

    value = normalize_for_search(
        text
    )

    one_child = [
        "bir oglum var",
        "bir qizim var",
        "bir usagim var",
        "bir ovladim var",
        "tek usagim var",
        "2 ci ovlad yoxdur",
        "ikinci ovlad yoxdur",
        "ikinci usaq yoxdur",
    ]

    if any(
        x in value
        for x in one_child
    ):
        return 1

    two_child = [
        "iki usagim",
        "iki ovladim",
        "2 usagim",
        "2 ovladim",
        "iki usaq var",
        "2 usaq var",
    ]

    if any(
        x in value
        for x in two_child
    ):
        return 2

    match = re.search(
        r"\b([2-5])\s*(?:usaq|ovlad)",
        value,
    )

    if match:
        return int(
            match.group(1)
        )

    return None


def set_child_count(
    lead: dict,
    count: int,
):

    ensure_lead_structure(
        lead
    )

    count = max(
        1,
        min(
            int(count),
            5,
        )
    )

    lead[
        "declared_child_count"
    ] = count

    lead[
        "multiple_children"
    ] = count > 1

    while len(
        lead["children"]
    ) < count:

        lead["children"].append(
            create_empty_child()
        )

    if count == 1:

        lead["children"] = [
            lead["children"][0]
        ]

        lead[
            "active_child_index"
        ] = 0

    else:

        lead["children"] = (
            lead["children"][:count]
        )

    sync_flat_fields(
        lead
    )


# =========================================================
# 10. TITLE
# =========================================================

def infer_parent_title_with_llm(
    name: str,
) -> str:

    if (
        not name
        or client is None
    ):
        return ""

    try:

        response = client.chat.completions.create(
            model="gpt-4o-mini",
            temperature=0,

            messages=[
                {
                    "role": "system",
                    "content": """
Azərbaycan adına əsasən müraciət formasını seç.

Yalnız:
xanım
bəy
neutral

Əmin deyilsənsə neutral.
"""
                },
                {
                    "role": "user",
                    "content": name,
                },
            ],

            response_format={
                "type": "json_schema",

                "json_schema": {
                    "name": "parent_title",
                    "strict": True,

                    "schema": {
                        "type": "object",

                        "properties": {
                            "title": {
                                "type": "string",

                                "enum": [
                                    "xanım",
                                    "bəy",
                                    "neutral",
                                ],
                            }
                        },

                        "required": [
                            "title"
                        ],

                        "additionalProperties": False,
                    },
                },
            },
        )

        result = json.loads(
            response.choices[0].message.content
        )

        title = result.get(
            "title",
            "neutral",
        )

        if title in {
            "xanım",
            "bəy",
        }:
            return title

    except Exception as exc:
        print(
            "TITLE ERROR:",
            exc,
        )

    return ""


def get_parent_display_name(
    lead: dict,
) -> str:

    name = lead.get(
        "parent_name"
    )

    title = lead.get(
        "parent_title"
    )

    if not name:
        return ""

    if title:
        return f"{name} {title}"

    return name


# =========================================================
# 11. KNOWLEDGE BASE
# =========================================================

def build_faq_index():

    if not os.path.exists(
        DATASET_PATH
    ):
        raise FileNotFoundError(
            DATASET_PATH
        )

    with open(
        DATASET_PATH,
        "r",
        encoding="utf-8",
    ) as file:

        raw = file.read()

    pattern = re.compile(
        r"(?:\d+\.\s*)?Sual:\s*(.*?)\s*"
        r"(?:Agent|Selnaz|Cavab):\s*(.*?)"
        r"(?=\n\s*(?:\d+\.\s*)?Sual:|\Z)",
        re.DOTALL | re.IGNORECASE,
    )

    pairs = []

    for question, answer in pattern.findall(
        raw
    ):

        question = re.sub(
            r"\s+",
            " ",
            question,
        ).strip()

        answer = re.sub(
            r"\s+",
            " ",
            answer,
        ).strip()

        if question and answer:

            pairs.append(
                (
                    question,
                    answer,
                )
            )

    if not pairs:
        raise ValueError(
            "FAQ faylında sual/cavab tapılmadı."
        )

    questions = [
        q
        for q, _
        in pairs
    ]

    answers = [
        a
        for _, a
        in pairs
    ]

    normalized_questions = [
        normalize_for_search(q)
        for q in questions
    ]

    vectorizer = FeatureUnion([
        (
            "word",
            TfidfVectorizer(
                analyzer="word",
                ngram_range=(1, 2),
                max_features=60000,
                sublinear_tf=True,
            ),
        ),
        (
            "char",
            TfidfVectorizer(
                analyzer="char_wb",
                ngram_range=(3, 5),
                max_features=80000,
                sublinear_tf=True,
            ),
        ),
    ])

    matrix = vectorizer.fit_transform(
        normalized_questions
    )

    return (
        questions,
        answers,
        vectorizer,
        matrix,
    )


(
    FAQ_QUESTIONS,
    FAQ_ANSWERS,
    FAQ_VECTORIZER,
    FAQ_MATRIX,
) = build_faq_index()


def retrieve_knowledge(
    query: str,
    k: int = 6,
) -> List[dict]:

    query_vector = FAQ_VECTORIZER.transform(
        [
            normalize_for_search(
                query
            )
        ]
    )

    scores = cosine_similarity(
        query_vector,
        FAQ_MATRIX,
    ).ravel()

    indexes = np.argsort(
        -scores
    )[:k]

    result = []

    for index in indexes:

        result.append({
            "question": FAQ_QUESTIONS[
                index
            ],

            "answer": FAQ_ANSWERS[
                index
            ],

            "score": float(
                scores[index]
            ),
        })

    return result


# =========================================================
# 12. MESSAGE UNDERSTANDING
# =========================================================

TOPICS = [
    "meeting_location",
    "meeting_day",
    "meeting_frequency",
    "meeting_duration",
    "parent_call_duration",
    "child_intro_call",
    "missed_session",
    "one_time_session",
    "price",
    "program_info",
    "program_age",
    "eligibility",
    "child_refusal",
    "language",
    "registration",
    "situation_advice",
    "state_recall",
    "other",
]


def analyze_message(
    user_text: str,
    lead: dict,
    history: Optional[List[dict]] = None,
) -> dict:

    ensure_lead_structure(
        lead
    )

    history = history or []

    state_payload = {
        "parent_name": lead.get(
            "parent_name"
        ),

        "children": lead.get(
            "children"
        ),

        "declared_child_count": lead.get(
            "declared_child_count",
            1,
        ),

        "active_child_index": lead.get(
            "active_child_index",
            0,
        ),

        "phone": lead.get(
            "phone"
        ),

        "preferred_call_time": lead.get(
            "preferred_call_time"
        ),
    }

    fallback = {
        "primary_intent": (
            "greeting"
            if is_greeting(
                user_text
            )
            else "field_answer"
        ),

        "secondary_intents": [],

        "questions": [],

        "parent_name": "",
        "child_name": "",
        "child_age": 0,

        "raw_concern": "",
        "concern_summary": "",

        "interpreted_needs": [],

        "consultation_update": {
            "context": "",
            "duration": "",
            "impact": "",
            "future_concern": "",
            "desired_outcome": "",
        },

        "phone": "",
        "preferred_call_time": "",

        "child_count": 0,

        "corrections": [],

        "entity_operation": "none",

        "new_child": {
            "name": "",
            "age": 0,
            "raw_concern": "",
        },

        "recall_fields": [],

        "has_situation_description": False,
        "should_reassess_hypothesis": False,

        "needs_clarification": False,
        "clarification_question": "",

        "needs_consultative_followup": False,
        "best_consultative_question": "",

        "should_pause_sales_flow": False,

        "confidence": 0.0,
    }

    if client is None:
        return fallback

    system_prompt = """
You are the conversation understanding and planning layer
for Junior Coaching Sales & Operations Assistant.

Analyze:
- current message
- previous conversation
- structured state

Do NOT answer the parent.
Return structured understanding.

========================================
PRIMARY INTENT
========================================

Identify what the user MOST wants from this turn.

Example:

"Qızımın 11 yaşı var, 3 aya 12 olacaq.
İndi Junior Coaching-ə başlaya bilər?"

PRIMARY intent:
eligibility_question

Age=11 is a FACT, not the primary response.

Never allow extracted state information to replace
the user's main question.

primary_intent must be one of:

greeting
field_answer
faq_question
eligibility_question
program_interest
registration_request
state_question
situation_analysis
correction
permission_question
human_agent_request
complaint
safety_risk
pause_request
unrelated


========================================
EXTRACT ALL FACTS
========================================

Extract all facts from the whole message.

Do not invent.

Unknown = empty.


========================================
NEED UNDERSTANDING
========================================

Parent does not need to use category keywords.

Example:

"Evdə rahat danışır,
məktəbdə bildiyi cavabı deməyə çəkinir."

raw_concern:
exact parent description

concern_summary:
"sosial mühitdə özünüifadədə çətinlik"

interpreted_needs:
["özünüifadə", "özgüvən"]

These are hypotheses, not diagnoses.


========================================
NEW EVIDENCE / HYPOTHESIS REVISION
========================================

If the parent provides new facts that change
the earlier picture, set:

should_reassess_hypothesis=true

Example:

Earlier:
"məktəbdə danışmır"

New:
"dostları ilə rahatdır,
tədbirlərdə çıxış edir,
fikrini sərbəst deyir,
sadəcə müəllimin sualına bəzən həvəsi olmur"

This contradicts a general self-confidence hypothesis.

Do NOT preserve stale reasoning just because it existed before.


========================================
CONSULTATIVE DISCOVERY
========================================

The assistant should use:

Minimum questions → maximum insight.

Only set needs_consultative_followup=true
when one short question would materially help understand:

- context
- duration
- impact
- parent's future concern
- desired outcome

Do NOT ask all of them like a checklist.

Choose ONE most useful unanswered question.

Examples:

"This more often happens in which situations?"

"How long have you noticed this?"

"How does this currently affect school or relationships?"

"If this changed, what would you most want to see differently?"

Only ask if genuinely useful.


========================================
MULTI-CHILD
========================================

Differentiate:

CORRECTION:
"Muradın 16 yox, 15 yaşı var."
→ update same child

NEW CHILD / SWITCH:
"Əslində Muradı yox, 8 yaşlı kiçik oğlumu
gətirmək istəyirəm."

→ entity_operation="switch_to_new_child"

Do NOT overwrite Murad's age to 8.

If a new child is mentioned but identity is not clear,
create new child only when wording clearly indicates another child.

If unclear:
needs_clarification=true.


========================================
STATE RECALL
========================================

"Mənim adımı və uşağın yaşını necə qeyd etmisiniz?"

recall_fields=[
 "parent_name",
 "child_age"
]

Do not treat as FAQ.


========================================
BUSINESS RULE QUESTIONS
========================================

Questions like:

"11 yaşı var, 3 aya 12 olacaq. İndi başlaya bilər?"

topic=eligibility

If exact exception is not known,
the answer layer must not invent it.


========================================
NAMES
========================================

Only extract names if clearly names.

"Bu nömrə ilə əlaqə saxlayın"
parent_name=""

Do not infer arbitrary phrase as name.


========================================
QUESTIONS
========================================

Split all distinct questions.

"Harada və hansı gün keçirilir?"
→ location + day.


========================================
FLOW PAUSE
========================================

should_pause_sales_flow=true when:
- unresolved question/objection exists
- parent is still exploring
- asking the next form question would feel pushy

false when it is natural to progress.


========================================
OUTPUT
========================================
"""

    payload = f"""
CURRENT STATE:
{json.dumps(state_payload, ensure_ascii=False)}

RECENT HISTORY:
{json.dumps(history[-12:], ensure_ascii=False)}

CURRENT MESSAGE:
{user_text}
"""

    try:

        response = client.chat.completions.create(
            model="gpt-4o-mini",
            temperature=0,

            messages=[
                {
                    "role": "system",
                    "content": system_prompt,
                },
                {
                    "role": "user",
                    "content": payload,
                },
            ],

            response_format={
                "type": "json_schema",

                "json_schema": {
                    "name": "conversation_plan",
                    "strict": True,

                    "schema": {
                        "type": "object",

                        "properties": {

                            "primary_intent": {
                                "type": "string",

                                "enum": [
                                    "greeting",
                                    "field_answer",
                                    "faq_question",
                                    "eligibility_question",
                                    "program_interest",
                                    "registration_request",
                                    "state_question",
                                    "situation_analysis",
                                    "correction",
                                    "permission_question",
                                    "human_agent_request",
                                    "complaint",
                                    "safety_risk",
                                    "pause_request",
                                    "unrelated",
                                ],
                            },

                            "secondary_intents": {
                                "type": "array",
                                "items": {
                                    "type": "string"
                                },
                            },

                            "questions": {
                                "type": "array",

                                "items": {
                                    "type": "object",

                                    "properties": {
                                        "text": {
                                            "type": "string"
                                        },

                                        "topic": {
                                            "type": "string",
                                            "enum": TOPICS,
                                        },
                                    },

                                    "required": [
                                        "text",
                                        "topic",
                                    ],

                                    "additionalProperties": False,
                                },
                            },

                            "parent_name": {
                                "type": "string"
                            },

                            "child_name": {
                                "type": "string"
                            },

                            "child_age": {
                                "type": "integer"
                            },

                            "raw_concern": {
                                "type": "string"
                            },

                            "concern_summary": {
                                "type": "string"
                            },

                            "interpreted_needs": {
                                "type": "array",
                                "items": {
                                    "type": "string"
                                },
                            },

                            "consultation_update": {
                                "type": "object",

                                "properties": {
                                    "context": {
                                        "type": "string"
                                    },

                                    "duration": {
                                        "type": "string"
                                    },

                                    "impact": {
                                        "type": "string"
                                    },

                                    "future_concern": {
                                        "type": "string"
                                    },

                                    "desired_outcome": {
                                        "type": "string"
                                    },
                                },

                                "required": [
                                    "context",
                                    "duration",
                                    "impact",
                                    "future_concern",
                                    "desired_outcome",
                                ],

                                "additionalProperties": False,
                            },

                            "phone": {
                                "type": "string"
                            },

                            "preferred_call_time": {
                                "type": "string"
                            },

                            "child_count": {
                                "type": "integer"
                            },

                            "corrections": {
                                "type": "array",

                                "items": {
                                    "type": "object",

                                    "properties": {
                                        "field": {
                                            "type": "string"
                                        },

                                        "value": {
                                            "type": "string"
                                        },

                                        "child_index": {
                                            "type": "integer"
                                        },
                                    },

                                    "required": [
                                        "field",
                                        "value",
                                        "child_index",
                                    ],

                                    "additionalProperties": False,
                                },
                            },

                            "entity_operation": {
                                "type": "string",

                                "enum": [
                                    "none",
                                    "switch_to_existing_child",
                                    "switch_to_new_child",
                                ],
                            },

                            "new_child": {
                                "type": "object",

                                "properties": {
                                    "name": {
                                        "type": "string"
                                    },

                                    "age": {
                                        "type": "integer"
                                    },

                                    "raw_concern": {
                                        "type": "string"
                                    },
                                },

                                "required": [
                                    "name",
                                    "age",
                                    "raw_concern",
                                ],

                                "additionalProperties": False,
                            },

                            "recall_fields": {
                                "type": "array",

                                "items": {
                                    "type": "string",

                                    "enum": [
                                        "parent_name",
                                        "child_name",
                                        "child_age",
                                        "main_concern",
                                        "phone",
                                        "preferred_call_time",
                                        "child_count",
                                    ],
                                },
                            },

                            "has_situation_description": {
                                "type": "boolean"
                            },

                            "should_reassess_hypothesis": {
                                "type": "boolean"
                            },

                            "needs_clarification": {
                                "type": "boolean"
                            },

                            "clarification_question": {
                                "type": "string"
                            },

                            "needs_consultative_followup": {
                                "type": "boolean"
                            },

                            "best_consultative_question": {
                                "type": "string"
                            },

                            "should_pause_sales_flow": {
                                "type": "boolean"
                            },

                            "confidence": {
                                "type": "number"
                            },
                        },

                        "required": [
                            "primary_intent",
                            "secondary_intents",
                            "questions",
                            "parent_name",
                            "child_name",
                            "child_age",
                            "raw_concern",
                            "concern_summary",
                            "interpreted_needs",
                            "consultation_update",
                            "phone",
                            "preferred_call_time",
                            "child_count",
                            "corrections",
                            "entity_operation",
                            "new_child",
                            "recall_fields",
                            "has_situation_description",
                            "should_reassess_hypothesis",
                            "needs_clarification",
                            "clarification_question",
                            "needs_consultative_followup",
                            "best_consultative_question",
                            "should_pause_sales_flow",
                            "confidence",
                        ],

                        "additionalProperties": False,
                    },
                },
            },
        )

        return json.loads(
            response.choices[
                0
            ].message.content
        )

    except Exception as exc:

        print(
            "ANALYZER ERROR:",
            exc,
        )

        return fallback


# =========================================================
# 13. ENTITY SWITCHING
# =========================================================

def apply_entity_operation(
    lead: dict,
    analysis: dict,
) -> Optional[str]:

    ensure_lead_structure(
        lead
    )

    operation = analysis.get(
        "entity_operation",
        "none",
    )

    if operation != "switch_to_new_child":
        return None

    new_child_data = analysis.get(
        "new_child",
        {},
    )

    new_child = create_empty_child()

    name = clean_name(
        new_child_data.get(
            "name",
            ""
        )
    )

    age = new_child_data.get(
        "age",
        0,
    )

    raw_concern = new_child_data.get(
        "raw_concern",
        ""
    ).strip()

    if name:
        new_child[
            "name"
        ] = name

    if (
        isinstance(
            age,
            int,
        )
        and 5 <= age <= 25
    ):
        new_child[
            "age"
        ] = age

    if raw_concern:
        new_child[
            "raw_concern"
        ] = raw_concern

        new_child[
            "main_concern"
        ] = raw_concern

    lead[
        "children"
    ].append(
        new_child
    )

    lead[
        "declared_child_count"
    ] = len(
        lead["children"]
    )

    lead[
        "multiple_children"
    ] = True

    lead[
        "active_child_index"
    ] = (
        len(
            lead["children"]
        )
        - 1
    )

    sync_flat_fields(
        lead
    )

    return (
        "Aydındır, indi kiçik övladınız üçün baxaq."
    )


# =========================================================
# 14. APPLY CORRECTIONS
# =========================================================

def apply_corrections(
    lead: dict,
    corrections: List[dict],
) -> List[str]:

    ensure_lead_structure(
        lead
    )

    confirmations = []

    for item in corrections:

        field = item.get(
            "field",
            "",
        )

        value = str(
            item.get(
                "value",
                "",
            )
        ).strip()

        child_index = max(
            0,
            int(
                item.get(
                    "child_index",
                    0,
                )
            ),
        )

        if not value:
            continue

        while len(
            lead["children"]
        ) <= child_index:

            lead[
                "children"
            ].append(
                create_empty_child()
            )

        if field == "parent_name":

            name = clean_name(
                value
            )

            if name:

                lead[
                    "parent_name"
                ] = name

                lead[
                    "parent_title"
                ] = infer_parent_title_with_llm(
                    name
                )

                confirmations.append(
                    f"Adınızı {name} olaraq yenilədim."
                )

        elif field == "child_name":

            name = clean_name(
                value
            )

            if name:

                lead[
                    "children"
                ][child_index][
                    "name"
                ] = name

                confirmations.append(
                    f"Övladınızın adını {name} olaraq yenilədim."
                )

        elif field == "child_age":

            match = re.search(
                r"\b(\d{1,2})\b",
                value,
            )

            if match:

                age = int(
                    match.group(1)
                )

                if 5 <= age <= 25:

                    lead[
                        "children"
                    ][child_index][
                        "age"
                    ] = age

                    confirmations.append(
                        f"Yaşı {age} olaraq yenilədim."
                    )

        elif field in {
            "raw_concern",
            "main_concern",
        }:

            child = lead[
                "children"
            ][child_index]

            child[
                "raw_concern"
            ] = value

            child[
                "main_concern"
            ] = value

            # stale interpretation reset
            child[
                "interpreted_needs"
            ] = []

            child[
                "reasoning_state"
            ] = create_empty_reasoning_state()

            confirmations.append(
                "Əsas ehtiyacla bağlı qeydi yenilədim."
            )

        elif field == "phone":

            phone = normalize_phone(
                value
            )

            if phone:

                lead[
                    "phone"
                ] = phone

                confirmations.append(
                    "Telefon nömrənizi yenilədim."
                )

        elif field == "preferred_call_time":

            lead[
                "preferred_call_time"
            ] = value

            confirmations.append(
                "Zəng vaxtını yenilədim."
            )

        elif field == "child_count":

            match = re.search(
                r"\d+",
                value,
            )

            if match:

                set_child_count(
                    lead,
                    int(
                        match.group()
                    ),
                )

    sync_flat_fields(
        lead
    )

    return list(
        dict.fromkeys(
            confirmations
        )
    )


# =========================================================
# 15. MERGE STATE
# =========================================================

def merge_analysis_into_state(
    lead: dict,
    analysis: dict,
    user_text: str,
    current_field: Optional[str],
) -> List[str]:

    ensure_lead_structure(
        lead
    )

    notes = []

    entity_note = apply_entity_operation(
        lead,
        analysis,
    )

    if entity_note:
        notes.append(
            entity_note
        )

    notes.extend(
        apply_corrections(
            lead,
            analysis.get(
                "corrections",
                [],
            ),
        )
    )

    explicit_count = detect_explicit_child_count(
        user_text
    )

    if explicit_count is not None:

        # Do not shrink children automatically
        # if multiple actual child states already exist
        if explicit_count > 1:

            set_child_count(
                lead,
                explicit_count,
            )

        elif (
            explicit_count == 1
            and len(
                lead["children"]
            ) == 1
        ):

            set_child_count(
                lead,
                1,
            )

    child = get_active_child(
        lead
    )

    # ---------------- Parent name ----------------

    parent_candidate = (
        deterministic_parent_name_extract(
            user_text
        )
        or analysis.get(
            "parent_name",
            "",
        ).strip()
    )

    if (
        parent_candidate
        and not lead.get(
            "parent_name"
        )
    ):

        name = clean_name(
            parent_candidate
        )

        if name:

            lead[
                "parent_name"
            ] = name

            lead[
                "parent_title"
            ] = infer_parent_title_with_llm(
                name
            )

    # ---------------- Child name ----------------

    child_name = analysis.get(
        "child_name",
        "",
    ).strip()

    if (
        child_name
        and not child.get(
            "name"
        )
    ):

        name = clean_name(
            child_name
        )

        if name:
            child[
                "name"
            ] = name

    # ---------------- Age ----------------

    ages = extract_contextual_ages(
        user_text
    )

    if (
        ages
        and not child.get(
            "age"
        )
    ):
        child[
            "age"
        ] = ages[0]

    elif (
        current_field == "child_age"
        and not child.get(
            "age"
        )
    ):

        age = extract_simple_age(
            user_text
        )

        if age:
            child[
                "age"
            ] = age

    elif (
        analysis.get(
            "child_age",
            0,
        )
        and not child.get(
            "age"
        )
        and normalize_phone(
            user_text
        ) is None
    ):

        age = int(
            analysis[
                "child_age"
            ]
        )

        if 5 <= age <= 25:
            child[
                "age"
            ] = age

    # ---------------- Concern ----------------

    raw_concern = analysis.get(
        "raw_concern",
        "",
    ).strip()

    concern_summary = analysis.get(
        "concern_summary",
        "",
    ).strip()

    interpreted_needs = analysis.get(
        "interpreted_needs",
        [],
    )

    if raw_concern:

        if analysis.get(
            "should_reassess_hypothesis",
            False,
        ):

            # New evidence changes the picture
            child[
                "raw_concern"
            ] = raw_concern

            child[
                "main_concern"
            ] = (
                concern_summary
                or raw_concern
            )

            child[
                "interpreted_needs"
            ] = list(
                dict.fromkeys(
                    interpreted_needs
                )
            )

            child[
                "reasoning_state"
            ] = create_empty_reasoning_state()

        else:

            if not child.get(
                "raw_concern"
            ):
                child[
                    "raw_concern"
                ] = raw_concern

            if concern_summary:

                child[
                    "main_concern"
                ] = concern_summary

            elif not child.get(
                "main_concern"
            ):

                child[
                    "main_concern"
                ] = raw_concern

            if interpreted_needs:

                child[
                    "interpreted_needs"
                ] = list(
                    dict.fromkeys(
                        interpreted_needs
                    )
                )

    # ---------------- Consultation ----------------

    consultation_update = analysis.get(
        "consultation_update",
        {},
    )

    consultation = child[
        "consultation"
    ]

    for key in [
        "context",
        "duration",
        "impact",
        "future_concern",
        "desired_outcome",
    ]:

        new_value = str(
            consultation_update.get(
                key,
                "",
            )
        ).strip()

        if new_value:
            consultation[
                key
            ] = new_value

    # ---------------- Phone ----------------

    phone = (
        normalize_phone(
            analysis.get(
                "phone",
                "",
            )
        )
        or normalize_phone(
            user_text
        )
    )

    if (
        phone
        and not lead.get(
            "phone"
        )
    ):
        lead[
            "phone"
        ] = phone

    # ---------------- Call time ----------------

    call_time = analysis.get(
        "preferred_call_time",
        "",
    ).strip()

    if (
        call_time
        and not lead.get(
            "preferred_call_time"
        )
    ):

        lead[
            "preferred_call_time"
        ] = call_time

    sync_flat_fields(
        lead
    )

    return notes


# =========================================================
# 16. SALES FLOW
# =========================================================

def get_next_missing_field(
    lead: dict,
):

    ensure_lead_structure(
        lead
    )

    child = get_active_child(
        lead
    )

    # requested simple flow:
    # age → need → parent name → phone → callback

    if not child.get(
        "age"
    ):
        return "child_age"

    if not child.get(
        "main_concern"
    ):
        return "main_concern"

    if not lead.get(
        "parent_name"
    ):
        return "parent_name"

    if not lead.get(
        "phone"
    ):
        return "phone"

    if not lead.get(
        "preferred_call_time"
    ):
        return "preferred_call_time"

    return None


def get_next_question(
    lead: dict,
) -> str:

    field = get_next_missing_field(
        lead
    )

    child = get_active_child(
        lead
    )

    if field == "child_age":

        if child.get(
            "name"
        ):

            return (
                f"{child_genitive(child['name'])} neçə yaşı var?"
            )

        return (
            "Övladınızın neçə yaşı var?"
        )

    if field == "main_concern":

        return (
            "Övladınızda hazırda ən çox dəyişməsini "
            "və ya inkişaf etməsini istədiyiniz məsələ nədir?"
        )

    if field == "parent_name":

        return (
            "Sizə necə müraciət edə bilərəm?"
        )

    if field == "phone":

        return (
            "Sizinlə əlaqə saxlamaq üçün telefon nömrənizi "
            "qeyd edə bilərsiniz?"
        )

    if field == "preferred_call_time":

        return (
            "Zəng üçün sizə hansı gün və saat aralığı uyğundur?"
        )

    return ""


# =========================================================
# 17. STATE RECALL
# =========================================================

def build_state_recall_answer(
    lead: dict,
    fields: List[str],
) -> str:

    ensure_lead_structure(
        lead
    )

    child = get_active_child(
        lead
    )

    parts = []

    for field in fields:

        if field == "parent_name":

            parent = get_parent_display_name(
                lead
            )

            parts.append(
                (
                    f"Sizi {parent} kimi qeyd etmişəm"
                    if parent
                    else "adınızı hələ qeyd etməmişəm"
                )
            )

        elif field == "child_name":

            parts.append(
                (
                    f"övladınızın adını {child['name']} kimi qeyd etmişəm"
                    if child.get("name")
                    else "övladınızın adını hələ qeyd etməmişəm"
                )
            )

        elif field == "child_age":

            parts.append(
                (
                    f"yaşını {child['age']} olaraq qeyd etmişəm"
                    if child.get("age")
                    else "yaşını hələ qeyd etməmişəm"
                )
            )

        elif field == "main_concern":

            value = (
                child.get(
                    "raw_concern"
                )
                or child.get(
                    "main_concern"
                )
            )

            parts.append(
                (
                    f"əsas narahatlığınızı “{value}” kimi qeyd etmişəm"
                    if value
                    else "əsas narahatlıqla bağlı hələ qeyd yoxdur"
                )
            )

        elif field == "phone":

            parts.append(
                (
                    f"nömrənizi {lead['phone']} kimi qeyd etmişəm"
                    if lead.get("phone")
                    else "telefon nömrənizi hələ qeyd etməmişəm"
                )
            )

        elif field == "preferred_call_time":

            parts.append(
                (
                    f"zəng vaxtını “{lead['preferred_call_time']}” kimi qeyd etmişəm"
                    if lead.get("preferred_call_time")
                    else "zəng vaxtını hələ qeyd etməmişəm"
                )
            )

        elif field == "child_count":

            count = lead.get(
                "declared_child_count",
                1,
            )

            parts.append(
                f"{count} övlad qeyd olunub"
            )

    if not parts:
        return ""

    return (
        "Hazırda belə qeyd etmişəm: "
        + "; ".join(parts)
        + "."
    )


# =========================================================
# 18. KNOWLEDGE COLLECTION
# =========================================================

def collect_relevant_knowledge(
    user_text: str,
    analysis: dict,
) -> List[dict]:

    queries = []

    for question in analysis.get(
        "questions",
        [],
    ):

        text = question.get(
            "text",
            "",
        ).strip()

        if text:
            queries.append(text)

    concern_query = " ".join([
        analysis.get(
            "raw_concern",
            "",
        ),

        analysis.get(
            "concern_summary",
            "",
        ),

        " ".join(
            analysis.get(
                "interpreted_needs",
                [],
            )
        ),
    ]).strip()

    if concern_query:
        queries.append(
            concern_query
        )

    if not queries:
        queries.append(
            user_text
        )

    result = []
    seen = set()

    for query in queries:

        for item in retrieve_knowledge(
            query,
            k=5,
        ):

            key = (
                item[
                    "question"
                ],
                item[
                    "answer"
                ],
            )

            if key in seen:
                continue

            if item[
                "score"
            ] < 0.08:
                continue

            seen.add(key)
            result.append(item)

    return result[:12]


# =========================================================
# 19. BUSINESS RULE CONFIDENCE
# =========================================================

def has_strong_business_rule_support(
    analysis: dict,
    knowledge: List[dict],
) -> bool:

    primary_intent = analysis.get(
        "primary_intent"
    )

    if primary_intent != "eligibility_question":
        return True

    if not knowledge:
        return False

    relevant_scores = [
        item.get(
            "score",
            0,
        )
        for item in knowledge
    ]

    if (
        relevant_scores
        and max(
            relevant_scores
        ) >= 0.30
    ):
        return True

    return False


# =========================================================
# 20. RESPONSE GENERATOR
# =========================================================

def generate_response(
    user_text: str,
    lead: dict,
    analysis: dict,
    history: Optional[List[dict]],
    knowledge: List[dict],
    business_rule_supported: bool,
) -> str:

    history = history or []

    if client is None:

        return (
            "Məlumatınızı nəzərə aldım. "
            "Dəqiq cavab üçün məsul əməkdaşla dəqiqləşdirə bilərik."
        )

    state_payload = {
        "parent_name": lead.get(
            "parent_name"
        ),

        "active_child_index": lead.get(
            "active_child_index",
            0,
        ),

        "active_child": get_active_child(
            lead
        ),

        "all_children": lead.get(
            "children"
        ),

        "phone": lead.get(
            "phone"
        ),

        "preferred_call_time": lead.get(
            "preferred_call_time"
        ),
    }

    knowledge_payload = [
        {
            "question": item[
                "question"
            ],

            "fact": item[
                "answer"
            ],

            "score": round(
                item[
                    "score"
                ],
                4,
            ),
        }

        for item in knowledge
    ]

    system_prompt = """
You are Junior Coaching Sales & Operations Assistant.

Generate ONE concise Azerbaijani response.

The user must feel:
"They are trying to understand me and my child",
not:
"They are trying to push me through a chatbot form."

================================================
PRIMARY INTENT FIRST
================================================

Always answer the user's primary intent first.

Example:

Parent:
"Qızım 11 yaşındadır, 3 aya 12 olacaq.
İndi başlaya bilər?"

Do NOT answer:
"Yaşını 11 qeyd etdim."

Age is state information.

Answer eligibility first.

================================================
BUSINESS RULES
================================================

Use only supported knowledge for:
- age eligibility
- price
- schedule
- location
- discounts
- program rules
- operational decisions

If an exact edge-case rule is not supported,
do NOT invent a decision.

Say briefly:
"Bu konkret halı dəqiqləşdirmək lazımdır."

================================================
CONSULTATIVE REASONING
================================================

When parent describes a situation:

- connect facts
- update interpretation when new facts contradict old facts
- do NOT keep a stale hypothesis

Use cautious language:
"əlaqəli ola bilər"
"bu əlavə məlumat mənzərəni dəyişir"
"tək bu məlumatla qəti nəticə demək olmaz"

Never diagnose.

================================================
NEED → SOLUTION
================================================

When enough context exists,
connect the parent's actual concern to relevant
Junior Coaching elements.

Do NOT dump a list of:
özgüvən, EQ, liderlik, ünsiyyət, fokus...

Use only what is relevant.

================================================
OBJECTIONS
================================================

Answer all parts of the objection.

If parent says:
"Məcbur etmək istəmirəm, belə halda nə edirsiniz?"

Acknowledge both:
- not wanting to force the child
- what the program/process does

only if supported by knowledge.

================================================
NO FORCED SALES
================================================

If fit is unclear:
clarify.

If not appropriate:
do not force-sell.

If human judgment is required:
recommend handoff.

================================================
REPEATED QUESTION
================================================

Do not copy the exact previous answer.

If user asks again,
infer what uncertainty remains.

================================================
LENGTH
================================================

This is WhatsApp / Instagram.

Default:
2–4 short sentences.

Simple FAQ:
20–40 words.

Consultative reply:
30–60 words.

Hard maximum:
70 words unless user explicitly asks for detail.

Do NOT explain everything you know.

Do NOT automatically end every answer with a question.
Another layer controls follow-up questions.
"""

    prompt = f"""
CURRENT MESSAGE:
{user_text}

ANALYSIS:
{json.dumps(analysis, ensure_ascii=False)}

STATE:
{json.dumps(state_payload, ensure_ascii=False)}

BUSINESS_RULE_SUPPORTED:
{business_rule_supported}

RELEVANT KNOWLEDGE:
{json.dumps(knowledge_payload, ensure_ascii=False)}

RECENT HISTORY:
{json.dumps(history[-10:], ensure_ascii=False)}

Write the concise response to the current message.
Do not append the next flow question.
"""

    try:

        response = client.chat.completions.create(
            model="gpt-4o-mini",
            temperature=0.2,

            messages=[
                {
                    "role": "system",
                    "content": system_prompt,
                },
                {
                    "role": "user",
                    "content": prompt,
                },
            ],
        )

        answer = response.choices[
            0
        ].message.content.strip()

        if answer:
            return answer

    except Exception as exc:

        print(
            "RESPONSE ERROR:",
            exc,
        )

    return (
        "Bu məqamı yanlış yönləndirməmək üçün "
        "məsul əməkdaşla dəqiqləşdirmək daha düzgün olar."
    )


# =========================================================
# 21. CONSULTATIVE FOLLOW-UP
# =========================================================

def choose_consultative_question(
    analysis: dict,
    lead: dict,
) -> Optional[str]:

    if not analysis.get(
        "needs_consultative_followup",
        False,
    ):
        return None

    proposed = analysis.get(
        "best_consultative_question",
        "",
    ).strip()

    if not proposed:
        return None

    child = get_active_child(
        lead
    )

    consultation = child.get(
        "consultation",
        {},
    )

    normalized = normalize_for_search(
        proposed
    )

    # avoid asking known dimensions again
    if (
        consultation.get(
            "duration"
        )
        and any(
            x in normalized
            for x in [
                "ne vaxtdan",
                "ne qeder muddet",
            ]
        )
    ):
        return None

    if (
        consultation.get(
            "context"
        )
        and any(
            x in normalized
            for x in [
                "hansi situasiya",
                "harada ozunu gosterir",
                "ne zaman olur",
            ]
        )
    ):
        return None

    if (
        consultation.get(
            "impact"
        )
        and "tesir" in normalized
    ):
        return None

    return proposed


# =========================================================
# 22. FINAL MESSAGE
# =========================================================

def build_final_message(
    lead: dict,
) -> str:

    parent = get_parent_display_name(
        lead
    )

    call_time = lead.get(
        "preferred_call_time"
    )

    result = "Qeydə alındı ✅"

    if (
        parent
        and call_time
    ):

        result += (
            f"\n\n{parent}, {call_time} sizinlə əlaqə "
            "saxlanılması üçün müraciətinizi qeyd etdim."
        )

    elif call_time:

        result += (
            f"\n\n{call_time} sizinlə əlaqə saxlanılması "
            "üçün müraciətinizi qeyd etdim."
        )

    result += (
        "\n\nİlkin zəng zamanı övladınızın iştirakı vacib deyil."
    )

    return result


# =========================================================
# 23. MAIN AGENT
# =========================================================

def lead_agent_reply(
    user_text: str,
    lead: dict,
    faq_min_score: float = 0.18,
    history: Optional[List[dict]] = None,
) -> str:

    del faq_min_score

    user_text = user_text.strip()

    history = history or []

    ensure_lead_structure(
        lead
    )

    field_before = get_next_missing_field(
        lead
    )

    # -----------------------------------------------------
    # UNDERSTAND / PLAN
    # -----------------------------------------------------

    analysis = analyze_message(
        user_text=user_text,
        lead=lead,
        history=history,
    )

    print(
        "V9 ANALYSIS:",
        analysis,
    )

    lead[
        "_last_intent"
    ] = analysis.get(
        "primary_intent"
    )

    lead[
        "_last_primary_intent"
    ] = analysis.get(
        "primary_intent"
    )

    lead[
        "_last_confidence"
    ] = analysis.get(
        "confidence"
    )

    # -----------------------------------------------------
    # EXTRACT / UPDATE STATE
    # -----------------------------------------------------

    update_notes = merge_analysis_into_state(
        lead=lead,
        analysis=analysis,
        user_text=user_text,
        current_field=field_before,
    )

    # -----------------------------------------------------
    # SAFETY
    # -----------------------------------------------------

    if analysis.get(
        "primary_intent"
    ) == "safety_risk":

        lead[
            "status"
        ] = "ESCALATED"

        return (
            "Bu vəziyyət daha diqqətli peşəkar qiymətləndirmə tələb edə bilər. "
            "Junior Coaching təcili psixoloji və ya tibbi yardımı əvəz etmir. "
            "Müraciətinizi məsul əməkdaşa yönləndirirəm."
        )

    # -----------------------------------------------------
    # HUMAN
    # -----------------------------------------------------

    if analysis.get(
        "primary_intent"
    ) == "human_agent_request":

        lead[
            "status"
        ] = "ESCALATED"

        return (
            "Əlbəttə. Müraciətinizi məsul əməkdaşa yönləndirirəm."
        )

    # -----------------------------------------------------
    # PERMISSION
    # -----------------------------------------------------

    if analysis.get(
        "primary_intent"
    ) == "permission_question":

        return (
            "Əlbəttə, buyurun 😊"
        )

    # -----------------------------------------------------
    # STATE RECALL
    # -----------------------------------------------------

    recall_fields = analysis.get(
        "recall_fields",
        [],
    )

    if (
        recall_fields
        or analysis.get(
            "primary_intent"
        ) == "state_question"
    ):

        answer = build_state_recall_answer(
            lead,
            recall_fields,
        )

        if answer:
            return answer

    # -----------------------------------------------------
    # GREETING
    # -----------------------------------------------------

    if (
        analysis.get(
            "primary_intent"
        ) == "greeting"
        and is_greeting(
            user_text
        )
    ):

        if not lead.get(
            "_greeted"
        ):

            lead[
                "_greeted"
            ] = True

            return (
                "Salam 😊 "
                + get_next_question(
                    lead
                )
            )

        return (
            get_next_question(
                lead
            )
            or "Buyurun 😊"
        )

    # -----------------------------------------------------
    # CLARIFICATION BEFORE WRONG ANSWER
    # -----------------------------------------------------

    if analysis.get(
        "needs_clarification",
        False,
    ):

        question = analysis.get(
            "clarification_question",
            "",
        ).strip()

        if question:
            return question

        return (
            "Sizi düzgün anlamaq üçün bir məqamı dəqiqləşdirə bilərsiniz?"
        )

    # -----------------------------------------------------
    # DOES THIS TURN NEED AN ANSWER?
    # -----------------------------------------------------

    primary = analysis.get(
        "primary_intent"
    )

    needs_response_generation = (
        bool(
            analysis.get(
                "questions"
            )
        )
        or analysis.get(
            "has_situation_description",
            False,
        )
        or primary in {
            "faq_question",
            "eligibility_question",
            "program_interest",
            "situation_analysis",
            "complaint",
            "pause_request",
            "unrelated",
        }
    )

    if needs_response_generation:

        knowledge = collect_relevant_knowledge(
            user_text=user_text,
            analysis=analysis,
        )

        if knowledge:

            lead[
                "_last_faq_score"
            ] = max(
                x[
                    "score"
                ]
                for x in knowledge
            )

        business_rule_supported = (
            has_strong_business_rule_support(
                analysis,
                knowledge,
            )
        )

        answer = generate_response(
            user_text=user_text,
            lead=lead,
            analysis=analysis,
            history=history,
            knowledge=knowledge,
            business_rule_supported=business_rule_supported,
        )

        # acknowledge correction / entity switch briefly
        if update_notes:

            note = " ".join(
                update_notes
            )

            answer = (
                note
                + " "
                + answer
            )

        # ---------------------------------------------
        # CONSULTATIVE QUESTION HAS PRIORITY OVER FLOW
        # ---------------------------------------------

        consultative_question = (
            choose_consultative_question(
                analysis,
                lead,
            )
        )

        if consultative_question:

            lead[
                "_consultative_turns"
            ] = (
                lead.get(
                    "_consultative_turns",
                    0,
                )
                + 1
            )

            return (
                answer
                + "\n\n"
                + consultative_question
            )

        # ---------------------------------------------
        # PAUSE SALES FLOW
        # ---------------------------------------------

        if analysis.get(
            "should_pause_sales_flow",
            False,
        ):

            return answer

        # ---------------------------------------------
        # NEXT SALES STEP
        # ---------------------------------------------

        next_question = get_next_question(
            lead
        )

        if next_question:

            return (
                answer
                + "\n\n"
                + next_question
            )

        lead[
            "status"
        ] = "CALL_REQUESTED"

        return (
            answer
            + "\n\n"
            + build_final_message(
                lead
            )
        )

    # -----------------------------------------------------
    # FIELD-ONLY TURN
    # -----------------------------------------------------

    next_question = get_next_question(
        lead
    )

    if next_question:
        return next_question

    lead[
        "status"
    ] = "CALL_REQUESTED"

    return build_final_message(
        lead
    )


# =========================================================
# 24. DATABASE
# =========================================================

def init_db():

    with sqlite3.connect(
        DB_PATH
    ) as conn:

        conn.execute(
            """
            CREATE TABLE IF NOT EXISTS leads (

                id INTEGER PRIMARY KEY AUTOINCREMENT,

                parent_name TEXT,
                parent_title TEXT,

                child_name TEXT,
                child_age INTEGER,
                main_concern TEXT,

                needs_concern_followup INTEGER DEFAULT 0,
                concern_duration TEXT,
                concern_onset TEXT,

                children_json TEXT,

                phone TEXT,
                preferred_call_time TEXT,

                source TEXT,
                status TEXT,

                created_at TEXT,
                updated_at TEXT
            )
            """
        )

        conn.execute(
            """
            CREATE TABLE IF NOT EXISTS children (

                id INTEGER PRIMARY KEY AUTOINCREMENT,

                lead_id INTEGER,
                child_index INTEGER,

                name TEXT,
                age INTEGER,

                raw_concern TEXT,
                main_concern TEXT,

                interpreted_needs TEXT,
                consultation_json TEXT,
                reasoning_state_json TEXT,

                concern_duration TEXT,
                concern_onset TEXT,

                created_at TEXT,

                FOREIGN KEY (lead_id)
                REFERENCES leads(id)
            )
            """
        )

        conn.execute(
            """
            CREATE TABLE IF NOT EXISTS conversation_logs (

                id INTEGER PRIMARY KEY AUTOINCREMENT,

                session_id TEXT NOT NULL,

                user_message TEXT,
                bot_response TEXT,

                intent TEXT,
                confidence REAL,
                faq_score REAL,

                current_field TEXT,

                parent_name TEXT,
                parent_title TEXT,

                child_name TEXT,
                child_age INTEGER,
                main_concern TEXT,

                children_json TEXT,

                phone TEXT,
                preferred_call_time TEXT,

                status TEXT,
                source TEXT,

                created_at TEXT
            )
            """
        )

        # ------- migrations -------

        child_columns = {
            row[1]
            for row in conn.execute(
                "PRAGMA table_info(children)"
            ).fetchall()
        }

        required_child_columns = {
            "raw_concern": "TEXT",
            "main_concern": "TEXT",
            "interpreted_needs": "TEXT",
            "consultation_json": "TEXT",
            "reasoning_state_json": "TEXT",
            "concern_duration": "TEXT",
            "concern_onset": "TEXT",
        }

        for column, dtype in required_child_columns.items():

            if column not in child_columns:

                conn.execute(
                    f"""
                    ALTER TABLE children
                    ADD COLUMN {column} {dtype}
                    """
                )

        lead_columns = {
            row[1]
            for row in conn.execute(
                "PRAGMA table_info(leads)"
            ).fetchall()
        }

        required_lead_columns = {
            "parent_title": "TEXT",
            "needs_concern_followup":
                "INTEGER DEFAULT 0",
            "concern_duration": "TEXT",
            "concern_onset": "TEXT",
            "children_json": "TEXT",
        }

        for column, dtype in required_lead_columns.items():

            if column not in lead_columns:

                conn.execute(
                    f"""
                    ALTER TABLE leads
                    ADD COLUMN {column} {dtype}
                    """
                )

        conn.commit()


# =========================================================
# 25. TIME
# =========================================================

def get_baku_time():

    return datetime.now(
        ZoneInfo(
            "Asia/Baku"
        )
    ).strftime(
        "%Y-%m-%d %H:%M:%S"
    )


# =========================================================
# 26. FIND LEAD
# =========================================================

def find_lead_by_phone(
    phone: str,
):

    if not phone:
        return None

    with sqlite3.connect(
        DB_PATH
    ) as conn:

        conn.row_factory = sqlite3.Row

        row = conn.execute(
            """
            SELECT *
            FROM leads
            WHERE phone = ?
            ORDER BY id DESC
            LIMIT 1
            """,
            (
                phone,
            ),
        ).fetchone()

        if row:
            return dict(row)

    return None


# =========================================================
# 27. SAVE LEAD
# =========================================================

def save_lead_to_db(
    lead: dict,
) -> int:

    ensure_lead_structure(
        lead
    )

    sync_flat_fields(
        lead
    )

    now = get_baku_time()

    children_json = json.dumps(
        lead[
            "children"
        ],
        ensure_ascii=False,
    )

    with sqlite3.connect(
        DB_PATH
    ) as conn:

        cursor = conn.cursor()

        cursor.execute(
            """
            INSERT INTO leads (

                parent_name,
                parent_title,

                child_name,
                child_age,
                main_concern,

                needs_concern_followup,
                concern_duration,
                concern_onset,

                children_json,

                phone,
                preferred_call_time,

                source,
                status,

                created_at,
                updated_at
            )

            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """,

            (
                lead.get(
                    "parent_name"
                ),

                lead.get(
                    "parent_title"
                ),

                lead.get(
                    "child_name"
                ),

                lead.get(
                    "child_age"
                ),

                lead.get(
                    "main_concern"
                ),

                int(
                    bool(
                        lead.get(
                            "needs_concern_followup",
                            False,
                        )
                    )
                ),

                lead.get(
                    "concern_duration"
                ),

                lead.get(
                    "concern_onset"
                ),

                children_json,

                lead.get(
                    "phone"
                ),

                lead.get(
                    "preferred_call_time"
                ),

                lead.get(
                    "source"
                ),

                lead.get(
                    "status"
                ),

                now,
                now,
            ),
        )

        lead_id = cursor.lastrowid

        for child_index, child in enumerate(
            lead[
                "children"
            ]
        ):

            conn.execute(
                """
                INSERT INTO children (

                    lead_id,
                    child_index,

                    name,
                    age,

                    raw_concern,
                    main_concern,

                    interpreted_needs,
                    consultation_json,
                    reasoning_state_json,

                    concern_duration,
                    concern_onset,

                    created_at
                )

                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                """,

                (
                    lead_id,
                    child_index,

                    child.get(
                        "name"
                    ),

                    child.get(
                        "age"
                    ),

                    child.get(
                        "raw_concern"
                    ),

                    child.get(
                        "main_concern"
                    ),

                    json.dumps(
                        child.get(
                            "interpreted_needs",
                            [],
                        ),
                        ensure_ascii=False,
                    ),

                    json.dumps(
                        child.get(
                            "consultation",
                            {},
                        ),
                        ensure_ascii=False,
                    ),

                    json.dumps(
                        child.get(
                            "reasoning_state",
                            {},
                        ),
                        ensure_ascii=False,
                    ),

                    child.get(
                        "concern_duration"
                    ),

                    child.get(
                        "concern_onset"
                    ),

                    now,
                ),
            )

        conn.commit()

        return lead_id


# =========================================================
# 28. SAVE CONVERSATION
# =========================================================

def save_conversation_log(
    session_id: str,
    user_message: str,
    bot_response: str,
    current_field: Optional[str],
    lead: dict,
):

    ensure_lead_structure(
        lead
    )

    sync_flat_fields(
        lead
    )

    now = get_baku_time()

    children_json = json.dumps(
        lead[
            "children"
        ],
        ensure_ascii=False,
    )

    with sqlite3.connect(
        DB_PATH
    ) as conn:

        conn.execute(
            """
            INSERT INTO conversation_logs (

                session_id,

                user_message,
                bot_response,

                intent,
                confidence,
                faq_score,

                current_field,

                parent_name,
                parent_title,

                child_name,
                child_age,
                main_concern,

                children_json,

                phone,
                preferred_call_time,

                status,
                source,

                created_at
            )

            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """,

            (
                session_id,

                user_message,
                bot_response,

                lead.get(
                    "_last_primary_intent"
                ),

                lead.get(
                    "_last_confidence"
                ),

                lead.get(
                    "_last_faq_score"
                ),

                current_field,

                lead.get(
                    "parent_name"
                ),

                lead.get(
                    "parent_title"
                ),

                lead.get(
                    "child_name"
                ),

                lead.get(
                    "child_age"
                ),

                lead.get(
                    "main_concern"
                ),

                children_json,

                lead.get(
                    "phone"
                ),

                lead.get(
                    "preferred_call_time"
                ),

                lead.get(
                    "status"
                ),

                lead.get(
                    "source"
                ),

                now,
            ),
        )

        conn.commit()


# =========================================================
# 29. ADMIN HELPERS
# =========================================================

def get_all_leads():

    with sqlite3.connect(
        DB_PATH
    ) as conn:

        conn.row_factory = sqlite3.Row

        rows = conn.execute(
            """
            SELECT *
            FROM leads
            ORDER BY id DESC
            """
        ).fetchall()

        return [
            dict(row)
            for row in rows
        ]


def get_all_children():

    with sqlite3.connect(
        DB_PATH
    ) as conn:

        conn.row_factory = sqlite3.Row

        rows = conn.execute(
            """
            SELECT *
            FROM children
            ORDER BY id DESC
            """
        ).fetchall()

        return [
            dict(row)
            for row in rows
        ]


def get_all_conversation_logs():

    with sqlite3.connect(
        DB_PATH
    ) as conn:

        conn.row_factory = sqlite3.Row

        rows = conn.execute(
            """
            SELECT *
            FROM conversation_logs
            ORDER BY id DESC
            """
        ).fetchall()

        return [
            dict(row)
            for row in rows
        ]


# =========================================================
# INIT
# =========================================================

init_db()