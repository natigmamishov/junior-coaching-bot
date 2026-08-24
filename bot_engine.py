"""
Junior Coaching — Conversation Engine V8

Architecture
------------
Understand
    ↓
Extract
    ↓
Update State
    ↓
Retrieve Knowledge
    ↓
Reason / Plan
    ↓
Answer naturally
    ↓
Decide next sales step


V8 goals
--------
1. Whole-message understanding
2. Multiple facts in one message
3. Multiple questions in one message
4. Contextual understanding of parent-described situations
5. State recall
6. Reliable corrections / overwrite
7. No guessed names
8. No fake second-child flows
9. Knowledge base = facts, not canned final answer
10. Natural contextual response generation
11. Maximum one sales-flow question
12. No psychological diagnosis
13. No invented business information
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
    "Junior_Coaching_sesli_AI_FAQ.txt"
)

DB_PATH = os.path.join(
    BASE_DIR,
    "junior_coaching.db"
)


# =========================================================
# 1. OPENAI
# =========================================================

load_dotenv(
    os.path.join(
        BASE_DIR,
        ".env"
    )
)

OPENAI_API_KEY = os.getenv(
    "OPENAI_API_KEY"
)

client: Optional[OpenAI] = None

if OPENAI_API_KEY:

    # Local testing environment.
    # Production-da verify=True istifadə etmək daha yaxşıdır.
    http_client = httpx.Client(
        verify=False,
        timeout=60,
    )

    client = OpenAI(
        api_key=OPENAI_API_KEY,
        http_client=http_client,
    )


# =========================================================
# 2. TEXT NORMALIZATION
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

    text = text.translate(
        AZ_TRANSLATION
    )

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

    # 0501234567
    if (
        len(digits) == 10
        and digits.startswith("0")
    ):
        return digits

    # 994501234567
    if (
        len(digits) == 12
        and digits.startswith("994")
    ):
        return digits

    return None


# =========================================================
# 4. AGE EXTRACTION
# =========================================================

def extract_contextual_ages(
    text: str,
) -> List[int]:

    """
    Yaşı yalnız yaş konteksti olduqda çıxarır.

    Qəbul:
        14 yaş
        14 yaşı var
        14 yaşlı
        14 yaşında
        13 və 15 yaşında

    Qəbul etmir:
        050 123 45 67
        14:00
        15:30
    """

    value = normalize_for_search(
        text
    )

    ages: List[int] = []

    patterns = [
        r"\b(\d{1,2})\s*yas\b",
        r"\b(\d{1,2})\s*yasi\b",
        r"\b(\d{1,2})\s*yasli\b",
        r"\b(\d{1,2})\s*yasinda\b",
    ]

    for pattern in patterns:

        matches = re.findall(
            pattern,
            value,
        )

        for item in matches:

            age = int(item)

            if (
                5 <= age <= 25
                and age not in ages
            ):
                ages.append(age)

    pair_patterns = [
        r"\b(\d{1,2})\s*(?:ve|,)\s*(\d{1,2})\s*yasinda\b",
        r"\b(\d{1,2})\s*(?:ve|,)\s*(\d{1,2})\s*yasli\b",
    ]

    for pattern in pair_patterns:

        match = re.search(
            pattern,
            value,
        )

        if match:

            for item in match.groups():

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

    """
    Bot konkret yaş soruşduqda:
        14
        12 tamam olacaq
    kimi cavablar üçün.
    """

    if normalize_phone(text):
        return None

    value = normalize_for_search(
        text
    )

    numbers = re.findall(
        r"\b\d{1,2}\b",
        value,
    )

    # Saat kimi görünürsə age sayma
    if re.search(
        r"\b\d{1,2}\s*[:.-]\s*\d{1,2}\b",
        value,
    ):
        return None

    if len(numbers) != 1:
        return None

    age = int(numbers[0])

    if 5 <= age <= 25:
        return age

    return None


# =========================================================
# 5. GREETING
# =========================================================

def is_greeting(text: str) -> bool:

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
# 6. NAME VALIDATION
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
    "demisdim",
    "buradan",
    "burdan",
    "maraq",
    "maraqlaniram",
    "proqram",
}


def clean_name(
    value: str,
) -> Optional[str]:

    if not value:
        return None

    value = str(value).strip()

    # honorific removal
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

    cleaned_words = []

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

        cleaned_words.append(
            word
        )

    result = " ".join(
        cleaned_words
    ).title()

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
# 7. NAME SUFFIX
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

        return (
            name
            + "n"
            + suffix
        )

    return (
        name
        + suffix
    )


# =========================================================
# 8. STATE STRUCTURE
# =========================================================

def create_empty_child() -> dict:

    return {
        "name": None,
        "age": None,

        # exactly what parent said
        "raw_concern": None,

        # short structured interpretation
        "main_concern": None,

        # LLM interpretation, not diagnosis
        "interpreted_needs": [],

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

        # backward compatibility
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
        "_last_confidence": None,
        "_last_faq_score": None,

        "_last_topic": None,
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

    # migrate old child dictionaries
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
        "_last_topic",
        None,
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

        lead[
            "active_child_index"
        ] = 0

    return lead[
        "children"
    ][index]


def sync_flat_fields(
    lead: dict,
):

    ensure_lead_structure(
        lead
    )

    first = lead[
        "children"
    ][0]

    lead[
        "child_name"
    ] = first.get(
        "name"
    )

    lead[
        "child_age"
    ] = first.get(
        "age"
    )

    lead[
        "main_concern"
    ] = first.get(
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
# 9. EXPLICIT CHILD COUNT
# =========================================================

def detect_explicit_child_count(
    text: str,
) -> Optional[int]:

    value = normalize_for_search(
        text
    )

    one_child_phrases = [
        "bir oglum var",
        "bir qizim var",
        "bir usagim var",
        "bir ovladim var",
        "tek usagim var",
        "tek ovladim var",
        "2 ci ovlad yoxdur",
        "ikinci ovlad yoxdur",
        "ikinci usaq yoxdur",
    ]

    if any(
        x in value
        for x in one_child_phrases
    ):
        return 1

    two_child_phrases = [
        "iki usagim",
        "iki ovladim",
        "2 usagim",
        "2 ovladim",
        "iki usaq var",
        "2 usaq var",
    ]

    if any(
        x in value
        for x in two_child_phrases
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

    if count == 1:

        lead["children"] = [
            lead["children"][0]
        ]

        lead[
            "active_child_index"
        ] = 0

    else:

        while len(
            lead["children"]
        ) < count:

            lead["children"].append(
                create_empty_child()
            )

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
Azərbaycan adına əsasən uyğun müraciət formasını seç.

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
                    "name": "title_result",
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

        data = json.loads(
            response.choices[0].message.content
        )

        if data["title"] in {
            "xanım",
            "bəy",
        }:

            return data[
                "title"
            ]

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

        return (
            f"{name} {title}"
        )

    return name


# =========================================================
# 11. FAQ / KNOWLEDGE INDEX
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
            "FAQ faylında Sual/Agent cütləri tapılmadı."
        )

    questions = [
        x[0]
        for x in pairs
    ]

    answers = [
        x[1]
        for x in pairs
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
                sublinear_tf=True,
                max_features=60000,
            ),
        ),
        (
            "char",
            TfidfVectorizer(
                analyzer="char_wb",
                ngram_range=(3, 5),
                sublinear_tf=True,
                max_features=80000,
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
) -> List[Dict[str, Any]]:

    normalized = normalize_for_search(
        query
    )

    vector = FAQ_VECTORIZER.transform(
        [normalized]
    )

    scores = cosine_similarity(
        vector,
        FAQ_MATRIX,
    ).ravel()

    indexes = np.argsort(
        -scores
    )[:k]

    result = []

    for index in indexes:

        result.append({
            "question": FAQ_QUESTIONS[index],
            "answer": FAQ_ANSWERS[index],
            "score": float(
                scores[index]
            ),
        })

    return result


# =========================================================
# 12. WHOLE CONVERSATION UNDERSTANDING
# =========================================================

QUESTION_TOPICS = [
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

    state_for_llm = {
        "parent_name": lead.get(
            "parent_name"
        ),

        "parent_title": lead.get(
            "parent_title"
        ),

        "children": lead.get(
            "children"
        ),

        "declared_child_count": lead.get(
            "declared_child_count",
            1,
        ),

        "phone": lead.get(
            "phone"
        ),

        "preferred_call_time": lead.get(
            "preferred_call_time"
        ),

        "status": lead.get(
            "status"
        ),
    }

    fallback = {
        "intent": (
            "greeting"
            if is_greeting(
                user_text
            )
            else "field_answer"
        ),

        "questions": [],

        "parent_name": "",
        "child_name": "",
        "child_age": 0,

        "raw_concern": "",
        "concern_summary": "",

        "interpreted_needs": [],

        "phone": "",
        "preferred_call_time": "",

        "child_count": 0,

        "corrections": [],

        "recall_fields": [],

        "has_situation_description": False,
        "needs_clarification": False,

        "clarification_reason": "",

        "should_pause_sales_flow": False,

        "confidence": 0.0,
    }

    if client is None:
        return fallback

    system_prompt = """
You are the CONVERSATION UNDERSTANDING layer for
Junior Coaching Sales & Operations Assistant.

You DO NOT answer the user.
You extract meaning and conversation structure.

Analyze together:
1. current user message,
2. recent conversation history,
3. current structured state.

CORE PRINCIPLE:
Understand meaning, not just keywords.

--------------------------------------------------
FACT EXTRACTION
--------------------------------------------------

Extract every fact present in the message.

The user may provide in one message:
- parent name
- child name
- age
- concern / situation
- phone
- preferred callback time
- questions
- corrections

Extract ALL of them.

Never invent a missing fact.

UNKNOWN means empty string / 0 / empty array.

--------------------------------------------------
NAME SAFETY
--------------------------------------------------

Only extract a person's name when clearly presented as a name.

Examples:

"Adım Günaydır"
parent_name = Günay

"Mən Günayam"
parent_name = Günay

"Oğlum Tunar üçün maraqlanıram"
child_name = Tunar

"Bu nömrə ilə əlaqə saxlayın 055..."
parent_name = ""

Never treat phrases such as:
"bu nömrə ilə"
"anasıyam"
"mənə"
"valideynəm"
"maraqlanıram"
as names.

If uncertain about a name, leave it empty.

--------------------------------------------------
BEHAVIOR → NEED UNDERSTANDING
--------------------------------------------------

Parents do NOT have to say category keywords.

Example:

"Evdə rahat danışır, məktəbdə bildiyi cavabı
deməyə çəkinir."

raw_concern:
the parent's actual description

concern_summary:
short neutral summary, e.g.
"sosial mühitdə özünüifadədə çətinlik"

interpreted_needs may include:
- özünüifadə
- özgüvən
- ünsiyyət

These are possible coaching needs, NOT diagnoses.

Another example:

"Qərar verməkdə çətinlik çəkir,
gələcəklə bağlı nə istədiyini bilmir."

concern_summary:
"qərarvermə və gələcək istiqamətini müəyyənləşdirmə"

interpreted_needs:
["qərarvermə", "məqsəd və gələcək"]

If the parent has already described a meaningful need,
do NOT leave concern_summary empty merely because
they did not use words like "özgüvən".

--------------------------------------------------
QUESTIONS
--------------------------------------------------

Split multiple questions.

Example:
"Görüşlər harada və hansı gün keçirilir?"

questions:
1. meeting_location
2. meeting_day

Example:
"Qiymət nə qədərdir və uşaq zəngdə olmalıdır?"
These are two separate questions.

--------------------------------------------------
STATE RECALL
--------------------------------------------------

Examples:

"Adımı necə qeyd etmisiniz?"
recall_fields=["parent_name"]

"Mənim adımı, qızımın adını, yaşını və
əsas narahatlığımı necə qeyd etmisiniz?"

recall_fields=[
 "parent_name",
 "child_name",
 "child_age",
 "main_concern"
]

"Neçə övladım olduğunu demişdim?"
recall_fields=["child_count"]

These are STATE questions.
Do NOT treat as FAQ.

--------------------------------------------------
CORRECTIONS
--------------------------------------------------

Corrections overwrite old values.

"16 yox, 15 yaşı var"
correction child_age=15

"Tunar yox, Turandır"
correction child_name=Turan

"Aygün mənəm, uşağın adı Ayxandır"
corrections:
parent_name=Aygün
child_name=Ayxan

"2-ci övlad yoxdur, bir oğlum var"
child_count=1

--------------------------------------------------
CHILD COUNT
--------------------------------------------------

Never infer multiple children from phone numbers,
ages, or random numbers.

Only set child_count > 1 if the user clearly says
they have multiple children.

--------------------------------------------------
SITUATION ANALYSIS
--------------------------------------------------

If parent describes behavior and asks:
"sizcə..."
"necə yanaşmaq olar?"
"bu nə ilə bağlı ola bilər?"
"proqram uyğun ola bilər?"

set:
has_situation_description=true

This is not necessarily FAQ.
It requires contextual reasoning.

Do NOT diagnose.

--------------------------------------------------
PAUSE FLOW
--------------------------------------------------

should_pause_sales_flow=true when:
- user is currently asking follow-up questions,
- user is exploring before deciding,
- user says "əvvəl məlumat almaq istəyirəm",
- user has an unresolved objection,
- immediately asking next lead-form question would feel pushy.

Otherwise false.

--------------------------------------------------
CLARIFICATION
--------------------------------------------------

needs_clarification=true only if:
- user's intended meaning is genuinely unclear,
- wrong confident answer would be risky.

Do not use clarification unnecessarily.

--------------------------------------------------
INTENTS
--------------------------------------------------

greeting
faq_question
field_answer
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
"""

    payload = f"""
CURRENT STATE:
{json.dumps(state_for_llm, ensure_ascii=False)}

RECENT HISTORY:
{json.dumps(history[-12:], ensure_ascii=False)}

CURRENT USER MESSAGE:
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
                    "name": "conversation_understanding",
                    "strict": True,

                    "schema": {
                        "type": "object",

                        "properties": {

                            "intent": {
                                "type": "string",

                                "enum": [
                                    "greeting",
                                    "faq_question",
                                    "field_answer",
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
                                            "enum": QUESTION_TOPICS,
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

                            "needs_clarification": {
                                "type": "boolean"
                            },

                            "clarification_reason": {
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
                            "intent",
                            "questions",
                            "parent_name",
                            "child_name",
                            "child_age",
                            "raw_concern",
                            "concern_summary",
                            "interpreted_needs",
                            "phone",
                            "preferred_call_time",
                            "child_count",
                            "corrections",
                            "recall_fields",
                            "has_situation_description",
                            "needs_clarification",
                            "clarification_reason",
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
# 13. APPLY CORRECTIONS
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

            while len(
                lead["children"]
            ) <= child_index:

                lead[
                    "children"
                ].append(
                    create_empty_child()
                )

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

            numbers = re.findall(
                r"\b\d{1,2}\b",
                value,
            )

            if numbers:

                age = int(
                    numbers[0]
                )

                if 5 <= age <= 25:

                    while len(
                        lead["children"]
                    ) <= child_index:

                        lead[
                            "children"
                        ].append(
                            create_empty_child()
                        )

                    lead[
                        "children"
                    ][child_index][
                        "age"
                    ] = age

                    confirmations.append(
                        f"Yaşı {age} olaraq yenilədim."
                    )

        elif field in {
            "main_concern",
            "raw_concern",
        }:

            while len(
                lead["children"]
            ) <= child_index:

                lead[
                    "children"
                ].append(
                    create_empty_child()
                )

            lead[
                "children"
            ][child_index][
                "raw_concern"
            ] = value

            lead[
                "children"
            ][child_index][
                "main_concern"
            ] = value

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

            numbers = re.findall(
                r"\d+",
                value,
            )

            if numbers:

                count = int(
                    numbers[0]
                )

                set_child_count(
                    lead,
                    count,
                )

                if count == 1:

                    confirmations.append(
                        "Bir övladınız olduğunu nəzərə aldım."
                    )

                else:

                    confirmations.append(
                        f"{count} övladınız olduğunu nəzərə aldım."
                    )

    sync_flat_fields(
        lead
    )

    return confirmations


# =========================================================
# 14. MERGE NEW INFORMATION
# =========================================================

def merge_analysis_into_state(
    lead: dict,
    analysis: dict,
    user_text: str,
    field_before: Optional[str],
) -> List[str]:

    ensure_lead_structure(
        lead
    )

    confirmations = apply_corrections(
        lead,
        analysis.get(
            "corrections",
            []
        ),
    )

    # -----------------------------------------------------
    # CHILD COUNT
    # -----------------------------------------------------

    explicit_count = detect_explicit_child_count(
        user_text
    )

    if explicit_count is not None:

        old_count = lead.get(
            "declared_child_count",
            1,
        )

        set_child_count(
            lead,
            explicit_count,
        )

        if (
            explicit_count != old_count
            and explicit_count == 1
        ):

            confirmations.append(
                "Bir övladınız olduğunu nəzərə aldım."
            )

    else:

        llm_count = analysis.get(
            "child_count",
            0,
        )

        if llm_count > 1:

            value = normalize_for_search(
                user_text
            )

            explicit_multi_markers = [
                "iki usaq",
                "2 usaq",
                "iki ovlad",
                "2 ovlad",
                "usaqlarim",
                "ovladlarim",
            ]

            if any(
                x in value
                for x in explicit_multi_markers
            ):

                set_child_count(
                    lead,
                    llm_count,
                )

    # -----------------------------------------------------
    # PARENT NAME
    # -----------------------------------------------------

    deterministic_name = (
        deterministic_parent_name_extract(
            user_text
        )
    )

    parent_candidate = (
        deterministic_name
        or analysis.get(
            "parent_name",
            ""
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

    # -----------------------------------------------------
    # ACTIVE CHILD
    # -----------------------------------------------------

    child = get_active_child(
        lead
    )

    # child name
    child_candidate = analysis.get(
        "child_name",
        ""
    ).strip()

    if (
        child_candidate
        and not child.get(
            "name"
        )
    ):

        name = clean_name(
            child_candidate
        )

        if name:

            child[
                "name"
            ] = name

    # -----------------------------------------------------
    # AGE
    # -----------------------------------------------------

    contextual_ages = extract_contextual_ages(
        user_text
    )

    if (
        contextual_ages
        and lead.get(
            "declared_child_count",
            1,
        ) > 1
        and len(
            contextual_ages
        ) > 1
    ):

        for index, age in enumerate(
            contextual_ages
        ):

            if index < len(
                lead["children"]
            ):

                if not lead[
                    "children"
                ][index].get(
                    "age"
                ):

                    lead[
                        "children"
                    ][index][
                        "age"
                    ] = age

    elif (
        contextual_ages
        and not child.get(
            "age"
        )
    ):

        child[
            "age"
        ] = contextual_ages[0]

    elif (
        field_before == "child_age"
        and not child.get(
            "age"
        )
    ):

        simple_age = extract_simple_age(
            user_text
        )

        if simple_age:

            child[
                "age"
            ] = simple_age

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

    # -----------------------------------------------------
    # CONCERN
    # -----------------------------------------------------

    raw_concern = analysis.get(
        "raw_concern",
        ""
    ).strip()

    summary = analysis.get(
        "concern_summary",
        ""
    ).strip()

    interpreted_needs = analysis.get(
        "interpreted_needs",
        [],
    )

    if raw_concern:

        # Important:
        # parent already described the need.
        # Do NOT ask "əsas ehtiyac nədir?" again.
        if not child.get(
            "raw_concern"
        ):

            child[
                "raw_concern"
            ] = raw_concern

        if summary:

            child[
                "main_concern"
            ] = summary

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

    # -----------------------------------------------------
    # PHONE
    # -----------------------------------------------------

    phone = (
        normalize_phone(
            analysis.get(
                "phone",
                ""
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

    # -----------------------------------------------------
    # CALLBACK TIME
    # -----------------------------------------------------

    call_time = analysis.get(
        "preferred_call_time",
        ""
    ).strip()

    if (
        call_time
        and not lead.get(
            "preferred_call_time"
        )
    ):

        normalized_time = normalize_for_search(
            call_time
        )

        # "sabah" təkdirsə qəbul etməyə bilərik.
        if normalized_time not in {
            "sabah",
            "bugun",
            "bu gun",
        }:

            lead[
                "preferred_call_time"
            ] = call_time

    sync_flat_fields(
        lead
    )

    return list(
        dict.fromkeys(
            confirmations
        )
    )


# =========================================================
# 15. SALES FLOW
# =========================================================

def child_is_complete(
    child: dict,
) -> bool:

    # V8 simple sales flow:
    # age + need are operationally the most important.
    # Name can still be requested but is not used
    # to infer multiple children.

    if not child.get(
        "age"
    ):
        return False

    if not child.get(
        "main_concern"
    ):
        return False

    return True


def advance_child_if_needed(
    lead: dict,
):

    ensure_lead_structure(
        lead
    )

    count = lead.get(
        "declared_child_count",
        1,
    )

    if count <= 1:

        lead[
            "active_child_index"
        ] = 0

        return

    current_index = lead.get(
        "active_child_index",
        0,
    )

    current_child = lead[
        "children"
    ][current_index]

    if not child_is_complete(
        current_child
    ):
        return

    for index in range(
        min(
            count,
            len(
                lead["children"]
            ),
        )
    ):

        if not child_is_complete(
            lead["children"][index]
        ):

            lead[
                "active_child_index"
            ] = index

            return


def get_next_missing_field(
    lead: dict,
):

    ensure_lead_structure(
        lead
    )

    advance_child_if_needed(
        lead
    )

    child = get_active_child(
        lead
    )

    # Main sales flow requested by client:
    # age → need → parent name + phone → callback time

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

    # multiple children only when explicit
    count = lead.get(
        "declared_child_count",
        1,
    )

    if count > 1:

        for index in range(count):

            child_item = lead[
                "children"
            ][index]

            if not child_is_complete(
                child_item
            ):

                lead[
                    "active_child_index"
                ] = index

                return get_next_missing_field(
                    lead
                )

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
            "Sizinlə əlaqə saxlaya bilməyimiz üçün "
            "telefon nömrənizi qeyd edin, zəhmət olmasa."
        )

    if field == "preferred_call_time":

        return (
            "Zəng üçün sizə hansı gün və saat aralığı "
            "daha uyğun olar?"
        )

    return ""


def has_any_lead_info(
    lead: dict,
) -> bool:

    if lead.get(
        "parent_name"
    ):
        return True

    if lead.get(
        "phone"
    ):
        return True

    if lead.get(
        "preferred_call_time"
    ):
        return True

    for child in lead.get(
        "children",
        []
    ):

        if any([
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
        ]):
            return True

    return False


# =========================================================
# 16. STATE RECALL
# =========================================================

def build_state_recall_answer(
    lead: dict,
    requested_fields: List[str],
) -> str:

    ensure_lead_structure(
        lead
    )

    if not requested_fields:
        return ""

    parts = []

    parent = get_parent_display_name(
        lead
    )

    child = get_active_child(
        lead
    )

    for field in requested_fields:

        if field == "parent_name":

            if parent:

                parts.append(
                    f"Sizi {parent} kimi qeyd etmişəm"
                )

            else:

                parts.append(
                    "adınızı hələ qeyd etməmişəm"
                )

        elif field == "child_name":

            if child.get(
                "name"
            ):

                parts.append(
                    f"övladınızın adını {child['name']} kimi qeyd etmişəm"
                )

            else:

                parts.append(
                    "övladınızın adını hələ qeyd etməmişəm"
                )

        elif field == "child_age":

            if child.get(
                "age"
            ):

                parts.append(
                    f"yaşını {child['age']} olaraq qeyd etmişəm"
                )

            else:

                parts.append(
                    "yaşını hələ qeyd etməmişəm"
                )

        elif field == "main_concern":

            raw = child.get(
                "raw_concern"
            )

            summary = child.get(
                "main_concern"
            )

            if raw:

                parts.append(
                    f"əsas narahatlığınızı “{raw}” kimi qeyd etmişəm"
                )

            elif summary:

                parts.append(
                    f"əsas ehtiyacı “{summary}” kimi qeyd etmişəm"
                )

            else:

                parts.append(
                    "əsas narahatlıqla bağlı hələ qeyd yoxdur"
                )

        elif field == "phone":

            if lead.get(
                "phone"
            ):

                parts.append(
                    f"telefon nömrənizi {lead['phone']} kimi qeyd etmişəm"
                )

            else:

                parts.append(
                    "telefon nömrənizi hələ qeyd etməmişəm"
                )

        elif field == "preferred_call_time":

            if lead.get(
                "preferred_call_time"
            ):

                parts.append(
                    f"zəng vaxtını “{lead['preferred_call_time']}” kimi qeyd etmişəm"
                )

            else:

                parts.append(
                    "zəng vaxtını hələ qeyd etməmişəm"
                )

        elif field == "child_count":

            count = lead.get(
                "declared_child_count",
                1,
            )

            if count == 1:

                parts.append(
                    "bir övladınız olduğunu qeyd etmişəm"
                )

            else:

                parts.append(
                    f"{count} övladınız olduğunu qeyd etmişəm"
                )

    if not parts:
        return ""

    if len(parts) == 1:

        return (
            parts[0][0].upper()
            + parts[0][1:]
            + "."
        )

    return (
        "Hazırda belə qeyd etmişəm: "
        + "; ".join(
            parts
        )
        + "."
    )


# =========================================================
# 17. RELEVANT KNOWLEDGE COLLECTION
# =========================================================

def collect_relevant_knowledge(
    user_text: str,
    analysis: dict,
) -> List[dict]:

    queries = []

    for item in analysis.get(
        "questions",
        []
    ):

        text = item.get(
            "text",
            ""
        ).strip()

        if text:
            queries.append(text)

    if analysis.get(
        "has_situation_description"
    ):

        situation_query = " ".join([
            analysis.get(
                "raw_concern",
                ""
            ),

            analysis.get(
                "concern_summary",
                ""
            ),

            " ".join(
                analysis.get(
                    "interpreted_needs",
                    []
                )
            ),
        ]).strip()

        if situation_query:
            queries.append(
                situation_query
            )

    if not queries:

        queries.append(
            user_text
        )

    knowledge = []

    seen = set()

    for query in queries:

        candidates = retrieve_knowledge(
            query,
            k=5,
        )

        for candidate in candidates:

            key = (
                candidate[
                    "question"
                ],
                candidate[
                    "answer"
                ],
            )

            if key in seen:
                continue

            # Very low TF-IDF matches are mostly noise.
            if candidate[
                "score"
            ] < 0.08:
                continue

            seen.add(key)

            knowledge.append(
                candidate
            )

    return knowledge[:12]


# =========================================================
# 18. LLM RESPONSE GENERATOR
# =========================================================

def generate_contextual_response(
    user_text: str,
    lead: dict,
    analysis: dict,
    history: Optional[List[dict]],
    knowledge: List[dict],
) -> str:

    history = history or []

    if client is None:

        return (
            "Məlumatınızı qeyd etdim. "
            "Proqramla bağlı sualınızı məsul əməkdaşla "
            "dəqiqləşdirə bilərik."
        )

    state_for_llm = {
        "parent_name": lead.get(
            "parent_name"
        ),

        "parent_title": lead.get(
            "parent_title"
        ),

        "children": lead.get(
            "children"
        ),

        "declared_child_count": lead.get(
            "declared_child_count",
            1,
        ),

        "phone": lead.get(
            "phone"
        ),

        "preferred_call_time": lead.get(
            "preferred_call_time"
        ),
    }

    knowledge_text = []

    for index, item in enumerate(
        knowledge,
        start=1,
    ):

        knowledge_text.append(
            {
                "id": index,
                "faq_question": item[
                    "question"
                ],
                "fact": item[
                    "answer"
                ],
                "retrieval_score": round(
                    item[
                        "score"
                    ],
                    4,
                ),
            }
        )

    system_prompt = """
You are the RESPONSE GENERATION layer of the
Junior Coaching Sales & Operations Assistant.

You receive:
- current user message
- recent conversation
- structured state
- conversation analysis
- retrieved knowledge-base facts

Your job is to create ONE natural Azerbaijani reply.

You are NOT a generic chatbot and NOT a form bot.

=================================================
PRIMARY BEHAVIOR
=================================================

1. First understand what the parent actually means.
2. Answer every meaningful question/objection in the current turn.
3. Use current state and previous conversation.
4. Do not ask for information already known.
5. Do not invent facts.
6. Knowledge-base facts are SOURCE MATERIAL, not canned text.
7. Rephrase facts naturally for the current situation.
8. Be concise and human.
9. Do NOT append a sales-flow question. Another layer handles it.
10. Do not repeat greetings if conversation has already started.

=================================================
SITUATION REASONING
=================================================

When the parent describes behavior, connect the observations.

Example:

"Evdə rahat danışır, məktəbdə bildiyi cavabı
deməyə çəkinir."

A good response may say:

"Təsvir etdiyiniz vəziyyət onun ümumiyyətlə
ünsiyyət qura bilməməsindən çox, sosial mühitdə
özünüifadə və özünəinamla bağlı çətinliklə də
əlaqəli ola bilər."

Never diagnose.

NEVER say:
"bu mütləq özgüvən problemidir"
"onda sosial fobiya var"
"psixoloji problemi var"

Prefer:
"əlaqəli ola bilər"
"bu yazışmaya əsasən qəti nəticə demək olmaz"
"təsvir etdiyiniz vəziyyət..."

=================================================
KNOWLEDGE BASE
=================================================

Only use business/program facts supported by
RETRIEVED KNOWLEDGE.

If exact answer is NOT supported:
- say you do not want to give inaccurate information,
- offer human clarification if appropriate.

Do not fabricate:
- prices
- discounts
- dates
- schedules
- addresses
- program rules
- guarantees

=================================================
OBJECTIONS
=================================================

Answer the full objection, not just one keyword.

Example:
"Məcbur etmək istəmirəm, belə halda nə edirsiniz?"

Explain both:
- forcing is not ideal,
- how Junior Coaching approaches this situation,
IF supported by knowledge.

=================================================
REPEATED QUESTION
=================================================

If the parent asks again, do not copy the exact same answer.
Infer why they are asking again and respond more directly.

Example price follow-up:
The parent may need budget certainty.
Acknowledge that need rather than repeating the first answer.

=================================================
STYLE
=================================================

Azerbaijani.
Warm, professional, concise.
Usually 2–5 sentences.
Do not overuse emojis.
Do not sound clinical.
Do not sound like a knowledge-base dump.
"""

    prompt = f"""
CURRENT USER MESSAGE:
{user_text}

STRUCTURED STATE:
{json.dumps(state_for_llm, ensure_ascii=False)}

CONVERSATION ANALYSIS:
{json.dumps(analysis, ensure_ascii=False)}

RECENT CONVERSATION:
{json.dumps(history[-12:], ensure_ascii=False)}

RETRIEVED KNOWLEDGE:
{json.dumps(knowledge_text, ensure_ascii=False)}

Write the best response to the parent's CURRENT message.
Do not add the next sales-flow question.
"""

    try:

        response = client.chat.completions.create(
            model="gpt-4o-mini",
            temperature=0.25,

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
            "RESPONSE GENERATION ERROR:",
            exc,
        )

    return (
        "Məlumatınızı nəzərə aldım. "
        "Bu mövzuda yanlış məlumat verməmək üçün "
        "lazım olduqda məsul əməkdaşla dəqiqləşdirə bilərik."
    )


# =========================================================
# 19. FIELD FALLBACK
# =========================================================

def save_current_field_fallback(
    lead: dict,
    field: str,
    user_text: str,
):

    ensure_lead_structure(
        lead
    )

    child = get_active_child(
        lead
    )

    text = user_text.strip()

    if field == "child_age":

        age = extract_simple_age(
            text
        )

        if age:

            child[
                "age"
            ] = age

    elif field == "main_concern":

        # Avoid treating obviously unrelated question
        # as concern.
        if "?" not in text:

            child[
                "raw_concern"
            ] = text

            child[
                "main_concern"
            ] = text

    elif field == "parent_name":

        # Important:
        # never turn arbitrary sentence into name.
        if len(
            text.split()
        ) <= 3:

            name = clean_name(
                text
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

    elif field == "phone":

        phone = normalize_phone(
            text
        )

        if phone:

            lead[
                "phone"
            ] = phone

    elif field == "preferred_call_time":

        value = normalize_for_search(
            text
        )

        has_time_signal = (
            bool(
                re.search(
                    r"\b\d{1,2}[:.-]?\d{0,2}\b",
                    value,
                )
            )
            or any(
                x in value
                for x in [
                    "seher",
                    "gunorta",
                    "nahardan sonra",
                    "axsam",
                    "isden sonra",
                ]
            )
        )

        if has_time_signal:

            lead[
                "preferred_call_time"
            ] = text

    sync_flat_fields(
        lead
    )


# =========================================================
# 20. FINAL MESSAGE
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

    result = (
        "Qeydə alındı ✅"
    )

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
            f"\n\n{call_time} sizinlə əlaqə "
            "saxlanılması üçün müraciətinizi qeyd etdim."
        )

    result += (
        "\n\nİlkin zəng zamanı övladınızın "
        "iştirakı vacib deyil."
    )

    return result


# =========================================================
# 21. MAIN AGENT
# =========================================================

def lead_agent_reply(
    user_text: str,
    lead: dict,
    faq_min_score: float = 0.18,
    history: Optional[List[dict]] = None,
) -> str:

    # Kept only for compatibility with existing app.py
    del faq_min_score

    user_text = user_text.strip()

    history = history or []

    ensure_lead_structure(
        lead
    )

    lead[
        "_last_faq_score"
    ] = None

    field_before = get_next_missing_field(
        lead
    )

    # =====================================================
    # A. UNDERSTAND
    # =====================================================

    analysis = analyze_message(
        user_text=user_text,
        lead=lead,
        history=history,
    )

    print(
        "V8 ANALYSIS DEBUG:",
        analysis,
    )

    lead[
        "_last_intent"
    ] = analysis.get(
        "intent"
    )

    lead[
        "_last_confidence"
    ] = analysis.get(
        "confidence"
    )

    # =====================================================
    # B. EXTRACT + UPDATE STATE
    # =====================================================

    correction_confirmations = (
        merge_analysis_into_state(
            lead=lead,
            analysis=analysis,
            user_text=user_text,
            field_before=field_before,
        )
    )

    # =====================================================
    # C. SAFETY
    # =====================================================

    if analysis.get(
        "intent"
    ) == "safety_risk":

        lead[
            "status"
        ] = "ESCALATED"

        return (
            "Təsvir etdiyiniz vəziyyət daha diqqətli "
            "qiymətləndirmə tələb edə bilər. Junior Coaching "
            "tibbi və ya təcili psixoloji yardımı əvəz etmir. "
            "Müraciətinizi məsul əməkdaşa yönləndirirəm."
        )

    # =====================================================
    # D. HUMAN HANDOFF
    # =====================================================

    if analysis.get(
        "intent"
    ) == "human_agent_request":

        lead[
            "status"
        ] = "ESCALATED"

        return (
            "Əlbəttə. Müraciətinizi məsul əməkdaşa "
            "yönləndirmək üçün qeydə aldım."
        )

    # =====================================================
    # E. PERMISSION QUESTION
    # =====================================================

    if analysis.get(
        "intent"
    ) == "permission_question":

        return (
            "Əlbəttə, buyurun 😊"
        )

    # =====================================================
    # F. STATE RECALL
    # =====================================================

    recall_fields = analysis.get(
        "recall_fields",
        [],
    )

    if (
        recall_fields
        or analysis.get(
            "intent"
        ) == "state_question"
    ):

        recall_answer = build_state_recall_answer(
            lead,
            recall_fields,
        )

        if recall_answer:

            return recall_answer

    # =====================================================
    # G. PURE GREETING
    # =====================================================

    if (
        analysis.get(
            "intent"
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

            next_question = get_next_question(
                lead
            )

            if next_question:

                return (
                    "Salam 😊\n\n"
                    + next_question
                )

            return (
                "Salam 😊 Buyurun."
            )

        next_question = get_next_question(
            lead
        )

        return (
            next_question
            or "Buyurun 😊"
        )

    # =====================================================
    # H. CLARIFICATION
    # =====================================================

    if analysis.get(
        "needs_clarification"
    ):

        reason = analysis.get(
            "clarification_reason",
            ""
        ).strip()

        if reason:

            return (
                "Sizi düzgün anlamaq üçün bir məqamı "
                f"dəqiqləşdirim: {reason}"
            )

        return (
            "Sizi düzgün anlamaq üçün bunu bir az "
            "dəqiqləşdirə bilərsiniz?"
        )

    # =====================================================
    # I. DETERMINE WHETHER CURRENT MESSAGE NEEDS
    # CONTEXTUAL ANSWER
    # =====================================================

    has_questions = bool(
        analysis.get(
            "questions"
        )
    )

    has_situation = analysis.get(
        "has_situation_description",
        False,
    )

    intent_needs_answer = analysis.get(
        "intent"
    ) in {
        "faq_question",
        "program_interest",
        "situation_analysis",
        "complaint",
        "pause_request",
        "unrelated",
    }

    needs_contextual_answer = (
        has_questions
        or has_situation
        or intent_needs_answer
    )

    # =====================================================
    # J. RETRIEVE KNOWLEDGE + REASON + ANSWER
    # =====================================================

    if needs_contextual_answer:

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

        answer = generate_contextual_response(
            user_text=user_text,
            lead=lead,
            analysis=analysis,
            history=history,
            knowledge=knowledge,
        )

        # correction should be acknowledged
        if correction_confirmations:

            correction_text = (
                "Düzəltdim ✅ "
                + " ".join(
                    correction_confirmations
                )
            )

            answer = (
                correction_text
                + "\n\n"
                + answer
            )

        # -------------------------------------------------
        # SALES FLOW DECISION
        # -------------------------------------------------

        should_pause = analysis.get(
            "should_pause_sales_flow",
            False,
        )

        if should_pause:

            return answer

        next_question = get_next_question(
            lead
        )

        if next_question:

            # FAQ-only visitor with no meaningful lead info:
            if not has_any_lead_info(
                lead
            ):

                return (
                    answer
                    + "\n\n"
                    + "Başqa sualınız varsa, buyurun 😊 "
                    + "Müraciətə keçmək istəyirsinizsə, "
                    + next_question
                )

            return (
                answer
                + "\n\n"
                + next_question
            )

        # Flow complete after answering question
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

    # =====================================================
    # K. PURE INFORMATION / FIELD MESSAGE
    # =====================================================

    field_after_merge = get_next_missing_field(
        lead
    )

    if (
        field_before
        and field_before == field_after_merge
    ):

        save_current_field_fallback(
            lead=lead,
            field=field_before,
            user_text=user_text,
        )

    # =====================================================
    # L. CORRECTION-ONLY TURN
    # =====================================================

    if correction_confirmations:

        correction_text = (
            "Düzəltdim ✅ "
            + " ".join(
                correction_confirmations
            )
        )

        next_question = get_next_question(
            lead
        )

        if next_question:

            return (
                correction_text
                + "\n\n"
                + next_question
            )

        lead[
            "status"
        ] = "CALL_REQUESTED"

        return (
            correction_text
            + "\n\n"
            + build_final_message(
                lead
            )
        )

    # =====================================================
    # M. NEXT STEP
    # =====================================================

    next_field = get_next_missing_field(
        lead
    )

    if next_field is None:

        lead[
            "status"
        ] = "CALL_REQUESTED"

        return build_final_message(
            lead
        )

    if has_any_lead_info(
        lead
    ):

        lead[
            "_flow_started"
        ] = True

    return get_next_question(
        lead
    )


# =========================================================
# 22. DATABASE INIT
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

                needs_concern_followup INTEGER DEFAULT 0,

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

        # -------------------------------------------------
        # leads migration
        # -------------------------------------------------

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

        # -------------------------------------------------
        # children migration
        # -------------------------------------------------

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
            "needs_concern_followup":
                "INTEGER DEFAULT 0",
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

        # -------------------------------------------------
        # logs migration
        # -------------------------------------------------

        log_columns = {
            row[1]
            for row in conn.execute(
                "PRAGMA table_info(conversation_logs)"
            ).fetchall()
        }

        required_log_columns = {
            "intent": "TEXT",
            "confidence": "REAL",
            "faq_score": "REAL",
            "parent_title": "TEXT",
            "child_age": "INTEGER",
            "main_concern": "TEXT",
            "children_json": "TEXT",
            "preferred_call_time": "TEXT",
            "source": "TEXT",
        }

        for column, dtype in required_log_columns.items():

            if column not in log_columns:

                conn.execute(
                    f"""
                    ALTER TABLE conversation_logs
                    ADD COLUMN {column} {dtype}
                    """
                )

        conn.commit()


# =========================================================
# 23. TIME
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
# 24. FIND LEAD
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

            return dict(
                row
            )

    return None


# =========================================================
# 25. SAVE LEAD
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
        lead.get(
            "children",
            []
        ),
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
            lead.get(
                "children",
                []
            )
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

                    needs_concern_followup,

                    concern_duration,
                    concern_onset,

                    created_at
                )

                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
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
                            []
                        ),
                        ensure_ascii=False,
                    ),

                    int(
                        bool(
                            child.get(
                                "needs_concern_followup",
                                False,
                            )
                        )
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
# 26. CONVERSATION LOG
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
        lead.get(
            "children",
            []
        ),
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
                    "_last_intent"
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
# 27. ADMIN HELPERS
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