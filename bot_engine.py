"""
Junior Coaching — Conversation Engine V7

V7 əsas məqsədləri
------------------
1. Whole-message understanding
2. Semantic question topics
3. Multiple questions in one message
4. Reliable state memory
5. Explicit corrections / overwrite
6. Safe multi-child handling
7. Contextual age extraction
8. Maximum one flow question per turn
9. FAQ + state + conversation context together
10. Existing Streamlit / SQLite compatibility
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
# 4. CONTEXTUAL AGE EXTRACTION
# =========================================================

def extract_contextual_ages(
    text: str,
) -> List[int]:

    """
    Telefon və saat rəqəmlərini yaş kimi götürmür.

    Qəbul edir:
        14 yaş
        14 yaşı var
        14 yaşlı
        13 və 15 yaşında
        uşaqlarım 12 və 16 yaşındadır

    Qəbul ETMİR:
        050 123 45 67
        14:00-16:00
    """

    normalized = normalize_for_search(
        text
    )

    ages: List[int] = []

    # 14 yaş / 14 yaşı / 14 yaşlı / 14 yaşında
    patterns = [
        r"\b(\d{1,2})\s*yas\b",
        r"\b(\d{1,2})\s*yasi\b",
        r"\b(\d{1,2})\s*yasli\b",
        r"\b(\d{1,2})\s*yasinda\b",
    ]

    for pattern in patterns:

        for value in re.findall(
            pattern,
            normalized,
        ):

            age = int(value)

            if (
                5 <= age <= 25
                and age not in ages
            ):
                ages.append(age)

    # 13 və 15 yaşında
    pair_patterns = [
        r"\b(\d{1,2})\s*(?:ve|və|,)\s*(\d{1,2})\s*yas",
        r"\b(\d{1,2})\s*(?:ve|və|,)\s*(\d{1,2})\s*yasinda",
    ]

    for pattern in pair_patterns:

        match = re.search(
            pattern,
            normalized,
        )

        if match:

            for value in match.groups():

                age = int(value)

                if (
                    5 <= age <= 25
                    and age not in ages
                ):
                    ages.append(age)

    return ages


def extract_single_age_if_current_field(
    text: str,
) -> Optional[int]:

    """
    Bot konkret yaş soruşubsa user sadəcə:
        14
        12 tamam olacaq
    deyə bilər.
    """

    normalized = normalize_for_search(
        text
    )

    numbers = re.findall(
        r"\b\d{1,2}\b",
        normalized,
    )

    if len(numbers) != 1:
        return None

    value = int(numbers[0])

    if 5 <= value <= 25:
        return value

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
# 6. NAME CLEANING
# =========================================================

NON_NAME_TOKENS = {
    "men",
    "mene",
    "anasi",
    "anasiyam",
    "atasi",
    "atasiyam",
    "valideyn",
    "valideynem",
    "valideyniyem",
    "oglum",
    "qizim",
    "usaq",
    "ovladim",
    "dedim",
    "deyirem",
    "demisdim",
    "yuxarida",
    "salam",
    "sagol",
    "sagolun",
    "tesekkur",
    "tesekkurler",
    "adim",
    "adi",
    "mən",
    "mənə",
}


def remove_honorific(
    text: str,
) -> str:

    blocked = {
        "bey",
        "bəy",
        "xanim",
        "xanım",
        "muellim",
        "müəllim",
    }

    result = []

    for word in text.strip().split():

        if normalize_for_search(
            word
        ) not in {
            normalize_for_search(x)
            for x in blocked
        }:
            result.append(word)

    return " ".join(result)


def remove_name_suffix(
    word: str,
) -> str:

    if not word:
        return word

    normalized = normalize_for_search(
        word
    )

    for suffix in [
        "dir",
        "dur",
    ]:

        if normalized.endswith(
            suffix
        ):

            base_length = (
                len(word)
                - len(suffix)
            )

            if base_length >= 4:
                return word[:base_length]

    return word


def clean_name(
    value: str,
) -> Optional[str]:

    if not value:
        return None

    value = remove_honorific(
        value
    )

    value = re.sub(
        r"[.,!?+():;]",
        "",
        value,
    ).strip()

    if not value:
        return None

    words = []

    for word in value.split():

        normalized_word = normalize_for_search(
            word
        )

        if normalized_word in NON_NAME_TOKENS:
            continue

        cleaned = remove_name_suffix(
            word
        )

        if cleaned:
            words.append(cleaned)

    if not words:
        return None

    # Valideyn adının 5 sözə çevrilməsinin qarşısı
    words = words[:2]

    result = " ".join(
        words
    )

    if not re.fullmatch(
        r"[A-Za-zƏəÖöÜüĞğÇçŞşİı\- ]+",
        result,
    ):
        return None

    return result.title()


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
# 7. AZERBAIJANI NAME SUFFIX
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
# 8. LEAD / CHILD STRUCTURE
# =========================================================

def create_empty_child() -> dict:

    return {
        "name": None,
        "age": None,
        "main_concern": None,
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

        # V7
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
    }


def ensure_lead_structure(
    lead: dict,
):

    lead.setdefault(
        "children",
        [
            create_empty_child()
        ]
    )

    if not lead["children"]:

        lead["children"] = [
            create_empty_child()
        ]

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
# 9. CHILD COUNT — IMPORTANT V7 FIX
# =========================================================

def detect_explicit_child_count(
    text: str,
) -> Optional[int]:

    """
    Yalnız explicit uşaq sayını qəbul edir.

    "050 123 45 67 + övladım" artıq multi-child yaratmır.
    """

    value = normalize_for_search(
        text
    )

    # 2-ci övlad yoxdur / bir oğlum var
    one_child_patterns = [
        "bir oglum var",
        "bir qizim var",
        "bir usagim var",
        "bir ovladim var",
        "tek usagim var",
        "tek ovladim var",
        "2 ci ovlad yoxdur",
        "ikinci ovlad yoxdur",
        "iki usaq deyil",
    ]

    if any(
        pattern in value
        for pattern in one_child_patterns
    ):
        return 1

    # iki uşağım / 2 uşağım
    two_child_patterns = [
        "iki usagim",
        "iki ovladim",
        "2 usagim",
        "2 ovladim",
        "2 usaqdir",
        "iki usaqdir",
        "iki usaq var",
        "2 usaq var",
    ]

    if any(
        pattern in value
        for pattern in two_child_patterns
    ):
        return 2

    # 3 uşağım və s.
    match = re.search(
        r"\b([2-5])\s*(?:usaq|ovlad)",
        value,
    )

    if match:

        return int(
            match.group(1)
        )

    # 13 və 15 yaşında iki uşağım var
    ages = extract_contextual_ages(
        text
    )

    if (
        len(ages) >= 2
        and any(
            marker in value
            for marker in [
                "iki usaq",
                "2 usaq",
                "iki ovlad",
                "2 ovlad",
            ]
        )
    ):
        return len(ages)

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

        if len(
            lead["children"]
        ) > count:

            lead["children"] = (
                lead["children"][:count]
            )

    sync_flat_fields(
        lead
    )


# =========================================================
# 10. PARENT TITLE
# =========================================================

def infer_parent_title_with_llm(
    parent_name: str,
) -> str:

    if (
        not parent_name
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

Yalnız bunlardan birini qaytar:
xanım
bəy
neutral

Əmin deyilsənsə neutral.
"""
                },
                {
                    "role": "user",
                    "content": parent_name,
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

        result = json.loads(
            response.choices[0].message.content
        )

        if result["title"] in [
            "xanım",
            "bəy",
        ]:

            return result[
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
        return f"{name} {title}"

    return name


# =========================================================
# 11. FAQ INDEX
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
            "FAQ faylında Sual / Agent cütləri tapılmadı."
        )

    questions = [
        q
        for q, _ in pairs
    ]

    answers = [
        a
        for _, a in pairs
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


def retrieve_faq_candidates(
    query: str,
    k: int = 7,
):

    normalized_query = normalize_for_search(
        query
    )

    vector = FAQ_VECTORIZER.transform(
        [normalized_query]
    )

    scores = cosine_similarity(
        vector,
        FAQ_MATRIX,
    ).ravel()

    top_indices = np.argsort(
        -scores
    )[:k]

    result = []

    for index in top_indices:

        result.append({
            "index": int(index),
            "question": FAQ_QUESTIONS[index],
            "answer": FAQ_ANSWERS[index],
            "score": float(scores[index]),
        })

    return result


# =========================================================
# 12. CANONICAL QUESTION TOPICS
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
    "other",
]


def deterministic_question_topic(
    question: str,
) -> Optional[str]:

    value = normalize_for_search(
        question
    )

    # LOCATION MUST OVERRIDE "keçirilir"
    if any(
        marker in value
        for marker in [
            "harada",
            "hardadir",
            "hardadi",
            "unvan",
            "adres",
            "mekan",
            "erazide",
            "hansi erazi",
            "yerlesir",
            "nece gele",
        ]
    ):
        return "meeting_location"

    if any(
        marker in value
        for marker in [
            "hansi gun",
            "ne vaxt",
            "bazar gunu",
            "heftenin hansi",
            "gunleri olur",
        ]
    ):
        return "meeting_day"

    if any(
        marker in value
        for marker in [
            "ayda nece",
            "heftede nece",
            "nece defe",
            "tezliyi",
        ]
    ):
        return "meeting_frequency"

    if (
        any(
            marker in value
            for marker in [
                "telefon zengi",
                "valideynle zeng",
                "ilkin zeng",
                "tanisliq zengi",
            ]
        )
        and any(
            marker in value
            for marker in [
                "nece deqiqe",
                "ne qeder cekir",
                "muddet",
                "davam edir",
            ]
        )
    ):
        return "parent_call_duration"

    if (
        any(
            marker in value
            for marker in [
                "qrup gorusu",
                "gorus",
                "sessiya",
            ]
        )
        and any(
            marker in value
            for marker in [
                "nece saat",
                "ne qeder davam",
                "muddet",
            ]
        )
    ):
        return "meeting_duration"

    if any(
        marker in value
        for marker in [
            "gele bilmediyimiz",
            "buraxdigimiz",
            "buraxilan gorus",
            "evez etmek",
            "evezlenir",
            "qacirdigimiz",
        ]
    ):
        return "missed_session"

    if any(
        marker in value
        for marker in [
            "bir defe",
            "bir goruse",
            "yalniz bir gorus",
            "tek gorus",
        ]
    ):
        return "one_time_session"

    if any(
        marker in value
        for marker in [
            "qiymet",
            "odenis",
            "budce",
            "ne qederdir",
            "texmini qiymet",
        ]
    ):
        return "price"

    if any(
        marker in value
        for marker in [
            "gelmek istemese",
            "istirak etmek istemese",
            "usaq istemir",
            "ovladim istemir",
            "proqrama gelmek istemir",
        ]
    ):
        return "child_refusal"

    if any(
        marker in value
        for marker in [
            "rus dili",
            "rus dilli",
            "ingilis dili",
            "dilinde",
            "hansi dil",
        ]
    ):
        return "language"

    if any(
        marker in value
        for marker in [
            "nece yas",
            "yasdan",
            "yas qrupu",
            "12 18",
        ]
    ):
        return "program_age"

    if any(
        marker in value
        for marker in [
            "proqram haqqinda",
            "proqram nedir",
            "melumat ver",
            "etrafli melumat",
            "ne edir",
        ]
    ):
        return "program_info"

    return None


# =========================================================
# 13. FAQ SELECTION WITH TOPIC
# =========================================================

def select_best_faq_with_llm(
    question: str,
    topic: str,
    candidates: List[Dict[str, Any]],
):

    if not candidates:
        return None

    if client is None:

        best = candidates[0]

        if best["score"] >= 0.22:
            return best

        return None

    candidate_text = "\n\n".join([
        (
            f"ID={i}\n"
            f"Sual: {candidate['question']}\n"
            f"Cavab: {candidate['answer']}"
        )
        for i, candidate in enumerate(
            candidates
        )
    ])

    try:

        response = client.chat.completions.create(
            model="gpt-4o-mini",
            temperature=0,

            messages=[
                {
                    "role": "system",
                    "content": """
FAQ seçicisən.

İstifadəçi sualının mənasına uyğun FAQ seç.

Canonical topic də verilib və ona ciddi əməl et.

Xüsusilə:
meeting_location = ÜNVAN / MƏKAN
meeting_day = hansı gün / nə vaxt
meeting_duration = qrup görüşünün neçə saat olması
parent_call_duration = valideynlə ilkin telefon zənginin müddəti
missed_session = buraxılmış görüşün əvəzlənməsi
one_time_session = yalnız bir görüşə gəlmək
price = qiymət / ödəniş

"harada keçirilir?" sualına "bazar günü keçirilir"
və ya "interaktiv keçirilir" cavabını seçmə.

Uyğun FAQ yoxdursa selected_id=-1.
"""
                },
                {
                    "role": "user",
                    "content": (
                        f"TOPIC: {topic}\n\n"
                        f"USER QUESTION:\n{question}\n\n"
                        f"CANDIDATES:\n{candidate_text}"
                    ),
                },
            ],

            response_format={
                "type": "json_schema",
                "json_schema": {
                    "name": "faq_selection",
                    "strict": True,
                    "schema": {
                        "type": "object",
                        "properties": {
                            "selected_id": {
                                "type": "integer"
                            },
                            "confidence": {
                                "type": "number"
                            },
                        },
                        "required": [
                            "selected_id",
                            "confidence",
                        ],
                        "additionalProperties": False,
                    },
                },
            },
        )

        data = json.loads(
            response.choices[0].message.content
        )

        selected_id = data.get(
            "selected_id",
            -1,
        )

        confidence = data.get(
            "confidence",
            0,
        )

        if (
            0 <= selected_id < len(candidates)
            and confidence >= 0.55
        ):

            return candidates[
                selected_id
            ]

    except Exception as exc:

        print(
            "FAQ SELECT ERROR:",
            exc,
        )

    best = candidates[0]

    if best["score"] >= 0.30:
        return best

    return None


# =========================================================
# 14. SPECIAL DETERMINISTIC ANSWERS
# =========================================================

LOCATION_ANSWER = (
    "Görüşlər Süleyman Sani Axundov küçəsində yerləşən "
    "ADAS Plaza-da, ELİT T/M yaxınlığında keçirilir."
)


def is_permission_to_ask(
    text: str,
) -> bool:

    value = normalize_for_search(
        text
    )

    return any(
        pattern in value
        for pattern in [
            "bir sual vere bilerem",
            "bir sual verim",
            "sual vere bilerem",
            "bir sey sorusa bilerem",
            "sizden bir sey sorusum",
        ]
    )


# =========================================================
# 15. MEMORY / STATE QUESTIONS
# =========================================================

def detect_state_question(
    text: str,
) -> Optional[str]:

    value = normalize_for_search(
        text
    )

    if any(
        x in value
        for x in [
            "adimi qeyd etdiniz",
            "adimi yazdiniz",
            "menim adim ne idi",
            "adimi goturdunuz",
        ]
    ):
        return "parent_name"

    if any(
        x in value
        for x in [
            "usaqin adi ne idi",
            "ovladimin adi ne idi",
            "usaqin adini qeyd",
        ]
    ):
        return "child_name"

    if (
        any(
            x in value
            for x in [
                "nece yasi",
                "yasi nece",
                "yasini demisdim",
                "yasini qeyd",
            ]
        )
        and any(
            x in value
            for x in [
                "usaq",
                "oglum",
                "qizim",
                "ovlad",
            ]
        )
    ):
        return "child_age"

    if any(
        x in value
        for x in [
            "nece ovladim",
            "nece usagim",
            "nece usaq",
            "nece ovlad",
        ]
    ):
        return "child_count"

    if any(
        x in value
        for x in [
            "nomremi qeyd",
            "telefonumu qeyd",
            "nomrem ne idi",
        ]
    ):
        return "phone"

    if any(
        x in value
        for x in [
            "zeng vaxtini qeyd",
            "ne vaxt demisdim",
            "hansi vaxt demisdim",
        ]
    ):
        return "preferred_call_time"

    return None


def answer_state_question(
    text: str,
    lead: dict,
) -> Optional[str]:

    field = detect_state_question(
        text
    )

    if field is None:
        return None

    if field == "parent_name":

        parent = get_parent_display_name(
            lead
        )

        if parent:

            return (
                f"Bəli, adınızı {parent} kimi qeyd etmişəm. 😊"
            )

        return (
            "Hələ adınızı qeyd etməmişəm."
        )

    if field == "child_name":

        child = get_active_child(
            lead
        )

        if child.get("name"):

            return (
                f"Bəli, övladınızın adını "
                f"{child['name']} kimi qeyd etmişəm."
            )

        return (
            "Hələ övladınızın adını qeyd etməmişəm."
        )

    if field == "child_age":

        child = get_active_child(
            lead
        )

        if child.get("age"):

            name = child.get(
                "name"
            )

            if name:

                return (
                    f"Bəli, {child_genitive(name)} "
                    f"{child['age']} yaşı olduğunu qeyd etmişəm."
                )

            return (
                f"Bəli, övladınızın {child['age']} yaşı "
                "olduğunu qeyd etmişəm."
            )

        return (
            "Hələ övladınızın yaşını qeyd etməmişəm."
        )

    if field == "child_count":

        count = lead.get(
            "declared_child_count",
            1,
        )

        if count == 1:

            return (
                "Bir övladınız olduğunu qeyd etmişəm."
            )

        return (
            f"{count} övladınız olduğunu qeyd etmişəm."
        )

    if field == "phone":

        phone = lead.get(
            "phone"
        )

        if phone:

            return (
                f"Bəli, telefon nömrənizi {phone} "
                "kimi qeyd etmişəm."
            )

        return (
            "Hələ telefon nömrənizi qeyd etməmişəm."
        )

    if field == "preferred_call_time":

        call_time = lead.get(
            "preferred_call_time"
        )

        if call_time:

            return (
                f"Bəli, zəng üçün uyğun vaxtı "
                f"“{call_time}” kimi qeyd etmişəm."
            )

        return (
            "Hələ zəng üçün uyğun vaxt qeyd etməmişəm."
        )

    return None


# =========================================================
# 16. CONVERSATION ANALYZER
# =========================================================

def analyze_message(
    user_text: str,
    lead: dict,
    history: Optional[List[dict]] = None,
) -> dict:

    ensure_lead_structure(
        lead
    )

    history = history or []

    state = {
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
            "declared_child_count"
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

    if client is None:

        return {
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
            "main_concern": "",
            "phone": "",
            "preferred_call_time": "",
            "child_count": 0,
            "corrections": [],
            "confidence": 0.0,
        }

    system_prompt = """
Sən Junior Coaching Conversation Analyzer-sən.

Cari mesajı:
1. əvvəlki conversation history,
2. hazırkı state
ilə birlikdə analiz et.

BİR MESAJDA BİRDƏN ÇOX MƏLUMAT VARSA HAMISINI ÇIXAR.

BİR MESAJDA BİRDƏN ÇOX SUAL VARSA HAMISINI AYRI questions ELEMENTİ ET.

questions elementinin:
- text
- topic
sahələri var.

Topic-lər:
meeting_location
meeting_day
meeting_frequency
meeting_duration
parent_call_duration
child_intro_call
missed_session
one_time_session
price
program_info
program_age
child_refusal
language
registration
other


MÜHÜM NÜMUNƏ:

"Görüşlər hansı gün olur və harada keçirilir?"

questions:
[
  {
    "text": "Görüşlər hansı gün olur?",
    "topic": "meeting_day"
  },
  {
    "text": "Görüşlər harada keçirilir?",
    "topic": "meeting_location"
  }
]


CORRECTION:

"Tunar yox, Turandır"
=> correction:
field=child_name
value=Turan
child_index=0

"Səhv yazmışdım, uşağın adı Turandır"
=> child_name correction.

"Aygün mənəm, uşağın adı Ayxandır"
=> iki correction:
parent_name=Aygün
child_name=Ayxan

"2-ci övlad yoxdur, bir oğlum var"
=> child_count=1


AD QAYDALARI:

"anasıyam"
"atasıyam"
"mənə"
"mən"
"övladım"
ad DEYİL.

"Adım Günaydır"
=> parent_name Günay.

"Oğlum Tunar"
=> child_name Tunar.


YAŞ:

Telefon nömrəsində və saatda olan rəqəmləri yaş kimi götürmə.

050 123 45 67
=> child_age DEYİL.

14:00
=> child_age DEYİL.

"14 yaşlı oğlum"
=> child_age=14.


MAIN CONCERN:

özgüvəni zəifdir
özünə qapanır
ünsiyyətdə çətinlik çəkir
məsuliyyətsizdir
məqsəd və gələcək
hamısı

bunlar complaint DEYİL.
Bunlar main_concern-dir.


STATE / MEMORY:

"neçə övladım olduğunu demişdim?"
"oğlumun neçə yaşı olduğunu demişdim?"
"adımı qeyd etdiniz?"

FAQ DEYİL.
intent=state_question.


İstifadəçi həm məlumat, həm sual verə bilər.
Hər ikisini çıxar.

Intent:
greeting
faq_question
field_answer
program_interest
registration_request
state_question
permission_question
correction
human_agent_request
complaint
safety_risk
meta_question
pause_request
unrelated
"""

    user_prompt = f"""
CURRENT STATE:
{json.dumps(state, ensure_ascii=False)}

RECENT HISTORY:
{json.dumps(history[-10:], ensure_ascii=False)}

USER:
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
                    "content": user_prompt,
                },
            ],

            response_format={
                "type": "json_schema",
                "json_schema": {
                    "name": "conversation_analysis",
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
                                    "permission_question",
                                    "correction",
                                    "human_agent_request",
                                    "complaint",
                                    "safety_risk",
                                    "meta_question",
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

                            "main_concern": {
                                "type": "string"
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
                            "main_concern",
                            "phone",
                            "preferred_call_time",
                            "child_count",
                            "corrections",
                            "confidence",
                        ],

                        "additionalProperties": False,
                    },
                },
            },
        )

        return json.loads(
            response.choices[0].message.content
        )

    except Exception as exc:

        print(
            "ANALYSIS ERROR:",
            exc,
        )

        return {
            "intent": "field_answer",
            "questions": [],
            "parent_name": "",
            "child_name": "",
            "child_age": 0,
            "main_concern": "",
            "phone": "",
            "preferred_call_time": "",
            "child_count": 0,
            "corrections": [],
            "confidence": 0.0,
        }


# =========================================================
# 17. APPLY CORRECTIONS
# =========================================================

def apply_corrections(
    lead: dict,
    corrections: List[dict],
) -> List[str]:

    ensure_lead_structure(
        lead
    )

    confirmations = []

    for correction in corrections:

        field = correction.get(
            "field",
            ""
        )

        value = str(
            correction.get(
                "value",
                ""
            )
        ).strip()

        child_index = correction.get(
            "child_index",
            0,
        )

        if not value:
            continue

        if field == "parent_name":

            cleaned = clean_name(
                value
            )

            if cleaned:

                lead[
                    "parent_name"
                ] = cleaned

                lead[
                    "parent_title"
                ] = infer_parent_title_with_llm(
                    cleaned
                )

                confirmations.append(
                    f"Adınızı {cleaned} olaraq düzəltdim."
                )

        elif field == "child_name":

            child_index = max(
                0,
                child_index,
            )

            while len(
                lead["children"]
            ) <= child_index:

                lead["children"].append(
                    create_empty_child()
                )

            cleaned = clean_name(
                value
            )

            if cleaned:

                lead[
                    "children"
                ][child_index][
                    "name"
                ] = cleaned

                confirmations.append(
                    f"Övladınızın adını {cleaned} olaraq düzəltdim."
                )

        elif field == "child_age":

            age = extract_single_age_if_current_field(
                value
            )

            if age:

                while len(
                    lead["children"]
                ) <= child_index:

                    lead["children"].append(
                        create_empty_child()
                    )

                lead[
                    "children"
                ][child_index][
                    "age"
                ] = age

                confirmations.append(
                    f"Yaşı {age} olaraq düzəltdim."
                )

        elif field == "main_concern":

            while len(
                lead["children"]
            ) <= child_index:

                lead["children"].append(
                    create_empty_child()
                )

            lead[
                "children"
            ][child_index][
                "main_concern"
            ] = value

            confirmations.append(
                "Qeyd etdiyiniz əsas ehtiyacı yenilədim."
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
                    "Telefon nömrəsini düzəltdim."
                )

        elif field == "preferred_call_time":

            lead[
                "preferred_call_time"
            ] = value

            confirmations.append(
                "Zəng üçün uyğun vaxtı yenilədim."
            )

        elif field == "child_count":

            numbers = re.findall(
                r"\d+",
                value,
            )

            if numbers:

                set_child_count(
                    lead,
                    int(numbers[0]),
                )

    sync_flat_fields(
        lead
    )

    return confirmations


# =========================================================
# 18. MERGE NEW INFORMATION
# =========================================================

def merge_analysis(
    lead: dict,
    analysis: dict,
    user_text: str,
    current_field: Optional[str] = None,
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

    # Explicit child count has precedence
    explicit_count = detect_explicit_child_count(
        user_text
    )

    if explicit_count is not None:

        set_child_count(
            lead,
            explicit_count,
        )

        if explicit_count == 1:

            confirmations.append(
                "Bir övladınız olduğunu nəzərə aldım."
            )

    elif analysis.get(
        "child_count",
        0,
    ) > 1:

        # LLM cannot create multi child unless wording
        # is actually explicit enough.
        normalized_user = normalize_for_search(
            user_text
        )

        if any(
            marker in normalized_user
            for marker in [
                "iki usaq",
                "2 usaq",
                "iki ovlad",
                "2 ovlad",
                "usaqlarim",
                "ovladlarim",
            ]
        ):

            set_child_count(
                lead,
                analysis[
                    "child_count"
                ],
            )

    # Parent name
    deterministic_name = (
        deterministic_parent_name_extract(
            user_text
        )
    )

    parent_name = (
        deterministic_name
        or analysis.get(
            "parent_name",
            ""
        ).strip()
    )

    if (
        parent_name
        and not lead.get(
            "parent_name"
        )
    ):

        cleaned = clean_name(
            parent_name
        )

        if cleaned:

            lead[
                "parent_name"
            ] = cleaned

            lead[
                "parent_title"
            ] = infer_parent_title_with_llm(
                cleaned
            )

    # Child
    child = get_active_child(
        lead
    )

    child_name = analysis.get(
        "child_name",
        ""
    ).strip()

    if (
        child_name
        and not child.get(
            "name"
        )
    ):

        cleaned = clean_name(
            child_name
        )

        if cleaned:
            child[
                "name"
            ] = cleaned

    # Contextual age only
    contextual_ages = extract_contextual_ages(
        user_text
    )

    if contextual_ages:

        if (
            lead.get(
                "declared_child_count",
                1,
            ) > 1
            and len(
                contextual_ages
            ) > 1
        ):

            for i, age in enumerate(
                contextual_ages
            ):

                if i < len(
                    lead["children"]
                ):

                    if not lead[
                        "children"
                    ][i].get(
                        "age"
                    ):

                        lead[
                            "children"
                        ][i][
                            "age"
                        ] = age

        elif not child.get(
            "age"
        ):

            child[
                "age"
            ] = contextual_ages[0]

    elif (
        current_field == "child_age"
        and not child.get(
            "age"
        )
    ):

        simple_age = extract_single_age_if_current_field(
            user_text
        )

        if simple_age:
            child["age"] = simple_age

    # LLM child_age only if no phone-style ambiguity
    elif (
        analysis.get(
            "child_age",
            0
        )
        and not child.get(
            "age"
        )
        and normalize_phone(
            user_text
        ) is None
    ):

        age = analysis[
            "child_age"
        ]

        if 5 <= age <= 25:
            child["age"] = age

    # Concern
    concern = analysis.get(
        "main_concern",
        ""
    ).strip()

    if (
        concern
        and not child.get(
            "main_concern"
        )
    ):

        normalized_concern = normalize_for_search(
            concern
        )

        if normalized_concern in {
            "hamisi",
            "her biri",
        }:

            concern = (
                "özgüvən, məqsəd və gələcək, "
                "məsuliyyət və intizam, ünsiyyət"
            )

        child[
            "main_concern"
        ] = concern

        if any(
            word in normalized_concern
            for word in [
                "fikirli",
                "ozune qapan",
                "danismir",
            ]
        ):

            child[
                "needs_concern_followup"
            ] = True

    # Phone
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

    # Call time
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

        normalized_call = normalize_for_search(
            call_time
        )

        # tək "sabah" qeyri-dəqiqdir
        if normalized_call not in {
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

    return confirmations


# =========================================================
# 19. FLOW
# =========================================================

def child_is_complete(
    child: dict,
) -> bool:

    if not child.get(
        "name"
    ):
        return False

    if not child.get(
        "age"
    ):
        return False

    if not child.get(
        "main_concern"
    ):
        return False

    if child.get(
        "needs_concern_followup"
    ):

        if not child.get(
            "concern_duration"
        ):
            return False

        if not child.get(
            "concern_onset"
        ):
            return False

    return True


def advance_child_if_needed(
    lead: dict,
):

    ensure_lead_structure(
        lead
    )

    # Only iterate over EXPLICITLY declared children.
    count = lead.get(
        "declared_child_count",
        1,
    )

    if count <= 1:

        lead[
            "active_child_index"
        ] = 0

        return

    index = lead.get(
        "active_child_index",
        0,
    )

    current = lead[
        "children"
    ][index]

    if not child_is_complete(
        current
    ):
        return

    for i in range(count):

        if not child_is_complete(
            lead["children"][i]
        ):

            lead[
                "active_child_index"
            ] = i

            return


def get_next_missing_field(
    lead: dict,
):

    ensure_lead_structure(
        lead
    )

    if not lead.get(
        "parent_name"
    ):
        return "parent_name"

    advance_child_if_needed(
        lead
    )

    child = get_active_child(
        lead
    )

    if not child.get(
        "name"
    ):
        return "child_name"

    if not child.get(
        "age"
    ):
        return "child_age"

    if not child.get(
        "main_concern"
    ):
        return "main_concern"

    if child.get(
        "needs_concern_followup"
    ):

        if not child.get(
            "concern_duration"
        ):
            return "concern_duration"

        if not child.get(
            "concern_onset"
        ):
            return "concern_onset"

    # Only explicit additional children
    count = lead.get(
        "declared_child_count",
        1,
    )

    if count > 1:

        for i in range(count):

            if not child_is_complete(
                lead["children"][i]
            ):

                lead[
                    "active_child_index"
                ] = i

                return get_next_missing_field(
                    lead
                )

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

    parent = get_parent_display_name(
        lead
    )

    if field == "parent_name":

        return (
            "Sizə necə müraciət edə bilərəm?"
        )

    if field == "child_name":

        if (
            lead.get(
                "declared_child_count",
                1,
            ) > 1
        ):

            number = (
                lead.get(
                    "active_child_index",
                    0,
                )
                + 1
            )

            return (
                f"{number}-ci övladınızın adını "
                "öyrənə bilərəm?"
            )

        if parent:

            return (
                f"Məmnun oldum, {parent}. "
                "Övladınızın adını öyrənə bilərəm?"
            )

        return (
            "Övladınızın adını öyrənə bilərəm?"
        )

    if field == "child_age":

        name = child.get(
            "name"
        )

        if name:

            return (
                f"{child_genitive(name)} neçə yaşı var?"
            )

        return (
            "Övladınızın neçə yaşı var?"
        )

    if field == "main_concern":

        return (
            "Ən çox hansı sahədə inkişaf etməsini istərdiniz?\n\n"
            "Məsələn: özgüvən, məqsəd və gələcək, "
            "məsuliyyət və intizam, ünsiyyət və s."
        )

    if field == "concern_duration":

        return (
            "Bu hal nə qədər müddətdir davam edir?"
        )

    if field == "concern_onset":

        return (
            "Sizcə bu vəziyyət hansısa hadisədən sonra "
            "başlayıb, yoxsa tədricən?"
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
            child.get("name"),
            child.get("age"),
            child.get("main_concern"),
        ]):
            return True

    return False


# =========================================================
# 20. ANSWER ONE QUESTION
# =========================================================

def answer_single_question(
    question: str,
    topic: str,
    lead: dict,
):

    # State is always answered from memory.
    state_answer = answer_state_question(
        question,
        lead,
    )

    if state_answer:
        return state_answer

    if is_permission_to_ask(
        question
    ):
        return (
            "Əlbəttə, buyurun 😊"
        )

    deterministic_topic = (
        deterministic_question_topic(
            question
        )
    )

    if deterministic_topic:
        topic = deterministic_topic

    # Important hard guarantee for location
    if topic == "meeting_location":

        return LOCATION_ANSWER

    candidates = retrieve_faq_candidates(
        question,
        k=7,
    )

    faq = select_best_faq_with_llm(
        question=question,
        topic=topic,
        candidates=candidates,
    )

    if faq:

        lead[
            "_last_faq_score"
        ] = faq[
            "score"
        ]

        return faq[
            "answer"
        ]

    return (
        "Bu sualla bağlı məlumat bazasında dəqiq cavab "
        "tapmadım. İstəsəniz bu sualı məsul əməkdaşa "
        "yönləndirə bilərik."
    )


def answer_all_questions(
    questions: List[dict],
    lead: dict,
) -> str:

    answers = []

    for item in questions:

        question = item.get(
            "text",
            ""
        ).strip()

        topic = item.get(
            "topic",
            "other",
        )

        if not question:
            continue

        answer = answer_single_question(
            question=question,
            topic=topic,
            lead=lead,
        )

        if (
            answer
            and answer not in answers
        ):
            answers.append(answer)

    return "\n\n".join(
        answers
    )


# =========================================================
# 21. FIELD FALLBACK
# =========================================================

def save_current_field_fallback(
    lead: dict,
    field: str,
    user_text: str,
):

    child = get_active_child(
        lead
    )

    value = user_text.strip()

    normalized = normalize_for_search(
        value
    )

    if field == "parent_name":

        name = (
            deterministic_parent_name_extract(
                value
            )
            or clean_name(
                remove_honorific(
                    value
                )
            )
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

    elif field == "child_name":

        name = clean_name(
            value
        )

        if name:
            child[
                "name"
            ] = name

    elif field == "child_age":

        age = extract_single_age_if_current_field(
            value
        )

        if age:
            child[
                "age"
            ] = age

    elif field == "main_concern":

        if normalized in {
            "hamisi",
            "her biri",
        }:

            value = (
                "özgüvən, məqsəd və gələcək, "
                "məsuliyyət və intizam, ünsiyyət"
            )

        child[
            "main_concern"
        ] = value

        if any(
            x in normalized
            for x in [
                "fikirli",
                "ozune qapan",
                "danismir",
            ]
        ):

            child[
                "needs_concern_followup"
            ] = True

    elif field == "concern_duration":

        child[
            "concern_duration"
        ] = value

    elif field == "concern_onset":

        child[
            "concern_onset"
        ] = value

    elif field == "phone":

        phone = normalize_phone(
            value
        )

        if phone:
            lead[
                "phone"
            ] = phone

    elif field == "preferred_call_time":

        if normalized not in {
            "sabah",
            "bugun",
            "bu gun",
        }:

            lead[
                "preferred_call_time"
            ] = value

    sync_flat_fields(
        lead
    )


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
            f"\n\n{call_time} sizinlə əlaqə saxlanılması "
            "üçün müraciətinizi qeyd etdim."
        )

    result += (
        "\n\nİlkin zəng zamanı övladınızın "
        "iştirakı vacib deyil."
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

    lead[
        "_last_faq_score"
    ] = None

    # ---------------------------------------------
    # 1. Analyze complete message
    # ---------------------------------------------

    analysis = analyze_message(
        user_text=user_text,
        lead=lead,
        history=history,
    )

    print(
        "ANALYSIS DEBUG:",
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

    # ---------------------------------------------
    # 2. Merge facts + corrections
    # ---------------------------------------------

    correction_confirmations = merge_analysis(
        lead=lead,
        analysis=analysis,
        user_text=user_text,
        current_field=field_before,
    )

    # ---------------------------------------------
    # 3. Safety
    # ---------------------------------------------

    if analysis.get(
        "intent"
    ) == "safety_risk":

        lead[
            "status"
        ] = "ESCALATED"

        return (
            "Bu vəziyyət peşəkar və təcili diqqət tələb edə bilər. "
            "Junior Coaching tibbi və ya psixoloji təcili yardımı "
            "əvəz etmir. Müraciətinizi məsul əməkdaşa yönləndirirəm."
        )

    # ---------------------------------------------
    # 4. Human agent
    # ---------------------------------------------

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

    # ---------------------------------------------
    # 5. Permission
    # ---------------------------------------------

    if (
        analysis.get(
            "intent"
        ) == "permission_question"
        or is_permission_to_ask(
            user_text
        )
    ):

        return (
            "Əlbəttə, buyurun 😊"
        )

    # ---------------------------------------------
    # 6. STATE QUESTION FIRST
    # ---------------------------------------------

    state_answer = answer_state_question(
        user_text,
        lead,
    )

    if state_answer:

        return state_answer

    # ---------------------------------------------
    # 7. Pure greeting
    # ---------------------------------------------

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

            return (
                "Salam 😊\n\n"
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

    # ---------------------------------------------
    # 8. Questions
    # ---------------------------------------------

    questions = analysis.get(
        "questions",
        []
    )

    # Deterministic recovery if LLM misses a direct question
    if (
        not questions
        and "?" in user_text
    ):

        questions = [{
            "text": user_text,
            "topic": (
                deterministic_question_topic(
                    user_text
                )
                or "other"
            ),
        }]

    question_answer = ""

    if questions:

        question_answer = answer_all_questions(
            questions,
            lead,
        )

    # ---------------------------------------------
    # 9. Correction confirmation
    # ---------------------------------------------

    correction_text = ""

    if correction_confirmations:

        # remove duplicate confirmation strings
        unique_confirmations = list(
            dict.fromkeys(
                correction_confirmations
            )
        )

        correction_text = (
            "Düzəltdim ✅ "
            + " ".join(
                unique_confirmations
            )
        )

    # ---------------------------------------------
    # 10. If we have user questions, answer first
    # ---------------------------------------------

    if question_answer:

        parts = []

        if correction_text:
            parts.append(
                correction_text
            )

        parts.append(
            question_answer
        )

        # No lead yet = soft bridge
        if not has_any_lead_info(
            lead
        ):

            parts.append(
                "Başqa sualınız varsa, buyurun 😊 "
                "Müraciət etmək istəyirsinizsə, "
                "sizə necə müraciət edə bilərəm?"
            )

            return "\n\n".join(
                parts
            )

        # Lead started: maximum ONE next question
        next_question = get_next_question(
            lead
        )

        if next_question:

            parts.append(
                next_question
            )

        return "\n\n".join(
            parts
        )

    # ---------------------------------------------
    # 11. Correction-only turn
    # ---------------------------------------------

    if correction_text:

        next_question = get_next_question(
            lead
        )

        if next_question:

            return (
                correction_text
                + "\n\n"
                + next_question
            )

        return correction_text

    # ---------------------------------------------
    # 12. Fallback save if extraction did not advance
    # ---------------------------------------------

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

    # ---------------------------------------------
    # 13. Complete?
    # ---------------------------------------------

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
                main_concern TEXT,

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

        # ---------- leads migration ----------

        existing = {
            row[1]
            for row in conn.execute(
                "PRAGMA table_info(leads)"
            ).fetchall()
        }

        required = {
            "parent_title": "TEXT",
            "needs_concern_followup":
                "INTEGER DEFAULT 0",
            "concern_duration": "TEXT",
            "concern_onset": "TEXT",
            "children_json": "TEXT",
        }

        for column, dtype in required.items():

            if column not in existing:

                conn.execute(
                    f"""
                    ALTER TABLE leads
                    ADD COLUMN {column} {dtype}
                    """
                )

        # ---------- log migration ----------

        existing_logs = {
            row[1]
            for row in conn.execute(
                "PRAGMA table_info(conversation_logs)"
            ).fetchall()
        }

        log_required = {
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

        for column, dtype in log_required.items():

            if column not in existing_logs:

                conn.execute(
                    f"""
                    ALTER TABLE conversation_logs
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

            return dict(
                row
            )

    return None


# =========================================================
# 27. SAVE LEAD + CHILDREN
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

        # Save each child separately
        for index, child in enumerate(
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
                    main_concern,

                    needs_concern_followup,
                    concern_duration,
                    concern_onset,

                    created_at
                )

                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
                """,
                (
                    lead_id,
                    index,

                    child.get(
                        "name"
                    ),
                    child.get(
                        "age"
                    ),
                    child.get(
                        "main_concern"
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
# 28. SAVE CONVERSATION LOG
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