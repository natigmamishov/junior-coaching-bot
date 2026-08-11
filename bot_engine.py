"""
Junior Coaching — Bot Engine V2

Əsas xüsusiyyətlər:
- OpenAI intent classification
- Interruptible conversation flow
- FAQ retrieval: word + character TF-IDF
- Azərbaycan dilində yazı normalizasiyası
- Ad extraction
- Multiple-child handling
- Meta questions
- Flexible user questions during form filling
- SQLite leads
- Conversation logs
"""

import os
import re
import json
import sqlite3

from datetime import datetime
from typing import List, Tuple, Optional
from zoneinfo import ZoneInfo

import numpy as np
import httpx

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
        ".env"
    )
)

_api_key = os.getenv(
    "OPENAI_API_KEY"
)

client: Optional[OpenAI] = None


if _api_key:

    _http_client = httpx.Client(
        verify=False,
        timeout=60,
    )

    client = OpenAI(
        api_key=_api_key,
        http_client=_http_client,
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


def normalize_text(
    text: str,
) -> str:

    text = str(
        text
    ).strip().lower()

    text = re.sub(
        r"\s+",
        " ",
        text,
    )

    return text


def normalize_for_search(
    text: str,
) -> str:
    """
    FAQ retrieval üçün daha tolerant normalizasiya.

    Məs:
    qiymət -> qiymet
    qoşulmaq -> qosulmaq
    uşağ -> usag
    """

    text = normalize_text(
        text
    )

    text = text.translate(
        AZ_TRANSLATION
    )

    text = re.sub(
        r"[^\w\s?]",
        " ",
        text,
    )

    text = re.sub(
        r"\s+",
        " ",
        text,
    ).strip()

    return text


# =========================================================
# 3. BASIC HELPERS
# =========================================================

def is_greeting(
    text: str,
) -> bool:

    normalized = normalize_for_search(
        text
    )

    normalized = normalized.strip(
        "!.,? "
    )

    greetings = {
        "salam",
        "salamlar",
        "slm",
        "hi",
        "hello",
        "hey",
        "salam necesiz",
        "salam necesiniz",
        "salam aleykum",
        "salamun aleykum",
        "aleykum salam",
    }

    # "salam proqramla maraqlaniram" da greeting olsun
    if normalized.startswith(
        "salam "
    ):
        return True

    return normalized in greetings


def normalize_phone(
    text: str,
):

    digits = re.sub(
        r"\D",
        "",
        text,
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


def extract_age_candidates(
    text: str,
) -> List[int]:

    numbers = re.findall(
        r"\b\d{1,2}\b",
        text,
    )

    result = []

    for number in numbers:

        value = int(
            number
        )

        if (
            1 <= value <= 99
            and value not in result
        ):
            result.append(
                value
            )

    return result


def extract_age(
    text: str,
):

    ages = extract_age_candidates(
        text
    )

    if len(ages) == 1:
        return ages[0]

    return None


# =========================================================
# 4. NAME EXTRACTION
# =========================================================

def clean_name_candidate(
    name: str,
) -> Optional[str]:

    if not name:
        return None

    name = name.strip()

    name = re.sub(
        r"[.,!?]+$",
        "",
        name,
    )

    # Birinci sözü götürürük.
    # "İsmayıl Məmmədov" kimi hallarda isə tam adı saxlamaq olar.
    words = name.split()

    stop_words = {
        "salam",
        "salamlar",
        "men",
        "mən",
        "adim",
        "adım",
        "adi",
        "adı",
        "ismim",
        "mənim",
        "menim",
        "usaqin",
        "uşağın",
        "ovladimin",
        "övladımın",
        "ovladimin",
        "övladın",
        "usaq",
        "uşaq",
    }

    words = [
        word
        for word in words
        if normalize_for_search(word)
        not in {
            normalize_for_search(x)
            for x in stop_words
        }
    ]

    if not words:
        return None

    candidate = " ".join(
        words[:2]
    )

    name_pattern = (
        r"[A-Za-zƏəÖöÜüĞğÇçŞşİı\- ]+"
    )

    if not re.fullmatch(
        name_pattern,
        candidate,
    ):
        return None

    return candidate.title()


def extract_person_name(
    text: str,
) -> Optional[str]:
    """
    Məsələn:

    "salam adim ismayildir"
    -> Ismayil

    "mənim adım Aygündür"
    -> Aygün

    "adı eli"
    -> Eli
    """

    original = text.strip()

    normalized = normalize_for_search(
        original
    )

    patterns = [
        r"\bad[iı]m\s+([a-zA-ZƏəÖöÜüĞğÇçŞşİı\-]+)",
        r"\bismim\s+([a-zA-ZƏəÖöÜüĞğÇçŞşİı\-]+)",
        r"\bmenim\s+ad[iı]m\s+([a-zA-ZƏəÖöÜüĞğÇçŞşİı\-]+)",
        r"\badi\s+([a-zA-ZƏəÖöÜüĞğÇçŞşİı\-]+)",
    ]

    # Orijinal text-də regex
    raw_patterns = [
        r"(?i)\bad[ıi]m\s+([A-Za-zƏəÖöÜüĞğÇçŞşİı\-]+)",
        r"(?i)\bismim\s+([A-Za-zƏəÖöÜüĞğÇçŞşİı\-]+)",
        r"(?i)\bmənim\s+ad[ıi]m\s+([A-Za-zƏəÖöÜüĞğÇçŞşİı\-]+)",
        r"(?i)\bmenim\s+ad[ıi]m\s+([A-Za-zƏəÖöÜüĞğÇçŞşİı\-]+)",
        r"(?i)\bad[ıi]\s+([A-Za-zƏəÖöÜüĞğÇçŞşİı\-]+)",
    ]

    for pattern in raw_patterns:

        match = re.search(
            pattern,
            original,
        )

        if match:

            candidate = match.group(
                1
            )

            # "İsmayıldır" -> "İsmayıl"
            endings = [
                "dır",
                "dir",
                "dur",
                "dür",
                "di",
                "di̇r",
                "dir.",
            ]

            lowered = candidate.lower()

            for ending in endings:

                if (
                    lowered.endswith(ending)
                    and len(candidate) > len(ending) + 2
                ):

                    candidate = (
                        candidate[:-len(ending)]
                    )

                    break

            return candidate.title()

    # Sadə ad cavabı
    if (
        len(original.split()) <= 2
        and not is_greeting(original)
    ):

        return clean_name_candidate(
            original
        )

    return None


# =========================================================
# 5. AZERBAIJANI NAME SUFFIXES
# =========================================================

def get_last_vowel(
    word: str,
) -> Optional[str]:

    vowels = (
        "aıoueəiöü"
    )

    for char in reversed(
        word.lower()
    ):

        if char in vowels:
            return char

    return None


def get_four_way_suffix(
    word: str,
) -> str:

    vowel = get_last_vowel(
        word
    )

    if vowel in [
        "a",
        "ı",
    ]:
        return "ın"

    if vowel in [
        "e",
        "ə",
        "i",
    ]:
        return "in"

    if vowel in [
        "o",
        "u",
    ]:
        return "un"

    if vowel in [
        "ö",
        "ü",
    ]:
        return "ün"

    return "ın"


def get_two_way_suffix(
    word: str,
) -> str:

    vowel = get_last_vowel(
        word
    )

    if vowel in [
        "a",
        "ı",
        "o",
        "u",
    ]:
        return "a"

    return "ə"


def child_genitive(
    name: str,
) -> str:

    if not name:
        return ""

    name = name.strip()

    suffix = get_four_way_suffix(
        name
    )

    vowels = (
        "aıoueəiöü"
    )

    if name[-1].lower() in vowels:

        return (
            name
            + "n"
            + suffix
        )

    return (
        name
        + suffix
    )


def child_dative(
    name: str,
) -> str:

    if not name:
        return ""

    name = name.strip()

    suffix = get_two_way_suffix(
        name
    )

    vowels = (
        "aıoueəiöü"
    )

    if name[-1].lower() in vowels:

        return (
            name
            + "y"
            + suffix
        )

    return (
        name
        + suffix
    )


# =========================================================
# 6. STRONG INTERRUPT DETECTORS
# =========================================================

def is_bot_meta_question(
    text: str,
) -> bool:

    normalized = normalize_for_search(
        text
    )

    patterns = [
        "siz botsuz",
        "sen botsan",
        "siz botsan",
        "bot musunuz",
        "botmusuz",
        "kimle danisiram",
        "men kimle danisiram",
        "kiminle danisiram",
        "siz kimsiniz",
        "sen kimsen",
        "kim cavab verir",
        "insansiniz",
        "insansan",
    ]

    return any(
        pattern in normalized
        for pattern in patterns
    )


def is_child_presence_question(
    text: str,
) -> bool:

    normalized = normalize_for_search(
        text
    )

    child_words = [
        "usaq",
        "ovlad",
        "qizim",
        "oglum",
    ]

    presence_words = [
        "yanimda",
        "yaninda",
        "olmalidir",
        "olmasi vacib",
        "olmalidi",
        "gelmelidir",
        "gelmelidi",
    ]

    return (
        any(
            word in normalized
            for word in child_words
        )
        and any(
            word in normalized
            for word in presence_words
        )
    )


def is_multiple_children_message(
    text: str,
) -> bool:

    normalized = normalize_for_search(
        text
    )

    patterns = [
        "iki usaq",
        "2 usaq",
        "iki ovlad",
        "2 ovlad",
        "mende iki usaq",
        "bizde iki usaq",
        "mende 2 usaq",
        "bizde 2 usaq",
    ]

    return any(
        pattern in normalized
        for pattern in patterns
    )


def is_pause_or_goodbye(
    text: str,
) -> bool:

    normalized = normalize_for_search(
        text
    )

    patterns = [
        "sonra elaqe saxlayariq",
        "sonra danisariq",
        "sonra yazaram",
        "sonra yazariq",
        "indi uygun deyil",
        "indi uygun deyiləm",
        "sag olun",
        "sagolun",
        "tesekkur sonra",
        "helelik",
    ]

    return any(
        pattern in normalized
        for pattern in patterns
    )


def is_not_available_today(
    text: str,
) -> bool:

    normalized = normalize_for_search(
        text
    )

    patterns = [
        "bugun danisa bilmeyeceyem",
        "bugun danisa bilmerem",
        "bu gun danisa bilmeyeceyem",
        "bu gun uygun deyil",
        "bugun uygun deyil",
        "bugun vaxtim yoxdur",
        "bu gun vaxtim yoxdur",
    ]

    return any(
        pattern in normalized
        for pattern in patterns
    )


def is_call_timing_question(
    text: str,
) -> bool:

    normalized = normalize_for_search(
        text
    )

    timing_words = [
        "sabah",
        "bugun",
        "bu gun",
        "hefteson",
        "seher",
        "axsam",
        "nahardan sonra",
    ]

    call_words = [
        "elaqe",
        "zeng",
        "danismaq",
        "danisa",
        "mumkundur",
        "olar",
    ]

    return (
        "?" in text
        or "mumkundur" in normalized
        or "olar" in normalized
    ) and (
        any(
            word in normalized
            for word in timing_words
        )
        and any(
            word in normalized
            for word in call_words
        )
    )


# =========================================================
# 7. FAQ DATASET
# =========================================================

def _build_faq_index():

    if not os.path.exists(
        DATASET_PATH
    ):
        raise FileNotFoundError(
            f"FAQ faylı tapılmadı: {DATASET_PATH}"
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

    pairs: List[
        Tuple[str, str]
    ] = []

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
            "FAQ faylından sual-cavab tapılmadı."
        )

    questions = [
        question
        for question, _ in pairs
    ]

    answers = [
        answer
        for _, answer in pairs
    ]

    normalized_questions = [
        normalize_for_search(
            question
        )
        for question in questions
    ]

    # Word + character TF-IDF
    vectorizer = FeatureUnion([
        (
            "word",
            TfidfVectorizer(
                analyzer="word",
                ngram_range=(1, 2),
                max_features=50000,
                sublinear_tf=True,
            )
        ),
        (
            "char",
            TfidfVectorizer(
                analyzer="char_wb",
                ngram_range=(3, 5),
                max_features=70000,
                sublinear_tf=True,
            )
        ),
    ])

    matrix = vectorizer.fit_transform(
        normalized_questions
    )

    return (
        questions,
        answers,
        normalized_questions,
        vectorizer,
        matrix,
    )


(
    questions,
    answers,
    normalized_questions,
    vectorizer,
    Q,
) = _build_faq_index()


def expand_faq_query(
    text: str,
) -> str:
    """
    FAQ query-yə yaxın sinonim sözlər əlavə edir.
    """

    normalized = normalize_for_search(
        text
    )

    additions = []

    if any(
        word in normalized
        for word in [
            "qiymet",
            "ne qederdir",
            "odenis",
            "pul",
            "cost",
        ]
    ):
        additions.extend([
            "qiymet",
            "odenis",
            "proqram qiymeti",
        ])

    if any(
        word in normalized
        for word in [
            "yasdan",
            "yas qrupu",
            "nece yas",
            "qosulmaq",
            "qebul",
        ]
    ):
        additions.extend([
            "yas qrupu",
            "12 18",
            "nece yas",
        ])

    if any(
        word in normalized
        for word in [
            "ne zaman baslayir",
            "ne vaxt baslayir",
            "baslama",
            "start",
        ]
    ):
        additions.extend([
            "proqram ne vaxt baslayir",
            "baslama tarixi",
        ])

    if any(
        word in normalized
        for word in [
            "endirim",
            "iki usaq",
            "2 usaq",
            "iki ovlad",
        ]
    ):
        additions.extend([
            "endirim",
            "iki usaq",
            "kampaniya",
        ])

    if additions:

        return (
            normalized
            + " "
            + " ".join(
                additions
            )
        )

    return normalized


def retrieve_similar(
    user_query: str,
    k: int = 4,
    min_score: float = 0.16,
) -> List[Tuple[str, str, float]]:

    query = expand_faq_query(
        user_query
    )

    user_vector = vectorizer.transform(
        [query]
    )

    similarities = cosine_similarity(
        user_vector,
        Q,
    ).ravel()

    top_indices = np.argsort(
        -similarities
    )[:k]

    results = []

    for index in top_indices:

        score = float(
            similarities[index]
        )

        if score >= min_score:

            results.append(
                (
                    questions[index],
                    answers[index],
                    score,
                )
            )

    return results


def get_best_faq_hit(
    user_text: str,
    min_score: float = 0.20,
):

    hits = retrieve_similar(
        user_query=user_text,
        k=1,
        min_score=min_score,
    )

    return hits[0] if hits else None


def answer_faq_question(
    user_text: str,
    min_score: float = 0.20,
):

    hit = get_best_faq_hit(
        user_text=user_text,
        min_score=min_score,
    )

    if hit is None:
        return None

    faq_question, faq_answer, score = hit

    return {
        "question": faq_question,
        "answer": faq_answer,
        "score": score,
    }


# =========================================================
# 8. FAQ / QUESTION DETECTION
# =========================================================

def looks_like_question(
    text: str,
) -> bool:

    normalized = normalize_for_search(
        text
    )

    question_patterns = [
        "nece",
        "ne qeder",
        "ne zaman",
        "ne vaxt",
        "harada",
        "kim",
        "hansi",
        "mumkundur",
        "olar",
        "varmi",
        "var",
        "isteyirem",
        "oyrenmek isteyirem",
        "melumat",
    ]

    return (
        "?" in text
        or any(
            pattern in normalized
            for pattern in question_patterns
        )
    )


def is_probable_faq_question(
    text: str,
) -> bool:

    normalized = normalize_for_search(
        text
    )

    faq_topics = [
        "proqram",
        "qiymet",
        "odenis",
        "endirim",
        "yas",
        "qosul",
        "baslay",
        "ne zaman",
        "ne vaxt",
        "gorus",
        "qrup",
        "ferdi",
        "online",
        "onlayn",
        "mekan",
        "harada",
        "telimci",
        "ismayil",
        "sertifikat",
        "sinaq",
        "kampaniya",
        "psixoloq",
        "valideyn",
        "tanisliq",
        "zeng",
    ]

    return (
        looks_like_question(
            text
        )
        and any(
            topic in normalized
            for topic in faq_topics
        )
    )


# =========================================================
# 9. LLM CLASSIFIER
# =========================================================

def classify_message_with_llm(
    user_text: str,
    current_field: Optional[str],
    model: str = "gpt-4o-mini",
) -> dict:

    if client is None:

        raise RuntimeError(
            "OpenAI client yoxdur."
        )

    system_message = """
Sən Junior Coaching üçün mesaj router modulusan.

Intent-lər:

1. greeting
2. faq_question
3. field_answer
4. registration_request
5. human_agent_request
6. complaint
7. safety_risk
8. meta_question
9. pause_request
10. unrelated

ÇOX VACİB QAYDA:

İstifadəçi hansı field mərhələsində olursa olsun,
əgər ayrıca sual verirsə həmin mesajı field_answer kimi məcbur etmə.

Məsələn:

current_field=phone
"user: uşaq yanımda olmalıdır?"
=> faq_question

current_field=phone
"user: sabah əlaqə saxlamaq mümkündür?"
=> faq_question

current_field=phone
"user: siz botsuz?"
=> meta_question

current_field=phone
"user: sonra əlaqə saxlayarıq sağ olun"
=> pause_request

current_field=phone
"user: 0501234567"
=> field_answer

current_field=child_age
"user: 14"
=> field_answer

current_field=child_age
"user: məndə iki uşaq var"
=> field_answer

current_field=main_concern
"user: dərsləri zəifdir"
=> field_answer

current_field=main_concern
"user: qiyməti nə qədərdir?"
=> faq_question

Meta suallar:
- siz botsuz?
- kimlə danışıram?
- siz kimsiniz?

pause_request:
- sonra əlaqə saxlayarıq
- sonra yazaram
- indi uyğun deyil
- sağ olun, sonra danışarıq

Yalnız JSON qaytar.
"""

    response = client.chat.completions.create(
        model=model,
        messages=[
            {
                "role": "system",
                "content": system_message,
            },
            {
                "role": "user",
                "content": (
                    f"current_field={current_field}\n"
                    f"user_message={user_text}"
                ),
            },
        ],
        temperature=0,
        response_format={
            "type": "json_schema",
            "json_schema": {
                "name": "junior_intent",
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
                                "registration_request",
                                "human_agent_request",
                                "complaint",
                                "safety_risk",
                                "meta_question",
                                "pause_request",
                                "unrelated",
                            ],
                        },
                        "is_question": {
                            "type": "boolean"
                        },
                        "should_escalate": {
                            "type": "boolean"
                        },
                        "confidence": {
                            "type": "number"
                        },
                    },
                    "required": [
                        "intent",
                        "is_question",
                        "should_escalate",
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


def safe_classify_message(
    user_text: str,
    current_field: Optional[str],
) -> dict:

    # Əvvəl deterministik güclü qaydalar
    if is_bot_meta_question(
        user_text
    ):

        return {
            "intent": "meta_question",
            "is_question": True,
            "should_escalate": False,
            "confidence": 1.0,
        }

    if is_pause_or_goodbye(
        user_text
    ):

        return {
            "intent": "pause_request",
            "is_question": False,
            "should_escalate": False,
            "confidence": 1.0,
        }

    if (
        is_child_presence_question(
            user_text
        )
        or is_call_timing_question(
            user_text
        )
        or is_probable_faq_question(
            user_text
        )
    ):

        return {
            "intent": "faq_question",
            "is_question": True,
            "should_escalate": False,
            "confidence": 0.99,
        }

    try:

        return classify_message_with_llm(
            user_text=user_text,
            current_field=current_field,
        )

    except Exception as exc:

        print(
            "LLM CLASSIFIER ERROR:",
            exc,
        )

        if is_greeting(
            user_text
        ):

            intent = "greeting"

        elif is_probable_faq_question(
            user_text
        ):

            intent = "faq_question"

        else:

            intent = "field_answer"

        return {
            "intent": intent,
            "is_question": (
                intent == "faq_question"
            ),
            "should_escalate": False,
            "confidence": 0.0,
        }


# =========================================================
# 10. PARENT TITLE
# =========================================================

def infer_parent_title_with_llm(
    parent_name: str,
    model: str = "gpt-4o-mini",
) -> str:

    if (
        not parent_name
        or client is None
    ):
        return ""

    try:

        response = client.chat.completions.create(
            model=model,
            messages=[
                {
                    "role": "system",
                    "content": """
Azərbaycan adına əsasən müraciət formasını seç:

xanım
bəy
neutral

Əmin deyilsənsə neutral seç.
"""
                },
                {
                    "role": "user",
                    "content": parent_name,
                },
            ],
            temperature=0,
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
            "title"
        )

        if title in [
            "xanım",
            "bəy",
        ]:
            return title

        return ""

    except Exception as exc:

        print(
            "TITLE ERROR:",
            exc,
        )

        return ""


# =========================================================
# 11. LEAD STRUCTURE
# =========================================================

def create_empty_lead(
    source: str = "CLI",
):

    return {
        "parent_name": None,
        "parent_title": None,

        "child_name": None,
        "child_age": None,

        "main_concern": None,

        "needs_concern_followup": False,
        "concern_duration": None,
        "concern_onset": None,

        "phone": None,
        "preferred_call_time": None,

        "source": source,
        "status": "NEW",

        "_last_intent": None,
        "_last_confidence": None,
        "_last_faq_score": None,
    }


FIELD_QUESTIONS = {

    "parent_name":
        "Sizə necə müraciət edə bilərəm?",

    "child_name":
        "Övladınızın adını öyrənə bilərəm?",

    "child_age":
        "Övladınızın neçə yaşı var?",

    "main_concern":
        (
            "Övladınızla bağlı hazırda sizi ən çox "
            "narahat edən məsələ nədir?"
        ),

    "concern_duration":
        (
            "Bu hal nə qədər müddətdir davam edir?"
        ),

    "concern_onset":
        (
            "Sizcə hansısa hadisədən sonra belə olub, "
            "yoxsa tədricən?"
        ),

    "phone":
        (
            "Sizinlə əlaqə saxlaya bilməyimiz üçün "
            "telefon nömrənizi qeyd edin, zəhmət olmasa."
        ),

    "preferred_call_time":
        (
            "Zəng üçün sizə hansı gün və saat aralığı "
            "daha uyğun olar?"
        ),
}


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
# 12. PERSONALIZED QUESTIONS
# =========================================================

def get_personalized_question(
    field: str,
    lead: dict,
) -> str:

    parent_display = (
        get_parent_display_name(
            lead
        )
    )

    child_name = lead.get(
        "child_name"
    )


    if field == "parent_name":

        return (
            "Sizə necə müraciət edə bilərəm?"
        )


    if field == "child_name":

        if parent_display:

            return (
                f"Məmnun oldum, {parent_display}. "
                "Övladınızın adını öyrənə bilərəm?"
            )

        return (
            "Övladınızın adını öyrənə bilərəm?"
        )


    if field == "child_age":

        if child_name:

            return (
                f"{child_genitive(child_name)} "
                "neçə yaşı var?"
            )

        return (
            "Övladınızın neçə yaşı var?"
        )


    if field == "main_concern":

        if child_name:

            return (
                "Aydındır. "
                f"{child_name} ilə bağlı hazırda sizi "
                "ən çox narahat edən məsələ nədir?"
            )

        return (
            FIELD_QUESTIONS[
                "main_concern"
            ]
        )


    if field == "concern_duration":

        return (
            "Başa düşürəm. "
            "Bu hal nə qədər müddətdir davam edir?"
        )


    if field == "concern_onset":

        return (
            "İcazənizlə bir sual da verim. "
            "Sizcə hansısa hadisədən sonra belə olub, "
            "yoxsa tədricən?"
        )


    if field == "phone":

        if child_name:

            return (
                "Başa düşürəm. "
                f"Bu istiqamətdə {child_dative(child_name)} "
                "dəstək olmaq mümkündür. "
                "Sizinlə əlaqə saxlaya bilməyimiz üçün "
                "telefon nömrənizi qeyd edin, zəhmət olmasa."
            )

        return (
            FIELD_QUESTIONS[
                "phone"
            ]
        )


    if field == "preferred_call_time":

        return (
            "Təşəkkür edirəm. "
            "Zəng üçün sizə hansı gün və saat aralığı "
            "daha uyğun olar?"
        )


    return FIELD_QUESTIONS.get(
        field,
        "",
    )


# =========================================================
# 13. NEXT FIELD
# =========================================================

def get_next_missing_field(
    lead: dict,
):

    fields = [
        "parent_name",
        "child_name",
        "child_age",
        "main_concern",
    ]

    for field in fields:

        if not lead.get(
            field
        ):

            return field


    if lead.get(
        "needs_concern_followup",
        False,
    ):

        if not lead.get(
            "concern_duration"
        ):

            return (
                "concern_duration"
            )

        if not lead.get(
            "concern_onset"
        ):

            return (
                "concern_onset"
            )


    final_fields = [
        "phone",
        "preferred_call_time",
    ]

    for field in final_fields:

        if not lead.get(
            field
        ):

            return field

    return None


# =========================================================
# 14. FIELD VALIDATION
# =========================================================

def save_user_answer(
    lead: dict,
    field: str,
    user_text: str,
):

    user_text = user_text.strip()

    if not user_text:

        return (
            False,
            "Zəhmət olmasa cavabınızı qeyd edin."
        )

    normalized = normalize_for_search(
        user_text
    )


    # =====================================================
    # PARENT NAME
    # =====================================================

    if field == "parent_name":

        name = extract_person_name(
            user_text
        )

        if not name:

            return (
                False,
                "Adınızı tam anlaya bilmədim. "
                "Məsələn: Aygün və ya "
                "\"Adım İsmayıldır\" şəklində yaza bilərsiniz."
            )

        lead[
            "parent_name"
        ] = name

        lead[
            "parent_title"
        ] = infer_parent_title_with_llm(
            name
        )

        return (
            True,
            None
        )


    # =====================================================
    # CHILD NAME
    # =====================================================

    if field == "child_name":

        name = extract_person_name(
            user_text
        )

        if not name:

            return (
                False,
                "Övladınızın adını tam anlaya bilmədim. "
                "Məsələn: Leyla və ya \"adı Eli\"."
            )

        lead[
            "child_name"
        ] = name

        return (
            True,
            None
        )


    # =====================================================
    # CHILD AGE
    # =====================================================

    if field == "child_age":

        if is_multiple_children_message(
            user_text
        ):

            child_name = lead.get(
                "child_name"
            )

            if child_name:

                return (
                    False,
                    "Başa düşürəm, iki övladınız var. "
                    "Hazırda müraciəti bir övlad üzrə "
                    "davam etdiririk. "
                    f"Əvvəlcə {child_genitive(child_name)} "
                    "yaşını qeyd edə bilərsiniz?"
                )

            return (
                False,
                "Başa düşürəm, iki övladınız var. "
                "Əvvəlcə müraciəti bir övlad üzrə davam etdirək. "
                "Hazırda qeyd etdiyiniz övladın yaşını yaza bilərsiniz?"
            )


        ages = extract_age_candidates(
            user_text
        )

        if len(ages) > 1:

            child_name = lead.get(
                "child_name"
            )

            if child_name:

                return (
                    False,
                    f"İki yaş qeyd etdiniz: "
                    f"{', '.join(map(str, ages))}. "
                    f"{child_genitive(child_name)} "
                    "yaşı hansıdır?"
                )

            return (
                False,
                "Bir neçə yaş qeyd etdiniz. "
                "Hazırda müraciət etdiyiniz övladın "
                "yaşını tək rəqəmlə qeyd edin."
            )


        age = extract_age(
            user_text
        )

        if age is None:

            return (
                False,
                "Zəhmət olmasa övladınızın yaşını "
                "rəqəmlə qeyd edin. Məsələn: 14."
            )


        if not (
            12 <= age <= 18
        ):

            return (
                False,
                "Junior Coaching proqramı 12–18 yaşlı "
                "yeniyetmələr üçün nəzərdə tutulub. "
                "Övladınızın yaşını yenidən "
                "dəqiqləşdirə bilərsiniz?"
            )


        lead[
            "child_age"
        ] = age

        return (
            True,
            None
        )


    # =====================================================
    # MAIN CONCERN
    # =====================================================

    if field == "main_concern":

        followup_answers = {

            "fikirli",
            "fikirlidir",
            "fikirli olur",
            "fikirli gezir",
            "fikirli gorunur",

            "cox fikirlidir",

            "ozune qapanir",
            "ozune qapanib",

            "qapalidir",

            "cox sakitdir",

            "danismir",
        }


        vague_answers = {

            "problemi var",
            "problem var",

            "cetinlik cekir",

            "yaxsi deyil",

            "narahatdir",

            "bilmirem",

            "hec ne",
        }


        if normalized in followup_answers:

            lead[
                "main_concern"
            ] = user_text

            lead[
                "needs_concern_followup"
            ] = True

            lead[
                "concern_duration"
            ] = None

            lead[
                "concern_onset"
            ] = None

            return (
                True,
                None
            )


        if normalized in vague_answers:

            return (
                False,
                "Bir qədər dəqiqləşdirə bilərsiniz? "
                "Övladınızla bağlı sizi narahat edən "
                "əsas məsələni qısa şəkildə qeyd edin."
            )


        if len(
            user_text
        ) < 2:

            return (
                False,
                "Bir qədər dəqiqləşdirə bilərsiniz?"
            )


        lead[
            "main_concern"
        ] = user_text

        lead[
            "needs_concern_followup"
        ] = False

        return (
            True,
            None
        )


    # =====================================================
    # DURATION
    # =====================================================

    if field == "concern_duration":

        duration_keywords = [
            "gun",
            "hefte",
            "ay",
            "il",
            "coxdan",
            "bir nece",
            "texminen",
            "usaqliqdan",
        ]

        has_keyword = any(
            keyword in normalized
            for keyword in duration_keywords
        )

        has_number = bool(
            re.search(
                r"\d+",
                normalized,
            )
        )

        if (
            not has_keyword
            and not has_number
        ):

            return (
                False,
                "Təxmini müddəti qeyd edə bilərsiniz? "
                "Məsələn: 2 həftədir, 3 aydır."
            )

        lead[
            "concern_duration"
        ] = user_text

        return (
            True,
            None
        )


    # =====================================================
    # ONSET
    # =====================================================

    if field == "concern_onset":

        unknown = [
            "bilmirem",
            "xeberim yoxdur",
            "bilinmir",
        ]

        if any(
            phrase == normalized
            for phrase in unknown
        ):

            lead[
                "concern_onset"
            ] = user_text

            return (
                True,
                None
            )


        if len(
            user_text.split()
        ) < 1:

            return (
                False,
                "Bir qədər dəqiqləşdirə bilərsiniz?"
            )


        lead[
            "concern_onset"
        ] = user_text

        return (
            True,
            None
        )


    # =====================================================
    # PHONE
    # =====================================================

    if field == "phone":

        phone = normalize_phone(
            user_text
        )

        if phone is None:

            return (
                False,
                "Telefon nömrəsi düzgün görünmür. "
                "Məsələn: 050 123 45 67."
            )

        lead[
            "phone"
        ] = phone

        return (
            True,
            None
        )


    # =====================================================
    # CALL TIME
    # =====================================================

    if field == "preferred_call_time":

        vague = {
            "ferqi yoxdur",
            "her zaman",
            "istenilen vaxt",
            "ne vaxt olsa",
            "bilmirem",
        }

        if normalized in vague:

            return (
                False,
                "Zəhmət olmasa uyğun gün və vaxtı "
                "bir qədər dəqiqləşdirin. "
                "Məsələn: sabah 14:00–15:00."
            )


        time_keywords = [
            "bugun",
            "sabah",
            "birisi gun",
            "bazar ertesi",
            "cersenbe",
            "cume",
            "senbe",
            "bazar",
            "heftesonu",
            "seher",
            "gunorta",
            "nahardan sonra",
            "isden sonra",
            "axsam",
        ]


        has_keyword = any(
            keyword in normalized
            for keyword in time_keywords
        )


        has_numeric_time = bool(
            re.search(
                r"\b\d{1,2}(?::\d{2})?"
                r"\s*[-–]\s*"
                r"\d{1,2}(?::\d{2})?\b"
                r"|\b\d{1,2}:\d{2}\b",
                normalized,
            )
        )


        if (
            not has_keyword
            and not has_numeric_time
        ):

            return (
                False,
                "Zəhmət olmasa uyğun gün və saatı qeyd edin. "
                "Məsələn: sabah 14:00–15:00."
            )


        lead[
            "preferred_call_time"
        ] = user_text

        return (
            True,
            None
        )


    return (
        False,
        f"'{field}' üçün validation yoxdur."
    )


# =========================================================
# 15. INTERRUPT RESPONSES
# =========================================================

def answer_meta_question(
    lead: dict,
) -> str:

    current_field = get_next_missing_field(
        lead
    )

    response = (
        "Mən Junior Coaching proqramı üzrə "
        "virtual müraciət köməkçisiyəm 😊 "
        "Proqram haqqında suallarınızı cavablandıra "
        "və müraciətinizi qeydə ala bilirəm."
    )

    if current_field:

        response += (
            "\n\n"
            + get_personalized_question(
                current_field,
                lead,
            )
        )

    return response


def answer_child_presence(
    lead: dict,
) -> str:

    current_field = get_next_missing_field(
        lead
    )

    child_name = lead.get(
        "child_name"
    )

    if child_name:

        response = (
            "İlkin zəng zamanı "
            f"{child_genitive(child_name)} "
            "yanınızda olması vacib deyil. "
            "Daha sonra ehtiyac olarsa övladınızla "
            "ayrıca təxminən 5 dəqiqəlik video "
            "tanışlıq görüşü keçirilə bilər."
        )

    else:

        response = (
            "İlkin zəng zamanı övladınızın "
            "yanınızda olması vacib deyil. "
            "Daha sonra ehtiyac olarsa övladınızla "
            "ayrıca təxminən 5 dəqiqəlik video "
            "tanışlıq görüşü keçirilə bilər."
        )

    if current_field:

        response += (
            "\n\n"
            + get_personalized_question(
                current_field,
                lead,
            )
        )

    return response


def answer_call_timing_question(
    lead: dict,
) -> str:

    current_field = get_next_missing_field(
        lead
    )

    response = (
        "Bəli, mümkündür 😊 "
        "Sizə uyğun gün və saat nəzərə alınaraq "
        "əlaqə saxlanıla bilər."
    )

    if current_field == "phone":

        response += (
            "\n\nƏvvəlcə sizinlə əlaqə saxlaya bilməyimiz "
            "üçün telefon nömrənizi qeyd edin, zəhmət olmasa."
        )

    elif current_field:

        response += (
            "\n\n"
            + get_personalized_question(
                current_field,
                lead,
            )
        )

    return response


def answer_not_available_today(
    lead: dict,
) -> str:

    current_field = get_next_missing_field(
        lead
    )

    response = (
        "Əlbəttə, problem deyil 😊 "
        "Sizə uyğun başqa gün və saat üçün "
        "əlaqə yaradıla bilər."
    )

    if current_field == "phone":

        response += (
            "\n\nƏlaqə üçün telefon nömrənizi "
            "qeyd edə bilərsiniz?"
        )

    elif current_field:

        response += (
            "\n\n"
            + get_personalized_question(
                current_field,
                lead,
            )
        )

    return response


# =========================================================
# 16. MAIN AGENT
# =========================================================

def lead_agent_reply(
    user_text: str,
    lead: dict,
    faq_min_score: float = 0.20,
) -> str:

    user_text = user_text.strip()

    current_field = get_next_missing_field(
        lead
    )

    # hər turn-də reset
    lead[
        "_last_faq_score"
    ] = None


    # =====================================================
    # STRONG INTERRUPTS BEFORE CLASSIFIER
    # =====================================================

    if is_child_presence_question(
        user_text
    ):

        lead[
            "_last_intent"
        ] = "faq_question"

        lead[
            "_last_confidence"
        ] = 1.0

        return answer_child_presence(
            lead
        )


    if is_call_timing_question(
        user_text
    ):

        lead[
            "_last_intent"
        ] = "faq_question"

        lead[
            "_last_confidence"
        ] = 1.0

        return answer_call_timing_question(
            lead
        )


    if is_not_available_today(
        user_text
    ):

        lead[
            "_last_intent"
        ] = "pause_request"

        lead[
            "_last_confidence"
        ] = 1.0

        return answer_not_available_today(
            lead
        )


    if is_pause_or_goodbye(
        user_text
    ):

        lead[
            "_last_intent"
        ] = "pause_request"

        lead[
            "_last_confidence"
        ] = 1.0

        return (
            "Əlbəttə 😊 İstədiyiniz zaman "
            "yenidən yaza bilərsiniz. "
            "Təşəkkür edirik."
        )


    # =====================================================
    # CLASSIFICATION
    # =====================================================

    classification = safe_classify_message(
        user_text=user_text,
        current_field=current_field,
    )

    intent = classification[
        "intent"
    ]

    lead[
        "_last_intent"
    ] = intent

    lead[
        "_last_confidence"
    ] = classification.get(
        "confidence"
    )

    print(
        "INTENT DEBUG:",
        classification,
    )


    # =====================================================
    # META QUESTION
    # =====================================================

    if intent == "meta_question":

        return answer_meta_question(
            lead
        )


    # =====================================================
    # PAUSE
    # =====================================================

    if intent == "pause_request":

        return (
            "Əlbəttə 😊 İstədiyiniz zaman "
            "yenidən yaza bilərsiniz. "
            "Təşəkkür edirik."
        )


    # =====================================================
    # GREETING
    # =====================================================

    if intent == "greeting":

        if current_field == "parent_name":

            return (
                "Salam 😊 "
                "Sizə necə müraciət edə bilərəm?"
            )

        if current_field:

            return (
                "Salam 😊\n\n"
                + get_personalized_question(
                    current_field,
                    lead,
                )
            )

        return (
            "Salam 😊 Müraciətiniz artıq qeydə alınıb."
        )


    # =====================================================
    # SAFETY
    # =====================================================

    if intent == "safety_risk":

        lead[
            "status"
        ] = "ESCALATED"

        return (
            "Bu hal təcili peşəkar diqqət tələb edə bilər. "
            "Junior Coaching psixoloji və ya tibbi yardımı "
            "əvəz etmir. Övladınız hazırda təhlükədədirsə, "
            "onu tək qoymayın və dərhal uyğun təcili yardım "
            "və psixi sağlamlıq mütəxəssisi ilə əlaqə saxlayın. "
            "Müraciətiniz məsul əməkdaşa yönləndirilir."
        )


    # =====================================================
    # COMPLAINT
    # =====================================================

    if intent == "complaint":

        lead[
            "status"
        ] = "ESCALATED"

        return (
            "Narahatlığınızı başa düşürəm. "
            "Müraciətinizi məsul əməkdaşa yönləndirmək "
            "üçün qeydə aldım."
        )


    # =====================================================
    # HUMAN AGENT
    # =====================================================

    if intent == "human_agent_request":

        lead[
            "status"
        ] = "ESCALATED"

        return (
            "Əlbəttə. Müraciətinizi Junior Coaching "
            "komandasından məsul əməkdaşa yönləndirmək "
            "üçün qeydə aldım."
        )


    # =====================================================
    # FAQ
    # =====================================================

    if intent == "faq_question":

        # əvvəl xüsusi qaydalar
        if is_child_presence_question(
            user_text
        ):

            return answer_child_presence(
                lead
            )

        if is_call_timing_question(
            user_text
        ):

            return answer_call_timing_question(
                lead
            )


        faq_result = answer_faq_question(
            user_text=user_text,
            min_score=faq_min_score,
        )


        if faq_result is not None:

            faq_answer = faq_result[
                "answer"
            ]

            lead[
                "_last_faq_score"
            ] = faq_result.get(
                "score"
            )

            if current_field:

                return (
                    f"{faq_answer}\n\n"
                    + get_personalized_question(
                        current_field,
                        lead,
                    )
                )

            return faq_answer


        if current_field:

            return (
                "Bu sualla bağlı məlumat bazasında "
                "dəqiq cavab tapmadım. "
                "İstəsəniz bu sualı məsul əməkdaşa "
                "yönləndirə bilərik.\n\n"
                + get_personalized_question(
                    current_field,
                    lead,
                )
            )

        return (
            "Bu sualla bağlı məlumat bazasında "
            "dəqiq cavab tapmadım."
        )


    # =====================================================
    # REGISTRATION
    # =====================================================

    if intent == "registration_request":

        if current_field:

            return (
                "Əlbəttə, müraciət prosesinə "
                "davam edə bilərik.\n\n"
                + get_personalized_question(
                    current_field,
                    lead,
                )
            )


    # =====================================================
    # UNRELATED
    # =====================================================

    if intent == "unrelated":

        if current_field:

            return (
                "Mən Junior Coaching proqramı və "
                "müraciət prosesi ilə bağlı kömək edirəm.\n\n"
                + get_personalized_question(
                    current_field,
                    lead,
                )
            )

        return (
            "Mən Junior Coaching proqramı ilə bağlı "
            "suallara cavab verirəm."
        )


    # =====================================================
    # FIELD ANSWER
    # =====================================================

    if current_field is None:

        lead[
            "status"
        ] = "CALL_REQUESTED"

        return (
            "Məlumatlarınız artıq tamamlanıb."
        )


    success, error_message = save_user_answer(
        lead=lead,
        field=current_field,
        user_text=user_text,
    )


    if not success:

        return error_message


    next_field = get_next_missing_field(
        lead
    )


    # =====================================================
    # COMPLETED
    # =====================================================

    if next_field is None:

        lead[
            "status"
        ] = "CALL_REQUESTED"

        parent_display = (
            get_parent_display_name(
                lead
            )
        )

        child_name = lead.get(
            "child_name"
        )

        call_time = lead.get(
            "preferred_call_time"
        )


        final_message = (
            f"Əla, {parent_display}. "
            f"Sizinlə {call_time} əlaqə saxlanılması "
            "üçün qeyd etdim. ✅"
        )


        if child_name:

            final_message += (
                "\n\n"
                "İlkin zəng zamanı "
                f"{child_genitive(child_name)} "
                "yanınızda olması vacib deyil."
            )


        return final_message


    return get_personalized_question(
        next_field,
        lead,
    )


# =========================================================
# 17. SQLITE
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

                phone TEXT,

                preferred_call_time TEXT,

                status TEXT,

                source TEXT,

                created_at TEXT

            )
            """
        )


        # ---------------------------------------------
        # LEADS MIGRATION
        # ---------------------------------------------

        cursor = conn.execute(
            "PRAGMA table_info(leads)"
        )

        existing = {
            row[1]
            for row in cursor.fetchall()
        }


        required = {

            "parent_title":
                "TEXT",

            "needs_concern_followup":
                "INTEGER DEFAULT 0",

            "concern_duration":
                "TEXT",

            "concern_onset":
                "TEXT",
        }


        for (
            column,
            dtype,
        ) in required.items():

            if column not in existing:

                conn.execute(
                    f"""
                    ALTER TABLE leads
                    ADD COLUMN {column} {dtype}
                    """
                )


        # ---------------------------------------------
        # LOG MIGRATION
        # ---------------------------------------------

        cursor = conn.execute(
            "PRAGMA table_info(conversation_logs)"
        )

        existing_logs = {
            row[1]
            for row in cursor.fetchall()
        }


        required_logs = {

            "intent": "TEXT",

            "confidence": "REAL",

            "faq_score": "REAL",

            "parent_title": "TEXT",

            "child_age": "INTEGER",

            "main_concern": "TEXT",

            "preferred_call_time": "TEXT",

            "source": "TEXT",
        }


        for (
            column,
            dtype,
        ) in required_logs.items():

            if column not in existing_logs:

                conn.execute(
                    f"""
                    ALTER TABLE conversation_logs
                    ADD COLUMN {column} {dtype}
                    """
                )


        conn.commit()


# =========================================================
# 18. TIME
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
# 19. FIND LEAD
# =========================================================

def find_lead_by_phone(
    phone: str,
):

    if not phone:
        return None


    with sqlite3.connect(
        DB_PATH
    ) as conn:

        conn.row_factory = (
            sqlite3.Row
        )

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


        return (
            dict(row)
            if row
            else None
        )


# =========================================================
# 20. SAVE LEAD
# =========================================================

def save_lead_to_db(
    lead: dict,
) -> int:

    now = get_baku_time()


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

                phone,

                preferred_call_time,

                source,

                status,

                created_at,

                updated_at

            )

            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
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


        conn.commit()

        return cursor.lastrowid


# =========================================================
# 21. SAVE LOG
# =========================================================

def save_conversation_log(
    session_id: str,
    user_message: str,
    bot_response: str,
    current_field: Optional[str],
    lead: dict,
):

    now = get_baku_time()


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

                phone,

                preferred_call_time,

                status,

                source,

                created_at

            )

            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
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
# 22. GET DATA
# =========================================================

def get_all_leads():

    with sqlite3.connect(
        DB_PATH
    ) as conn:

        conn.row_factory = (
            sqlite3.Row
        )

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


def get_all_conversation_logs():

    with sqlite3.connect(
        DB_PATH
    ) as conn:

        conn.row_factory = (
            sqlite3.Row
        )

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
# DB INIT
# =========================================================

init_db()