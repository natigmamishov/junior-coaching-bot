"""
Junior Coaching — Bot Engine V2.1

Əsas funksiyalar:
- OpenAI intent routing
- Interruptible conversation flow
- FAQ retrieval: word + character TF-IDF
- Azərbaycan dilində tolerant normalization
- Ad extraction və honorific cleaning
- Multiple children handling
- Meta questions
- FAQ suallarının flow-un istənilən yerində cavablandırılması
- English general programme question support
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
        ".env",
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

    text = str(text).strip().lower()

    text = re.sub(
        r"\s+",
        " ",
        text,
    )

    return text


def normalize_for_search(text: str) -> str:

    text = normalize_text(
        text
    )

    text = text.translate(
        AZ_TRANSLATION
    )

    # +, vergül, nöqtə və s. təmizlənir
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
# 3. LANGUAGE HELPERS
# =========================================================

def looks_english(text: str) -> bool:

    normalized = normalize_for_search(
        text
    )

    english_words = [
        "could you",
        "can you",
        "tell me",
        "information",
        "programme",
        "program",
        "about",
        "price",
        "how much",
        "age",
        "child",
        "teenager",
    ]

    score = sum(
        phrase in normalized
        for phrase in english_words
    )

    return score >= 2


def is_english_program_info_question(
    text: str,
) -> bool:

    normalized = normalize_for_search(
        text
    )

    if not looks_english(text):
        return False

    programme_words = [
        "programme",
        "program",
        "junior coaching",
        "information",
        "tell me more",
    ]

    return any(
        word in normalized
        for word in programme_words
    )


# =========================================================
# 4. GREETING
# =========================================================

def is_greeting(text: str) -> bool:

    normalized = normalize_for_search(
        text
    ).strip()

    greetings = {
        "salam",
        "salamlar",
        "slm",
        "hello",
        "hi",
        "hey",
        "salam necesiz",
        "salam necesiniz",
        "salam aleykum",
        "salamun aleykum",
        "aleykum salam",
    }

    return normalized in greetings


def starts_with_greeting(text: str) -> bool:

    normalized = normalize_for_search(
        text
    )

    return (
        normalized == "salam"
        or normalized.startswith("salam ")
        or normalized.startswith("hello ")
        or normalized.startswith("hi ")
    )


# =========================================================
# 5. PHONE
# =========================================================

def normalize_phone(text: str):

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


# =========================================================
# 6. AGE
# =========================================================

def extract_age_candidates(
    text: str,
) -> List[int]:

    numbers = re.findall(
        r"\b\d{1,2}\b",
        text,
    )

    result = []

    for number in numbers:

        value = int(number)

        if (
            1 <= value <= 99
            and value not in result
        ):
            result.append(value)

    return result


def extract_age(text: str):

    ages = extract_age_candidates(
        text
    )

    if len(ages) == 1:
        return ages[0]

    return None


def detect_embedded_child_age(
    text: str,
) -> Optional[int]:

    """
    Məs:
    '11 yasli oglum ucun'
    -> 11
    """

    normalized = normalize_for_search(
        text
    )

    match = re.search(
        r"\b(\d{1,2})\s*yas",
        normalized,
    )

    if match:

        return int(
            match.group(1)
        )

    return None


# =========================================================
# 7. NAME HELPERS
# =========================================================

HONORIFICS = {
    "bey",
    "bəy",
    "xanim",
    "xanım",
    "müəllim",
    "muellim",
}


def strip_name_statement_suffix(
    word: str,
) -> str:

    """
    Elvindir -> Elvin
    Orxandir -> Orxan
    Ismayildir -> Ismayil

    Nadir -> Nadir, çünki 'Na' çox qısadır.
    """

    if not word:
        return word

    original = word.strip()

    lower_normalized = normalize_for_search(
        original
    )

    suffixes = [
        "dir",
        "dur",
        "dur",
    ]

    for suffix in suffixes:

        if lower_normalized.endswith(
            suffix
        ):

            base_length = (
                len(original)
                - len(suffix)
            )

            # Yanlış şəkildə Nadir -> Na olmasın
            if base_length >= 4:

                return original[
                    :base_length
                ]

    return original


def strip_honorifics(
    text: str,
) -> str:

    words = text.strip().split()

    cleaned = []

    for word in words:

        normalized = normalize_for_search(
            word
        )

        if normalized in {
            "bey",
            "xanim",
            "muellim",
        }:
            continue

        cleaned.append(
            word
        )

    return " ".join(
        cleaned
    ).strip()


def clean_extracted_name(
    name: str,
) -> Optional[str]:

    if not name:
        return None

    name = name.strip()

    name = strip_honorifics(
        name
    )

    name = re.sub(
        r"[.,!?+]+",
        "",
        name,
    ).strip()

    if not name:
        return None

    words = name.split()

    cleaned_words = []

    for word in words:

        word = strip_name_statement_suffix(
            word
        )

        if word:
            cleaned_words.append(
                word
            )

    if not cleaned_words:
        return None

    # Ad mərhələsində maksimum 2 söz saxlayaq
    cleaned_words = cleaned_words[:2]

    result = " ".join(
        cleaned_words
    ).strip()

    pattern = (
        r"[A-Za-zƏəÖöÜüĞğÇçŞşİı\- ]+"
    )

    if not re.fullmatch(
        pattern,
        result,
    ):
        return None

    return result.title()


def extract_person_name(
    text: str,
) -> Optional[str]:

    """
    Examples:

    salam adim ismayildir -> Ismayil
    mənim adım Aygündür -> Aygün
    İsmayıl bəy -> İsmayıl
    elvindir -> Elvin
    orxandir -> Orxan
    adi eli -> Eli
    """

    original = text.strip()

    # ---------------------------------------------
    # Explicit "adım ..." patterns
    # ---------------------------------------------

    patterns = [
        r"(?i)\bad[ıi]m\s+([A-Za-zƏəÖöÜüĞğÇçŞşİı\-]+)",
        r"(?i)\bmənim\s+ad[ıi]m\s+([A-Za-zƏəÖöÜüĞğÇçŞşİı\-]+)",
        r"(?i)\bmenim\s+ad[ıi]m\s+([A-Za-zƏəÖöÜüĞğÇçŞşİı\-]+)",
        r"(?i)\bismim\s+([A-Za-zƏəÖöÜüĞğÇçŞşİı\-]+)",
        r"(?i)\bad[ıi]\s+([A-Za-zƏəÖöÜüĞğÇçŞşİı\-]+)",
    ]

    for pattern in patterns:

        match = re.search(
            pattern,
            original,
        )

        if match:

            return clean_extracted_name(
                match.group(1)
            )

    # ---------------------------------------------
    # "Salam İsmayıl" tipli hallarda
    # salam sözünü təmizlə
    # ---------------------------------------------

    normalized = normalize_for_search(
        original
    )

    if normalized.startswith(
        "salam "
    ):

        without_greeting = re.sub(
            r"(?i)^salam[,\s]+",
            "",
            original,
        ).strip()

        # "salam adim..." yuxarıdakı pattern-də tutulmalı idi.
        if (
            len(without_greeting.split()) <= 2
            and "adim" not in normalize_for_search(
                without_greeting
            )
        ):

            return clean_extracted_name(
                without_greeting
            )

    # ---------------------------------------------
    # Sadə cavab
    # ---------------------------------------------

    if len(
        original.split()
    ) <= 3:

        candidate = strip_honorifics(
            original
        )

        return clean_extracted_name(
            candidate
        )

    return None


# =========================================================
# 8. AZERBAIJANI NAME SUFFIXES
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

    vowels = "aıoueəiöü"

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

    vowels = "aıoueəiöü"

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
# 9. SPECIAL ROUTING DETECTORS
# =========================================================

def is_bot_meta_question(
    text: str,
) -> bool:

    normalized = normalize_for_search(
        text
    )

    patterns = [
        "siz botsuz",
        "siz botsunuz",
        "sen botsan",
        "botmusuz",
        "bot musunuz",
        "men kimle danisiram",
        "kimle danisiram",
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

    child_terms = [
        "usaq",
        "ovlad",
        "oglum",
        "qizim",
        "yeniyetme",
    ]

    presence_terms = [
        "yanimda",
        "yaninda",
        "olmalidir",
        "olmalidi",
        "olmasi vacib",
        "gelmelidir",
        "gelmelidi",
    ]

    return (
        any(
            term in normalized
            for term in child_terms
        )
        and any(
            term in normalized
            for term in presence_terms
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
        "iki usağim",
        "2 usagim",
        "mende iki usaq",
        "mende 2 usaq",
        "bizde iki usaq",
        "bizde 2 usaq",
        "iki usaqdir",
        "2 usaqdir",
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
        "sag olun sonra",
        "sagolun sonra",
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
        "bugun uygun deyil",
        "bu gun uygun deyil",
        "bugun vaxtim yoxdur",
        "bu gun vaxtim yoxdur",
    ]

    return any(
        pattern in normalized
        for pattern in patterns
    )


def is_contact_here_question(
    text: str,
) -> bool:

    normalized = normalize_for_search(
        text
    )

    here_terms = [
        "burdan",
        "buradan",
        "burda",
        "burada",
        "chatdan",
        "burdan elaqe",
        "buradan elaqe",
    ]

    contact_terms = [
        "elaqe",
        "danismaq",
        "yazismaq",
        "mumkundur",
        "olar",
    ]

    return (
        any(
            term in normalized
            for term in here_terms
        )
        and any(
            term in normalized
            for term in contact_terms
        )
    )


def is_call_timing_question(
    text: str,
) -> bool:

    normalized = normalize_for_search(
        text
    )

    time_terms = [
        "sabah",
        "bugun",
        "bu gun",
        "axsam",
        "axsam saatlari",
        "seher",
        "gunorta",
        "nahardan sonra",
        "heftesonu",
    ]

    contact_terms = [
        "zeng",
        "elaqe",
        "danismaq",
        "danisa",
        "alinar",
        "mumkundur",
        "olar",
    ]

    return (
        any(
            term in normalized
            for term in time_terms
        )
        and any(
            term in normalized
            for term in contact_terms
        )
    )


def is_general_program_interest(
    text: str,
) -> bool:

    normalized = normalize_for_search(
        text
    )

    patterns = [
        "proqramla maraqlaniram",
        "proqramlarla maraqlaniram",
        "proqramla maraqlanirdim",
        "junior coaching proqramiyla maraqlaniram",
        "junior coaching proqrami ile maraqlaniram",
        "sizin proqramla maraqlaniram",
        "sizin proqramla maraqlanirdim",
        "esas sizin proqramla maraqlanirdim",
        "proqram haqqinda melumat",
        "proqram barede melumat",
        "proqrami oyrenmek isteyirem",
    ]

    return any(
        pattern in normalized
        for pattern in patterns
    )


# =========================================================
# 10. FAQ INDEX
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
                max_features=50000,
                sublinear_tf=True,
            ),
        ),
        (
            "char",
            TfidfVectorizer(
                analyzer="char_wb",
                ngram_range=(3, 5),
                max_features=70000,
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


questions, answers, vectorizer, Q = (
    _build_faq_index()
)


def expand_faq_query(
    text: str,
) -> str:

    normalized = normalize_for_search(
        text
    )

    additions = []

    if any(
        x in normalized
        for x in [
            "qiymet",
            "odenis",
            "ne qeder",
            "pul",
        ]
    ):

        additions.extend([
            "qiymet",
            "odenis",
            "proqram qiymeti",
        ])

    if any(
        x in normalized
        for x in [
            "yasdan",
            "yas qrupu",
            "nece yas",
            "qosulmaq",
        ]
    ):

        additions.extend([
            "yas qrupu",
            "12 18",
            "proqrama nece yasdan",
        ])

    if any(
        x in normalized
        for x in [
            "ne zaman baslayir",
            "ne vaxt baslayir",
            "baslama tarixi",
        ]
    ):

        additions.extend([
            "proqram ne vaxt baslayir",
            "baslama tarixi",
        ])

    if any(
        x in normalized
        for x in [
            "endirim",
            "iki usaq",
            "2 usaq",
        ]
    ):

        additions.extend([
            "endirim",
            "kampaniya",
            "iki usaq",
        ])

    if additions:

        return (
            normalized
            + " "
            + " ".join(additions)
        )

    return normalized


def retrieve_similar(
    user_query: str,
    k: int = 4,
    min_score: float = 0.15,
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


def answer_faq_question(
    user_text: str,
    min_score: float = 0.20,
):

    hits = retrieve_similar(
        user_query=user_text,
        k=1,
        min_score=min_score,
    )

    if not hits:
        return None

    question, answer, score = hits[0]

    return {
        "question": question,
        "answer": answer,
        "score": score,
    }


# =========================================================
# 11. QUESTION DETECTION
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
        "olarmi",
        "olar",
        "varmi",
        "oyrenmek isteyirem",
        "melumat isteyirem",
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

    topics = [
        "proqram",
        "qiymet",
        "odenis",
        "endirim",
        "yas",
        "qosul",
        "baslay",
        "gorus",
        "qrup",
        "ferdi",
        "onlayn",
        "online",
        "mekan",
        "telimci",
        "sertifikat",
        "sinaq",
        "kampaniya",
        "psixoloq",
        "tanisliq",
        "zeng",
    ]

    return (
        looks_like_question(text)
        and any(
            topic in normalized
            for topic in topics
        )
    )


# =========================================================
# 12. LLM ROUTER
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
Sən Junior Coaching mesaj router-isən.

Intent-lər:

greeting
faq_question
field_answer
registration_request
human_agent_request
complaint
safety_risk
meta_question
pause_request
unrelated

ƏSAS QAYDA:

current_field istifadəçini məcbur etmir.

İstifadəçi formanın istənilən mərhələsində proqram haqqında
sual verə bilər.

Məsələn:

current_field=parent_name
"junior coaching proqramıyla maraqlanıram"
=> faq_question

current_field=parent_name
"11 yaşlı oğlum üçün proqramla maraqlanıram"
=> faq_question

current_field=main_concern
"mən əsas sizin proqramla maraqlanırdım"
=> faq_question

current_field=phone
"uşaq yanımda olmalıdır?"
=> faq_question

current_field=phone
"axşam saatları zəng etmək alınar?"
=> faq_question

current_field=phone
"0555555555"
=> field_answer

current_field=child_name
"2 uşaqdır məndə"
=> field_answer

current_field=parent_name
"İsmayıl bəy"
=> field_answer

current_field=parent_name
"Salam adım İsmayıldır"
=> field_answer

current_field=child_age
"12 tamam olacaq"
=> field_answer

Meta:
"siz botsuz?"
"kimlə danışıram?"
=> meta_question

Pause:
"sonra əlaqə saxlayarıq sağ olun"
=> pause_request

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

    # ---------------------------------------------
    # Explicit deterministic routes first
    # ---------------------------------------------

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
        or is_contact_here_question(
            user_text
        )
        or is_general_program_interest(
            user_text
        )
        or is_english_program_info_question(
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
            "confidence": 1.0,
        }

    try:

        return classify_message_with_llm(
            user_text=user_text,
            current_field=current_field,
        )

    except Exception as exc:

        print(
            "LLM ROUTER ERROR:",
            exc,
        )

        if is_greeting(
            user_text
        ):

            intent = "greeting"

        else:

            intent = "field_answer"

        return {
            "intent": intent,
            "is_question": False,
            "should_escalate": False,
            "confidence": 0.0,
        }


# =========================================================
# 13. PARENT TITLE
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
                    "content": (
                        "Azərbaycan adına əsasən müraciət "
                        "formasını seç: xanım, bəy və ya neutral. "
                        "Əmin deyilsənsə neutral seç."
                    ),
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
# 14. LEAD
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
        "Bu hal nə qədər müddətdir davam edir?",

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
# 15. PERSONALIZED QUESTIONS
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

        return FIELD_QUESTIONS[
            "main_concern"
        ]

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

        return FIELD_QUESTIONS[
            "phone"
        ]

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
# 16. NEXT FIELD
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
            return "concern_duration"

        if not lead.get(
            "concern_onset"
        ):
            return "concern_onset"

    for field in [
        "phone",
        "preferred_call_time",
    ]:

        if not lead.get(
            field
        ):
            return field

    return None


# =========================================================
# 17. FIELD VALIDATION
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

    # ---------------------------------------------
    # PARENT NAME
    # ---------------------------------------------

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

        return True, None

    # ---------------------------------------------
    # CHILD NAME
    # ---------------------------------------------

    if field == "child_name":

        if is_multiple_children_message(
            user_text
        ):

            return (
                False,
                "Başa düşürəm 😊 İki övladınız üçün də "
                "məlumat ala bilərik. Hələlik birinci "
                "övladınızdan başlayaq. "
                "Onun adını qeyd edə bilərsiniz?"
            )

        name = extract_person_name(
            user_text
        )

        if not name:

            return (
                False,
                "Övladınızın adını tam anlaya bilmədim. "
                "Məsələn: Leyla, Orxan və ya \"adı Eli\"."
            )

        lead[
            "child_name"
        ] = name

        return True, None

    # ---------------------------------------------
    # AGE
    # ---------------------------------------------

    if field == "child_age":

        if is_multiple_children_message(
            user_text
        ):

            child_name = lead.get(
                "child_name"
            )

            return (
                False,
                "Başa düşürəm, iki övladınız var. "
                "Hazırda müraciəti bir övlad üzrə "
                "davam etdiririk. "
                f"Əvvəlcə {child_genitive(child_name)} "
                "yaşını qeyd edə bilərsiniz?"
            )

        ages = extract_age_candidates(
            user_text
        )

        if len(ages) > 1:

            child_name = lead.get(
                "child_name"
            )

            return (
                False,
                f"Bir neçə yaş qeyd etdiniz: "
                f"{', '.join(map(str, ages))}. "
                f"{child_genitive(child_name)} yaşı hansıdır?"
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

        return True, None

    # ---------------------------------------------
    # MAIN CONCERN
    # ---------------------------------------------

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

            return True, None

        if normalized in vague_answers:

            return (
                False,
                "Bir qədər dəqiqləşdirə bilərsiniz? "
                "Övladınızla bağlı sizi narahat edən "
                "əsas məsələni qısa şəkildə qeyd edin."
            )

        lead[
            "main_concern"
        ] = user_text

        lead[
            "needs_concern_followup"
        ] = False

        return True, None

    # ---------------------------------------------
    # DURATION
    # ---------------------------------------------

    if field == "concern_duration":

        duration_terms = [
            "gun",
            "hefte",
            "ay",
            "il",
            "coxdan",
            "bir nece",
            "texminen",
            "usaqliqdan",
        ]

        if not (
            any(
                term in normalized
                for term in duration_terms
            )
            or re.search(
                r"\d+",
                normalized,
            )
        ):

            return (
                False,
                "Təxmini müddəti qeyd edə bilərsiniz? "
                "Məsələn: 2 həftədir və ya 3 aydır."
            )

        lead[
            "concern_duration"
        ] = user_text

        return True, None

    # ---------------------------------------------
    # ONSET
    # ---------------------------------------------

    if field == "concern_onset":

        lead[
            "concern_onset"
        ] = user_text

        return True, None

    # ---------------------------------------------
    # PHONE
    # ---------------------------------------------

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

        return True, None

    # ---------------------------------------------
    # CALL TIME
    # ---------------------------------------------

    if field == "preferred_call_time":

        # "sabah" təkbaşına kifayət deyil
        vague_day_only = {
            "sabah",
            "bugun",
            "bu gun",
            "heftesonu",
            "bazar",
        }

        if normalized in vague_day_only:

            return (
                False,
                f"Əlbəttə. {user_text.capitalize()} sizə "
                "hansı saat aralığı daha uyğun olar? "
                "Məsələn: 14:00–16:00."
            )

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
                "Zəhmət olmasa uyğun gün və saat aralığını "
                "bir qədər dəqiqləşdirin. "
                "Məsələn: sabah 14:00–16:00."
            )

        time_terms = [
            "bugun",
            "bu gun",
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

        has_daypart = any(
            term in normalized
            for term in time_terms
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

        if not (
            has_daypart
            or has_numeric_time
        ):

            return (
                False,
                "Zəhmət olmasa uyğun gün və saat aralığını "
                "qeyd edin. Məsələn: sabah 14:00–16:00."
            )

        lead[
            "preferred_call_time"
        ] = user_text

        return True, None

    return (
        False,
        f"'{field}' üçün validation müəyyən edilməyib."
    )


# =========================================================
# 18. SPECIAL RESPONSES
# =========================================================

def append_current_question(
    response: str,
    lead: dict,
) -> str:

    current_field = get_next_missing_field(
        lead
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


def answer_meta_question(
    lead: dict,
) -> str:

    return append_current_question(
        (
            "Mən Junior Coaching proqramı üzrə "
            "virtual müraciət köməkçisiyəm 😊 "
            "Proqram haqqında suallarınızı cavablandıra "
            "və müraciətinizi qeydə ala bilirəm."
        ),
        lead,
    )


def answer_child_presence(
    lead: dict,
) -> str:

    child_name = lead.get(
        "child_name"
    )

    if child_name:

        response = (
            f"İlkin zəng zamanı {child_genitive(child_name)} "
            "yanınızda olması vacib deyil. "
            "Daha sonra proqramdan əvvəl övladınızla "
            "təxminən 5 dəqiqəlik video tanışlıq "
            "görüşü keçirilə bilər."
        )

    else:

        response = (
            "İlkin zəng zamanı övladınızın yanınızda "
            "olması vacib deyil. Daha sonra proqramdan "
            "əvvəl övladınızla təxminən 5 dəqiqəlik "
            "video tanışlıq görüşü keçirilə bilər."
        )

    return append_current_question(
        response,
        lead,
    )


def answer_contact_here(
    lead: dict,
) -> str:

    return append_current_question(
        (
            "Bəli 😊 Buradan müraciətinizi qeyd edə bilərsiniz. "
            "Əlaqə nömrənizi və sizə uyğun zəng vaxtını "
            "qeyd etdikdən sonra Junior Coaching komandası "
            "sizinlə əlaqə saxlayacaq."
        ),
        lead,
    )


def answer_call_timing(
    lead: dict,
) -> str:

    current_field = get_next_missing_field(
        lead
    )

    response = (
        "Bəli, mümkündür 😊 Sizə uyğun gün və saat "
        "aralığı nəzərə alınaraq əlaqə saxlanıla bilər."
    )

    if current_field == "phone":

        response += (
            "\n\nƏvvəlcə sizinlə əlaqə saxlaya bilməyimiz "
            "üçün telefon nömrənizi qeyd edə bilərsiniz?"
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
        "Əlbəttə, problem deyil 😊 Sizə uyğun başqa gün "
        "və saat üçün əlaqə yaradıla bilər."
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


def answer_general_program_interest(
    lead: dict,
) -> str:

    response = (
        "Əlbəttə 😊 Junior Coaching 12–18 yaşlı "
        "yeniyetmələrin şəxsi və sosial inkişafına "
        "dəstək verən proqramdır. Yeniyetmələr proqram "
        "çərçivəsində özlərini daha yaxşı tanımaq, "
        "fikirlərini ifadə etmək, qərarvermə, məsuliyyət "
        "və sağlam ünsiyyət kimi bacarıqlar üzərində işləyirlər."
    )

    return append_current_question(
        response,
        lead,
    )


def answer_english_program_interest(
    lead: dict,
) -> str:

    response = (
        "Of course 😊 Junior Coaching is a development "
        "programme designed for teenagers aged 12–18. "
        "It supports teenagers in areas such as self-awareness, "
        "communication, decision-making and responsibility."
    )

    current_field = get_next_missing_field(
        lead
    )

    if current_field == "parent_name":

        response += (
            "\n\nMay I ask how I should address you?"
        )

    elif current_field:

        # Flow hazırda AZ saxlanılır
        response += (
            "\n\n"
            + get_personalized_question(
                current_field,
                lead,
            )
        )

    return response


def answer_embedded_age_interest(
    age: int,
    lead: dict,
) -> str:

    if age < 12:

        response = (
            f"Əlbəttə, məlumat verə bilərəm 😊 "
            f"Junior Coaching proqramı hazırda 12–18 yaşlı "
            f"yeniyetmələr üçün nəzərdə tutulub. "
            f"Qeyd etdiyiniz {age} yaş hazırkı yaş "
            f"qrupuna uyğun deyil."
        )

    elif age <= 18:

        response = (
            f"Əlbəttə 😊 {age} yaş Junior Coaching "
            f"proqramının 12–18 yaş aralığına uyğundur."
        )

    else:

        response = (
            f"Junior Coaching proqramı hazırda 12–18 yaşlı "
            f"yeniyetmələr üçün nəzərdə tutulub."
        )

    return append_current_question(
        response,
        lead,
    )


# =========================================================
# 19. MAIN AGENT
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

    lead[
        "_last_faq_score"
    ] = None

    # =====================================================
    # HARD ROUTES — CURRENT FIELD-DƏN ƏVVƏL
    # =====================================================

    if is_child_presence_question(
        user_text
    ):

        lead["_last_intent"] = "faq_question"
        lead["_last_confidence"] = 1.0

        return answer_child_presence(
            lead
        )

    if is_contact_here_question(
        user_text
    ):

        lead["_last_intent"] = "faq_question"
        lead["_last_confidence"] = 1.0

        return answer_contact_here(
            lead
        )

    if is_call_timing_question(
        user_text
    ):

        lead["_last_intent"] = "faq_question"
        lead["_last_confidence"] = 1.0

        return answer_call_timing(
            lead
        )

    if is_not_available_today(
        user_text
    ):

        lead["_last_intent"] = "pause_request"
        lead["_last_confidence"] = 1.0

        return answer_not_available_today(
            lead
        )

    if is_pause_or_goodbye(
        user_text
    ):

        lead["_last_intent"] = "pause_request"
        lead["_last_confidence"] = 1.0

        return (
            "Əlbəttə 😊 İstədiyiniz zaman yenidən "
            "yaza bilərsiniz. Təşəkkür edirik."
        )

    if is_bot_meta_question(
        user_text
    ):

        lead["_last_intent"] = "meta_question"
        lead["_last_confidence"] = 1.0

        return answer_meta_question(
            lead
        )

    # -----------------------------------------------------
    # Embedded age + programme interest
    # -----------------------------------------------------

    embedded_age = detect_embedded_child_age(
        user_text
    )

    if (
        embedded_age is not None
        and is_general_program_interest(
            user_text
        )
    ):

        lead["_last_intent"] = "faq_question"
        lead["_last_confidence"] = 1.0

        return answer_embedded_age_interest(
            embedded_age,
            lead,
        )

    # -----------------------------------------------------
    # English
    # -----------------------------------------------------

    if is_english_program_info_question(
        user_text
    ):

        lead["_last_intent"] = "faq_question"
        lead["_last_confidence"] = 1.0

        return answer_english_program_interest(
            lead
        )

    # -----------------------------------------------------
    # Programme interest
    # -----------------------------------------------------

    if is_general_program_interest(
        user_text
    ):

        lead["_last_intent"] = "faq_question"
        lead["_last_confidence"] = 1.0

        return answer_general_program_interest(
            lead
        )

    # -----------------------------------------------------
    # Multiple children must work before field validation
    # -----------------------------------------------------

    if (
        current_field
        in [
            "child_name",
            "child_age",
        ]
        and is_multiple_children_message(
            user_text
        )
    ):

        lead["_last_intent"] = "field_answer"
        lead["_last_confidence"] = 1.0

        if current_field == "child_name":

            return (
                "Başa düşürəm 😊 İki övladınız üçün də "
                "məlumat ala bilərik. Hələlik birinci "
                "övladınızdan başlayaq. "
                "Onun adını qeyd edə bilərsiniz?"
            )

        child_name = lead.get(
            "child_name"
        )

        return (
            "Başa düşürəm, iki övladınız var. "
            "Hazırda müraciəti bir övlad üzrə davam etdiririk. "
            f"Əvvəlcə {child_genitive(child_name)} "
            "yaşını qeyd edə bilərsiniz?"
        )

    # =====================================================
    # ROUTER
    # =====================================================

    classification = safe_classify_message(
        user_text=user_text,
        current_field=current_field,
    )

    intent = classification[
        "intent"
    ]

    lead["_last_intent"] = intent

    lead["_last_confidence"] = (
        classification.get(
            "confidence"
        )
    )

    print(
        "INTENT DEBUG:",
        classification,
    )

    # =====================================================
    # GREETING
    # =====================================================

    if intent == "greeting":

        if current_field == "parent_name":

            return (
                "Salam 😊 Sizə necə müraciət edə bilərəm?"
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
    # META
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
            "Əlbəttə 😊 İstədiyiniz zaman yenidən "
            "yaza bilərsiniz. Təşəkkür edirik."
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
            "əvəz etmir. Müraciətiniz məsul əməkdaşa "
            "yönləndirilir."
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
            "Müraciətinizi məsul əməkdaşa "
            "yönləndirmək üçün qeydə aldım."
        )

    # =====================================================
    # HUMAN
    # =====================================================

    if intent == "human_agent_request":

        lead[
            "status"
        ] = "ESCALATED"

        return (
            "Əlbəttə. Müraciətinizi Junior Coaching "
            "komandasından məsul əməkdaşa "
            "yönləndirmək üçün qeydə aldım."
        )

    # =====================================================
    # FAQ
    # =====================================================

    if intent == "faq_question":

        # Təkrar special handlers
        if is_child_presence_question(
            user_text
        ):
            return answer_child_presence(
                lead
            )

        if is_contact_here_question(
            user_text
        ):
            return answer_contact_here(
                lead
            )

        if is_call_timing_question(
            user_text
        ):
            return answer_call_timing(
                lead
            )

        if is_general_program_interest(
            user_text
        ):
            return answer_general_program_interest(
                lead
            )

        faq_result = answer_faq_question(
            user_text=user_text,
            min_score=faq_min_score,
        )

        if faq_result is not None:

            lead[
                "_last_faq_score"
            ] = faq_result.get(
                "score"
            )

            response = faq_result[
                "answer"
            ]

            if current_field:

                response += (
                    "\n\n"
                    + get_personalized_question(
                        current_field,
                        lead,
                    )
                )

            return response

        response = (
            "Bu sualla bağlı məlumat bazasında "
            "dəqiq cavab tapmadım. İstəsəniz sualınızı "
            "məsul əməkdaşa yönləndirə bilərik."
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
    # FINISHED
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
                f"İlkin zəng zamanı "
                f"{child_genitive(child_name)} "
                "yanınızda olması vacib deyil."
            )

        return final_message

    return get_personalized_question(
        next_field,
        lead,
    )


# =========================================================
# 20. SQLITE INIT
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
        # leads migrations
        # ---------------------------------------------

        existing = {
            row[1]
            for row in conn.execute(
                "PRAGMA table_info(leads)"
            ).fetchall()
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

        for column, dtype in required.items():

            if column not in existing:

                conn.execute(
                    f"""
                    ALTER TABLE leads
                    ADD COLUMN {column} {dtype}
                    """
                )

        # ---------------------------------------------
        # conversation_logs migrations
        # ---------------------------------------------

        existing_logs = {
            row[1]
            for row in conn.execute(
                "PRAGMA table_info(conversation_logs)"
            ).fetchall()
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

        for column, dtype in required_logs.items():

            if column not in existing_logs:

                conn.execute(
                    f"""
                    ALTER TABLE conversation_logs
                    ADD COLUMN {column} {dtype}
                    """
                )

        conn.commit()


# =========================================================
# 21. TIME
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
# 22. FIND LEAD
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
# 23. SAVE LEAD
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
# 24. SAVE CONVERSATION LOG
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
# 25. ADMIN HELPERS
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
# INIT
# =========================================================

init_db()