"""
Junior Coaching — Bot Engine V6 Conversation Manager

Əsas prinsiplər
---------------
1. Hər user mesajı bütöv analiz olunur.
2. Bir mesajdan bir neçə məlumat eyni anda çıxarıla bilər.
3. Artıq məlum olan məlumat yenidən soruşulmur.
4. User arada sual verirsə:
      əvvəl suala cavab verilir,
      sonra maksimum 1 çatışmayan məlumat soruşulur.
5. main_concern mərhələsində:
      "məsuliyyətsizdir",
      "özgüvəni zəifdir",
      "məqsəd və gələcək",
      "hamısı"
   complaint / FAQ kimi qəbul edilmir.
6. İki və daha çox uşaq üçün children strukturu saxlanılır.
7. SQLite + Streamlit app.py ilə backward compatible-dir.
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


def normalize_text(
    text: str,
) -> str:

    return re.sub(
        r"\s+",
        " ",
        str(text).strip().lower(),
    )


def normalize_for_search(
    text: str,
) -> str:

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

    return re.sub(
        r"\s+",
        " ",
        text,
    ).strip()


# =========================================================
# 3. GENERAL HELPERS
# =========================================================

def normalize_phone(
    text: str,
) -> Optional[str]:

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


def extract_all_ages(
    text: str,
) -> List[int]:

    result = []

    for value in re.findall(
        r"\b\d{1,2}\b",
        text,
    ):

        number = int(
            value
        )

        if (
            1 <= number <= 99
            and number not in result
        ):
            result.append(
                number
            )

    return result


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
# 4. NAME CLEANING
# =========================================================

def remove_honorific(
    text: str,
) -> str:

    words = text.strip().split()

    blocked = {
        "bey",
        "xanim",
        "muellim",
    }

    result = []

    for word in words:

        if normalize_for_search(
            word
        ) not in blocked:

            result.append(
                word
            )

    return " ".join(
        result
    )


# Ad ola bilməyən sözlər.
# Bunlar rol, işarə əvəzliyi, təsdiq/inkar və ya
# söhbət ifadələridir — heç vaxt ad kimi qəbul edilmir.

NON_NAME_TOKENS = {

    # Əvəzliklər
    "men", "mene", "menim", "menem",
    "sen", "sene", "senin",
    "siz", "size", "sizin", "sizde",
    "o", "ona", "onun", "onlar",
    "biz", "bize", "bizim",

    # Rollar
    "ana", "anasi", "anasiyam", "anayam",
    "ata", "atasi", "atasiyam", "atayam",
    "valideyn", "valideyniyem", "valideynem",
    "nene", "nenesiyem", "baba", "babasiyam",
    "xala", "xalasiyam", "bibi", "bibisiyem",
    "emi", "emisiyem", "dayi", "dayisiyam",

    # Uşağa istinad
    "usaq", "usagi", "usagin", "ovlad", "ovladi",
    "oglum", "oglu", "qizim", "qizi", "bala", "balam",

    # Söhbət ifadələri
    "dedim", "dedin", "demisdim", "yazdim", "yazmisam",
    "yuxarida", "asagida", "evvel", "evvelde", "hemin",
    "bilmirem", "bilmirik", "yoxdur", "var", "yoxumdur",
    "beli", "xeyr", "yox", "he", "hee", "hmm", "ok", "okey",
    "tesekkur", "sagolun", "sagol", "salam", "hello",
    "adi", "adim", "adin", "adiniz",
    "hamisi", "her", "biri", "hec", "bir",

    # Müraciət formaları
    "bey", "xanim", "muellim", "muellime",
}


# Sərbəst mətn sahələrinə (main_concern, zəng vaxtı və s.)
# yazılmamalı olan "cavab olmayan" cavablar.

NON_ANSWER_TOKENS = {
    "beli", "he", "hee", "yox", "xeyr", "hmm",
    "ok", "okey", "bilmirem", "bilmirik",
    "dedim", "dedim yuxarida", "yuxarida",
    "hec ne", "hecne", "sonra", "sonra deyerem",
    "tesekkur", "sagolun", "sagol",
}


# Söhbət tarixçəsində saxlanılan maksimum mesaj sayı.
# LLM-ə yalnız son hissəsi göndərilir.

MAX_HISTORY_MESSAGES = 20


# İmtina ifadələri.
#
# Valideyn məlumat verməkdən imtina edirsə:
#   - bu şikayət DEYİL (ESCALATED olmamalıdır),
#   - eyni sual sonsuz təkrarlanmamalıdır,
#   - imtina mətni heç bir sahəyə yazılmamalıdır.

REFUSAL_MARKERS = [
    "istemirem",
    "istemir",
    "vermeyeceyem",
    "vermerem",
    "paylasmaq istemirem",
    "uygun deyilem",
    "uygun deyil",
    "yazmaq istemirem",
    "demek istemirem",
    "gerek yoxdur",
    "lazim deyil",
    "imtina",
]


def is_refusal(
    text: str,
) -> bool:

    """
    "vermək istəmirəm", "uyğun deyiləm" -> True
    """

    value = normalize_for_search(
        text
    )

    return any(
        marker in value
        for marker in REFUSAL_MARKERS
    )


# main_concern mərhələsində birbaşa cavab sayılan ifadələr.
#
# "hamısı" FAQ sualı deyil — bu, soruşulan suala cavabdır.

CONCERN_ANSWER_MARKERS = [
    "hamisi",
    "hamsi",
    "her biri",
    "her sey",
    "ozguven",
    "ozune inam",
    "ozuneinam",
    "unsiyyet",
    "mesuliyyet",
    "intizam",
    "fokus",
    "liderlik",
    "meqsed",
    "gelecek",
    "emosional",
    "qerarverme",
    "sosiallasma",
    "telefon asililigi",
]


def is_direct_concern_answer(
    text: str,
) -> bool:

    """
    Qısa, sualsız və inkişaf sahəsi bildirən cavab.
    """

    value = normalize_for_search(
        text
    )

    if "?" in text:
        return False

    if len(
        value.split()
    ) > 6:
        return False

    return any(
        marker in value
        for marker in CONCERN_ANSWER_MARKERS
    )


# "-dır/-dir/-dur/-dür" şəkilçisi kəsilməməli olan
# real adlar (Bahadır -> Baha olmamalıdır).

NAME_SUFFIX_EXCEPTIONS = {
    "bahadir",
    "qadir",
    "kadir",
    "nadir",
    "qedir",
    "abdulqadir",
    "cavidan",
}


def remove_name_suffix(
    word: str,
) -> str:

    """
    ismayildir -> ismayil
    orxandir   -> orxan
    elvindir   -> elvin

    Amma Nadir -> Nadir, Bahadır -> Bahadır qalır.
    """

    if not word:
        return word

    normalized = normalize_for_search(
        word
    )

    if normalized in NAME_SUFFIX_EXCEPTIONS:
        return word

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

                return word[
                    :base_length
                ]

    return word


def looks_like_person_name(
    value: str,
) -> bool:

    """
    Sərbəst mətnin ad kimi yadda saxlanmasının qarşısını alır.

    "Dedim yuxarıda", "anasıyam", "mənə" -> False
    "Orxan", "Ayxan Məmmədov"            -> True
    """

    if not value:
        return False

    words = value.split()

    if not words or len(words) > 2:
        return False

    for word in words:

        normalized = normalize_for_search(
            word
        )

        if len(normalized) < 2:
            return False

        if normalized in NON_NAME_TOKENS:
            return False

    return True


def clean_name(
    value: str,
) -> Optional[str]:

    if not value:
        return None

    value = remove_honorific(
        value
    )

    value = re.sub(
        r"[.,!?+]",
        "",
        value,
    ).strip()

    if not value:
        return None

    words = []

    for word in value.split():

        cleaned = remove_name_suffix(
            word
        )

        if cleaned:
            words.append(
                cleaned
            )

    if not words:
        return None

    # Maksimum 2 söz
    value = " ".join(
        words[:2]
    )

    if not re.fullmatch(
        r"[A-Za-zƏəÖöÜüĞğÇçŞşİı\- ]+",
        value,
    ):
        return None

    if not looks_like_person_name(
        value
    ):
        return None

    return value.title()


# =========================================================
# 4b. PARENT ROLE -> TITLE
# =========================================================

PARENT_ROLE_TITLES = [

    (
        "xanım",
        [
            "anasiyam", "anayam", "anasi kimi",
            "nenesiyem", "xalasiyam", "bibisiyem",
        ],
    ),

    (
        "bəy",
        [
            "atasiyam", "atayam",
            "babasiyam", "emisiyem", "dayisiyam",
        ],
    ),
]


def detect_parent_title_from_role(
    text: str,
) -> str:

    """
    "anasıyam" -> xanım, "atasıyam" -> bəy.

    LLM çağırışına ehtiyac qalmır.
    """

    value = normalize_for_search(
        text
    )

    for title, markers in PARENT_ROLE_TITLES:

        if any(
            marker in value
            for marker in markers
        ):
            return title

    return ""


def deterministic_name_extract(
    text: str,
) -> Optional[str]:

    patterns = [

        r"(?i)\bmənim\s+adım\s+([A-Za-zƏəÖöÜüĞğÇçŞşİı\-]+)",

        r"(?i)\bmenim\s+adim\s+([A-Za-zƏəÖöÜüĞğÇçŞşİı\-]+)",

        r"(?i)\badım\s+([A-Za-zƏəÖöÜüĞğÇçŞşİı\-]+)",

        r"(?i)\badim\s+([A-Za-zƏəÖöÜüĞğÇçŞşİı\-]+)",

        r"(?i)\bmən\s+([A-Za-zƏəÖöÜüĞğÇçŞşİı\-]+)[əa]m\b",

        r"(?i)\bmen\s+([A-Za-zƏəÖöÜüĞğÇçŞşİı\-]+)[ae]m\b",
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
# 5. AZERBAIJANI SUFFIXES
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
# 6. LEAD STRUCTURE
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

        # Backward compatibility
        "child_name": None,
        "child_age": None,
        "main_concern": None,

        "needs_concern_followup": False,
        "concern_duration": None,
        "concern_onset": None,

        # V3
        "children": [
            create_empty_child()
        ],

        "active_child_index": 0,
        "multiple_children": False,

        "phone": None,
        "preferred_call_time": None,

        "source": source,
        "status": "NEW",

        "lead_stage": "discovery",
        "application_status": "in_progress",
        "objections": [],
        "questions": [],
        "previous_actions": [],
        "pending_actions": [],
        "handoff_status": "none",
        "owner": "ai",

        "_last_intent": None,
        "_last_confidence": None,
        "_last_faq_score": None,

        # V4 — söhbət kontekstİ və axın vəziyyəti.
        # "_" ilə başlayan sahələr DB-yə yazılmır.
        "_history": [],
        "_last_asked_field": None,
        "_ask_repeat_count": 0,

        # Valideyn imtina etdiyi sahələr bir daha soruşulmur.
        "_skipped_fields": [],
        "_refusal_counts": {},
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

    lead.setdefault(
        "active_child_index",
        0,
    )

    lead.setdefault(
        "multiple_children",
        False,
    )

    lead.setdefault(
        "_skipped_fields",
        [],
    )

    lead.setdefault(
        "_refusal_counts",
        {},
    )

    lead.setdefault("lead_stage", "discovery")
    lead.setdefault("application_status", "in_progress")
    lead.setdefault("objections", [])
    lead.setdefault("questions", [])
    lead.setdefault("previous_actions", [])
    lead.setdefault("pending_actions", [])
    lead.setdefault("handoff_status", "none")
    lead.setdefault("owner", "ai")


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

    if index >= len(
        lead["children"]
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

    """
    app.py və köhnə DB strukturu üçün
    birinci uşağın məlumatlarını flat saxlayırıq.
    """

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
# 7. PARENT TITLE
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
Azərbaycan adına əsasən müraciət formasını müəyyən et.

Yalnız:
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

        data = json.loads(
            response.choices[0].message.content
        )

        if data["title"] in [
            "xanım",
            "bəy",
        ]:

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
# 8. FAQ INDEX
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
    ) as f:

        raw = f.read()

    pattern = re.compile(
        r"(?:\d+\.\s*)?Sual:\s*(.*?)\s*"
        r"(?:Agent|Selnaz|Cavab):\s*(.*?)"
        r"(?=\n\s*(?:\d+\.\s*)?Sual:|\Z)",
        re.DOTALL | re.IGNORECASE,
    )

    pairs = []

    for q, a in pattern.findall(
        raw
    ):

        q = re.sub(
            r"\s+",
            " ",
            q,
        ).strip()

        a = re.sub(
            r"\s+",
            " ",
            a,
        ).strip()

        if q and a:

            pairs.append(
                (
                    q,
                    a,
                )
            )

    questions = [
        q
        for q, _ in pairs
    ]

    answers = [
        a
        for _, a in pairs
    ]

    # Yalnız sual deyil, cavab mətni də indekslənir.
    #
    # Səbəb: "görüşlər harada keçirilir?" sualının doğru
    # cavabı "Məkan haradadır?" bəndidir, lakin sual mətnləri
    # leksik olaraq uyğun gəlmir — cavabın içindəki
    # "Görüşlər ... keçirilir" ifadəsi uyğunluğu tapır.
    #
    # Sual 2 dəfə yazılır ki, çəkisi cavabdan yüksək qalsın.

    normalized = [
        normalize_for_search(
            f"{q} {q} {a}"
        )
        for q, a in zip(
            questions,
            answers,
        )
    ]

    vectorizer = FeatureUnion([

        (
            "word",
            TfidfVectorizer(
                analyzer="word",
                ngram_range=(1, 2),
                sublinear_tf=True,
            ),
        ),

        (
            "char",
            TfidfVectorizer(
                analyzer="char_wb",
                ngram_range=(3, 5),
                sublinear_tf=True,
            ),
        ),

    ])

    matrix = vectorizer.fit_transform(
        normalized
    )

    return (
        questions,
        answers,
        vectorizer,
        matrix,
    )


FAQ_QUESTIONS, FAQ_ANSWERS, FAQ_VECTORIZER, FAQ_MATRIX = (
    build_faq_index()
)


def answer_faq_question(
    user_text: str,
    min_score: float = 0.18,
):

    query = normalize_for_search(
        user_text
    )

    vector = FAQ_VECTORIZER.transform(
        [query]
    )

    scores = cosine_similarity(
        vector,
        FAQ_MATRIX,
    ).ravel()

    best_index = int(
        np.argmax(scores)
    )

    best_score = float(
        scores[
            best_index
        ]
    )

    if best_score < min_score:

        return None

    return {
        "question": FAQ_QUESTIONS[
            best_index
        ],
        "answer": FAQ_ANSWERS[
            best_index
        ],
        "score": best_score,
    }


def retrieve_faq_candidates(
    user_text: str,
    k: int = 6,
    min_score: float = 0.06,
) -> List[Dict[str, Any]]:

    """
    TF-IDF yalnız RECALL üçün istifadə olunur.

    Ən yaxşı bir cavabı seçmək əvəzinə namizədləri qaytarır,
    son seçimi LLM kontekstlə birlikdə edir.

    Səbəb: leksik oxşarlıq mənanı tutmur.
    "görüşlər harada keçirilir?" sualı
    "Görüşlər hansı gün keçirilir?" ilə 0.57 bal alır.
    """

    query = normalize_for_search(
        user_text
    )

    vector = FAQ_VECTORIZER.transform(
        [query]
    )

    scores = cosine_similarity(
        vector,
        FAQ_MATRIX,
    ).ravel()

    top_indexes = np.argsort(
        scores
    )[::-1][:k]

    candidates = []

    for index in top_indexes:

        score = float(
            scores[index]
        )

        if score < min_score:
            continue

        candidates.append(
            {
                "id": int(index),
                "question": FAQ_QUESTIONS[index],
                "answer": FAQ_ANSWERS[index],
                "score": score,
            }
        )

    return candidates


# =========================================================
# 9. SPECIAL QUESTION DETECTORS
# =========================================================

def is_child_presence_question(
    text: str,
) -> bool:

    value = normalize_for_search(
        text
    )

    return (
        any(
            x in value
            for x in [
                "usaq",
                "ovlad",
                "oglum",
                "qizim",
            ]
        )
        and any(
            x in value
            for x in [
                "yanimda",
                "olmalidir",
                "gelmelidir",
                "istirak",
                "zengde",
            ]
        )
    )


def is_contact_here_question(
    text: str,
) -> bool:

    value = normalize_for_search(
        text
    )

    return (
        any(
            x in value
            for x in [
                "burdan",
                "buradan",
                "burda",
                "burada",
            ]
        )
        and any(
            x in value
            for x in [
                "elaqe",
                "danismaq",
                "yazismaq",
                "mumkundur",
                "olar",
            ]
        )
    )


def is_bot_question(
    text: str,
) -> bool:

    value = normalize_for_search(
        text
    )

    patterns = [
        "siz botsuz",
        "sen botsan",
        "kimle danisiram",
        "siz kimsiniz",
        "kim cavab verir",
    ]

    return any(
        x in value
        for x in patterns
    )


def is_call_time_question(
    text: str,
) -> bool:

    value = normalize_for_search(
        text
    )

    return (
        any(
            x in value
            for x in [
                "sabah",
                "bugun",
                "axsam",
                "seher",
                "gunorta",
                "heftesonu",
            ]
        )
        and any(
            x in value
            for x in [
                "zeng",
                "elaqe",
                "danismaq",
                "mumkundur",
                "olar",
                "alinar",
            ]
        )
    )


# =========================================================
# 10. WHOLE MESSAGE EXTRACTION — V3 CORE
# =========================================================

def build_fallback_extraction(
    user_text: str,
) -> dict:

    """
    LLM olmayanda (açar yoxdur və ya sorğu xəta verib)
    qayda əsaslı minimal analiz.

    Salamlaşma və telefon nömrəsi burada da tanınmalıdır,
    əks halda "salam" parent_name kimi yadda saxlanılır.
    """

    return {
        "intent": (
            "greeting"
            if is_greeting(
                user_text
            )
            else "field_answer"
        ),
        "is_question": (
            "?" in user_text
        ),
        "question_text": (
            user_text
            if "?" in user_text
            else ""
        ),
        "faq_choice": -1,
        "topic_open": False,
        "parent_name": "",
        "parent_title": detect_parent_title_from_role(
            user_text
        ),
        "children": [],
        "corrections": [],
        "phone": (
            normalize_phone(
                user_text
            )
            or ""
        ),
        "preferred_call_time": "",
        "multiple_children": False,
        "children_count": 0,
        "confidence": 0.0,
    }


def build_history_text(
    history: Optional[List[Dict[str, str]]],
    max_turns: int = 8,
) -> str:

    """
    Son mesajları LLM üçün mətnə çevirir.

    Kontekst olmadan düzəlişləri və davam edən mövzunu
    başa düşmək mümkün deyil.
    """

    if not history:
        return "(əvvəlki mesaj yoxdur)"

    lines = []

    for message in history[-max_turns:]:

        role = (
            "Valideyn"
            if message.get("role") == "user"
            else "Agent"
        )

        lines.append(
            f"{role}: {message.get('content', '')}"
        )

    return "\n".join(
        lines
    )


def analyze_message(
    user_text: str,
    lead: dict,
    history: Optional[List[Dict[str, str]]] = None,
    faq_candidates: Optional[List[Dict[str, Any]]] = None,
) -> dict:
    """
    V6 Conversation Analyzer.

    Bir mesajı tək intent kimi yox, bütöv dialoq addımı kimi analiz edir:
      - bütün slotları çıxarır,
      - bir neçə sualı ayrıca ayırır,
      - düzəlişləri müəyyən edir,
      - state haqqında sualları tanıyır,
      - flow-a indi qayıtmağın təbii olub-olmadığını seçir.
    """

    ensure_lead_structure(lead)

    state = {
        "parent_name": lead.get("parent_name"),
        "parent_title": lead.get("parent_title"),
        "phone": lead.get("phone"),
        "preferred_call_time": lead.get("preferred_call_time"),
        "children": lead.get("children"),
        "next_missing_field": get_next_missing_field(lead),
    }

    if client is None:
        data = build_fallback_extraction(user_text)
        data.setdefault("questions", [user_text] if data.get("is_question") else [])
        data.setdefault("state_question_type", "")
        data.setdefault("resume_flow", False)
        return data

    candidates_text = "\n".join(
        f"[{item['id']}] {item['question']}"
        for item in (faq_candidates or [])
    ) or "(namizəd yoxdur)"

    system_prompt = r"""
Sən Junior Coaching müraciət sisteminin CONVERSATION ANALYZER modulusan.

HƏR DƏFƏ bunları birlikdə nəzərə al:
1. SÖHBƏT TARİXÇƏSİ
2. CARİ STATE
3. KNOWLEDGE BASE NAMİZƏDLƏRİ
4. VALİDEYNİN SON MESAJI

MƏQSƏD: son mesajı bütöv emal et. Flow-u mesaja zorla tətbiq etmə.

A) BİR MESAJDA BÜTÜN MƏLUMATLARI ÇIXAR

Misal:
"Mən Nərgizəm. Oğlum Orxanın 15 yaşı var. Özünəinamı zəifdir.
Nömrəm 0501234567. Sabah 15:00-dan sonra danışa bilərəm."
=> parent_name=Nərgiz
=> children=[{name:Orxan, age:15, main_concern:özünəinam}]
=> phone=0501234567
=> preferred_call_time=sabah 15:00-dan sonra

Bir mesajda həm məlumat, həm sual ola bilər. Məlumatı itirmə.

B) BİR NEÇƏ SUALI AYIR

"Görüşlər harada keçirilir və qiyməti nə qədərdir?"
=> questions=["Görüşlər harada keçirilir?", "Qiyməti nə qədərdir?"]

"Telefon zəngi nə qədər davam edir, uşağın yanında olması lazımdır?"
=> 2 ayrı sual.

questions massivində istifadəçinin REAL suallarını ayrı-ayrı saxla.
question_text geriyə uyğunluq üçün ilk sual olsun, sual yoxdursa boş olsun.

C) MƏNA FƏRQLƏRİNİ QORU

- "Görüşlər harada keçirilir?" = MƏKAN
- "Görüşlər hansı gün keçirilir?" = GÜN/TARİX
- "Görüşlər neçə saatdır?" = QRUP GÖRÜŞÜNÜN MÜDDƏTİ
- "Telefon zəngi nə qədər davam edir?" = VALİDEYNLƏ İLKİN TELEFON ZƏNGİNİN MÜDDƏTİ
- "Gələ bilmədiyimiz görüşü əvəz etmək olur?" = BURAXILMIŞ GÖRÜŞÜN ƏVƏZLƏNMƏSİ
- "Yalnız bir görüşə gələ bilərik?" = BİRDƏFƏLİK İŞTİRAK
Bunları bir-biri ilə qarışdırma.

D) DÜZƏLİŞLƏR

Əvvəlki state səhvdirsə corrections massivinə yaz və yeni dəyəri də çıxar.

"Aygün mənəm, uşağın adı Ayxandır"
=> corrections=[
 {field:parent_name, child_index:0, value:Aygün},
 {field:child_name, child_index:0, value:Ayxan}
]

"mənə yox, Tunar"
Əgər tarixçədən bunun uşaq adı düzəlişi olduğu aydındırsa:
=> correction child_name=Tunar

"yox, 14 yaşı var"
=> correction child_age=14

E) AD VƏ ROL

"anasıyam", "atasıyam", "valideyniyəm", "mənə", "mənim" AD DEYİL.
"Mən Aygünəm" => parent_name=Aygün
"Adım İsmayıldır" => parent_name=İsmayıl
"Aygün xanım" => parent_name=Aygün, parent_title=xanım
"Salam, anasıyam Afət" => parent_name=Afət, parent_title=xanım
"adı Ayxandır" => child_name=Ayxan

Valideyn və uşaq adını qarışdırma. "mənəm" valideynə, "uşağın adı/övladım/oğlum/qızım" uşağa aiddir.

F) STATE HAQQINDA SUALLAR

Aşağıdakılar FAQ deyil, state_question intentidir:
- "Adımı qeyd etdiniz?"
- "Nömrəmi götürdünüz?"
- "Uşağın yaşını qeyd etdiniz?"
- "Mən nə demişdim?"
- "Məlumatlarımı qeyd etmisiniz?"

state_question_type:
parent_name, child_name, child_age, main_concern, phone,
preferred_call_time, summary və ya boş string.

G) INTENT

greeting              yalnız salamlaşma
smalltalk             təşəkkür, sağ olun, necəsiniz
permission_to_ask     "bir sual verə bilərəm?"
faq_question          proqram haqqında real sual(lar)
state_question        artıq toplanmış məlumat haqqında sual
field_answer          flow-dakı sahəyə cavab
correction            əvvəlki məlumatı düzəldir
refusal               məlumat verməkdən imtina
program_interest      proqramla maraq
registration_request  qeydiyyat/müraciət istəyi
human_agent_request   canlı əməkdaş istəyi
complaint             Junior Coaching xidmətindən narazılıq
safety_risk           özünə/başqasına zərər və təcili risk
meta_question         "niyə soruşursunuz?", "siz botsunuz?"
pause_request         "sonra danışaq"
unrelated             əlaqəsiz

"məsuliyyətsizdir", "özgüvəni zəifdir", "məqsəd və gələcək",
"ünsiyyəti zəifdir", "hamısı", "hər biri" = main_concern/field_answer,
FAQ və complaint DEYİL.

"bir sual verə bilərəm?" => permission_to_ask, is_question=false.

H) SALAMLAŞMA

Tarixçədə agent artıq salamlaşıbsa, istifadəçinin son mesajında "Salam" sözünün olması
intent-i greeting etməməlidir, əgər mesajda başqa məna/sual da varsa.
"Salam, qiymət nə qədərdir?" => faq_question, greeting deyil.

I) topic_open və resume_flow

topic_open=true:
- valideyn eyni mövzunu davam etdirir,
- follow-up edir,
- bir cavabdan sonra əlavə izah gözləyir,
- eyni sualı başqa cür yenidən soruşur.

resume_flow=true yalnız o zaman ki:
- valideynin sualı tam cavablandırıldıqdan sonra anketə qayıtmaq təbii görünür,
- cavabın ardınca maksimum BİR flow sualı vermək münasibdir.

Əgər valideyn israrla qiyməti, məkanı, proqramı və s. müzakirə edir, resume_flow=false.
Flow-u valideynin mövzusundan üstün tutma.

J) BİR NEÇƏ UŞAQ

children massivdir; hər uşaq ayrıca elementdir.
"13 və 15 yaşında iki uşağım var" => 2 uşaq.
Uşaqların sırasını valideynin dediyi sıra ilə saxla.

K) İMTİNA

"nömrə vermək istəmirəm", "demək istəmirəm" => refusal.
İmtina mətnini slot kimi yazma.
"""

    system_prompt += r"""

L) MULTI-INTENT, UNCERTAINTY AND HANDOFF
- Put every detected intent in `intents`; `intent` is only the primary one.
- Infer needs semantically from behaviour, but never make a psychological diagnosis.
- If an entity or meaning is genuinely ambiguous, set clarification_needed=true
  and ask one short clarification_question instead of guessing.
- Put objections/decision blockers in `objections`.
- Set handoff_required=true for explicit human requests, safety issues, facts absent
  from the approved knowledge base, or decisions that require a person.
- `reasoning_summary` must be short and must not invent facts.
"""

    user_prompt = f"""
SÖHBƏT TARİXÇƏSİ:
{build_history_text(history)}

CARİ STATE:
{json.dumps(state, ensure_ascii=False)}

KNOWLEDGE BASE NAMİZƏDLƏRİ:
{candidates_text}

VALİDEYNİN SON MESAJI:
{user_text}
"""

    try:
        response = client.chat.completions.create(
            model="gpt-4o-mini",
            temperature=0,
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt},
            ],
            response_format={
                "type": "json_schema",
                "json_schema": {
                    "name": "junior_message_analysis_v6",
                    "strict": True,
                    "schema": {
                        "type": "object",
                        "properties": {
                            "intent": {
                                "type": "string",
                                "enum": [
                                    "greeting", "smalltalk", "permission_to_ask",
                                    "faq_question", "state_question", "field_answer",
                                    "correction", "refusal", "program_interest",
                                    "registration_request", "human_agent_request",
                                    "complaint", "safety_risk", "meta_question",
                                    "pause_request", "unrelated"
                                ]
                            },
                            "intents": {
                                "type": "array",
                                "items": {"type": "string"}
                            },
                            "is_question": {"type": "boolean"},
                            "question_text": {"type": "string"},
                            "questions": {"type": "array", "items": {"type": "string"}},
                            "faq_choice": {"type": "integer"},
                            "state_question_type": {
                                "type": "string",
                                "enum": [
                                    "", "parent_name", "child_name", "child_age",
                                    "main_concern", "phone", "preferred_call_time", "summary"
                                ]
                            },
                            "topic_open": {"type": "boolean"},
                            "resume_flow": {"type": "boolean"},
                            "parent_name": {"type": "string"},
                            "parent_title": {"type": "string", "enum": ["xanım", "bəy", ""]},
                            "children": {
                                "type": "array",
                                "items": {
                                    "type": "object",
                                    "properties": {
                                        "name": {"type": "string"},
                                        "age": {"type": "integer"},
                                        "main_concern": {"type": "string"}
                                    },
                                    "required": ["name", "age", "main_concern"],
                                    "additionalProperties": False
                                }
                            },
                            "corrections": {
                                "type": "array",
                                "items": {
                                    "type": "object",
                                    "properties": {
                                        "field": {
                                            "type": "string",
                                            "enum": [
                                                "parent_name", "child_name", "child_age",
                                                "main_concern", "phone", "preferred_call_time"
                                            ]
                                        },
                                        "child_index": {"type": "integer"},
                                        "value": {"type": "string"}
                                    },
                                    "required": ["field", "child_index", "value"],
                                    "additionalProperties": False
                                }
                            },
                            "phone": {"type": "string"},
                            "preferred_call_time": {"type": "string"},
                            "multiple_children": {"type": "boolean"},
                            "children_count": {"type": "integer"},
                            "confidence": {"type": "number"},
                            "clarification_needed": {"type": "boolean"},
                            "clarification_question": {"type": "string"},
                            "ambiguity_present": {"type": "boolean"},
                            "objections": {
                                "type": "array",
                                "items": {"type": "string"}
                            },
                            "handoff_required": {"type": "boolean"},
                            "reasoning_summary": {"type": "string"}
                        },
                        "required": [
                            "intent", "intents", "is_question", "question_text", "questions",
                            "faq_choice", "state_question_type", "topic_open", "resume_flow",
                            "parent_name", "parent_title", "children", "corrections",
                            "phone", "preferred_call_time", "multiple_children",
                            "children_count", "confidence", "clarification_needed",
                            "clarification_question", "ambiguity_present", "objections",
                            "handoff_required", "reasoning_summary"
                        ],
                        "additionalProperties": False
                    }
                }
            }
        )
        data = json.loads(response.choices[0].message.content)
        if not data.get("questions") and data.get("question_text"):
            data["questions"] = [data["question_text"]]
        return data

    except Exception as exc:
        print("MESSAGE ANALYSIS ERROR:", exc)
        data = build_fallback_extraction(user_text)
        data.setdefault("questions", [user_text] if data.get("is_question") else [])
        data.setdefault("state_question_type", "")
        data.setdefault("resume_flow", False)
        return data


# Köhnə ad — geriyə uyğunluq üçün saxlanılır.
extract_message_with_llm = analyze_message


# =========================================================
# 11. MERGE EXTRACTED INFORMATION
# =========================================================

def ensure_child_slot(
    lead: dict,
    index: int,
) -> dict:

    """
    Verilmiş indeksə qədər uşaq strukturunu genişləndirir.
    """

    ensure_lead_structure(
        lead
    )

    while len(
        lead["children"]
    ) <= index:

        lead[
            "children"
        ].append(
            create_empty_child()
        )

    return lead[
        "children"
    ][index]


def apply_corrections(
    lead: dict,
    corrections: List[Dict[str, Any]],
) -> List[str]:

    """
    Valideyn səhvi düzəldəndə state YENİLƏNİR.

    V3-də bütün slotlar yalnız boş olanda yazılırdı,
    ona görə "uşağın adı Tunardır" deyəndən sonra da
    agent köhnə ad üzərindən sual verirdi.

    Qaytarır: dəyişdirilmiş sahələrin adları.
    """

    if not corrections:
        return []

    changed = []

    def record(
        field_name: str,
        old_value,
        new_value,
    ):

        """
        Yalnız REAL dəyişiklik düzəliş sayılır.

        Boş slotun ilk dəfə dolması düzəliş deyil —
        əks halda agent "Düzəldim" deyib valideyni çaşdırır.
        """

        if old_value and old_value != new_value:

            changed.append(
                field_name
            )

    for item in corrections:

        field = item.get(
            "field"
        )

        raw_value = str(
            item.get(
                "value",
                "",
            )
        ).strip()

        if not field or not raw_value:
            continue

        index = item.get(
            "child_index",
            0,
        )

        if not isinstance(
            index,
            int,
        ) or index < 0:

            index = 0

        if field == "parent_name":

            name = clean_name(
                raw_value
            )

            if name:

                record(
                    "parent_name",
                    lead.get(
                        "parent_name"
                    ),
                    name,
                )

                lead[
                    "parent_name"
                ] = name

                title = detect_parent_title_from_role(
                    raw_value
                )

                if title:

                    lead[
                        "parent_title"
                    ] = title



        elif field == "child_name":

            name = clean_name(
                raw_value
            )

            if name:

                slot = ensure_child_slot(
                    lead,
                    index,
                )

                record(
                    "child_name",
                    slot.get(
                        "name"
                    ),
                    name,
                )

                slot[
                    "name"
                ] = name

        elif field == "child_age":

            ages = extract_all_ages(
                raw_value
            )

            if ages:

                slot = ensure_child_slot(
                    lead,
                    index,
                )

                record(
                    "child_age",
                    slot.get(
                        "age"
                    ),
                    ages[0],
                )

                slot[
                    "age"
                ] = ages[0]

        elif field == "main_concern":

            slot = ensure_child_slot(
                lead,
                index,
            )

            record(
                "main_concern",
                slot.get(
                    "main_concern"
                ),
                raw_value,
            )

            slot[
                "main_concern"
            ] = raw_value

        elif field == "phone":

            phone = normalize_phone(
                raw_value
            )

            if phone:

                record(
                    "phone",
                    lead.get(
                        "phone"
                    ),
                    phone,
                )

                lead[
                    "phone"
                ] = phone

        elif field == "preferred_call_time":

            record(
                "preferred_call_time",
                lead.get(
                    "preferred_call_time"
                ),
                raw_value,
            )

            lead[
                "preferred_call_time"
            ] = raw_value

    # Düzəlişdən sonra aktiv uşaq yenidən hesablanmalıdır.
    if changed:

        lead[
            "active_child_index"
        ] = 0

    return changed


def merge_children(
    lead: dict,
    extracted_children: List[Dict[str, Any]],
    user_text: str,
):

    """
    LLM-in qaytardığı uşaq massivini state ilə birləşdirir.

    V3-də sxem yalnız BİR uşaq daşıya bilirdi
    ("child_name", "child_age"), ona görə
    "2 uşaqdır Ayxan və Orxan" mesajında ikinci uşaq itirdi.

    V5 — sızma qoruması:
    LLM hər cavabda bütün uşaq siyahısını təkrar qaytarır və
    aktiv uşağın cavabını digərlərinə də yazır. "hamısı"
    cavabından sonra ikinci uşaq da "özgüvən" alırdı, hətta
    heç deyilməmiş yaş (15) da uydurulurdu.

    Ona görə aktiv olmayan uşaq üçün yalnız valideynin
    mesajında REAL dəlili olan məlumat qəbul edilir.
    """

    if not extracted_children:
        return

    active_index = lead.get(
        "active_child_index",
        0,
    )

    normalized_text = normalize_for_search(
        user_text
    )

    ages_in_text = extract_all_ages(
        user_text
    )

    # Aktiv uşaq üçün gələn qayğı — digər uşaqlara
    # eyni ilə köçürülübsə, bu sızmadır.
    active_concern = ""

    if active_index < len(
        extracted_children
    ):

        candidate = extracted_children[
            active_index
        ]

        if isinstance(
            candidate,
            dict,
        ):

            active_concern = normalize_for_search(
                str(
                    candidate.get(
                        "main_concern",
                        "",
                    )
                ).strip()
            )

    for index, incoming in enumerate(
        extracted_children
    ):

        if not isinstance(
            incoming,
            dict,
        ):
            continue

        # Cari sualın ünvanlandığı uşaq — cavab birbaşa
        # ona aiddir, əlavə yoxlama lazım deyil.
        is_active = (
            index == active_index
        )

        # ---- Ad ----

        name = clean_name(
            str(
                incoming.get(
                    "name",
                    "",
                )
            ).strip()
        )

        # Uydurulmuş qardaş/bacı adının qarşısını alır:
        # ad mesajda keçmirsə, qəbul edilmir.
        if name and not (
            is_active
            or normalize_for_search(
                name
            ) in normalized_text
        ):

            name = None

        # ---- Yaş ----

        age = incoming.get(
            "age",
            0,
        )

        if not (
            isinstance(age, int)
            and 1 <= age <= 99
            # Aktiv olmayan uşağın yaşı yalnız mesajda
            # həqiqətən yazılıbsa qəbul edilir.
            and (
                is_active
                or age in ages_in_text
            )
        ):

            age = None

        # ---- Qayğı ----

        concern = str(
            incoming.get(
                "main_concern",
                "",
            )
        ).strip()

        # Aktiv uşağın qayğısı olduğu kimi digərinə
        # köçürülübsə — bu ayrıca cavab deyil, sızmadır.
        if (
            concern
            and not is_active
            and active_concern
            and normalize_for_search(
                concern
            ) == active_concern
        ):

            concern = ""

        # Mövcud olmayan slot yalnız real məlumat
        # gələndə açılır — əks halda bot uydurulmuş
        # uşaq haqqında sual verməyə başlayır.
        slot_exists = index < len(
            lead["children"]
        )

        if not slot_exists and not (
            name
            or age
            or concern
        ):
            continue

        child = ensure_child_slot(
            lead,
            index,
        )

        if name and not child.get(
            "name"
        ):

            child[
                "name"
            ] = name

        if age and not child.get(
            "age"
        ):

            child[
                "age"
            ] = age

        if concern and not child.get(
            "main_concern"
        ):

            child[
                "main_concern"
            ] = expand_concern(
                concern
            )

            if concern_needs_followup(
                concern
            ):

                child[
                    "needs_concern_followup"
                ] = True

    if len(
        lead["children"]
    ) > 1:

        lead[
            "multiple_children"
        ] = True


def expand_concern(
    concern: str,
) -> str:

    """
    "hamısı" / "hər biri" -> bütün istiqamətlər.
    """

    normalized = normalize_for_search(
        concern
    )

    if normalized in {
        "hamisi",
        "her biri",
        "hamisi olsun",
        "her biri olsun",
        "hamsi",
    }:

        return (
            "özgüvən, məqsəd və gələcək, "
            "məsuliyyət və intizam, ünsiyyət"
        )

    return concern


def concern_needs_followup(
    concern: str,
) -> bool:

    normalized = normalize_for_search(
        concern
    )

    return any(
        word in normalized
        for word in [
            "fikirli",
            "ozune qapan",
            "danismir",
        ]
    )


def merge_extracted_information(
    lead: dict,
    data: dict,
    user_text: str,
) -> List[str]:

    """
    Analiz nəticəsini state-ə yazır.

    Ardıcıllıq vacibdir:
      1. düzəlişlər (overwrite edə bilər),
      2. yeni məlumatlar (yalnız boş slotlara).

    Qaytarır: düzəldilmiş sahələrin siyahısı.
    """

    ensure_lead_structure(
        lead
    )

    # ---------------------------------------------
    # 1. Düzəlişlər — overwrite icazəlidir
    # ---------------------------------------------

    corrected = apply_corrections(
        lead,
        data.get(
            "corrections",
        ) or [],
    )

    # ---------------------------------------------
    # 2. Parent name
    # ---------------------------------------------

    parent_name = str(
        data.get(
            "parent_name",
            "",
        )
    ).strip()

    deterministic_name = (
        deterministic_name_extract(
            user_text
        )
    )

    if deterministic_name:

        parent_name = (
            deterministic_name
        )

    if (
        parent_name
        and not lead.get(
            "parent_name"
        )
    ):

        parent_name = clean_name(
            parent_name
        )

        if parent_name:

            lead[
                "parent_name"
            ] = parent_name

    # Rol ifadəsi ("anasıyam") LLM-siz də başlığı verir.
    if not lead.get(
        "parent_title"
    ):

        title = (
            detect_parent_title_from_role(
                user_text
            )
            or str(
                data.get(
                    "parent_title",
                    "",
                )
            ).strip()
        )

        if title in (
            "xanım",
            "bəy",
        ):

            lead[
                "parent_title"
            ] = title

    # ---------------------------------------------
    # 3. Uşaqlar
    # ---------------------------------------------

    children_count = data.get(
        "children_count",
        0,
    )

    if (
        data.get(
            "multiple_children"
        )
        and isinstance(children_count, int)
        and children_count >= 2
    ):

        lead[
            "multiple_children"
        ] = True

        ensure_child_slot(
            lead,
            children_count - 1,
        )

    merge_children(
        lead,
        data.get(
            "children",
        ) or [],
        user_text,
    )

    # Deterministik: "13 və 15 yaşında iki uşağım var"
    ages = extract_all_ages(
        user_text
    )

    if (
        len(ages) >= 2
        and any(
            x in normalize_for_search(
                user_text
            )
            for x in [
                "usaq",
                "ovlad",
            ]
        )
    ):

        lead[
            "multiple_children"
        ] = True

        for i, age in enumerate(
            ages
        ):

            child = ensure_child_slot(
                lead,
                i,
            )

            if not child.get(
                "age"
            ):

                child[
                    "age"
                ] = age

    # ---------------------------------------------
    # 4. Phone
    # ---------------------------------------------

    phone = (
        normalize_phone(
            str(
                data.get(
                    "phone",
                    "",
                )
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

    # ---------------------------------------------
    # 5. Call time
    # ---------------------------------------------

    call_time = str(
        data.get(
            "preferred_call_time",
            "",
        )
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

    return corrected


# =========================================================
# 12. NEXT MISSING FIELD
# =========================================================

def child_is_complete(
    child: dict,
    skipped: Optional[List[str]] = None,
) -> bool:

    skipped = skipped or []

    def missing(
        field: str,
        value,
    ) -> bool:

        if field in skipped:
            return False

        return not value

    if missing(
        "child_name",
        child.get(
            "name"
        ),
    ):
        return False

    if missing(
        "child_age",
        child.get(
            "age"
        ),
    ):
        return False

    if missing(
        "main_concern",
        child.get(
            "main_concern"
        ),
    ):
        return False

    if child.get(
        "needs_concern_followup"
    ):

        if missing(
            "concern_duration",
            child.get(
                "concern_duration"
            ),
        ):
            return False

        if missing(
            "concern_onset",
            child.get(
                "concern_onset"
            ),
        ):
            return False

    return True


def advance_child_if_needed(
    lead: dict,
):

    ensure_lead_structure(
        lead
    )

    skipped = lead.get(
        "_skipped_fields",
        [],
    )

    index = lead.get(
        "active_child_index",
        0,
    )

    current = lead[
        "children"
    ][index]

    if not child_is_complete(
        current,
        skipped,
    ):
        return

    # Növbəti incomplete child
    for i, child in enumerate(
        lead["children"]
    ):

        if not child_is_complete(
            child,
            skipped,
        ):

            lead[
                "active_child_index"
            ] = i

            return


def get_next_missing_field(
    lead: dict,
):

    """
    Növbəti çatışmayan sahəni qaytarır.

    Valideynin imtina etdiyi sahələr ("_skipped_fields")
    soruşulmur — əks halda söhbət sonsuz döngəyə düşür.

    Keçilən sahə state-ə YAZILMIR, sadəcə nəzərə alınmır,
    beləliklə bazada saxta dəyər yaranmır.
    """

    ensure_lead_structure(
        lead
    )

    skipped = lead.get(
        "_skipped_fields",
        [],
    )

    def is_missing(
        field: str,
        value,
    ) -> bool:

        if field in skipped:
            return False

        return not value

    if is_missing(
        "parent_name",
        lead.get(
            "parent_name"
        ),
    ):

        return "parent_name"

    advance_child_if_needed(
        lead
    )

    child = get_active_child(
        lead
    )

    if is_missing(
        "child_name",
        child.get(
            "name"
        ),
    ):

        return "child_name"

    if is_missing(
        "child_age",
        child.get(
            "age"
        ),
    ):

        return "child_age"

    if is_missing(
        "main_concern",
        child.get(
            "main_concern"
        ),
    ):

        return "main_concern"

    if child.get(
        "needs_concern_followup"
    ):

        if is_missing(
            "concern_duration",
            child.get(
                "concern_duration"
            ),
        ):

            return "concern_duration"

        if is_missing(
            "concern_onset",
            child.get(
                "concern_onset"
            ),
        ):

            return "concern_onset"

    # Başqa natamam uşaq?
    for i, other_child in enumerate(
        lead["children"]
    ):

        if not child_is_complete(
            other_child,
            skipped,
        ):

            if i != lead.get(
                "active_child_index"
            ):

                lead[
                    "active_child_index"
                ] = i

                return get_next_missing_field(
                    lead
                )

    if is_missing(
        "phone",
        lead.get(
            "phone"
        ),
    ):

        return "phone"

    # Telefon verilməyibsə zəng vaxtının mənası yoxdur.
    if not lead.get(
        "phone"
    ):

        return None

    if is_missing(
        "preferred_call_time",
        lead.get(
            "preferred_call_time"
        ),
    ):

        return "preferred_call_time"

    return None


# =========================================================
# 13. ONE QUESTION ONLY
# =========================================================

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

    # Eyni sahə ardıcıl soruşulursa ifadə dəyişməlidir,
    # əks halda agent mexaniki təkrar edir.
    repeated = lead.get(
        "_ask_repeat_count",
        0,
    ) > 0

    if field == "parent_name":

        if repeated:

            return (
                "Adınızı qeyd edə bilərsinizmi?"
            )

        return (
            "Sizə necə müraciət edə bilərəm?"
        )

    if field == "child_name":

        if lead.get(
            "multiple_children"
        ):

            index = (
                lead.get(
                    "active_child_index",
                    0,
                )
                + 1
            )

            return (
                f"{index}-ci övladınızın adını "
                "öyrənə bilərəm?"
            )

        # Təkrar soruşulanda ifadə dəyişir.
        if repeated:

            return (
                "Övladınızın adı nədir?"
            )

        # "Məmnun oldum" yalnız BİR dəfə deyilir.
        if parent and not lead.get(
            "_parent_greeted"
        ):

            lead[
                "_parent_greeted"
            ] = True

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


# =========================================================
# 14. FIELD-SPECIFIC FALLBACK SAVE
# =========================================================

def save_current_field_fallback(
    lead: dict,
    field: str,
    user_text: str,
):

    """
    LLM slot çıxarmasa belə cari suala verilən
    sadə cavabı itirmirik.

    Amma hər mətn qəbul edilmir — "Dedim yuxarıda" kimi
    ifadələr ad kimi yadda saxlanmamalıdır.
    """

    child = get_active_child(
        lead
    )

    value = user_text.strip()

    normalized = normalize_for_search(
        value
    )

    # Sual, boş cavab və ya imtina heç bir sahəyə yazılmır.
    #
    # "men telefonla elaqe ucun uygun deyilem" ifadəsi
    # uşağın inkişaf ehtiyacı kimi yadda saxlanmamalıdır.
    if (
        not value
        or value.endswith("?")
        or is_refusal(
            value
        )
    ):
        return

    if field == "parent_name":

        name = (
            deterministic_name_extract(
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

        title = detect_parent_title_from_role(
            value
        )

        if title and not lead.get(
            "parent_title"
        ):

            lead[
                "parent_title"
            ] = title

    elif field == "child_name":

        name = clean_name(
            value
        )

        if name:

            child[
                "name"
            ] = name

    elif field == "child_age":

        ages = extract_all_ages(
            value
        )

        if len(
            ages
        ) == 1:

            child[
                "age"
            ] = ages[0]

    elif field == "main_concern":

        # Bu sahədə complaint/FAQ kimi səhv classification
        # etməmək üçün qısa development cavabları
        # birbaşa qəbul edilir.

        if normalized in NON_ANSWER_TOKENS:
            return

        child[
            "main_concern"
        ] = expand_concern(
            value
        )

        if concern_needs_followup(
            value
        ):

            child[
                "needs_concern_followup"
            ] = True

    elif field == "concern_duration":

        if normalized in NON_ANSWER_TOKENS:
            return

        child[
            "concern_duration"
        ] = value

    elif field == "concern_onset":

        if normalized in NON_ANSWER_TOKENS:
            return

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

        if normalized in NON_ANSWER_TOKENS:
            return

        lead[
            "preferred_call_time"
        ] = value

    sync_flat_fields(
        lead
    )


# =========================================================
# 15. SPECIAL QUESTION ANSWERS
# =========================================================

def answer_special_question(
    user_text: str,
    lead: dict,
) -> Optional[str]:

    if is_child_presence_question(
        user_text
    ):

        return (
            "İlkin zəng zamanı övladınızın iştirakı vacib deyil."
        )

    if is_contact_here_question(
        user_text
    ):

        return (
            "Bəli 😊 Buradan müraciətinizi qeyd edə bilərsiniz. "
            "Məlumatlar tamamlandıqdan sonra Junior Coaching "
            "komandası sizinlə əlaqə saxlayacaq."
        )

    if is_call_time_question(
        user_text
    ):

        return (
            "Bəli, mümkündür 😊 Sizə uyğun gün və saat "
            "nəzərə alınaraq əlaqə saxlanıla bilər."
        )

    if is_bot_question(
        user_text
    ):

        return (
            "Mən Junior Coaching proqramı üzrə virtual "
            "müraciət köməkçisiyəm 😊"
        )

    return None


# =========================================================
# 16. FAQ ANSWER + ONE NEXT QUESTION
# =========================================================

def history_has_greeting(
    history: Optional[List[Dict[str, str]]],
) -> bool:
    """Agent artıq salamlaşıbsa True."""

    if not history:
        return False

    for message in history:
        if message.get("role") != "assistant":
            continue
        value = normalize_for_search(
            message.get("content", "")
        )
        if value.startswith("salam") or "xos gelmisiniz" in value:
            return True

    return False


def answer_state_question(
    lead: dict,
    state_question_type: str,
) -> str:
    """FAQ-a getmədən artıq toplanmış state-dən cavab verir."""

    ensure_lead_structure(lead)
    parent = get_parent_display_name(lead)
    child = get_active_child(lead)

    if state_question_type == "parent_name":
        if parent:
            return f"Bəli, adınızı {parent} kimi qeyd etmişəm."
        return "Hələ adınızı qeyd etməmişəm."

    if state_question_type == "child_name":
        name = child.get("name")
        if name:
            return f"Bəli, övladınızın adını {name} kimi qeyd etmişəm."
        return "Hələ övladınızın adını qeyd etməmişəm."

    if state_question_type == "child_age":
        age = child.get("age")
        if age:
            return f"Bəli, övladınızın yaşını {age} olaraq qeyd etmişəm."
        return "Hələ övladınızın yaşını qeyd etməmişəm."

    if state_question_type == "main_concern":
        concern = child.get("main_concern")
        if concern:
            return f"Bəli, əsas ehtiyac kimi “{concern}” qeyd etmişəm."
        return "Hələ əsas ehtiyac barədə məlumat qeyd etməmişəm."

    if state_question_type == "phone":
        phone = lead.get("phone")
        if phone:
            return f"Bəli, əlaqə nömrənizi {phone} kimi qeyd etmişəm."
        return "Hələ telefon nömrənizi qeyd etməmişəm."

    if state_question_type == "preferred_call_time":
        call_time = lead.get("preferred_call_time")
        if call_time:
            return f"Bəli, uyğun zəng vaxtını “{call_time}” kimi qeyd etmişəm."
        return "Hələ zəng üçün uyğun vaxt qeyd etməmişik."

    # summary
    parts = []
    if parent:
        parts.append(f"adınız: {parent}")

    for idx, item in enumerate(lead.get("children", []), start=1):
        child_parts = []
        if item.get("name"):
            child_parts.append(f"ad: {item['name']}")
        if item.get("age"):
            child_parts.append(f"yaş: {item['age']}")
        if item.get("main_concern"):
            child_parts.append(f"ehtiyac: {item['main_concern']}")
        if child_parts:
            prefix = "övlad" if len(lead.get("children", [])) == 1 else f"{idx}-ci övlad"
            parts.append(prefix + ": " + ", ".join(child_parts))

    if lead.get("phone"):
        parts.append(f"telefon: {lead['phone']}")
    if lead.get("preferred_call_time"):
        parts.append(f"zəng vaxtı: {lead['preferred_call_time']}")

    if not parts:
        return "Hələ müraciət üzrə məlumat toplamamışıq."

    return "Bəli, indiyə qədər bunları qeyd etmişəm: " + "; ".join(parts) + "."


def generate_contextual_kb_answer(
    question: str,
    lead: dict,
    history: Optional[List[Dict[str, str]]] = None,
    faq_min_score: float = 0.18,
) -> str:
    """
    Hər REAL sual üçün ayrıca retrieval + semantic selection edir.

    TF-IDF yalnız namizəd gətirir. Son seçim LLM tərəfindən sualın
    mənasına görə edilir. Cavab yalnız knowledge base faktlarına söykənir.
    """

    special = answer_special_question(question, lead)
    if special:
        return special

    candidates = retrieve_faq_candidates(
        question,
        k=10,
        min_score=0.04,
    )

    if not candidates:
        return (
            "Bu sualla bağlı məlumat bazasında dəqiq cavab tapmadım. "
            "İstəsəniz bu sualı məsul əməkdaşa yönləndirə bilərik."
        )

    if client is None:
        best = candidates[0]
        if best["score"] >= faq_min_score:
            lead["_last_faq_score"] = best["score"]
            return best["answer"]
        return (
            "Bu sualla bağlı məlumat bazasında dəqiq cavab tapmadım. "
            "İstəsəniz bu sualı məsul əməkdaşa yönləndirə bilərik."
        )

    candidate_text = "\n\n".join(
        f"ID={item['id']}\nSUAL: {item['question']}\nCAVAB: {item['answer']}"
        for item in candidates
    )

    state = {
        "parent_name": lead.get("parent_name"),
        "parent_title": lead.get("parent_title"),
        "children": lead.get("children"),
        "phone": lead.get("phone"),
        "preferred_call_time": lead.get("preferred_call_time"),
    }

    system_prompt = r"""
Sən Junior Coaching üçün cavab seçən və təbii formada ifadə edən modulusan.

QAYDALAR:
1. İstifadəçinin sualının MƏNASINA tam uyğun knowledge-base namizədini seç.
2. Leksik oxşarlığa aldanma.
   - "harada" = məkan, "hansı gün" = gün, "neçə saat" = müddət.
   - "telefon zəngi neçə dəqiqə" = valideynlə ilkin telefon zənginin müddəti.
   - "gələ bilmədiyimiz görüşü əvəz etmək" = buraxılmış görüşün əvəzlənməsi.
   - "yalnız bir görüşə gəlmək" = birdəfəlik iştirak.
3. Dəqiq uyğun namizəd yoxdursa matched=false.
4. Heç bir fakt uydurma. Qiymət/rəqəm knowledge base-də yoxdursa özündən rəqəm demə.
5. Knowledge-base cavabının sonunda başqa sual varsa, istifadəçi həmin sualı verməyibsə
   onu mexaniki şəkildə cavaba əlavə etmə.
6. Tarixçədə eyni mövzuya artıq cavab verilib və istifadəçi yenidən soruşursa,
   eyni cümləni sözbəsöz təkrarlama. Onun narahatlığını qısa şəkildə qəbul et,
   sonra knowledge-base-də olan faktı fərqli və təbii formada izah et.
7. Xüsusilə qiymət mövzusunda: knowledge base dəqiq rəqəm vermirsə bunu açıq de.
   İstifadəçi ikinci dəfə büdcə üçün təxmini rəqəm istəyirsə, yenə saxta rəqəm demə;
   dəqiq/təxmini rəqəmin bu məlumat bazasında olmadığını izah et və məsul əməkdaşdan
   dəqiqləşdirmə təklif et. Eyni flow sualını təkrarlama.
8. Cavab 1-3 qısa abzas olsun. Yeni anket sualı yaratma.
"""

    user_prompt = f"""
SÖHBƏT TARİXÇƏSİ:
{build_history_text(history)}

CARİ STATE:
{json.dumps(state, ensure_ascii=False)}

İSTİFADƏÇİNİN SUALI:
{question}

KNOWLEDGE BASE NAMİZƏDLƏRİ:
{candidate_text}
"""

    try:
        response = client.chat.completions.create(
            model="gpt-4o-mini",
            temperature=0.15,
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt},
            ],
            response_format={
                "type": "json_schema",
                "json_schema": {
                    "name": "junior_contextual_answer_v6",
                    "strict": True,
                    "schema": {
                        "type": "object",
                        "properties": {
                            "matched": {"type": "boolean"},
                            "selected_id": {"type": "integer"},
                            "answer": {"type": "string"}
                        },
                        "required": ["matched", "selected_id", "answer"],
                        "additionalProperties": False
                    }
                }
            }
        )
        result = json.loads(response.choices[0].message.content)

        if result.get("matched"):
            selected_id = result.get("selected_id", -1)
            for item in candidates:
                if item["id"] == selected_id:
                    lead["_last_faq_score"] = item["score"]
                    break
            answer = str(result.get("answer", "")).strip()
            if answer:
                return answer

    except Exception as exc:
        print("CONTEXTUAL FAQ ERROR:", exc)

    # Təhlükəsiz fallback: yalnız yüksək confidence namizədi qaytar.
    best = candidates[0]
    if best["score"] >= faq_min_score:
        lead["_last_faq_score"] = best["score"]
        return best["answer"]

    return (
        "Bu sualla bağlı məlumat bazasında dəqiq cavab tapmadım. "
        "İstəsəniz bu sualı məsul əməkdaşa yönləndirə bilərik."
    )


def answer_user_question(
    user_text: str,
    lead: dict,
    faq_min_score: float,
    data: Optional[dict] = None,
    faq_candidates: Optional[List[Dict[str, Any]]] = None,
    history: Optional[List[Dict[str, str]]] = None,
) -> str:
    """Bir mesajdakı bütün sualları ayrı-ayrılıqda cavablandırır."""

    data = data or {}

    if data.get("intent") == "state_question":
        return answer_state_question(
            lead,
            data.get("state_question_type", "summary") or "summary",
        )

    questions = [
        str(q).strip()
        for q in (data.get("questions") or [])
        if str(q).strip()
    ]

    if not questions and data.get("question_text"):
        questions = [str(data["question_text"]).strip()]

    if not questions:
        questions = [user_text]

    answers = []
    seen = set()

    for question in questions[:4]:
        normalized = normalize_for_search(question)
        if not normalized or normalized in seen:
            continue
        seen.add(normalized)

        answer = generate_contextual_kb_answer(
            question=question,
            lead=lead,
            history=history,
            faq_min_score=faq_min_score,
        )
        if answer:
            answers.append(answer)

    if not answers:
        return (
            "Bu sualla bağlı məlumat bazasında dəqiq cavab tapmadım. "
            "İstəsəniz bu sualı məsul əməkdaşa yönləndirə bilərik."
        )

    return "\n\n".join(answers)


# =========================================================
# 17. FINAL MESSAGE
# =========================================================

def finalize_lead(
    lead: dict,
) -> str:

    """
    Müraciəti yekunlaşdırır və DÜZGÜN statusu qoyur.

    Telefon yoxdursa "zəng edəcəyik" deyilmir —
    bunun əvəzinə valideynə əlaqə yolu təklif olunur.
    """

    if not lead.get(
        "phone"
    ):

        lead[
            "status"
        ] = "NO_CONTACT"

        parent = get_parent_display_name(
            lead
        )

        return (
            f"{parent + ', m' if parent else 'M'}əlumatlarınızı "
            "qeyd etdim ✅\n\n"
            "Telefon nömrəsi olmadığı üçün sizinlə "
            "birbaşa əlaqə saxlaya bilməyəcəyik. "
            "İstədiyiniz vaxt buradan yaza bilərsiniz — "
            "suallarınızı cavablandırmağa hazıram."
        )

    lead[
        "status"
    ] = "CALL_REQUESTED"

    lead["lead_stage"] = "callback_requested"
    lead["application_status"] = "completed"

    action = {
        "type": "create_callback",
        "status": "pending",
        "phone": lead.get("phone"),
        "preferred_call_time": lead.get("preferred_call_time"),
    }
    actions = lead.setdefault("previous_actions", [])
    if action not in actions:
        actions.append(action)

    return build_final_message(
        lead
    )


def build_final_message(
    lead: dict,
) -> str:

    parent = get_parent_display_name(
        lead
    )

    call_time = lead.get(
        "preferred_call_time"
    )

    if call_time:

        result = (
            f"Qeydə alındı ✅\n\n"
            f"{parent + ', ' if parent else ''}"
            f"{call_time} sizinlə əlaqə saxlanılması "
            f"üçün müraciətinizi qeyd etdim."
        )

    else:

        result = (
            "Qeydə alındı ✅"
        )

    result += (
        "\n\nİlkin zəng zamanı övladınızın "
        "iştirakı vacib deyil."
    )

    return result


# =========================================================
# 18. MAIN AGENT
# =========================================================

def handle_refusal(
    lead: dict,
    field: Optional[str],
) -> str:

    """
    Valideyn məlumat verməkdən imtina edəndə.

    Prinsip:
      1-ci imtina  — seçimi qəbul et, səbəbi izah et, bir dəfə soruş.
      2-ci imtina  — sahəni KEÇ, təkrar soruşma.

    İmtina şikayət deyil — lead ESCALATED olmur.
    """

    if not field:

        return (
            "Əlbəttə, bu sizin seçiminizdir 😊"
        )

    counts = lead.setdefault(
        "_refusal_counts",
        {},
    )

    counts[field] = counts.get(
        field,
        0,
    ) + 1

    # -------------------------------------------------
    # İkinci imtina — sahə keçilir
    # -------------------------------------------------

    if counts[field] >= 2:

        skipped = lead.setdefault(
            "_skipped_fields",
            [],
        )

        if field not in skipped:

            skipped.append(
                field
            )

        # Telefon yoxdursa zəng vaxtının mənası qalmır.
        if field == "phone" and (
            "preferred_call_time"
            not in skipped
        ):

            skipped.append(
                "preferred_call_time"
            )

        if field == "phone":

            return (
                "Başa düşdüm, nömrənizi qeyd etmirik 😊\n\n"
                "İstədiyiniz vaxt buradan yaza və ya "
                "Junior Coaching komandası ilə birbaşa "
                "əlaqə saxlaya bilərsiniz. "
                "Sizi maraqlandıran sualları burada "
                "cavablandırmağa davam edə bilərəm."
            )

        return (
            "Başa düşdüm, bu sualı keçirik 😊"
        )

    # -------------------------------------------------
    # Birinci imtina — səbəb izah edilir
    # -------------------------------------------------

    if field == "phone":

        # Cavab sualla bitir ki, üstünə anket sualı
        # əlavə olunmasın — əks halda "istəməsəniz
        # verməyin, nömrənizi yazın" kimi ziddiyyət yaranır.
        return (
            "Əlbəttə, bu sizin seçiminizdir 😊\n\n"
            "Nömrəni yalnız sizinlə əlaqə saxlamaq üçün "
            "istəyirik, başqa məqsədlə istifadə olunmur. "
            "İstəsəniz nömrəsiz də davam edə bilərik.\n\n"
            "Nömrənizi qeyd etmək istəyirsinizmi?"
        )

    if field == "preferred_call_time":

        return (
            "Problem deyil, uyğun vaxtı sonra da "
            "dəqiqləşdirə bilərik 😊"
        )

    return (
        "Əlbəttə, məcbur deyilsiniz 😊 "
        "İstəsəniz bu sualı keçə bilərik."
    )


def build_correction_ack(
    corrected: List[str],
) -> str:

    """
    Düzəliş edilibsə valideyn bunu görməlidir.
    """

    if not corrected:
        return ""

    if "parent_name" in corrected and len(
        set(corrected)
    ) == 1:

        return "Düzəliş üçün təşəkkür edirəm."

    return "Düzəldim, təşəkkür edirəm."


def track_asked_field(
    lead: dict,
) -> int:

    """
    Eyni sahənin neçə dəfə ardıcıl soruşulduğunu sayır.

    Bu sayğac həm keçid ifadəsini, həm də sualın
    ifadə variantını seçir — mexaniki təkrarın qarşısını alır.
    """

    field = get_next_missing_field(
        lead
    )

    previous = lead.get(
        "_last_asked_field"
    )

    if field and field == previous:

        repeat_count = lead.get(
            "_ask_repeat_count",
            0,
        ) + 1

    else:

        repeat_count = 0

    lead[
        "_last_asked_field"
    ] = field

    lead[
        "_ask_repeat_count"
    ] = repeat_count

    return repeat_count


def build_flow_bridge(
    repeat_count: int,
) -> str:

    """
    Anket sualına qayıdanda təbii keçid ifadəsi.
    """

    if repeat_count == 0:
        return ""

    if repeat_count == 1:
        return "Davam edək."

    return "Qayıdaq müraciətinizə."


def append_next_question(
    answer: str,
    lead: dict,
    with_bridge: bool = True,
) -> str:

    """
    Cavabın üstünə MAKSİMUM bir anket sualı əlavə edir.
    """

    # Knowledge base cavabı özü sualla bitirsə,
    # üstünə anket sualı əlavə etmirik —
    # valideyn eyni anda iki sual görməməlidir.
    if answer.rstrip().endswith(
        "?"
    ):
        return answer

    # Sayğac hər halda yenilənir ki, sual variantı dəyişsin.
    repeat_count = track_asked_field(
        lead
    )

    bridge = (
        build_flow_bridge(
            repeat_count
        )
        if with_bridge
        else ""
    )

    question = get_next_question(
        lead
    )

    if not question:
        return answer

    tail = (
        f"{bridge} {question}".strip()
        if bridge
        else question
    )

    if not answer:
        return tail

    return (
        answer
        + "\n\n"
        + tail
    )


def _detect_state_question_type(
    user_text: str,
) -> str:
    """Ən çox rast gəlinən state suallarını deterministik tutur."""

    value = normalize_for_search(user_text)

    if any(x in value for x in ["adimi qeyd", "adimi gotur", "adimi yaz", "adimi bilirsiz", "adimi bilirsiniz"]):
        return "parent_name"
    if any(x in value for x in ["usagin adini qeyd", "ovladimin adini qeyd", "usaqin adi yadinizda"]):
        return "child_name"
    if any(x in value for x in ["yasini qeyd", "yasi yadinizda", "yasini yazdiniz"]):
        return "child_age"
    if any(x in value for x in ["nomremi qeyd", "nomremi gotur", "telefonumu qeyd"]):
        return "phone"
    if any(x in value for x in ["vaxti qeyd", "zeng vaxtini qeyd"]):
        return "preferred_call_time"
    if any(x in value for x in ["melumatlarimi qeyd", "ne qeyd etmisiniz", "men ne demisdim", "neleri qeyd"]):
        return "summary"

    return ""



# =========================================================
# V11 ORCHESTRATION LAYER
# LLM reasons, policy decides next action.
# =========================================================

def decide_next_step_policy(lead: dict, analysis: dict) -> str:
    """
    Final business decision layer.
    LLM cannot directly force sales flow transition.
    """

    # Human ownership always wins
    if (
        analysis.get("handoff_required")
        or lead.get("handoff_status") == "assigned"
        or lead.get("owner") == "human"
        or lead.get("status") == "ESCALATED"
    ):
        return "HUMAN_HANDOFF"

    # Uncertainty blocks progression
    if analysis.get("clarification_needed"):
        return "CLARIFY"

    intent = analysis.get("intent", "")

    # Explicit ready signals only
    ready = (
        analysis.get("ready_to_proceed") is True
        and not analysis.get("ambiguity_present")
        and intent not in (
            "spouse_skepticism",
            "child_resistance",
            "decision_dependency",
        )
    )

    if ready:
        # Parent call is always before child introduction
        if not lead.get("parent_call_completed"):
            return "PARENT_CALL"

        if lead.get("child_intro_required", True):
            return "CHILD_INTRO"

        return "NEXT_SALES_STEP"

    # Never jump to booking only because lead is interested
    if intent in ("program_interest", "faq_question"):
        return "CONTINUE"

    return "CONTINUE"


def verify_analysis(analysis: dict) -> dict:
    """Normalize and validate the LLM output before it can mutate state."""
    analysis.setdefault("intents", [analysis.get("intent", "field_answer")])
    analysis.setdefault("questions", [])
    analysis.setdefault("objections", [])
    analysis.setdefault("clarification_needed", False)
    analysis.setdefault("clarification_question", "")
    analysis.setdefault("ambiguity_present", False)
    analysis.setdefault("handoff_required", False)
    analysis.setdefault("reasoning_summary", "")

    confidence = float(analysis.get("confidence") or 0)
    uncertain_entities = any(
        analysis.get(key)
        for key in ("parent_name", "children", "preferred_call_time")
    )
    if confidence < 0.45 and uncertain_entities:
        analysis["clarification_needed"] = True
        analysis["ambiguity_present"] = True
        analysis["clarification_question"] = analysis.get(
            "clarification_question"
        ) or (
            "Düzgün qeyd etməyim üçün, paylaşdığınız məlumatın kimə aid "
            "olduğunu dəqiqləşdirə bilərsiniz?"
        )
        analysis["parent_name"] = ""
        analysis["children"] = []
        analysis["preferred_call_time"] = ""

    return analysis


def update_conversation_state(lead: dict, analysis: dict) -> None:
    """Store questions/objections and expose actions for a future tool layer."""
    ensure_lead_structure(lead)

    for key in ("questions", "objections"):
        target = lead.setdefault(key, [])
        for value in analysis.get(key) or []:
            value = str(value).strip()
            if value and value not in target:
                target.append(value)

    if analysis.get("handoff_required"):
        lead["handoff_status"] = "requested"
        lead["owner"] = "human"
        if "human_handoff" not in lead["pending_actions"]:
            lead["pending_actions"].append("human_handoff")

    missing = get_next_missing_field(lead)
    if missing is None and lead.get("phone"):
        lead["lead_stage"] = "callback_requested"
        lead["application_status"] = "completed"
        if "create_callback" not in lead["pending_actions"]:
            lead["pending_actions"].append("create_callback")
    elif lead.get("phone"):
        lead["lead_stage"] = "contact_captured"
    elif any(not child_is_empty(c) for c in lead.get("children", [])):
        lead["lead_stage"] = "qualified"


def apply_orchestration_guard(reply: str, lead: dict, analysis: dict) -> str:
    """
    Prevent premature flow jumps.
    """

    action = decide_next_step_policy(lead, analysis)

    if action == "CLARIFY":
        q = analysis.get("clarification_question")
        if q:
            return q

    if action == "HUMAN_HANDOFF":
        return (
            "Müraciətinizi məsul əməkdaşa yönləndirirəm. "
            "Komandamız sizinlə əlaqə saxlayacaq."
        )

    return reply

def lead_agent_reply(
    user_text: str,
    lead: dict,
    faq_min_score: float = 0.18,
    history: Optional[List[Dict[str, str]]] = None,
) -> str:
    """
    V6 Conversation Manager.

    Prinsip:
      1. Cari mesaj + tarixçə + state birlikdə analiz olunur.
      2. Mesajdakı BÜTÜN yeni məlumatlar state-ə yazılır.
      3. Düzəlişlər köhnə state-i overwrite edir.
      4. Mesajda sual varsa əvvəl bütün suallar cavablandırılır.
      5. Flow yalnız təbii olduqda və maksimum bir sualla davam edir.
    """

    user_text = user_text.strip()
    ensure_lead_structure(lead)

    if history is None:
        history = lead.setdefault("_history", [])

    field_before = get_next_missing_field(lead)
    lead["_last_faq_score"] = None

    # -----------------------------------------------------
    # 1. İlk recall + conversation analysis
    # -----------------------------------------------------
    faq_candidates = retrieve_faq_candidates(
        user_text,
        k=12,
        min_score=0.03,
    )

    data = analyze_message(
        user_text=user_text,
        lead=lead,
        history=history,
        faq_candidates=faq_candidates,
    )
    data = verify_analysis(data)

    # Deterministik state sualı LLM səhvindən daha etibarlıdır.
    deterministic_state_question = _detect_state_question_type(user_text)
    if deterministic_state_question:
        data["intent"] = "state_question"
        data["state_question_type"] = deterministic_state_question
        data["is_question"] = True
        data["resume_flow"] = False

    lead["_last_intent"] = data.get("intent")
    lead["_last_confidence"] = data.get("confidence")

    # -----------------------------------------------------
    # 2. Bütün məlumatları merge et
    # -----------------------------------------------------
    corrected = merge_extracted_information(
        lead,
        data,
        user_text,
    )
    update_conversation_state(lead, data)

    intent = data.get("intent")
    reply = None

    # -----------------------------------------------------
    # 3. Təhlükəsizlik / handoff / real complaint
    # -----------------------------------------------------
    if intent == "safety_risk":
        lead["status"] = "ESCALATED"
        lead["handoff_status"] = "requested"
        lead["owner"] = "human"
        reply = (
            "Bu vəziyyət peşəkar və təcili diqqət tələb edə bilər. "
            "Junior Coaching tibbi və ya psixoloji təcili yardımı əvəz etmir. "
            "Müraciətinizi məsul əməkdaşa yönləndirirəm."
        )

    elif intent == "human_agent_request":
        lead["status"] = "ESCALATED"
        lead["handoff_status"] = "requested"
        lead["owner"] = "human"
        reply = (
            "Əlbəttə. Müraciətinizi məsul əməkdaşa yönləndirmək üçün qeydə aldım."
        )

    elif intent == "complaint" and field_before != "main_concern":
        lead["status"] = "ESCALATED"
        reply = (
            "Narahatlığınızı başa düşürəm. Müraciətinizi məsul əməkdaşa "
            "yönləndirmək üçün qeydə aldım."
        )

    # -----------------------------------------------------
    # 4. İmtina
    # -----------------------------------------------------
    elif intent == "refusal" or is_refusal(user_text):
        base = handle_refusal(lead, field_before)
        # İlk imtinada cavab özü sualla bitə bilər; əlavə flow sualı vermirik.
        if base.rstrip().endswith("?"):
            reply = base
        elif get_next_missing_field(lead) is not None:
            reply = append_next_question(base, lead, with_bridge=False)
        else:
            reply = base

    # -----------------------------------------------------
    # 5. Bir sual vermək üçün icazə
    # -----------------------------------------------------
    elif intent == "permission_to_ask":
        reply = "Əlbəttə, buyurun 😊"

    # -----------------------------------------------------
    # 6. State haqqında sual
    # -----------------------------------------------------
    elif intent == "state_question":
        reply = answer_state_question(
            lead,
            data.get("state_question_type", "summary") or "summary",
        )
        # State sualından sonra avtomatik flow-u məcbur etmirik.

    # -----------------------------------------------------
    # 7. Pause
    # -----------------------------------------------------
    elif intent == "pause_request":
        reply = (
            "Əlbəttə 😊 İstədiyiniz zaman buradan davam edə bilərsiniz. "
            "İndiyə qədər paylaşdığınız məlumatları nəzərə alacağam."
        )

    # -----------------------------------------------------
    # 8. Salam / smalltalk — salam yalnız bir dəfə
    # -----------------------------------------------------
    elif intent in ("greeting", "smalltalk"):
        already_greeted = history_has_greeting(history)

        if intent == "greeting":
            opening = "Salam 😊" if not already_greeted else "Buyurun 😊"
        else:
            opening = "Təşəkkür edirəm 😊"

        # Əgər sırf salam/smalltalk-dırsa bir növbəti sual verilə bilər.
        reply = append_next_question(
            opening,
            lead,
            with_bridge=False,
        )

    # -----------------------------------------------------
    # 9. main_concern-a birbaşa qısa cavab
    # -----------------------------------------------------
    elif (
        field_before == "main_concern"
        and is_direct_concern_answer(user_text)
        and not data.get("questions")
    ):
        # Analyzer artıq concern-i merge etmiş ola bilər.
        if get_next_missing_field(lead) == "main_concern":
            save_current_field_fallback(
                lead=lead,
                field="main_concern",
                user_text=user_text,
            )

        if get_next_missing_field(lead) is None:
            reply = finalize_lead(lead)
        else:
            reply = append_next_question("", lead, with_bridge=False)

    # -----------------------------------------------------
    # 10. REAL sual(lar) — ƏVVƏL hamısını cavablandır
    # -----------------------------------------------------
    elif (
        data.get("questions")
        or data.get("is_question")
        or intent in ("faq_question", "program_interest", "meta_question")
    ):
        answer = answer_user_question(
            user_text=user_text,
            lead=lead,
            faq_min_score=faq_min_score,
            data=data,
            faq_candidates=faq_candidates,
            history=history,
        )

        correction_ack = build_correction_ack(corrected)
        if correction_ack:
            answer = correction_ack + "\n\n" + answer

        # Flow yalnız analyzer bunun təbii olduğunu deyəndə qayıdır.
        # Eyni mövzuda follow-up / israr varsa qayıtmır.
        if data.get("resume_flow") and not data.get("topic_open"):
            reply = append_next_question(
                answer,
                lead,
                with_bridge=True,
            )
        else:
            reply = answer

    # -----------------------------------------------------
    # 11. Düzəliş
    # -----------------------------------------------------
    elif corrected:
        ack = build_correction_ack(corrected)
        if get_next_missing_field(lead) is None:
            reply = finalize_lead(lead)
            if ack:
                reply = ack + "\n\n" + reply
        else:
            reply = append_next_question(
                ack,
                lead,
                with_bridge=False,
            )

    # -----------------------------------------------------
    # 12. Adi field cavabı
    # -----------------------------------------------------
    if reply is None:
        field_after_merge = get_next_missing_field(lead)

        # Analyzer heç nə çıxarmayıbsa yalnız cari field üçün təhlükəsiz fallback.
        if field_before and field_before == field_after_merge:
            save_current_field_fallback(
                lead=lead,
                field=field_before,
                user_text=user_text,
            )

        if get_next_missing_field(lead) is None:
            reply = finalize_lead(lead)
        else:
            prefix = build_correction_ack(corrected)
            reply = append_next_question(
                prefix,
                lead,
                with_bridge=not bool(prefix),
            )

    # -----------------------------------------------------
    # 13. Bütün sahələr tamamlanıbsa yekunlaşdır
    # -----------------------------------------------------
    if (
        lead.get("status") == "NEW"
        and get_next_missing_field(lead) is None
    ):
        final_message = finalize_lead(lead)

        # Əgər indicə sual cavablandırılıbsa yekunu ayrıca əlavə etmək olar,
        # amma eyni final mesajı ikinci dəfə yazmırıq.
        if final_message not in (reply or ""):
            reply = f"{reply}\n\n{final_message}" if reply else final_message

    # -----------------------------------------------------
    # V11 Decision orchestration guard
    # -----------------------------------------------------
    reply = apply_orchestration_guard(reply, lead, data)

    # -----------------------------------------------------
    # 14. Tarixçə
    # -----------------------------------------------------
    history.append({"role": "user", "content": user_text})
    history.append({"role": "assistant", "content": reply})
    del history[:-MAX_HISTORY_MESSAGES]

    return reply


# =========================================================
# 19. DATABASE
# =========================================================

def child_is_empty(
    child: dict,
) -> bool:

    """
    Heç bir məlumatı olmayan slot bazaya yazılmır.
    """

    return not any(
        child.get(key)
        for key in (
            "name",
            "age",
            "main_concern",
            "concern_duration",
            "concern_onset",
        )
    )


def insert_children_rows(
    conn,
    lead_id: int,
    children: List[Dict[str, Any]],
    created_at: str,
):

    """
    Hər uşaq üçün ayrıca sətir yazır.

    children_json saxlanılır (geriyə uyğunluq üçün),
    amma əsl sorğu mənbəyi bu cədvəldir.
    """

    for index, child in enumerate(
        children or []
    ):

        if not isinstance(
            child,
            dict,
        ):
            continue

        if child_is_empty(
            child
        ):
            continue

        conn.execute(
            """
            INSERT OR REPLACE INTO children (

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

                created_at,
            ),
        )


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

                lead_id INTEGER NOT NULL,
                child_index INTEGER NOT NULL,

                name TEXT,
                age INTEGER,
                main_concern TEXT,

                needs_concern_followup INTEGER DEFAULT 0,
                concern_duration TEXT,
                concern_onset TEXT,

                created_at TEXT,

                FOREIGN KEY (lead_id)
                    REFERENCES leads(id)
                    ON DELETE CASCADE
            )
            """
        )

        conn.execute(
            """
            CREATE INDEX IF NOT EXISTS
                idx_children_lead_id
            ON children (lead_id)
            """
        )

        conn.execute(
            """
            CREATE UNIQUE INDEX IF NOT EXISTS
                idx_children_lead_slot
            ON children (lead_id, child_index)
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

        # ---------------------------------------------
        # Leads migration
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

            "children_json":
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
        # Logs migration
        # ---------------------------------------------

        existing_logs = {
            row[1]
            for row in conn.execute(
                "PRAGMA table_info(conversation_logs)"
            ).fetchall()
        }

        log_required = {

            "intent":
                "TEXT",

            "confidence":
                "REAL",

            "faq_score":
                "REAL",

            "parent_title":
                "TEXT",

            "child_age":
                "INTEGER",

            "main_concern":
                "TEXT",

            "children_json":
                "TEXT",

            "preferred_call_time":
                "TEXT",

            "source":
                "TEXT",
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

        # ---------------------------------------------
        # Children backfill
        #
        # Köhnə lead-lərdə uşaqlar yalnız children_json
        # (və ya flat sütunlar) içində qalıb. Onları
        # children cədvəlinə köçürürük ki, ikinci uşaq
        # adi SQL ilə də görünsün.
        # ---------------------------------------------

        backfill_rows = conn.execute(
            """
            SELECT
                l.id,
                l.children_json,
                l.child_name,
                l.child_age,
                l.main_concern,
                l.needs_concern_followup,
                l.concern_duration,
                l.concern_onset,
                l.created_at
            FROM leads AS l
            WHERE NOT EXISTS (
                SELECT 1
                FROM children AS c
                WHERE c.lead_id = l.id
            )
            """
        ).fetchall()

        for row in backfill_rows:

            (
                lead_id,
                raw_json,
                child_name,
                child_age,
                main_concern,
                followup,
                duration,
                onset,
                created_at,
            ) = row

            children = []

            if raw_json:

                try:

                    parsed = json.loads(
                        raw_json
                    )

                    if isinstance(
                        parsed,
                        list,
                    ):

                        children = [
                            item
                            for item in parsed
                            if isinstance(
                                item,
                                dict,
                            )
                        ]

                except (
                    ValueError,
                    TypeError,
                ):

                    children = []

            # children_json yoxdursa, flat sütunlardan
            # tək uşaq bərpa edilir.
            if not children and child_name:

                children = [
                    {
                        "name": child_name,
                        "age": child_age,
                        "main_concern": main_concern,
                        "needs_concern_followup": bool(
                            followup
                        ),
                        "concern_duration": duration,
                        "concern_onset": onset,
                    }
                ]

            insert_children_rows(
                conn,
                lead_id,
                children,
                created_at
                or get_baku_time(),
            )

        conn.commit()


# =========================================================
# 20. TIME
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
# 21. FIND LEAD
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

        if row:

            return dict(
                row
            )

    return None


# =========================================================
# 22. SAVE LEAD
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
            [],
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

        insert_children_rows(
            conn,
            lead_id,
            lead.get(
                "children",
                [],
            ),
            now,
        )

        conn.commit()

        lead["_db_id"] = lead_id

        return lead_id


def update_lead_in_db(lead: dict) -> bool:
    """Persist post-completion corrections for the current application."""
    lead_id = lead.get("_db_id")
    if not lead_id:
        return False

    ensure_lead_structure(lead)
    sync_flat_fields(lead)
    now = get_baku_time()
    children_json = json.dumps(lead.get("children", []), ensure_ascii=False)

    with sqlite3.connect(DB_PATH) as conn:
        cursor = conn.execute(
            """
            UPDATE leads SET
                parent_name=?, parent_title=?, child_name=?, child_age=?,
                main_concern=?, needs_concern_followup=?, concern_duration=?,
                concern_onset=?, children_json=?, phone=?, preferred_call_time=?,
                source=?, status=?, updated_at=?
            WHERE id=?
            """,
            (
                lead.get("parent_name"), lead.get("parent_title"),
                lead.get("child_name"), lead.get("child_age"),
                lead.get("main_concern"),
                int(bool(lead.get("needs_concern_followup", False))),
                lead.get("concern_duration"), lead.get("concern_onset"),
                children_json, lead.get("phone"),
                lead.get("preferred_call_time"), lead.get("source"),
                lead.get("status"), now, lead_id,
            ),
        )
        if cursor.rowcount != 1:
            return False
        conn.execute("DELETE FROM children WHERE lead_id = ?", (lead_id,))
        insert_children_rows(conn, lead_id, lead.get("children", []), now)
        conn.commit()
        return True


# =========================================================
# 23. SAVE CONVERSATION LOG
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
            [],
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
# 24. ADMIN HELPERS
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


def get_children_for_lead(
    lead_id: int,
):

    with sqlite3.connect(
        DB_PATH
    ) as conn:

        conn.row_factory = (
            sqlite3.Row
        )

        rows = conn.execute(
            """
            SELECT *
            FROM children
            WHERE lead_id = ?
            ORDER BY child_index
            """,
            (
                lead_id,
            ),
        ).fetchall()

        return [
            dict(row)
            for row in rows
        ]


def get_all_leads_with_children():

    """
    Hər lead-i uşaq siyahısı ilə birlikdə qaytarır.
    """

    leads = get_all_leads()

    with sqlite3.connect(
        DB_PATH
    ) as conn:

        conn.row_factory = (
            sqlite3.Row
        )

        rows = conn.execute(
            """
            SELECT *
            FROM children
            ORDER BY lead_id, child_index
            """
        ).fetchall()

    by_lead: Dict[int, List[dict]] = {}

    for row in rows:

        by_lead.setdefault(
            row["lead_id"],
            [],
        ).append(
            dict(row)
        )

    for lead in leads:

        lead[
            "children"
        ] = by_lead.get(
            lead["id"],
            [],
        )

    return leads


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
