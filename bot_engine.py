"""
Junior Coaching — Bot Engine

Funksiyalar:
- OpenAI ilə intent classification
- TF-IDF FAQ retrieval
- Lead məlumatlarının mərhələli toplanması
- Fərdiləşdirilmiş dialoq
- Valideynə xanım/bəy müraciəti
- Azərbaycan dilində ad şəkilçiləri
- SQLite lead yaddaşı
- Streamlit testləri üçün conversation log

Bu modul:
- cli.py
- app.py
- gələcək FastAPI / Instagram / WhatsApp webhook

tərəfindən import edilə bilər.
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

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity


# =========================================================
# 0. FAYL YOLLARI
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
# 1. OPENAI CLIENT
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

    # Lokal test üçün.
    # Production-da verify=False istifadə etmək tövsiyə edilmir.
    _http_client = httpx.Client(
        verify=False,
        timeout=60,
    )

    client = OpenAI(
        api_key=_api_key,
        http_client=_http_client,
    )


# =========================================================
# 2. LLM INTENT CLASSIFICATION
# =========================================================

def classify_message_with_llm(
    user_text: str,
    current_field: Optional[str],
    model: str = "gpt-4o-mini",
) -> dict:

    if client is None:
        raise RuntimeError(
            "OpenAI client yoxdur. "
            "OPENAI_API_KEY .env faylında təyin edilməyib."
        )

    system_message = """
Sən Junior Coaching üçün mesaj təsnifat modulusan.

İstifadəçi mesajını aşağıdakı intent-lərdən yalnız birinə aid et.

1. greeting

Salamlaşma mesajları:
salam, slm, hi, hello və s.

2. faq_question

Junior Coaching proqramı haqqında məlumat sualları.

Məsələn:
- proqram neçə ay davam edir?
- qiymət nə qədərdir?
- harada keçirilir?
- neçə görüş olur?
- sınaq dərsi varmı?
- təlimçi kimdir?
- qrupda neçə nəfər olur?
- bir dəfə iştirak etmək olar?
- valideynlə zəng neçə dəqiqədir?
- uşaqla tanışlıq zəngi neçə dəqiqədir?

3. field_answer

Agentin hazırda soruşduğu suala verilmiş cavabdır.

Məsələn:

current_field=parent_name
mesaj=Aygün
=> field_answer

current_field=child_name
mesaj=Leyla
=> field_answer

current_field=child_age
mesaj=14
=> field_answer

current_field=main_concern
mesaj=dərsə qulaq asmır
=> field_answer

current_field=main_concern
mesaj=fikirlidir
=> field_answer

current_field=concern_duration
mesaj=bir neçə aydır
=> field_answer

current_field=concern_onset
mesaj=məktəbdə dava olmuşdu
=> field_answer

current_field=phone
mesaj=051 373 22 44
=> field_answer

current_field=preferred_call_time
mesaj=sabah 14:00-15:00
=> field_answer

4. registration_request

İstifadəçi qeydiyyatdan keçmək və ya proqrama qoşulmaq istəyir.

5. human_agent_request

İstifadəçi canlı insanla, İsmayıl müəllimlə
və ya əməkdaşla danışmaq istəyir.

should_escalate=true olmalıdır.

6. complaint

İstifadəçi xidmətdən narazıdır, şikayət edir,
cavab almadığını bildirir və s.

should_escalate=true olmalıdır.

7. safety_risk

Özünə zərər, başqasına zərər,
ciddi təhlükə və ya təcili risk bildirir.

should_escalate=true olmalıdır.

8. unrelated

Junior Coaching və müraciət prosesi ilə əlaqəsi olmayan mesaj.

Qaydalar:

- Mesajlarda yazı səhvləri ola bilər.
- Azərbaycan dilində sadə latın yazılışı ola bilər.
- current_field varsa, istifadəçinin qısa cavabını
  mümkün qədər həmin sualın cavabı kimi qiymətləndir.
- field_answer-ları səhvən unrelated etmə.
- complaint, human_agent_request və safety_risk üçün
  should_escalate=true olmalıdır.
- Yalnız JSON schema-ya uyğun cavab ver.
"""

    user_message = f"""
Cari gözlənilən sahə:
{current_field}

İstifadəçi mesajı:
{user_text}
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
                "content": user_message,
            },
        ],
        temperature=0,
        response_format={
            "type": "json_schema",
            "json_schema": {
                "name": "junior_message_intent",
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


# =========================================================
# 3. XANIM / BƏY
# =========================================================

def infer_parent_title_with_llm(
    parent_name: str,
    model: str = "gpt-4o-mini",
) -> str:

    if not parent_name:
        return ""

    if client is None:
        return ""

    try:

        system_message = """
Sən Azərbaycan adları üçün müraciət formasını müəyyən edən modulsan.

Verilən ada əsasən aşağıdakılardan birini qaytar:

- xanım
- bəy
- neutral

Əgər addan cinsi etibarlı müəyyən etmək mümkün deyilsə,
neutral seç.

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
                    "content": f"Ad: {parent_name}",
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
            "title",
            "neutral"
        )

        if title in [
            "xanım",
            "bəy",
        ]:
            return title

        return ""

    except Exception as exc:

        print(
            "Parent title detection error:",
            exc,
        )

        return ""


# =========================================================
# 4. FAQ DATASET + TF-IDF
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

    vectorizer = TfidfVectorizer(
        lowercase=True,
        analyzer="word",
        ngram_range=(1, 2),
        max_features=50000,
    )

    matrix = vectorizer.fit_transform(
        questions
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


def retrieve_similar(
    user_query: str,
    k: int = 4,
    min_score: float = 0.10,
) -> List[Tuple[str, str, float]]:

    user_vector = vectorizer.transform(
        [user_query]
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
    min_score: float = 0.25,
):

    hits = retrieve_similar(
        user_query=user_text,
        k=1,
        min_score=min_score,
    )

    return hits[0] if hits else None


def answer_faq_question(
    user_text: str,
    min_score: float = 0.25,
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
# 5. TEXT HELPERS
# =========================================================

def normalize_text(
    text: str,
) -> str:

    return re.sub(
        r"\s+",
        " ",
        text.lower().strip(),
    )


def is_greeting(
    text: str,
) -> bool:

    text = normalize_text(
        text
    )

    text = text.strip(
        "!.,? "
    )

    greetings = {
        "salam",
        "salamlar",
        "slm",
        "salam necesiz",
        "salam necəsiz",
        "salam necesiniz",
        "salam necəsiniz",
        "salam aleykum",
        "salamun aleykum",
        "aleykum salam",
        "hello",
        "hi",
        "hey",
    }

    return text in greetings


def is_faq_question(
    user_text: str,
) -> bool:

    text = normalize_text(
        user_text
    )

    faq_keywords = [
        "proqram",
        "junior coaching",
        "coaching",
        "qiymət",
        "qiymeti",
        "ödəniş",
        "odenis",
        "endirim",
        "neçə ay",
        "nece ay",
        "neçə saat",
        "nece saat",
        "yaş qrupu",
        "yas qrupu",
        "hansı yaş",
        "hansi yas",
        "dərs",
        "ders",
        "görüş",
        "gorus",
        "məkan",
        "mekan",
        "harada",
        "sınaq",
        "sinaq",
        "sertifikat",
        "qeydiyyat",
        "təlimçi",
        "telimci",
        "ismayıl",
        "ismayil",
        "fərdi",
        "ferdi",
        "qrup",
        "onlayn",
        "canlı",
        "canli",
        "psixoloq",
        "nəticə",
        "netice",
        "tanışlıq",
        "tanisliq",
    ]

    question_words = [
        "nədir",
        "nedir",
        "necə",
        "nece",
        "nə qədər",
        "ne qeder",
        "neçə",
        "hansı",
        "hansi",
        "harada",
        "varmı",
        "varmi",
        "olurmu",
        "olarmı",
        "olarmi",
        "mümkündür",
        "mumkundur",
        "olar",
    ]

    has_topic = any(
        keyword in text
        for keyword in faq_keywords
    )

    has_question = (
        "?" in text
        or any(
            word in text
            for word in question_words
        )
    )

    return (
        has_topic
        and has_question
    )


def extract_age(
    text: str,
):

    numbers = re.findall(
        r"\d+",
        text,
    )

    if not numbers:
        return None

    return int(
        numbers[0]
    )


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


# =========================================================
# 6. AZƏRBAYCAN DİLİNDƏ AD ŞƏKİLÇİLƏRİ
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
# 7. SAFE CLASSIFIER
# =========================================================

def safe_classify_message(
    user_text: str,
    current_field: Optional[str],
) -> dict:

    try:

        return classify_message_with_llm(
            user_text=user_text,
            current_field=current_field,
        )

    except Exception as exc:

        print(
            "LLM classifier error:",
            exc,
        )

        if is_greeting(
            user_text
        ):

            intent = "greeting"

        elif is_faq_question(
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
# 8. LEAD STRUKTURU
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
    }


FIELD_QUESTIONS = {

    "parent_name": (
        "Sizə necə müraciət edə bilərəm?"
    ),

    "child_name": (
        "Övladınızın adını öyrənə bilərəm?"
    ),

    "child_age": (
        "Övladınızın neçə yaşı var?"
    ),

    "main_concern": (
        "Övladınızla bağlı hazırda sizi ən çox "
        "narahat edən məsələ nədir?"
    ),

    "concern_duration": (
        "Bu hal nə qədər müddətdir davam edir?"
    ),

    "concern_onset": (
        "Sizcə hansısa hadisədən sonra belə olub, "
        "yoxsa tədricən?"
    ),

    "phone": (
        "Sizinlə əlaqə saxlaya bilməyimiz üçün "
        "telefon nömrənizi qeyd edin, zəhmət olmasa."
    ),

    "preferred_call_time": (
        "Zəng üçün sizə hansı gün və saat aralığı "
        "daha uyğun olar?"
    ),
}


# =========================================================
# 9. VALİDEYN DISPLAY NAME
# =========================================================

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
# 10. FƏRDİLƏŞDİRİLMİŞ SUALLAR
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
            "Övladınızla bağlı hazırda sizi ən çox "
            "narahat edən məsələ nədir?"
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
            "Başa düşürəm. "
            "Sizinlə əlaqə saxlaya bilməyimiz üçün "
            "telefon nömrənizi qeyd edin, zəhmət olmasa."
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
# 11. NÖVBƏTİ MISSING FIELD
# =========================================================

def get_next_missing_field(
    lead: dict,
):

    standard_fields = [
        "parent_name",
        "child_name",
        "child_age",
        "main_concern",
    ]

    for field in standard_fields:

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
# 12. USER ANSWER VALIDATION
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

    normalized = normalize_text(
        user_text
    )


    # =====================================================
    # VALİDEYN ADI
    # =====================================================

    if field == "parent_name":

        if is_greeting(
            user_text
        ):

            return (
                False,
                "Salam 😊 Sizə necə müraciət edə bilərəm?"
            )

        if any(
            char.isdigit()
            for char in user_text
        ):

            return (
                False,
                "Ad düzgün görünmür. "
                "Zəhmət olmasa adınızı qeyd edin."
            )

        if len(
            user_text
        ) < 2:

            return (
                False,
                "Zəhmət olmasa adınızı qeyd edin."
            )

        name_pattern = (
            r"[A-Za-zƏəÖöÜüĞğÇçŞşİı\- ]+"
        )

        if not re.fullmatch(
            name_pattern,
            user_text,
        ):

            return (
                False,
                "Ad düzgün görünmür. "
                "Zəhmət olmasa yalnız adınızı qeyd edin."
            )

        parent_name = (
            user_text.title()
        )

        lead[
            "parent_name"
        ] = parent_name

        lead[
            "parent_title"
        ] = infer_parent_title_with_llm(
            parent_name
        )

        return (
            True,
            None
        )


    # =====================================================
    # UŞAQ ADI
    # =====================================================

    if field == "child_name":

        if is_greeting(
            user_text
        ):

            return (
                False,
                "Zəhmət olmasa övladınızın adını qeyd edin."
            )

        if any(
            char.isdigit()
            for char in user_text
        ):

            return (
                False,
                "Ad düzgün görünmür. "
                "Zəhmət olmasa övladınızın adını qeyd edin."
            )

        if len(
            user_text
        ) < 2:

            return (
                False,
                "Zəhmət olmasa övladınızın adını qeyd edin."
            )

        name_pattern = (
            r"[A-Za-zƏəÖöÜüĞğÇçŞşİı\- ]+"
        )

        if not re.fullmatch(
            name_pattern,
            user_text,
        ):

            return (
                False,
                "Ad düzgün görünmür. "
                "Zəhmət olmasa yalnız adı qeyd edin."
            )

        lead[
            "child_name"
        ] = user_text.title()

        return (
            True,
            None
        )


    # =====================================================
    # YAŞ
    # =====================================================

    if field == "child_age":

        age = extract_age(
            user_text
        )

        if age is None:

            return (
                False,
                "Zəhmət olmasa övladınızın yaşını "
                "rəqəmlə qeyd edin. Məsələn: 14."
            )

        if (
            age < 12
            or age > 18
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
    # ƏSAS MƏSƏLƏ
    # =====================================================

    if field == "main_concern":

        followup_answers = {

            "fikirli",
            "fikirlidir",
            "fikirli olur",
            "fikirli gəzir",
            "fikirli gezir",
            "fikirli görünür",
            "fikirli gorunur",

            "çox fikirlidir",
            "cox fikirlidir",

            "çox fikirli olur",
            "cox fikirli olur",

            "özünə qapanır",
            "ozune qapanir",

            "özünə qapanıb",
            "ozune qapanib",

            "qapalıdır",
            "qapalidir",

            "çox sakitdir",
            "cox sakitdir",

            "danışmır",
            "danismir",
        }

        vague_answers = {

            "problemi var",
            "problem var",

            "çətinlik çəkir",
            "cetinlik cekir",

            "yaxşı deyil",
            "yaxsi deyil",

            "narahatdır",
            "narahatdir",

            "bilmirəm",
            "bilmirem",

            "heç nə",
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


    # =====================================================
    # MÜDDƏT
    # =====================================================

    if field == "concern_duration":

        invalid_answers = {
            "bilmirəm",
            "bilmirem",
            "dəqiq bilmirəm",
            "deqiq bilmirem",
            "bilinmir",
        }

        if normalized in invalid_answers:

            return (
                False,
                "Təxmini müddəti qeyd edə bilərsiniz? "
                "Məsələn: bir neçə həftədir, "
                "3 aydır və ya 1 ilə yaxındır."
            )

        duration_keywords = [
            "gün",
            "gun",
            "həftə",
            "hefte",
            "ay",
            "il",
            "çoxdan",
            "coxdan",
            "bir neçə",
            "bir nece",
            "təxminən",
            "texminen",
            "uşaqlıqdan",
            "usaqliqdan",
        ]

        has_duration_keyword = any(
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
            not has_duration_keyword
            and not has_number
        ):

            return (
                False,
                "Təxmini müddəti qeyd edə bilərsiniz? "
                "Məsələn: 2 həftədir, 3 aydır "
                "və ya 1 ilə yaxındır."
            )

        lead[
            "concern_duration"
        ] = user_text

        return (
            True,
            None
        )


    # =====================================================
    # HADİSƏ / TƏDRİCƏN
    # =====================================================

    if field == "concern_onset":

        unknown_answers = {
            "bilmirəm",
            "bilmirem",
            "bilinmir",
            "xəbərim yoxdur",
            "xeberim yoxdur",
        }

        if normalized in unknown_answers:

            lead[
                "concern_onset"
            ] = user_text

            return (
                True,
                None
            )


        gradual_keywords = [
            "tədricən",
            "tedricen",
            "yavaş",
            "yavas",
            "zamanla",
            "getdikcə",
            "getdikce",
        ]

        event_keywords = [
            "hadisə",
            "hadise",
            "sonra",
            "məktəb",
            "mekteb",
            "dava",
            "ağlayıb",
            "aglayib",
            "köç",
            "koc",
            "boşan",
            "bosan",
            "imtahan",
            "dost",
            "müəllim",
            "muellim",
        ]

        has_gradual = any(
            keyword in normalized
            for keyword in gradual_keywords
        )

        has_event = any(
            keyword in normalized
            for keyword in event_keywords
        )

        if (
            not has_gradual
            and not has_event
            and len(
                user_text.split()
            ) < 2
        ):

            return (
                False,
                "Bir qədər dəqiqləşdirə bilərsiniz? "
                "Bu vəziyyət hansısa hadisədən sonra "
                "başladı, yoxsa tədricən?"
            )

        lead[
            "concern_onset"
        ] = user_text

        return (
            True,
            None
        )


    # =====================================================
    # TELEFON
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
    # ZƏNG VAXTI
    # =====================================================

    if field == "preferred_call_time":

        vague_times = {

            "sonra",

            "fərqi yoxdur",
            "ferqi yoxdur",

            "istənilən vaxt",
            "istenilen vaxt",

            "hər zaman",
            "her zaman",

            "nə vaxt olsa",
            "ne vaxt olsa",

            "istədiyiniz vaxt",
            "istediyiniz vaxt",

            "bilmirəm",
            "bilmirem",
        }

        if normalized in vague_times:

            return (
                False,
                "Zəhmət olmasa zəng üçün uyğun gün "
                "və saat aralığını bir qədər dəqiq "
                "qeyd edin. Məsələn: sabah "
                "14:00–15:00 arası."
            )


        time_keywords = [

            "bu gün",
            "bugun",

            "sabah",

            "birisi gün",
            "birisigun",

            "bazar ertəsi",
            "bazar ertesi",

            "çərşənbə",
            "cersenbe",

            "cümə",
            "cume",

            "şənbə",
            "senbe",

            "bazar",

            "həftəsonu",
            "heftesonu",

            "səhər",
            "seher",

            "günorta",
            "gunorta",

            "nahardan sonra",

            "işdən sonra",
            "isden sonra",

            "axşam",
            "axsam",
        ]


        has_time_keyword = any(
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
            not has_time_keyword
            and not has_numeric_time
        ):

            return (
                False,
                "Zəhmət olmasa zəng üçün uyğun gün "
                "və saat aralığını qeyd edin. "
                "Məsələn: sabah 14:00–15:00 arası."
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
        f"'{field}' sahəsi üçün məlumat emalı müəyyən edilməyib."
    )


# =========================================================
# 13. ƏSAS BOT FUNKSİYASI
# =========================================================

def lead_agent_reply(
    user_text: str,
    lead: dict,
    faq_min_score: float = 0.25,
) -> str:

    user_text = user_text.strip()

    current_field = (
        get_next_missing_field(
            lead
        )
    )

    classification = (
        safe_classify_message(
            user_text=user_text,
            current_field=current_field,
        )
    )

    intent = classification[
        "intent"
    ]

    # Analiz üçün lead-a yazırıq.
    lead["_last_intent"] = intent
    lead["_last_confidence"] = classification.get(
        "confidence"
    )

    print(
        "INTENT DEBUG:",
        classification,
    )


    free_text_fields = {
        "main_concern",
        "concern_duration",
        "concern_onset",
    }


    if (
        current_field in free_text_fields
        and intent == "unrelated"
    ):

        intent = "field_answer"

        lead["_last_intent"] = intent


    # =====================================================
    # GREETING
    # =====================================================

    if intent == "greeting":

        if current_field:

            if current_field == "parent_name":

                return (
                    "Salam 😊 Sizə necə müraciət edə bilərəm?"
                )

            return (
                "Salam 😊\n\n"
                + get_personalized_question(
                    current_field,
                    lead,
                )
            )

        return (
            "Salam 😊 Məlumatlarınız artıq qeydə alınıb."
        )


    # =====================================================
    # SAFETY RISK
    # =====================================================

    if intent == "safety_risk":

        lead[
            "status"
        ] = "ESCALATED"

        return (
            "Bu, təcili peşəkar diqqət tələb edən ciddi haldır. "
            "Junior Coaching psixoloji və ya tibbi yardımı "
            "əvəz etmir. Övladınız hazırda təhlükədədirsə, "
            "onu tək qoymayın və uyğun təcili yardım və "
            "psixi sağlamlıq mütəxəssisi ilə əlaqə saxlayın. "
            "Müraciət məsul əməkdaşa ötürülür."
        )


    # =====================================================
    # COMPLAINT
    # =====================================================

    if intent == "complaint":

        lead[
            "status"
        ] = "ESCALATED"

        parent_display = (
            get_parent_display_name(
                lead
            )
        )

        if parent_display:

            return (
                f"Başa düşürəm, {parent_display}. "
                "Müraciətinizi məsul əməkdaşa "
                "yönləndirmək üçün qeydə aldım."
            )

        return (
            "Başa düşürəm. Müraciətinizi məsul "
            "əməkdaşa yönləndirmək üçün qeydə aldım."
        )


    # =====================================================
    # HUMAN AGENT
    # =====================================================

    if intent == "human_agent_request":

        lead[
            "status"
        ] = "ESCALATED"

        return (
            "Əlbəttə. Müraciətinizi İsmayıl müəllimə "
            "və ya məsul əməkdaşa yönləndirmək üçün "
            "qeydə aldım."
        )


    # =====================================================
    # REGISTRATION
    # =====================================================

    if intent == "registration_request":

        if current_field:

            return (
                "Əlbəttə, qeydiyyat prosesinə "
                "davam edə bilərik.\n\n"
                + get_personalized_question(
                    current_field,
                    lead,
                )
            )

        lead[
            "status"
        ] = "CALL_REQUESTED"

        return (
            "Məlumatlarınız tamamlandı. "
            "Müraciətiniz qeydə alınır."
        )


    # =====================================================
    # FAQ
    # =====================================================

    if intent == "faq_question":

        faq_result = (
            answer_faq_question(
                user_text=user_text,
                min_score=faq_min_score,
            )
        )

        if faq_result is not None:

            faq_answer = (
                faq_result[
                    "answer"
                ]
            )

            lead["_last_faq_score"] = (
                faq_result.get(
                    "score"
                )
            )

            if current_field:

                return (
                    f"{faq_answer}\n\n"
                    + get_personalized_question(
                        current_field,
                        lead,
                    )
                )

            return (
                faq_answer
            )


        lead["_last_faq_score"] = None

        return (
            "Bu sualla bağlı məlumat bazasında "
            "dəqiq cavab tapmadım. Məlumatınızı "
            "məsul əməkdaşa yönləndirə bilərəm."
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


    success, error_message = (
        save_user_answer(
            lead=lead,
            field=current_field,
            user_text=user_text,
        )
    )


    if not success:

        return (
            error_message
        )


    next_field = (
        get_next_missing_field(
            lead
        )
    )


    # =====================================================
    # MÜRACİƏT TAMAMLANDI
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

        child_name = (
            lead.get(
                "child_name"
            )
        )

        call_time = (
            lead.get(
                "preferred_call_time"
            )
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


        return (
            final_message
        )


    return (
        get_personalized_question(
            next_field,
            lead,
        )
    )


# =========================================================
# 14. SQLITE INITIALIZATION
# =========================================================

def init_db():

    with sqlite3.connect(
        DB_PATH
    ) as conn:


        # =================================================
        # LEADS
        # =================================================

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


        # =================================================
        # CONVERSATION LOGS
        # =================================================

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


        # =================================================
        # MÖVCUD LEADS DB MIGRATION
        # =================================================

        cursor = conn.execute(
            "PRAGMA table_info(leads)"
        )

        existing_columns = {
            row[1]
            for row in cursor.fetchall()
        }


        required_columns = {

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
            column_name,
            column_type,
        ) in required_columns.items():

            if (
                column_name
                not in existing_columns
            ):

                conn.execute(
                    f"""
                    ALTER TABLE leads
                    ADD COLUMN {column_name} {column_type}
                    """
                )


        # =================================================
        # CONVERSATION LOG MIGRATION
        # =================================================

        cursor = conn.execute(
            "PRAGMA table_info(conversation_logs)"
        )

        existing_log_columns = {
            row[1]
            for row in cursor.fetchall()
        }


        required_log_columns = {

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
            column_name,
            column_type,
        ) in required_log_columns.items():

            if (
                column_name
                not in existing_log_columns
            ):

                conn.execute(
                    f"""
                    ALTER TABLE conversation_logs
                    ADD COLUMN {column_name} {column_type}
                    """
                )


        conn.commit()


# =========================================================
# 15. TIME
# =========================================================

def get_baku_time():

    return (
        datetime.now(
            ZoneInfo(
                "Asia/Baku"
            )
        ).strftime(
            "%Y-%m-%d %H:%M:%S"
        )
    )


# =========================================================
# 16. FIND EXISTING LEAD
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

        cursor = (
            conn.cursor()
        )

        cursor.execute(
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
        )


        row = (
            cursor.fetchone()
        )


        if row:

            return dict(
                row
            )


        return None


# =========================================================
# 17. SAVE LEAD
# =========================================================

def save_lead_to_db(
    lead: dict,
) -> int:

    current_time = (
        get_baku_time()
    )


    with sqlite3.connect(
        DB_PATH
    ) as conn:

        cursor = (
            conn.cursor()
        )


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

                current_time,

                current_time,
            ),
        )


        conn.commit()


        return (
            cursor.lastrowid
        )


# =========================================================
# 18. SAVE CONVERSATION LOG
# =========================================================

def save_conversation_log(
    session_id: str,
    user_message: str,
    bot_response: str,
    current_field: Optional[str],
    lead: dict,
):

    """
    Hər user -> bot mesaj cütünü SQLite-a yazır.

    Streamlit testlərindən sonra analiz etmək üçün istifadə olunacaq.
    """

    current_time = (
        get_baku_time()
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

                current_time,
            ),
        )


        conn.commit()


# =========================================================
# 19. OPTIONAL ANALYSIS HELPERS
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
# DB INITIALIZATION
# =========================================================

init_db()