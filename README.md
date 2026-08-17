# Junior Coaching Assistant

Junior Coaching proqramı üçün satış və müraciət (lead) köməkçisi.
Valideynlərin suallarına cavab verir, müraciət məlumatlarını toplayır və
SQLite bazasında saxlayır.

**Necə işləyir (V4):**

Hər mesaj üçün agent üç mənbəni **birlikdə** nəzərə alır:

1. **Söhbətin konteksti** — son mesajlar
2. **Toplanmış məlumatlar** — cari state (valideyn, uşaqlar, telefon)
3. **Knowledge base** — FAQ namizədləri

- **TF-IDF** — yalnız *recall* üçün: sual+cavab mətnini indeksləyib
  ən yaxın 6 namizədi verir
- **LLM (OpenAI)** — *precision* üçün: namizədlərdən mənaca uyğun olanı
  seçir, eyni anda bütün slotları çıxarır və düzəlişləri müəyyən edir
- **Qaydalar** — ad validasiyası, rol tanınması, telefon formatı
- **SQLite** — müraciətləri `junior_coaching.db` faylında saxlayır

**V4-də həll olunanlar:**

| Problem | Həll |
|---|---|
| Bir mesajdakı məlumatlar itirdi | `children` massivi — bir neçə uşaq eyni anda |
| Sualın mənası səhv tutulurdu | top-6 namizəd + LLM seçimi (tək TF-IDF nəticəsi deyil) |
| "anasıyam", "dedim yuxarıda" ad kimi yazılırdı | ad validasiyası + rol tanınması |
| Düzəliş state-i yeniləmirdi | `corrections` — mövcud dəyər üzərinə yazıla bilir |
| "bir sual verə bilərəm?" KB-də axtarılırdı | `permission_to_ask` intent |
| Anket sualı mexaniki təkrarlanırdı | `topic_open` + təkrar sayğacı və ifadə variantları |
| **Nömrə verməyəndə sonsuz döngə** | `refusal` intent — 2-ci imtinadan sonra sahə keçilir |
| **İmtina şikayət sayılıb eskalasiya olunurdu** | imtina ≠ complaint, lead `NO_CONTACT` statusu alır |
| **İmtina mətni uşağın ehtiyacı kimi yazılırdı** | sərbəst mətn sahələrində imtina filtri |
| **"hamısı" FAQ sualı sayılırdı** | `is_direct_concern_answer` — cavab kimi qəbul edilir |

**Lead statusları:**

| Status | Mənası |
|---|---|
| `CALL_REQUESTED` | Bütün məlumatlar var, zəng gözlənilir |
| `NO_CONTACT` | Valideyn nömrə vermək istəmədi — məlumat saxlanılır |
| `ESCALATED` | Canlı əməkdaşa yönləndirilib (şikayət, təhlükə, operator istəyi) |

> Qeyd: `OPENAI_API_KEY` təyin edilməsə, bot qayda əsaslı rejimə keçir və
> yenə də işləyir (LLM təsnifatı olmadan).

---

## Fayllar

| Fayl | Təyinat |
|------|---------|
| `app.py` | Streamlit chat interfeysi |
| `bot_engine.py` | Bütün məntiq (LLM, FAQ, lead, SQLite) — UI-dan asılı deyil |
| `Junior_Coaching_sesli_AI_FAQ.txt` | FAQ bazası (sual/cavab) |
| `junior_coaching.db` | Müraciətlərin saxlandığı SQLite baza |
| `test_scenarios.py` | Reqressiya testləri (müştəri rəyindəki hallar) |
| `requirements.txt` | Python asılılıqları |
| `.env` | Gizli açar: `OPENAI_API_KEY=...` (git-ə düşmür) |
| `.streamlit/secrets.toml` | Admin şifrəsi: `ADMIN_PASSWORD=...` (git-ə düşmür) |

---

## 1. Quraşdırma (bir dəfə)

```bash
cd "new bot"

# (tövsiyə olunur) virtual mühit
python3 -m venv .venv
source .venv/bin/activate        # Windows: .venv\Scripts\activate

# asılılıqları qur
pip install -r requirements.txt
```

`.env` faylında OpenAI açarınızın olduğundan əmin olun:

```
OPENAI_API_KEY=sk-...
```

Admin panel üçün şifrəni təyin edin (koda yazılmır):

```bash
cp .streamlit/secrets.toml.example .streamlit/secrets.toml
# faylı açıb ADMIN_PASSWORD dəyərini dəyişin
```

Alternativ olaraq mühit dəyişəni ilə:

```bash
export ADMIN_PASSWORD="..."
```

Şifrə təyin edilməsə, admin panel bağlı qalır.

---

## 2. İşə salmaq (Run)

```bash
streamlit run app.py
```

Brauzer avtomatik açılır. Açılmasa, bu ünvana keçin:

**http://localhost:8501**

Başqa portda işə salmaq üçün:

```bash
streamlit run app.py --server.port 8600
```

---

## 3. Dayandırmaq (Stop)

- Terminalda işləyirsə: həmin terminalda **`Ctrl + C`** basın.
- Arxa planda işləyirsə, prosesi tapıb dayandırın:

```bash
pkill -f "streamlit run app.py"
```

Dayandığını yoxlamaq üçün:

```bash
curl -s -o /dev/null -w "%{http_code}\n" http://localhost:8501/_stcore/health
# 000 qayıdırsa — server dayanıb
```

---

## Testlər

```bash
python3 test_scenarios.py           # offline (açar tələb etmir)
python3 test_scenarios.py --live    # + real LLM ssenariləri
```

`--live` müştəri rəyində qeyd olunan konkret halları yoxlayır:
bir mesajda çox məlumat, düzəliş, sualın mənası, iki uşaq.

---

## Demo ssenarisi

1. `salam` yazın → bot salamlayır və adınızı soruşur
2. Adınızı yazın, məsələn `Natiq`
3. Orta prosesdə sual verin: `proqram neçə ay davam edir?` → FAQ-dan cavab verir
4. Yaş (12–18), telefon (`050 123 45 67`), zəng vaxtı (`sabah 14:00–16:00`)
   daxil edin → müraciət yadda saxlanılır və nömrə verilir
5. Aşağıdakı **📋 Qeydə alınmış müraciətlər** bölməsində bazadakı leadləri görün
