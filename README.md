# SilentICU — Semantic Alarm Deduplication Engine

> Reducerea zgomotului de alarme în unitățile ICU prin înțelegere semantică și agent AI

---

## Echipa

| Nume              |
|-------------------|
| Andrada Sigmirean |
| Carina Sturza     |

---

## Problema abordată

În unitățile de terapie intensivă (ICU), personalul medical primește aproximativ **820 de alarme per pacient pe zi**, o cifră care poate ajunge la **1503** în cazuri extreme *(Lobo et al., 2024)*. Studiile arată că **74–99% dintre acestea sunt false sau clinic nesemnificative** *(Nyarko et al., 2023)*, ducând la:

- **Zgomot excesiv** — personalul nu mai poate distinge alarmele importante
- **Alarm fatigue** — desensibilizare și răspuns întârziat la alarme critice
- **Risc clinic direct** — The Joint Commission a raportat **80 de decese** legate de alarm fatigue între 2009–2012

### De ce eșuează regulile simple?

Același eveniment clinic apare sub formulări complet diferite din surse diferite:

| Eveniment | Monitor A | Monitor B | EHR |
|---|---|---|---|
| Tahicardie | *"Heart rate too high"* | *"Tachycardia detected"* | *"HR exceeds threshold"* |
| Hipoxie | *"SpO2 below 90%"* | *"O2 saturation low"* | *"Oxygen saturation critical"* |
| Hipotensiune | *"Blood pressure low"* | *"Hypotension alert"* | *"BP drop detected"* |
| Bradicardie | *"Heart rate too low"* | *"Bradycardia detected"* | *"Low HR warning"* |

Un sistem bazat pe **text identic** nu detectează că aceste alarme descriu același eveniment. Avem nevoie de **înțelegere semantică**.

---

## Soluția propusă — SilentICU

Un agent inteligent care:

1. **Transformă** alarmele în vectori semantici de 384 dimensiuni (Sentence-BERT / MiniLM)
2. **Detectează** alarme similare în ferestre de timp de 10 min (cosine similarity ≥ 0.80)
3. **Grupează** alarmele redundante într-un singur cluster, cu trasabilitate la sursele originale
4. **Generează** o notificare consolidată cu explicație în limbaj natural (agent Gemini 1.5 Flash cu tool use)
5. **Evaluează** performanța prin metrici KPI: rată deduplicare, false merge rate, ARI, NMI

### Schema arhitecturii

```
Surse ICU (Monitor A, Monitor B, EHR, IV Pump, Ventilator)
        ↓
   Alarme brute — "Heart rate too high", "Tachycardia detected", ...
        ↓
   Sentence-BERT (MiniLM) — text → vector de 384 dimensiuni
        ↓
   Deduplicare — similaritate ≥ 0.80 + fereastră 10 min → cluster
        ↓
   Agent Gemini — classify_severity · generate_unified_alarm · explain_grouping
        ↓
   [HIGH] Tahicardie (+2 alarme similare) | Explicație | Dashboard Streamlit
```

---

## Date folosite

| Sursă | Tip | Alarme | Descriere |
|---|---|---|---|
| Date simulate | Generat programatic | 1000 | 11 tipuri clinice, 6 surse, 10 pacienți |
| PhysioNet Challenge 2015 | Date reale ICU | 750 | 5 tipuri aritmii, etichetate de medici |
| **Total** | **combinat** | **1750** | **Dataset final de lucru** |

### Date simulate — structura unui record

| Câmp          | Tip           | Exemplu                                |
|---------------|---------------|----------------------------------------|
| `alarm_id`    | string        | `1`                                    |
| `timestamp`   | datetime      | `2024-01-15 06:00:13`                  |
| `message`     | text liber | `Poor electrode contact`               |
| `source_type` | enum          | `IV Pump`, `EHR System`, `Monitor C`   |
| `patient_id` | string | `Patient_009`                          |
| `alarm_type` | string | `disconnected_lead`                    |
| `severity`    | enum          | `LOW` / `MEDIUM` / `HIGH` / `CRITICAL` |
| `is_duplicate` | bool | `True` / `False` |
| `data_source` | enum | `simulated` / `physionet` |
| `is_true_alarm` | bool | `True` / `False` |

### Tipuri de variație semantică injectată

| Tip | Exemplu |
|---|---|
| Sinonimie directă | *"Tachycardia"* vs. *"High Heart Rate"* |
| Terminologie clinică vs. laică | *"HR exceeds threshold"* vs. *"Heartbeat too fast"* |
| Abrevieri | *"SpO2 low"* vs. *"Blood oxygen saturation critical"* |
| Nivel de detaliu | *"High BP"* vs. *"Blood pressure 165/95 mmHg"* |
| Variație gramaticală | *"Tachycardia detected"* vs. *"Detected tachycardia"* |
| Alarme complet diferite (fals merge) | *"Sensor disconnected"* vs. *"Tachycardia alarm"* |

### Date reale — PhysioNet/CinC Challenge 2015

Setul conține 750 de alarme din ICU real, etichetate de experți ca **true alarm** sau **false alarm**. Deoarece datele conțin semnale fiziologice brute (ECG, PPG, BP), nu mesaje text, tipul aritmiei din header-ul `.hea` este mapat la un mesaj reprezentativ:

| Tip aritmie | Mesaj text mapat |
|---|---|
| Asystole | *"Cardiac arrest / No heartbeat detected"* |
| Extreme Bradycardia | *"Heart rate critically low / HR < 40 bpm"* |
| Extreme Tachycardia | *"Heart rate critically high / HR > 150 bpm"* |
| Ventricular Flutter/Fib | *"V-fib detected / Ventricular fibrillation"* |
| Ventricular Tachycardia | *"V-tach / Ventricular tachycardia alarm"* |

---

## Tehnologii folosite

| Componentă | Tehnologie | Rol |
|---|---|---|
| Embeddings | `sentence-transformers` (MiniLM-L6-v2) | Text → vectori semantici de 384 dim. |
| Similaritate | `scikit-learn` cosine similarity | Detectare alarme similare |
| Clustering | Threshold-based + fereastră timp | Grupare alarme redundante |
| Agent LLM | Gemini 1.5 Flash (Google AI) | Alarmă unificată + explicații |
| Evaluare | `scikit-learn` metrics (ARI, NMI) | Măsurare performanță clusterizare |
| Dashboard | Streamlit | Vizualizare rezultate |

---


## Obiective
 
### Obiective tehnice
 
1. Implementarea unui pipeline de ingestie și normalizare a alarmelor din surse multiple simulate
2. Transformarea mesajelor de alarmă în vectori semantici folosind Sentence-BERT (MiniLM-L6-v2)
3. Calculul similarității cosinus și gruparea alarmelor cu similaritate ≥ 0.80 în ferestre de 10 minute
4. Generarea automată a unei alarme consolidate cu sumar, surse originale și prioritate, prin agent LLM
5. Integrarea unui agent Gemini 1.5 Flash cu tool use: `classify_severity`, `generate_unified_alarm`, `explain_grouping`
6. Construirea unui dashboard Streamlit cu alarme brute, alarme grupate și metrici
7. Evaluarea performanței cu ARI, NMI și false merge rate

### Obiective de cercetare
 
1. Compararea performanței mai multor modele de embeddings pe același set de date
2. Analiza influenței pragului de similaritate (0.70, 0.80, 0.85, 0.90) asupra calității grupării
3. Investigarea cazurilor de false merge și strategii de minimizare
4. Evaluarea utilității practice a explicațiilor generate de LLM în context clinic

---

## Limite și ipoteze
 
### Limite
 
- Datele de alarme sunt simulate sau provin din PhysioNet Challenge 2015 — nu din sisteme clinice live
- Nu se procesează semnale fiziologice brute (ECG, SpO2 raw) — doar textul mesajelor de alarmă

### Ipoteze
 
- Mesajele de alarmă cu similaritate semantică ≥ 0.80 descriu același eveniment clinic
- Modelele pre-antrenate de embeddings captează suficient de bine semantica domeniului medical general
- Limba de lucru este engleza (standardul alarmelor medicale internaționale)
- Prioritatea alarmei consolidate este determinată de alarma cu severitatea maximă din grup
- Fereastra de timp de 10 minute și pragul de 0.80 sunt configurabile, nu fixe

---



## Instalare și rulare

```bash
# 1. Clonează repo-ul
git clone https://github.com/LauraDiosan-CS/projects-silenticu.git
cd projects-silenticu

# 2. Creează mediul virtual
python -m venv .venv
source .venv/bin/activate  # Mac/Linux
# .venv\Scripts\activate   # Windows

# 3. Instalează dependențele
pip install -r requirements.txt

# 4. Configurează cheia API
cp .env.example .env
# editează .env și adaugă GOOGLE_API_KEY=...

# 5. Generează datele
python src/load_data.py

# 6. Rulează dashboard-ul
streamlit run app/dashboard.py
```

---

## Surse

- Uçak, S. et al. (2025). *Nurses' Alarm Fatigue Levels in Adult ICUs.* Journal of Clinical Nursing. doi:10.1111/jocn.17644
- Wunderlich, C. et al. (2025). *Assessment of alarm fatigue among ICU nurses.* BMC Nursing. PMC12406432
- Nyarko, D. et al. (2023). *Alarm fatigue in ICU — prevalence and clinical impact.*
- Lobo, et al. (2024). *Alarm burden in intensive care units — updated analysis.*
- PhysioNet/CinC Challenge 2015. *Reducing False Arousals from Life-Threatening Arrhythmias.* [physionet.org/content/challenge-2015](https://physionet.org/content/challenge-2015/1.0.0/)
- The Joint Commission (2013). *Medical Device Alarm Safety in Hospitals.* Sentinel Event Alert 50.
- AHRQ — Alarm Fatigue Primer. [psnet.ahrq.gov/primer/alarm-fatigue](https://psnet.ahrq.gov/primer/alarm-fatigue)
- Reimers, N. & Gurevych, I. (2019). *Sentence-BERT.* [sbert.net](https://www.sbert.net/)