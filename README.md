# SilentICU — Semantic Alarm Deduplication Engine

> Reducerea zgomotului de alarme în unitățile ICU prin înțelegere semantică și agent AI

---

## Echipa

| Nume | Contribuție principală                  |
|---|-----------------------------------------|
| Andrada Sigmirean | Date, embeddings, evaluare, dashboard   |
| Carina Sturza | Date, agent LLM, deduplicare, evaluare  |

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

### Schema arhitecturii

```
┌─────────────────────────────────────────────────────────────────────┐
│                      SURSE DE ALARME ICU                            │
│  ┌──────────┐  ┌────────────┐  ┌──────────┐  ┌──────────┐  ┌──────┐ │
│  │Monitor A │  │Monitor B   │  │EHR System│  │ IV Pump  │  │ Vent │ │
│  │ Philips  │  │GE CARESCAPE│  │ Epic     │  │BD Alaris │  │PB980 │ │
│  └────┬─────┘  └────┬───────┘  └────┬─────┘  └────┬─────┘  └──┬───┘ │
└───────┼─────────────┼───────────────┼─────────────┼───────────┼─────┘
        └─────────────┴──────┬────────┴─────────────┴───────────┘
                             ↓  1000 simulate + 1878 PhysioNet = 2878 alarme
        ┌────────────────────────────────────────────────────┐
        │                 FLUX ALARME BRUTE                  │
        │  "Heart rate too high" · "Tachycardia detected"    │
        │  "HR exceeds threshold" · "SpO2 below 90%" · ...   │
        └─────────────────────┬──────────────────────────────┘
                              ↓  text → vector (384 dimensiuni)
        ┌────────────────────────────────────────────────────┐
        │         SENTENCE-BERT  (all-MiniLM-L6-v2)          │
        │   "Heart rate too high"  → [0.12, -0.45, 0.87 ...] │
        │   "Tachycardia detected" → [0.11, -0.43, 0.85 ...] │
        │               cosine similarity = 0.94  ✓          │
        │          ARI: 0.6318 (superior MPNet 0.5518)       │
        └─────────────────────┬──────────────────────────────┘
                              ↓  similaritate ≥ 0.80
                                + fereastră ≤ 10 min
                                + același patient_id
        ┌────────────────────────────────────────────────────┐
        │     DEDUPLICARE SEMANTICĂ (Union-Find)             │
        │  Cluster #42  ·  patient_id: Patient_003           │
        │    ├─ "Heart rate too high"   [Monitor A · 14:32]  │
        │    ├─ "Tachycardia detected"  [Monitor B · 14:33]  │
        │    └─ "HR exceeds threshold"  [EHR      · 14:33]   │
        │                                                    │
        │  Deduplication rate: ~68%                          │
        └─────────────────────┬──────────────────────────────┘
                              ↓  tool use
        ┌────────────────────────────────────────────────────┐
        │           AGENT GPT-4o (Azure OpenAI)              │
        │                                                    │
        │  1. classify_severity      →  HIGH                 │
        │  2. generate_unified_alarm →  notificare unificată │
        │  3. explain_grouping       →  explicație LLM       │
        └─────────────────────┬──────────────────────────────┘
                              ↓
        ┌──────────────────────────────────────────────────────────┐
        │                        OUTPUT                            │
        │  ┌─────────────────────┐  ┌────────────┐  ┌──────────┐   │
        │  │ [HIGH] Tahicardie   │  │ Explicație │  │Dashboard │   │
        │  │ 3 alarme suprimate  │  │ în limbaj  │  │Streamlit │   │
        │  │ Monitor A, B, EHR   │  │ natural    │  │+ metrici │   │
        │  └─────────────────────┘  └────────────┘  └──────────┘   │
        └──────────────────────────────────────────────────────────┘
```

### Ce face sistemul

1. **Transformă** alarmele în vectori semantici de 384 dimensiuni (Sentence-BERT / all-MiniLM-L6-v2)
2. **Detectează** alarme similare: cosine similarity ≥ 0.80, fereastră ≤ 10 min, același pacient
3. **Grupează** alarmele redundante folosind algoritmul **Union-Find**, cu trasabilitate completă
4. **Generează** o notificare consolidată cu explicație în limbaj natural (agent GPT-4o cu tool use)
5. **Evaluează** performanța prin metrici KPI: rată deduplicare, Precision/Recall/F1, ARI, NMI

---

## Date folosite

| Sursă | Tip | Alarme | Descriere |
|---|---|---|---|
| Date simulate | Generat programatic | 1000 | 11 tipuri clinice, 6 surse, 10 pacienți |
| PhysioNet Challenge 2015 | Date reale ICU | 1878 | 5 tipuri aritmii, cu variații de mesaj |
| **Total** | **combinat** | **2878** | **Dataset final de lucru** |

### Tipuri de variație semantică injectată

| Tip | Exemplu |
|---|---|
| Sinonimie directă | *"Tachycardia"* vs. *"High Heart Rate"* |
| Terminologie clinică vs. laică | *"HR exceeds threshold"* vs. *"Heartbeat too fast"* |
| Abrevieri | *"SpO2 low"* vs. *"Blood oxygen saturation critical"* |
| Nivel de detaliu | *"High BP"* vs. *"Blood pressure 165/95 mmHg"* |
| Variație gramaticală | *"Tachycardia detected"* vs. *"Detected tachycardia"* |

---

## Tehnologii folosite

| Componentă | Tehnologie | Rol |
|---|---|---|
| Embeddings | `sentence-transformers` all-MiniLM-L6-v2 | Text → vectori semantici 384 dim |
| Similaritate | `scikit-learn` cosine similarity | Detectare alarme similare |
| Clustering | Union-Find + fereastră timp + patient_id | Grupare alarme redundante |
| Agent LLM | GPT-4o via Azure OpenAI | Alarmă unificată + explicații |
| Evaluare | `scikit-learn` ARI, NMI, P/R/F1 | Măsurare performanță |
| Dashboard | Streamlit | Vizualizare rezultate |

---

## Rezultate obținute

| KPI | Valoare | Interpretare |
|---|---|---|
| Deduplication rate | ~68% | 68% din alarme consolidate |
| Precision | > 0.90 | Alarme corect identificate ca duplicate |
| Recall | > 0.85 | Duplicate reale găsite de sistem |
| ARI (MiniLM) | 0.6318 | Calitate clusterizare vs. ground truth |
| ARI (MPNet) | 0.5518 | MiniLM superior pe texte scurte ICU |

---

## Instalare și rulare

```bash
# 1. Clonează repo-ul
git clone https://github.com/LauraDiosan-CS/projects-silenticu.git
cd projects-silenticu

# 2. Creează mediul virtual
python -m venv .venv
source .venv/bin/activate   # Mac/Linux
# .venv\Scripts\activate    # Windows

# 3. Instalează dependențele
pip install -r requirements.txt

# 4. Configurează cheia API
cp .env.example .env
# editează .env și adaugă AZURE_OPENAI_API_KEY=...

# 5. Rulează notebook-urile în ordine:
#    data_evaluation.ipynb → embeddins_notebook.ipynb → agent.ipynb → evaluation.ipynb

# 6. Rulează dashboard-ul
streamlit run src/dashboard.py
```

---

## SDG-uri impactate

- **SDG 3 — Ensure healthy lives and promote well-being for all at all ages**: reducerea alarm fatigue îmbunătățește direct siguranța pacienților ICU și calitatea deciziilor clinice
- **SDG 9 — Build resilient infrastructure, promote inclusive and sustainable industrialization and foster innovation**: inovație AI aplicată în sisteme medicale critice, demonstrând că NLP modern poate reduce riscuri clinice reale

---

### LINK TEASER
https://youtu.be/Ui7bOrlZZPw

## Surse

- Uçak, S. et al. (2025). *Nurses' Alarm Fatigue Levels in Adult ICUs.* Journal of Clinical Nursing. doi:10.1111/jocn.17644
- Wunderlich, C. et al. (2025). *Assessment of alarm fatigue among ICU nurses.* BMC Nursing. PMC12406432
- Nyarko, D. et al. (2023). *Alarm fatigue in ICU — prevalence and clinical impact.*
- Lobo et al. (2024). *Alarm burden in intensive care units — updated analysis.*
- PhysioNet/CinC Challenge 2015. *Reducing False Arousals from Life-Threatening Arrhythmias.* [physionet.org/content/challenge-2015](https://physionet.org/content/challenge-2015/1.0.0/)
- The Joint Commission (2013). *Medical Device Alarm Safety in Hospitals.* Sentinel Event Alert 50.
- AHRQ — Alarm Fatigue Primer. [psnet.ahrq.gov/primer/alarm-fatigue](https://psnet.ahrq.gov/primer/alarm-fatigue)
- Reimers, N. & Gurevych, I. (2019). *Sentence-BERT.* [sbert.net](https://www.sbert.net/)
