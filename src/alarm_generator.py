import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import random

ALARM_GROUPS = {
    "tachycardia": {
        "messages": [
            "Heart rate too high",
            "Tachycardia detected",
            "HR exceeds threshold",
            "Pulse rate elevated",
            "Heart rate above 100 bpm",
            "Rapid heart rate alert",
            "HR > 120 bpm warning",
            "Increased heart rate detected",
            "Tachycardia alarm triggered",
            "High pulse rate detected",
            "Cardiac rate elevated above limit",
            "Heart rate critical: above 130",
        ],
        "severity": "HIGH",
    },
    "hypoxia": {
        "messages": [
            "O2 saturation low",
            "SpO2 below 90%",
            "Oxygen saturation critical",
            "Low blood oxygen level",
            "SpO2 dropping",
            "Oxygen saturation below normal",
            "Hypoxemia detected",
            "Blood oxygen too low",
            "SpO2 alarm: 88%",
            "Pulse oximetry low",
            "Oxygen level critical",
            "Desaturation event detected",
        ],
        "severity": "CRITICAL",
    },
    "hypotension": {
        "messages": [
            "Blood pressure dropping",
            "Low BP alert",
            "Hypotension detected",
            "Systolic BP below 90",
            "Blood pressure critically low",
            "MAP below threshold",
            "Low arterial pressure",
            "BP below lower limit",
            "Arterial hypotension alarm",
            "Systolic pressure alarm",
            "NIBP low systolic",
            "Mean arterial pressure low",
        ],
        "severity": "HIGH",
    },
    "apnea": {
        "messages": [
            "Apnea detected",
            "Breathing stopped",
            "Respiratory pause detected",
            "No breathing detected",
            "Apnea alarm triggered",
            "Respiratory arrest warning",
            "No respiratory effort detected",
            "Breathing pattern interrupted",
            "Apnea episode alarm",
            "Respiratory pause > 20 seconds",
            "Ventilation alarm - apnea",
            "Patient not breathing",
        ],
        "severity": "CRITICAL",
    },
    "hyperglycemia": {
        "messages": [
            "Blood glucose high",
            "Hyperglycemia alert",
            "Blood sugar elevated",
            "Glucose above 180 mg/dL",
            "High blood glucose level",
            "Elevated blood sugar detected",
            "BG critical high",
            "Glucose above target range",
            "Hyperglycemia detected",
            "Blood sugar above threshold",
            "High glucose alarm",
            "Glucose level elevated",
        ],
        "severity": "MEDIUM",
    },
    "disconnected_lead": {
        "messages": [
            "Lead disconnected",
            "ECG lead off",
            "Electrode detached",
            "Lead II disconnection",
            "Poor electrode contact",
            "Signal loss - lead",
            "ECG signal lost",
            "Electrode contact poor",
            "Lead detached: check patient",
            "ECG lead disconnect alarm",
            "Monitoring lead off",
            "Skin electrode detached",
        ],
        "severity": "LOW",
    },
    "bradycardia": {
        "messages": [
            "Bradycardia detected",
            "Heart rate too low",
            "Low heart rate alarm",
            "HR below threshold",
            "Bradycardia alarm triggered",
            "Heart rate below 50 bpm",
            "Slow heart rate detected",
            "Pulse rate critically low",
            "HR below lower limit",
            "Cardiac rate low",
            "Bradycardia: HR 42 bpm",
            "Heart rate below minimum",
        ],
        "severity": "HIGH",
    },
    "arrhythmia": {
        "messages": [
            "Irregular heartbeat detected",
            "Arrhythmia alarm",
            "Abnormal heart rhythm",
            "Cardiac arrhythmia detected",
            "Atrial fibrillation detected",
            "AFib alarm triggered",
            "Irregular rhythm detected",
            "Heart rhythm abnormal",
            "PVC detected",
            "Ventricular ectopy alarm",
            "Rhythm irregularity alarm",
            "Abnormal cardiac rhythm",
        ],
        "severity": "HIGH",
    },
}

SOURCES = {
    "Monitor A":   "Philips IntelliVue MX800",
    "Monitor B":   "GE CARESCAPE B650",
    "Monitor C":   "Drager Infinity Delta",
    "EHR System":  "Epic Systems",
    "IV Pump":     "BD Alaris System",
    "Ventilator":  "Puritan Bennett 980",
}

SEVERITY_WEIGHTS = {
    "tachycardia": 0.04,
    "hypoxia": 0.03,
    "hypotension": 0.04,
    "apnea": 0.03,
    "hyperglycemia": 0.28,
    "disconnected_lead": 0.46,
    "bradycardia": 0.06,
    "arrhythmia": 0.06,
}

PATIENTS = [f"Patient_{i:03d}" for i in range(1, 11)]


def generate_alarm_stream(
    n_alarms: int = 1000,
    start_time: datetime = None,
    duration_hours: int = 8,
    duplicate_rate: float = 0.65,
    seed: int = 42,
) -> pd.DataFrame:
    """
        genereaza un flux simulat de alarme ICU realist.
    :param n_alarms: numarul total de alarme
    :param start_time: momentul de start al simularii
    :param duration_hours: durata in ore
    :param duplicate_rate: proportia de alarme duplicate semantic
    :param seed: pentru reproductibilitate

    :return DataFrame cu coloanele:
        alarm_id, timestamp, message, source, source_device,
        patient_id, true_group, severity, is_duplicate
    """
    random.seed(seed)
    np.random.seed(seed)

    if start_time is None:
        start_time = datetime(2024, 1, 15, 6, 0, 0)

    total_seconds = duration_hours * 3600

    # mai multe alarme la inceput si sfarsit de tura
    timestamps = sorted([
        start_time + timedelta(seconds=random.randint(0, total_seconds))
        for _ in range(n_alarms)
    ])

    groups = list(ALARM_GROUPS.keys())
    source_names = list(SOURCES.keys())
    alarms = []

    # simulam clustere de alarme: acelasi eveniment
    # genereaza alarme din surse multiple
    i = 0
    alarm_id = 1

    while i < len(timestamps):
        weights = [SEVERITY_WEIGHTS[g] for g in groups]
        group = random.choices(groups, weights=weights, k=1)[0]

        group_data = ALARM_GROUPS[group]
        patient = random.choice(PATIENTS)

        # decide cate alarme in acest cluster (1-4)
        if random.random() < duplicate_rate:
            cluster_size = random.choices(
                [2, 3, 4], weights=[0.5, 0.3, 0.2]
            )[0]
        else:
            cluster_size = 1

        cluster_size = min(cluster_size, len(timestamps) - i)

        for j in range(cluster_size):
            if i + j >= len(timestamps):
                break

            message = random.choice(group_data["messages"])
            source = random.choice(source_names)

            alarms.append({
                "alarm_id": alarm_id,
                "timestamp": timestamps[i + j],
                "message": message,
                "source": source,
                "source_device": SOURCES[source],
                "patient_id": patient,
                "true_group": group,
                "severity": group_data["severity"],
                "is_duplicate": j > 0,
            })
            alarm_id += 1

        i += cluster_size

    df = pd.DataFrame(alarms)
    df = df.sort_values("timestamp").reset_index(drop=True)
    return df


def save_alarms(
    df: pd.DataFrame,
    path: str = "data/raw/alarms_simulated.csv",
):
    import os
    os.makedirs(os.path.dirname(path), exist_ok=True)
    df.to_csv(path, index=False)
    print(f"Salvate {len(df)} alarme in {path}")


if __name__ == "__main__":
    df = generate_alarm_stream(n_alarms=1000, seed=42)
    save_alarms(df)
    print(f"\nPrimele 5 randuri:")
    print(df.head().to_string())
    print(f"\nDistributie grupe:")
    print(df["true_group"].value_counts())
    print(f"\nDistributie severitate:")
    print(df["severity"].value_counts())