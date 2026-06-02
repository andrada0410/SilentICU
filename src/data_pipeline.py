from pathlib import Path
import os
import pandas as pd
import numpy as np
from datetime import datetime, timedelta

from src.alarm_generator import generate_alarm_stream
from src.physionet_loader import process_all_records


# grupuri semantice comune (simulate + PhysioNet)
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
    # PhysioNet Challenge 2015
    "asystole": {
        "messages": [
            "Asystole detected - no cardiac activity",
            "Cardiac arrest alarm - asystole",
            "No heartbeat detected - asystole",
            "ECG flatline detected",
            "Heart activity stopped - check patient",
            "Cardiac standstill detected",
        ],
        "severity": "CRITICAL",
    },
    "extreme_bradycardia": {
        "messages": [
            "Extreme bradycardia: HR below 20 bpm",
            "Heart rate critically low",
            "Severe bradycardia detected",
            "HR below minimum threshold",
            "Critically low heart rate detected",
            "Extreme low heart rate alarm",
        ],
        "severity": "CRITICAL",
    },
    "extreme_tachycardia": {
        "messages": [
            "Extreme tachycardia: HR above 180 bpm",
            "Heart rate dangerously high",
            "Severe tachycardia detected",
            "Tachycardia alarm - HR > 180",
            "Critically high heart rate",
            "Extreme high heart rate alarm",
        ],
        "severity": "HIGH",
    },
    "ventricular_tachycardia": {
        "messages": [
            "Ventricular tachycardia detected",
            "VT alarm triggered",
            "V-tach detected - check patient",
            "Sustained ventricular tachycardia",
            "VT episode detected",
            "Ventricular tachycardia alarm",
        ],
        "severity": "CRITICAL",
    },
    "ventricular_flutter": {
        "messages": [
            "Ventricular fibrillation detected",
            "VF alarm - immediate attention required",
            "V-fib detected",
            "Ventricular flutter alarm",
            "VF episode detected",
            "Ventricular flutter/fibrillation alarm",
        ],
        "severity": "CRITICAL",
    },
}




ROOT = Path(__file__).parent.parent  # urca din src/


# 1. Date simulate - 3 ture consecutive

# tura 1 (zi): 06:00 - 14:00 -> 600 alarme
tura1 = generate_alarm_stream(n_alarms=1000, seed=42,
            start_time=datetime(2024, 1, 15, 6, 0, 0))


df_simulat = pd.concat([tura1], ignore_index=True)
df_simulat['data_source'] = 'simulated'


# 2. Date PhysioNet - mapare la mesaje text

# Mapare tip alarma PhysioNet → mesaje din ALARM_GROUPS
PHYSIONET_TO_GROUP = {
    'Ventricular Tachycardia':    'ventricular_tachycardia',
    'Tachycardia':                'tachycardia',
    'Bradycardia':                'bradycardia',
    'Asystole':                   'asystole',
    'Ventricular Flutter Fib':    'ventricular_flutter',
}


def physionet_to_alarms(df_physionet):
    """converteste recordurile PhysioNet in format compatibil cu simulatorul."""
    rows = []
    base_time = datetime(2024, 1, 16, 6, 0, 0)
    rng = np.random.default_rng(seed=99)

    for i, row in df_physionet.iterrows():
        group_key = PHYSIONET_TO_GROUP.get(row['alarm_type'])
        if group_key is None:
            continue

        messages = ALARM_GROUPS[group_key]['messages']
        severity = ALARM_GROUPS[group_key]['severity']
        patient_id = f"PHY_Patient_{i % 10 + 1:03d}"
        base_ts = base_time + timedelta(minutes=i * 2)

        # alegem 2-3 mesaje diferite pentru acelasi eveniment clinic
        n_msgs = rng.integers(2, 4)  # 2 sau 3
        chosen = rng.choice(messages, size=min(n_msgs, len(messages)), replace=False)

        for j, msg in enumerate(chosen):
            rows.append({
                'alarm_id': f"PHY_{row['record_id']}_{j}",
                'timestamp': base_ts + timedelta(seconds=int(rng.integers(0, 180))),
                'message': msg,
                'source': 'PhysioNet_Monitor' if j == 0 else rng.choice([
                    'EHR System', 'Monitor B', 'Nurse Station'
                ]),
                'source_device': 'PhysioNet ICU Monitor',
                'patient_id': patient_id,
                'true_group': group_key,
                'severity': severity,
                'is_duplicate': j > 0,  # primul e original, restul duplicate
                'is_true_alarm': row['is_true_alarm'],
                'data_source': 'physionet',
            })

    return pd.DataFrame(rows)

df_physionet_raw = process_all_records()
df_physionet = physionet_to_alarms(df_physionet_raw)


# 3. Combinare si salvare

df_simulat['is_true_alarm'] = True  # simulatele sunt toate "reale"

df_combined = pd.concat([df_simulat, df_physionet], ignore_index=True)
df_combined = df_combined.sort_values('timestamp').reset_index(drop=True)


os.makedirs(ROOT / 'data/raw', exist_ok=True)
df_combined.to_csv(ROOT / 'data/raw/alarms_combined.csv', index=False)


print(f"Total alarme: {len(df_combined)}")
print(f"  - Simulate:  {len(df_simulat)}")
print(f"  - PhysioNet: {len(df_physionet)}")
print(f"\nDistributie sursa:")
print(df_combined['data_source'].value_counts())
print(f"\nDistributie grup:")
print(df_combined['true_group'].value_counts())