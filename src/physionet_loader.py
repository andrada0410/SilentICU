import scipy.io
import numpy as np
import pandas as pd
from pathlib import Path

TRAINING_DIR = Path('../data/raw/physionet/training')


def load_alarms_index():
    """citeste fisierul ALARMS - sursa de adevar pentru tot dataset-ul."""
    alarms_path = TRAINING_DIR / 'ALARMS'

    rows = []
    for line in alarms_path.read_text().strip().splitlines():
        parts = line.strip().split(',')
        if len(parts) == 3:
            record_id, alarm_type, label = parts
            rows.append({
                'record_id': record_id.strip(),
                'alarm_type': alarm_type.strip().replace('_', ' '),
                # in PhysioNet Challenge 2015: 0 = true alarm, 1 = false alarm
                'is_true_alarm': label.strip() == '0',
            })

    return pd.DataFrame(rows)


def parse_hea(hea_path):
    """extrage fs, n_samples si canale din .hea"""
    with open(hea_path, 'r') as f:
        lines = f.readlines()

    header = lines[0].split()
    fs = int(header[2])
    n_samples = int(header[3])

    channels = []
    for line in lines[1:]:
        if line.startswith('#') or line.strip() == '':
            continue
        parts = line.split()
        if len(parts) >= 9:
            channels.append(parts[8])

    return fs, n_samples, channels


def load_signals(mat_path, channels):
    """incarca si scaleaza semnalele din .mat"""
    data = scipy.io.loadmat(str(mat_path))
    val = data['val'].astype(float)

    scale_map = {'II': 1477, 'V': 2834, 'PLETH': 1568, 'ABP': 54.44}

    signals = {}
    for i, ch in enumerate(channels):
        if i < val.shape[0]:
            scale = scale_map.get(ch, 1)
            signals[ch] = val[i] / scale

    return signals


def signals_to_features(signals, fs):
    """features simple din ultimele 10 secunde de semnal."""
    last_n = int(10 * fs)
    features = {}

    if 'ABP' in signals:
        abp = signals['ABP'][-last_n:]
        features['abp_mean'] = round(np.mean(abp), 1)
        features['abp_min'] = round(np.min(abp), 1)

    if 'PLETH' in signals:
        pleth = signals['PLETH'][-last_n:]
        features['pleth_mean'] = round(np.mean(pleth), 1)

    if 'II' in signals:
        ecg = signals['II'][-last_n:]
        features['ecg_std'] = round(np.std(ecg), 4)

    return features


def process_all_records():
    """proceseaza toate recordurile din ALARMS."""

    df_index = load_alarms_index()
    print(f"Gasit {len(df_index)} recorduri in ALARMS")

    rows = []
    errors = []

    for _, row in df_index.iterrows():
        name = row['record_id']
        mat_path = TRAINING_DIR / f'{name}.mat'
        hea_path = TRAINING_DIR / f'{name}.hea'

        if not mat_path.exists() or not hea_path.exists():
            errors.append(f"Lipsa fisiere: {name}")
            continue

        try:
            fs, n_samples, channels = parse_hea(hea_path)
            signals = load_signals(mat_path, channels)
            features = signals_to_features(signals, fs)

            rows.append({
                'record_id': name,
                'alarm_type': row['alarm_type'],
                'is_true_alarm': row['is_true_alarm'],
                'fs': fs,
                'duration_sec': n_samples / fs,
                'channels': ', '.join(channels),
                **features
            })

        except Exception as e:
            errors.append(f"{name}: {e}")

    df = pd.DataFrame(rows)

    out_path = TRAINING_DIR.parent / 'physionet_processed.csv'
    df.to_csv(out_path, index=False)

    # summary
    print(f"\nProcesate:  {len(rows)}")
    print(f"Erori:      {len(errors)}")
    if errors:
        for e in errors[:5]:
            print(f"  - {e}")

    print(f"\nDistributie tip alarma:")
    print(df['alarm_type'].value_counts().to_string())

    print(f"\nTrue alarm vs False alarm:")
    print(df['is_true_alarm'].value_counts().rename({True: 'True alarm', False: 'False alarm'}).to_string())

    print(f"\nSalvat: {out_path}")
    return df


if __name__ == '__main__':
    df = process_all_records()
    print(df.head())