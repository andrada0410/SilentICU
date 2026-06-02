"""
Grupare alarme similare semantic

Strategia: Union-Find pe perechi similare
  1. Se calculeaza cosine similarity intre toate perechile de alarme
  2. Perechile cu sim >= THRESHOLD si in aceeasi fereastra de timp (TIME_WINDOW_MIN)
     sunt conectate intr-un graf
  3. Union-Find gaseste componentele conexe → clustere finale

De ce Union-Find si nu threshold simplu?
  Daca A~B si B~C (dar A si C nu sunt direct similare), tot descriu
  acelasi eveniment clinic → trebuie sa fie în acelasi cluster.
  Union-Find propaga tranzitiv aceste conexiuni.
"""

import numpy as np
import pandas as pd
from sklearn.metrics.pairwise import cosine_similarity
from dataclasses import dataclass, field
from typing import Optional
import logging

logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
logger = logging.getLogger(__name__)

# Configurare implicita (configurabila la instantiere)
DEFAULT_THRESHOLD = 0.80  # similaritate minima pentru a considera 2 alarme duplicate
DEFAULT_TIME_WINDOW = 10  # fereastra de timp in minute


# Union-Find (Disjoint Set Union) cu path compression + rank
class UnionFind:
    """
        structura de date Union-Find pentru gruparea eficienta a nodurilor conexe.
    Exemplu:
        uf = UnionFind(5)
        uf.union(0, 1)  # 0 si 1 in acelasi grup
        uf.union(1, 2)  # 2 se alatura grupului {0, 1}
        uf.find(2) == uf.find(0)  # True
    """

    def __init__(self, n: int):
        self.parent = list(range(n))
        self.rank = [0] * n

    def find(self, x: int) -> int:
        """ gaseste radacina componentei lui x (cu path compression)."""
        if self.parent[x] != x:
            self.parent[x] = self.find(self.parent[x])
        return self.parent[x]

    def union(self, x: int, y: int) -> bool:
        """ uneste componentele lui x si y. returneaza True daca au fost diferite."""
        rx, ry = self.find(x), self.find(y)
        if rx == ry:
            return False
        # union by rank — arborele mai mic se ataseaza la cel mai mare
        if self.rank[rx] < self.rank[ry]:
            rx, ry = ry, rx
        self.parent[ry] = rx
        if self.rank[rx] == self.rank[ry]:
            self.rank[rx] += 1
        return True

    def components(self) -> dict[int, list[int]]:
        """ returneaza un dict {radacina: [indici membri]}."""
        groups: dict[int, list[int]] = {}
        for i in range(len(self.parent)):
            root = self.find(i)
            groups.setdefault(root, []).append(i)
        return groups



# Cluster — reprezentare unui grup de alarme
@dataclass
class AlarmCluster:
    """ un cluster de alarme similare semantic."""
    cluster_id: int
    indices: list[int]  # indicii din DataFrame-ul original
    alarm_ids: list[str]  # alarm_id-urile originale
    messages: list[str]  # mesajele originale
    timestamps: list[pd.Timestamp]  # timestamp-urile
    sources: list[str]  # sursele (Monitor A, EHR, etc.)
    severities: list[str]  # severităţile individuale
    is_duplicate: bool  # True daca clusterul are > 1 alarma
    representative_message: str = ""  # mesajul cel mai reprezentativ
    max_severity: str = "LOW"  # severitatea maxima din cluster

    # scorurile de similaritate ale membrilor fata de reprezentant
    similarity_scores: list[float] = field(default_factory=list)

    def __post_init__(self):
        self.max_severity = self._compute_max_severity()
        if not self.representative_message:
            self.representative_message = self.messages[0]

    def _compute_max_severity(self) -> str:
        order = {"LOW": 0, "MEDIUM": 1, "HIGH": 2, "CRITICAL": 3}
        return max(self.severities, key=lambda s: order.get(s, 0), default="LOW")

    @property
    def size(self) -> int:
        return len(self.indices)

    @property
    def time_span_minutes(self) -> float:
        if len(self.timestamps) < 2:
            return 0.0
        return (max(self.timestamps) - min(self.timestamps)).total_seconds() / 60

    def to_dict(self) -> dict:
        return {
            "cluster_id": self.cluster_id,
            "size": self.size,
            "is_duplicate": self.is_duplicate,
            "representative_message": self.representative_message,
            "max_severity": self.max_severity,
            "alarm_ids": self.alarm_ids,
            "messages": self.messages,
            "sources": self.sources,
            "time_span_minutes": round(self.time_span_minutes, 2),
            "earliest_timestamp": min(self.timestamps),
            "latest_timestamp": max(self.timestamps),
        }


# AlarmDeduplicator — clasa principala
class AlarmDeduplicator:
    """
        grupeaza alarme similare semantic folosind Union-Find.
    Pipeline:
        1. embed()     — primeste vectorii pre-calculati din AlarmEmbedder
        2. _find_pairs() — identifica perechile (sim >= threshold, in fereastra de timp)
        3. _cluster()  — aplica Union-Find pe perechi → clustere finale
        4. fit()       — ruleaza tot pipeline-ul si adauga coloanele in df

    :param threshold (float): similaritate minima cosinus (default 0.80)
    :param time_window_min (int): fereastra de timp maxima intre 2 alarme (default 10 min)
    ex complet:
        dedup = AlarmDeduplicator(threshold=0.80, time_window_min=10)
        df_result = dedup.fit(df, embeddings)
        clusters = dedup.clusters          # lista AlarmCluster
        summary  = dedup.cluster_summary() # DataFrame rezumat
    """

    def __init__(self, threshold: float = DEFAULT_THRESHOLD,
                 time_window_min: int = DEFAULT_TIME_WINDOW):
        self.threshold = threshold
        self.time_window_min = time_window_min
        self.clusters: list[AlarmCluster] = []
        self._df: Optional[pd.DataFrame] = None
        self._embeddings: Optional[np.ndarray] = None
        self._sim_matrix: Optional[np.ndarray] = None

    # calculare matrice similaritate
    def _compute_similarity(self, embeddings: np.ndarray) -> np.ndarray:
        """ calculeaza matricea NxN de cosine similarity."""
        logger.info(f"Calculare matrice similaritate pentru {len(embeddings)} alarme...")
        sim = cosine_similarity(embeddings)
        logger.info(f"Shape matrice: {sim.shape}")
        return sim

    # gasire perechi valide (sim + fereastra timp)
    def _find_pairs(self, df: pd.DataFrame,
                    sim_matrix: np.ndarray) -> list[tuple[int, int, float]]:
        """
            identifica perechile de alarme care sunt:
        - semantic similare (sim >= threshold)
        - in aceeasi fereastra de timp (|t_i - t_j| <= time_window_min)
        - de la pacienti diferiti NU sunt grupate (patient_id diferit)

        :return: lista de (i, j, similarity_score), i < j
        """
        n = len(df)
        # convertim la secunde float — functioneaza cu datetime, Timestamp si datetime64
        timestamps_s = pd.to_datetime(df["timestamp"]).astype(np.int64) / 1e9
        patient_ids = df["patient_id"].values if "patient_id" in df.columns else None
        window_s = self.time_window_min * 60  # secunde

        pairs = []
        # optimizare: iteram doar triunghiul superior (i < j)
        for i in range(n):
            for j in range(i + 1, n):
                sim = sim_matrix[i, j]
                if sim < self.threshold:
                    continue

                # verificare fereastra timp
                dt = abs(timestamps_s.iloc[j] - timestamps_s.iloc[i])
                if dt > window_s:
                    continue

                # nu grupam alarme de la pacienti diferiti
                if patient_ids is not None:
                    if patient_ids[i] != patient_ids[j]:
                        continue

                pairs.append((i, j, float(sim)))

        logger.info(f"Perechi similare gasite: {len(pairs)} "
                    f"(threshold={self.threshold}, fereastra={self.time_window_min} min)")
        return pairs

    # Union-Find -> clustere
    def _cluster(self, n: int,
                 pairs: list[tuple[int, int, float]],
                 df: pd.DataFrame) -> list[AlarmCluster]:
        """ aplica Union-Find pe perechi si construieste obiectele AlarmCluster."""
        uf = UnionFind(n)
        # retinem si scorurile de similaritate per pereche pentru analiza ulterioara
        pair_scores: dict[tuple[int, int], float] = {}

        for i, j, score in pairs:
            uf.union(i, j)
            pair_scores[(i, j)] = score

        components = uf.components()  # {radacina: [indici]}

        clusters = []
        for cluster_id, (root, indices) in enumerate(components.items()):
            rows = df.iloc[indices]

            # cel mai reprezentativ mesaj = primul in timp
            sorted_idx = rows["timestamp"].argsort()
            rep_message = rows.iloc[sorted_idx.iloc[0]]["message"]

            # scorurile de similaritate ale membrilor fata de primul (reprezentant)
            scores = []
            for idx in indices:
                key = (min(indices[0], idx), max(indices[0], idx))
                scores.append(pair_scores.get(key, 1.0 if idx == indices[0] else 0.0))

            cluster = AlarmCluster(
                cluster_id=cluster_id,
                indices=indices,
                alarm_ids=rows["alarm_id"].tolist(),
                messages=rows["message"].tolist(),
                timestamps=rows["timestamp"].tolist(),
                sources=rows["source"].tolist() if "source" in rows.columns else [],
                severities=rows["severity"].tolist() if "severity" in rows.columns else ["LOW"] * len(indices),
                is_duplicate=len(indices) > 1,
                representative_message=rep_message,
                similarity_scores=scores,
            )
            clusters.append(cluster)

        # statistici
        n_duplicate_clusters = sum(1 for c in clusters if c.is_duplicate)
        n_suppressed = sum(c.size - 1 for c in clusters if c.is_duplicate)
        logger.info(f"Clustere formate: {len(clusters)} total "
                    f"({n_duplicate_clusters} cu duplicate, "
                    f"{len(clusters) - n_duplicate_clusters} singleton)")
        logger.info(f"Alarme suprimate (reduse la reprezentant): {n_suppressed} "
                    f"din {n} ({100 * n_suppressed / n:.1f}%)")

        return clusters

    # adaugare coloane in DataFrame
    def _annotate_df(self, df: pd.DataFrame,
                     clusters: list[AlarmCluster]) -> pd.DataFrame:
        """
            adauga coloanele de deduplicare in DataFrame-ul original:
        - cluster_id: ID-ul clusterului
        - is_dedup_duplicate: True daca alarma face parte dintr-un cluster cu > 1 element
        - cluster_size: dimensiunea clusterului
        - cluster_severity: severitatea maxima a clusterului
        - is_representative: True pentru alarma reprezentanta a clusterului
        """
        df = df.copy()
        df["cluster_id"] = -1
        df["is_dedup_duplicate"] = False
        df["cluster_size"] = 1
        df["cluster_severity"] = df.get("severity", "LOW")
        df["is_representative"] = True  # implicit fiecare alarma e reprezentanta

        for cluster in clusters:
            df.loc[df.index[cluster.indices], "cluster_id"] = cluster.cluster_id
            df.loc[df.index[cluster.indices], "cluster_size"] = cluster.size
            df.loc[df.index[cluster.indices], "cluster_severity"] = cluster.max_severity

            if cluster.is_duplicate:
                df.loc[df.index[cluster.indices], "is_dedup_duplicate"] = True
                # doar prima alarma (cea mai veche) e reprezentanta
                sorted_times = df.iloc[cluster.indices]["timestamp"].argsort()
                rep_pos = cluster.indices[sorted_times.iloc[0]]
                # toate celelalte nu sunt reprezentante
                non_rep = [i for i in cluster.indices if i != rep_pos]
                df.loc[df.index[non_rep], "is_representative"] = False

        return df

    # API principal
    def fit(self, df: pd.DataFrame, embeddings: np.ndarray) -> pd.DataFrame:
        """
            ruleaza pipeline-ul complet de deduplicare.
        :param df: DataFrame cu coloanele: alarm_id, timestamp, message, source_type, severity, patient_id
        :param embeddings: matrice (N, 384) din AlarmEmbedder
        :return: df adnotat cu coloanele cluster_id, is_dedup_duplicate, etc.
        """
        assert len(df) == len(embeddings), (
            f"df ({len(df)} randuri) si embeddings ({len(embeddings)} vectori) "
            f"trebuie sa aiba aceeasi lungime!"
        )

        # reset index pentru indexare sigura
        df = df.reset_index(drop=True)
        self._df = df
        self._embeddings = embeddings

        logger.info(f"=== AlarmDeduplicator: {len(df)} alarme ===")
        logger.info(f"Threshold: {self.threshold} | Fereastra timp: {self.time_window_min} min")

        self._sim_matrix = self._compute_similarity(embeddings)
        pairs = self._find_pairs(df, self._sim_matrix)
        self.clusters = self._cluster(len(df), pairs, df)
        result_df = self._annotate_df(df, self.clusters)

        return result_df

    # Rezumat clustere
    def cluster_summary(self) -> pd.DataFrame:
        """
            returneaza un DataFrame cu un rand per cluster.
        Util pentru vizualizare si pentru agentul LLM.
        Coloane: cluster_id, size, is_duplicate, representative_message,
                 max_severity, sources, time_span_minutes, earliest_timestamp
        """
        if not self.clusters:
            raise ValueError("Ruleaza fit() mai intai!")

        rows = [c.to_dict() for c in self.clusters]
        summary = pd.DataFrame(rows)
        # sorteaza dupa dimensiune descrescator
        summary = summary.sort_values("size", ascending=False).reset_index(drop=True)
        return summary

    def duplicate_clusters(self) -> list[AlarmCluster]:
        """ returneaza doar clusterele cu mai mult de o alarma (duplicatele reale)."""
        return [c for c in self.clusters if c.is_duplicate]

    # Metrici rapide
    def deduplication_rate(self) -> float:
        """ procentul de alarme suprimate (reduse la reprezentant)."""
        if not self.clusters:
            return 0.0
        total = sum(c.size for c in self.clusters)
        suppressed = sum(c.size - 1 for c in self.clusters if c.is_duplicate)
        return suppressed / total if total > 0 else 0.0

    def stats(self) -> dict:
        """ returneaza un dict cu statistici generale despre deduplicare."""
        if not self.clusters:
            return {}
        total_alarms = sum(c.size for c in self.clusters)
        n_dup_clusters = len(self.duplicate_clusters())
        suppressed = sum(c.size - 1 for c in self.clusters if c.is_duplicate)

        return {
            "total_alarms": total_alarms,
            "total_clusters": len(self.clusters),
            "duplicate_clusters": n_dup_clusters,
            "singleton_clusters": len(self.clusters) - n_dup_clusters,
            "alarms_suppressed": suppressed,
            "deduplication_rate": round(self.deduplication_rate(), 4),
            "threshold": self.threshold,
            "time_window_min": self.time_window_min,
        }

    def __repr__(self) -> str:
        return (f"AlarmDeduplicator(threshold={self.threshold}, "
                f"time_window_min={self.time_window_min}, "
                f"clusters={len(self.clusters)})")