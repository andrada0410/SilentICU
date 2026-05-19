"""
Calcul KPI-uri

Metrici calculate:
  Deduplicare:
    - deduplication_rate    : % alarme suprimate
    - false_merge_rate      : % perechi grupate incorect (alarme diferite clinic)
    - precision / recall / f1 pe detectia duplicatelor

  Clustering (vs. ground truth true_group):
    - ARI  (Adjusted Rand Index)   — robusta la clustere dezechilibrate
    - NMI  (Normalized Mutual Info) — masoara informatia comuna
    - homogeneity / completeness / v_measure

  Agent LLM:
    - severity_accuracy     : cat de des agentul da severitatea corecta
    - tool_call_rate        : % clustere unde toate 3 tool-urile au fost apelate
    - avg_processing_time   : timp mediu per cluster
"""

import json
import logging
import os
from dataclasses import dataclass, field
from typing import Optional

import numpy as np
import pandas as pd
from sklearn.metrics import (
    adjusted_rand_score,
    normalized_mutual_info_score,
    homogeneity_completeness_v_measure,
    precision_score,
    recall_score,
    f1_score,
    confusion_matrix,
)

from src.deduplicator import AlarmCluster
from src.agent import AgentResult

logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
logger = logging.getLogger(__name__)

SEVERITY_ORDER = {"LOW": 0, "MEDIUM": 1, "HIGH": 2, "CRITICAL": 3}


# Structuri de rezultate
@dataclass
class DeduplicationMetrics:
    """ metrici de deduplicare (ce a suprimar sistemul)."""
    total_alarms: int = 0
    total_clusters: int = 0
    duplicate_clusters: int = 0
    alarms_suppressed: int = 0
    deduplication_rate: float = 0.0     # alarme_suprimate / total_alarme

    # detectie duplicate (binary classification)
    precision: float = 0.0             # din ce am zis duplicate, cate chiar sunt
    recall: float = 0.0                # din toate duplicatele reale, cate am gasit
    f1: float = 0.0
    false_merge_rate: float = 0.0      # alarme diferite grupate impreuna (erori)
    true_positive: int = 0
    false_positive: int = 0
    false_negative: int = 0
    true_negative: int = 0


@dataclass
class ClusteringMetrics:
    """ metrici de calitate a clusterizarii vs. ground truth (true_group)."""
    ari: float = 0.0                   # Adjusted Rand Index [-1, 1]
    nmi: float = 0.0                   # Normalized Mutual Information [0, 1]
    homogeneity: float = 0.0           # fiecare cluster contine doar o clasa
    completeness: float = 0.0          # toate alarmele unei clase sunt in acelasi cluster
    v_measure: float = 0.0             # media armonica homogeneity + completeness
    n_true_groups: int = 0             # grupuri clinice distincte in ground truth
    n_predicted_clusters: int = 0      # clustere prezise


@dataclass
class AgentMetrics:
    """ metrici pentru performanta agentului LLM."""
    total_processed: int = 0
    success_rate: float = 0.0
    tool_call_completeness: float = 0.0  # % clustere cu toate 3 tool-urile apelate
    severity_accuracy: float = 0.0       # severitate prezisa == severitate reala
    severity_within_one: float = 0.0     # severitate la max 1 nivel distanta
    avg_processing_time_s: float = 0.0
    total_processing_time_s: float = 0.0
    errors: int = 0
    action_distribution: dict = field(default_factory=dict)


@dataclass
class KPIReport:
    """ raport complet KPI pentru o rulare SilentICU."""
    timestamp: str = ""
    threshold: float = 0.80
    time_window_min: int = 10
    model_embeddings: str = "all-MiniLM-L6-v2"
    model_agent: str = "gpt-4o"

    deduplication: DeduplicationMetrics = field(default_factory=DeduplicationMetrics)
    clustering: ClusteringMetrics = field(default_factory=ClusteringMetrics)
    agent: AgentMetrics = field(default_factory=AgentMetrics)

    # comparatie praguri (optional)
    threshold_comparison: list[dict] = field(default_factory=list)

    def to_dict(self) -> dict:
        import dataclasses
        return dataclasses.asdict(self)


# SilentICUEvaluator
class SilentICUEvaluator:
    """
        calculeaza toate metricile KPI pentru o rulare SilentICU.

    :param df: DataFrame adnotat de AlarmDeduplicator.fit()
        Trebuie sa contina coloanele:
            - is_duplicate (ground truth din data_pipeline)
            - is_dedup_duplicate (prezis de deduplicator)
            - true_group (grupul clinic real, ex: 'tachycardia')
            - cluster_id (prezis de deduplicator)
            - severity (severitatea reala)
            - cluster_severity (severitatea prezisa de agent sau deduplicator)
    :param clusters (list[AlarmCluster]): clusterele din AlarmDeduplicator.clusters
    :param agent_results (list[AgentResult], optional): rezultatele agentului Gemini
    """

    def __init__(
        self,
        df: pd.DataFrame,
        clusters: list[AlarmCluster],
        agent_results: Optional[list[AgentResult]] = None,
    ):
        self.df = df.copy().reset_index(drop=True)
        self.clusters = clusters
        self.agent_results = agent_results or []
        self._validate_columns()

    def _validate_columns(self):
        required = ["is_dedup_duplicate", "cluster_id"]
        missing = [c for c in required if c not in self.df.columns]
        if missing:
            raise ValueError(
                f"Coloane lipsa din df: {missing}. "
                f"Asigura-te ca ai rulat AlarmDeduplicator.fit() mai intai."
            )

    # 1. Metrici deduplicare
    def compute_deduplication_metrics(self) -> DeduplicationMetrics:
        """
            calculeaza metricile de deduplicare (precision, recall, f1 score), comparand:
        - is_duplicate (ground truth: stiam ca e duplicat la generare)
        - is_dedup_duplicate (prezis de sistemul nostru)
        False merge = am grupat alarme care NU sunt duplicate (tipuri clinice diferite)
        """
        m = DeduplicationMetrics()
        m.total_alarms = len(self.df)
        m.total_clusters = len(self.clusters)
        m.duplicate_clusters = sum(1 for c in self.clusters if c.is_duplicate)
        m.alarms_suppressed = sum(
            c.size - 1 for c in self.clusters if c.is_duplicate
        )
        m.deduplication_rate = (
            m.alarms_suppressed / m.total_alarms if m.total_alarms > 0 else 0.0
        )

        # binary classification: duplicate sau nu
        if "is_duplicate" in self.df.columns:
            y_true = self.df["is_duplicate"].astype(int).values
            y_pred = self.df["is_dedup_duplicate"].astype(int).values

            m.precision = float(precision_score(y_true, y_pred, zero_division=0))
            m.recall = float(recall_score(y_true, y_pred, zero_division=0))
            m.f1 = float(f1_score(y_true, y_pred, zero_division=0))

            tn, fp, fn, tp = confusion_matrix(y_true, y_pred, labels=[0, 1]).ravel()
            m.true_positive = int(tp)
            m.false_positive = int(fp)
            m.false_negative = int(fn)
            m.true_negative = int(tn)

            # false merge rate = FP / (FP + TN) — ce % din non-duplicate am grupat gresit
            # cel mai important din context clinic: am grupat alarme care descriu eevenimente complet diferite
            m.false_merge_rate = (
                fp / (fp + tn) if (fp + tn) > 0 else 0.0
            )
        else:
            logger.warning(
                "Coloana 'is_duplicate' nu exista in df — "
                "metricile binary classification nu pot fi calculate."
            )

        return m


    # 2. Metrici clustering
    def compute_clustering_metrics(self) -> ClusteringMetrics:
        """
            calculeaza:
        - ARI (Adjusted Rand Index): masoara cat de bine se suprapun clusterele prezise peste grupurile reale
        - NMI (Normalized Mutual Information): masoara cata informatie depsre true_group se regaseste in cluster_id
        - homogeneity: fiecare cluster contine alarme dintr-un singur tip clinic?
        - completeness: toate alarmele de acelasi tip clinic sunt in acelasi cluster?
        - v_measure: media armonica homogeneity si completeness

        comparand cluster_id (prezis) cu true_group (ground truth).
        Necesita coloana 'true_group' in df.
        """
        m = ClusteringMetrics()

        if "true_group" not in self.df.columns:
            logger.warning(
                "Coloana 'true_group' nu exista — "
                "metricile de clustering nu pot fi calculate."
            )
            return m

        labels_true = self.df["true_group"].values
        labels_pred = self.df["cluster_id"].values

        # sklearn asteapta etichete numerice sau string-uri — ambele merg
        m.ari = float(adjusted_rand_score(labels_true, labels_pred))
        m.nmi = float(normalized_mutual_info_score(labels_true, labels_pred))

        h, c, v = homogeneity_completeness_v_measure(labels_true, labels_pred)
        m.homogeneity = float(h)
        m.completeness = float(c)
        m.v_measure = float(v)

        m.n_true_groups = int(self.df["true_group"].nunique())
        m.n_predicted_clusters = int(self.df["cluster_id"].nunique())

        return m


    # 3. Metrici agent
    def compute_agent_metrics(self) -> AgentMetrics:
        """
            calculeaza metricile pentru agentul Gemini.
        Necesita agent_results nevide.
        """
        m = AgentMetrics()

        if not self.agent_results:
            logger.warning("Nu exista agent_results — metricile agentului nu sunt calculate.")
            return m

        m.total_processed = len(self.agent_results)
        m.errors = sum(1 for r in self.agent_results if not r.success)
        m.success_rate = (
            (m.total_processed - m.errors) / m.total_processed
            if m.total_processed > 0 else 0.0
        )

        # tool call completeness — toate 3 tool-urile apelate
        all_tools = {"classify_severity", "generate_unified_alarm", "explain_grouping"}
        m.tool_call_completeness = sum(
            1 for r in self.agent_results
            if all_tools.issubset(set(r.tools_called))
        ) / m.total_processed if m.total_processed > 0 else 0.0

        # timing
        times = [r.processing_time_s for r in self.agent_results]
        m.avg_processing_time_s = float(np.mean(times)) if times else 0.0
        m.total_processing_time_s = float(np.sum(times)) if times else 0.0

        # distributie actiuni recomandate
        actions = [r.action_required for r in self.agent_results if r.success]
        m.action_distribution = {
            action: actions.count(action)
            for action in set(actions)
        }

        # severity accuracy — comparam cu cluster_severity din df (max din cluster)
        # cream un dict cluster_id -> cluster_severity din df
        if "cluster_severity" in self.df.columns and self.agent_results:
            cluster_sev_map = (
                self.df.groupby("cluster_id")["cluster_severity"]
                .first()
                .to_dict()
            )

            correct = 0
            within_one = 0
            compared = 0

            for r in self.agent_results:
                if not r.success:
                    continue
                true_sev = cluster_sev_map.get(r.cluster_id)
                if true_sev is None:
                    continue
                compared += 1
                if r.severity == true_sev:
                    correct += 1
                if abs(
                    SEVERITY_ORDER.get(r.severity, 0) -
                    SEVERITY_ORDER.get(true_sev, 0)
                ) <= 1:
                    within_one += 1

            if compared > 0:
                m.severity_accuracy = correct / compared
                m.severity_within_one = within_one / compared

        return m


    # 4. Comparatie praguri (sensitivity analysis)
    def threshold_sensitivity(
        self,
        embeddings: np.ndarray,
        thresholds: Optional[list[float]] = None,
        time_window_min: int = 10,
    ) -> list[dict]:
        """
            ruleaza deduplicatorul cu praguri diferite si compara rezultatele.
        Util pentru analiza influentei threshold-ului (0.70, 0.80, 0.85, 0.90).

        :param embeddings: vectorii MiniLM (N, 384)
        :param thresholds: lista de praguri testate (default: [0.70, 0.75, 0.80, 0.85, 0.90])
        :param time_window_min: fereastra de timp (same for all)
        :return: lista de dict cu metricile per prag
        """
        # import local pentru a evita import circular
        from src.deduplicator import AlarmDeduplicator

        if thresholds is None:
            thresholds = [0.70, 0.75, 0.80, 0.85, 0.90]

        results = []
        logger.info(f"Sensitivity analysis pe {len(thresholds)} praguri...")

        for thr in thresholds:
            dedup = AlarmDeduplicator(
                threshold=thr,
                time_window_min=time_window_min,
            )
            df_t = dedup.fit(self.df.copy(), embeddings)

            stats = dedup.stats()

            # ARI si false merge rate
            ari = 0.0
            fmr = 0.0

            if "true_group" in df_t.columns:
                ari = float(adjusted_rand_score(
                    df_t["true_group"].values,
                    df_t["cluster_id"].values,
                ))

            if "is_duplicate" in df_t.columns:
                y_true = df_t["is_duplicate"].astype(int).values
                y_pred = df_t["is_dedup_duplicate"].astype(int).values
                tn, fp, fn, tp = confusion_matrix(
                    y_true, y_pred, labels=[0, 1]
                ).ravel()
                fmr = fp / (fp + tn) if (fp + tn) > 0 else 0.0

            results.append({
                "threshold": thr,
                "deduplication_rate": round(stats["deduplication_rate"], 4),
                "alarms_suppressed": stats["alarms_suppressed"],
                "total_clusters": stats["total_clusters"],
                "duplicate_clusters": stats["duplicate_clusters"],
                "ari": round(ari, 4),
                "false_merge_rate": round(fmr, 4),
            })
            logger.info(
                f"  thr={thr}: dedup_rate={stats['deduplication_rate']:.2%}, "
                f"ARI={ari:.4f}, FMR={fmr:.4f}"
            )

        return results


    # 5. Raport complet
    def full_report(
        self,
        embeddings: Optional[np.ndarray] = None,
        threshold: float = 0.80,
        time_window_min: int = 10,
        run_threshold_comparison: bool = False,
    ) -> KPIReport:
        """
            calculeaza toate metricile si returneaza un KPIReport complet.

        :param embeddings: necesar doar daca run_threshold_comparison=True
        :param threshold / time_window_min: configuratia curenta (pentru metadata)
        :param run_threshold_comparison: daca sa ruleze sensitivity analysis
        """
        from datetime import datetime

        report = KPIReport(
            timestamp=datetime.now().isoformat(),
            threshold=threshold,
            time_window_min=time_window_min,
        )

        logger.info("=== Calcul KPI complet ===")

        report.deduplication = self.compute_deduplication_metrics()
        logger.info(
            f"Deduplication rate: {report.deduplication.deduplication_rate:.2%} | "
            f"F1: {report.deduplication.f1:.4f} | "
            f"FMR: {report.deduplication.false_merge_rate:.4f}"
        )

        report.clustering = self.compute_clustering_metrics()
        logger.info(
            f"ARI: {report.clustering.ari:.4f} | "
            f"NMI: {report.clustering.nmi:.4f} | "
            f"V-measure: {report.clustering.v_measure:.4f}"
        )

        report.agent = self.compute_agent_metrics()
        if self.agent_results:
            logger.info(
                f"Agent success: {report.agent.success_rate:.2%} | "
                f"Tool completeness: {report.agent.tool_call_completeness:.2%} | "
                f"Severity acc: {report.agent.severity_accuracy:.2%}"
            )

        if run_threshold_comparison and embeddings is not None:
            report.threshold_comparison = self.threshold_sensitivity(embeddings)

        return report


    # 6. Afisare si salvare
    @staticmethod
    def print_report(report: KPIReport):
        """Afiseaza raportul formatat in consola."""
        d = report.deduplication
        c = report.clustering
        a = report.agent

        print(f"### KPI REPORT")
        print(f"  {report.timestamp}")
        print(f"  Threshold: {report.threshold} | Window: {report.time_window_min} min")
        print(f"  Embeddings: {report.model_embeddings} | Agent: {report.model_agent}")

        print(f"\n>> DEDUPLICARE")
        print(f"  Total alarme        : {d.total_alarms}")
        print(f"  Total clustere      : {d.total_clusters}")
        print(f"  Clustere duplicate  : {d.duplicate_clusters}")
        print(f"  Alarme suprimate    : {d.alarms_suppressed}")
        print(f"  Deduplication rate  : {d.deduplication_rate:.2%}")
        print(f"  Precision           : {d.precision:.4f}")
        print(f"  Recall              : {d.recall:.4f}")
        print(f"  F1 Score            : {d.f1:.4f}")
        print(f"  False Merge Rate    : {d.false_merge_rate:.4f}")
        print(f"  Confusion: TP={d.true_positive} FP={d.false_positive} "
              f"FN={d.false_negative} TN={d.true_negative}")

        print(f"\n>> CLUSTERING (vs. ground truth)")
        print(f"  Grupuri clinice reale : {c.n_true_groups}")
        print(f"  Clustere prezise      : {c.n_predicted_clusters}")
        print(f"  ARI                   : {c.ari:.4f}")
        print(f"  NMI                   : {c.nmi:.4f}")
        print(f"  Homogeneity           : {c.homogeneity:.4f}")
        print(f"  Completeness          : {c.completeness:.4f}")
        print(f"  V-measure             : {c.v_measure:.4f}")

        if a.total_processed > 0:
            print(f"\n>>> AGENT GPT-4o (Azure OpenAI)")
            print(f"  Clustere procesate    : {a.total_processed}")
            print(f"  Success rate          : {a.success_rate:.2%}")
            print(f"  Tool completeness     : {a.tool_call_completeness:.2%}")
            print(f"  Severity accuracy     : {a.severity_accuracy:.2%}")
            print(f"  Severity within ±1    : {a.severity_within_one:.2%}")
            print(f"  Timp mediu/cluster    : {a.avg_processing_time_s:.2f}s")
            print(f"  Timp total            : {a.total_processing_time_s:.1f}s")
            print(f"  Actiuni recomandate   : {a.action_distribution}")

        if report.threshold_comparison:
            print(f"\n>>> SENSITIVITY ANALYSIS (threshold)")
            print(f"  {'Thr':>5} | {'Dedup%':>7} | {'Suppressed':>10} | "
                  f"{'ARI':>7} | {'FMR':>7}")
            for row in report.threshold_comparison:
                print(
                    f"  {row['threshold']:>5.2f} | "
                    f"{row['deduplication_rate']:>7.2%} | "
                    f"{row['alarms_suppressed']:>10} | "
                    f"{row['ari']:>7.4f} | "
                    f"{row['false_merge_rate']:>7.4f}"
                )


    @staticmethod
    def save_report(report: KPIReport, path: str = "data/results/kpi_report.json"):
        """ salveaza raportul KPI ca fisier JSON."""
        os.makedirs(os.path.dirname(path), exist_ok=True)
        with open(path, "w", encoding="utf-8") as f:
            json.dump(report.to_dict(), f, indent=2, default=str)
        logger.info(f"Raport KPI salvat: {path}")

    @staticmethod
    def load_report(path: str) -> dict:
        """ incarca un raport KPI salvat anterior."""
        with open(path, "r", encoding="utf-8") as f:
            return json.load(f)