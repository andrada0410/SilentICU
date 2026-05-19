"""
Agent Azure OpenAI (gpt-4o) cu tool use

Foloseste SDK-ul: openai (pip install openai)
    from openai import AzureOpenAI

Agentul primeste clustere de alarme similare si apeleaza 3 tool-uri:
  1. classify_severity      -- determina severitatea consolidata a clusterului
  2. generate_unified_alarm -- genereaza mesajul unificat in limbaj natural
  3. explain_grouping        -- explica de ce alarmele au fost grupate impreuna

Variabile de mediu necesare in .env:
    AZURE_OPENAI_KEY=...
    AZURE_OPENAI_ENDPOINT=https://proiect-open-ai.openai.azure.com/
    AZURE_OPENAI_DEPLOYMENT=gpt-4o
    AZURE_OPENAI_API_VERSION=2024-02-01

"""

import os
import json
import time
import logging
from dataclasses import dataclass, field
from typing import Optional

from openai import AzureOpenAI
from dotenv import load_dotenv

from src.deduplicator import AlarmCluster

load_dotenv()
logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
logger = logging.getLogger(__name__)

SEVERITY_ORDER = {"LOW": 0, "MEDIUM": 1, "HIGH": 2, "CRITICAL": 3}


# Tool definitions
TOOLS = [
    {
        "type": "function",
        "function": {
            "name": "classify_severity",
            "description": (
                "Analizeaza mesajele de alarma dintr-un cluster si determina "
                "severitatea consolidata a grupului (LOW, MEDIUM, HIGH sau CRITICAL). "
                "Ia in considerare toate mesajele, nu doar cel mai sever."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "severity": {
                        "type": "string",
                        "enum": ["LOW", "MEDIUM", "HIGH", "CRITICAL"],
                        "description": "Severitatea consolidata a clusterului de alarme.",
                    },
                    "reasoning": {
                        "type": "string",
                        "description": "Scurta justificare clinica (1-2 propozitii).",
                    },
                },
                "required": ["severity", "reasoning"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "generate_unified_alarm",
            "description": (
                "Genereaza un singur mesaj de alarma consolidat care inlocuieste "
                "toate alarmele redundante din cluster. Mesajul trebuie sa fie clar, "
                "concis, in engleza medicala standard si sa mentioneze numarul de surse."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "unified_message": {
                        "type": "string",
                        "description": (
                            "Mesajul unificat, ex: "
                            "'[HIGH] Tachycardia detected across 3 sources - "
                            "HR consistently elevated. Immediate assessment recommended.'"
                        ),
                    },
                    "action_required": {
                        "type": "string",
                        "enum": ["none", "monitor", "assess", "intervene", "emergency"],
                        "description": "Actiunea recomandata personalului medical.",
                    },
                },
                "required": ["unified_message", "action_required"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "explain_grouping",
            "description": (
                "Explica in limbaj natural de ce aceste alarme au fost grupate impreuna. "
                "Destinata personalului medical, nu inginerilor. "
                "Scurta (2-3 propozitii), fara jargon tehnic."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "explanation": {
                        "type": "string",
                        "description": (
                            "Explicatia gruparii, ex: "
                            "'These 3 alerts all describe the same cardiac event - "
                            "elevated heart rate - reported by different monitors within "
                            "a 4-minute window. They have been merged to reduce noise.'"
                        ),
                    },
                    "clinical_context": {
                        "type": "string",
                        "description": "Contextul clinic relevant (optional, 1 propozitie).",
                    },
                },
                "required": ["explanation"],
            },
        },
    },
]



# AgentResult
@dataclass
class AgentResult:
    """ rezultatul complet al agentului pentru un cluster de alarme."""
    cluster_id: int
    cluster_size: int
    original_messages: list[str]

    severity: str = "UNKNOWN"
    severity_reasoning: str = ""
    unified_message: str = ""
    action_required: str = "monitor"
    explanation: str = ""
    clinical_context: str = ""

    tools_called: list[str] = field(default_factory=list)
    processing_time_s: float = 0.0
    error: Optional[str] = None

    @property
    def success(self) -> bool:
        return self.error is None and bool(self.unified_message)

    def to_dict(self) -> dict:
        return {
            "cluster_id": self.cluster_id,
            "cluster_size": self.cluster_size,
            "severity": self.severity,
            "severity_reasoning": self.severity_reasoning,
            "unified_message": self.unified_message,
            "action_required": self.action_required,
            "explanation": self.explanation,
            "clinical_context": self.clinical_context,
            "tools_called": self.tools_called,
            "processing_time_s": round(self.processing_time_s, 3),
            "success": self.success,
            "error": self.error,
        }

    def display(self):
        print(f"\n>>> CLUSTER #{self.cluster_id}  |  {self.cluster_size} alarme  |  {self.severity}")
        print("Alarme originale:")
        for i, msg in enumerate(self.original_messages, 1):
            print(f"  {i}. {msg}")
        print(f"\n[UNIFIED] {self.unified_message}")
        print(f"[ACTION]  {self.action_required.upper()}")
        print(f"\n[EXPLANATION] {self.explanation}")
        if self.clinical_context:
            print(f"[CONTEXT]    {self.clinical_context}")
        print(f"\n[JUSTIFICATION] {self.severity_reasoning}")
        print(f"Tools apelate: {', '.join(self.tools_called)}")
        print(f"Timp: {self.processing_time_s:.2f}s")



# SilentICUAgent
class SilentICUAgent:
    """ Agent Azure OpenAI (gpt-4o) pentru procesarea clusterelor de alarme ICU."""

    def __init__(
        self,
        api_key: Optional[str] = None,
        endpoint: Optional[str] = None,
        deployment: Optional[str] = None,
        api_version: Optional[str] = None,
        rate_limit_delay: float = 0.5,
    ):
        api_key    = api_key    or os.getenv("AZURE_OPENAI_KEY")
        endpoint   = endpoint   or os.getenv("AZURE_OPENAI_ENDPOINT")
        deployment = deployment or os.getenv("AZURE_OPENAI_DEPLOYMENT", "gpt-4o")
        api_version = api_version or os.getenv("AZURE_OPENAI_API_VERSION", "2024-02-01")

        if not api_key or not endpoint:
            raise ValueError(
                "AZURE_OPENAI_KEY si AZURE_OPENAI_ENDPOINT trebuie setate in .env"
            )

        self.client = AzureOpenAI(
            api_key=api_key,
            azure_endpoint=endpoint,
            api_version=api_version,
        )
        self.deployment = deployment
        self.rate_limit_delay = rate_limit_delay
        logger.info(f"Agent initializat: Azure OpenAI / {deployment}")

    # prompt
    def _build_prompt(self, cluster: AlarmCluster) -> str:
        messages_block = "\n".join(
            f"  [{i+1}] \"{msg}\" (source: {src}, severity: {sev})"
            for i, (msg, src, sev) in enumerate(zip(
                cluster.messages,
                cluster.sources if cluster.sources else ["unknown"] * cluster.size,
                cluster.severities if cluster.severities else ["LOW"] * cluster.size,
            ))
        )
        time_info = (
            f"{cluster.time_span_minutes:.1f} minutes"
            if cluster.size > 1 else "single alarm"
        )
        return f"""You are SilentICU, an AI assistant for ICU alarm deduplication.

You have identified a cluster of {cluster.size} semantically similar alarms that occurred within {time_info}.
These alarms likely describe the same clinical event from different monitoring sources.

ALARM CLUSTER #{cluster.cluster_id}:
{messages_block}

Your task - call ALL THREE tools in sequence:
1. classify_severity      - determine the consolidated severity for this cluster
2. generate_unified_alarm - create ONE clear consolidated alarm message
3. explain_grouping       - explain (in plain language for nurses/doctors) why these were grouped

Be concise and clinically accurate. The unified alarm replaces all individual alarms in the dashboard.
"""

    # aplicare tool call
    def _apply_tool_call(self, name: str, args: dict, result: AgentResult) -> None:
        result.tools_called.append(name)
        if name == "classify_severity":
            result.severity = args.get("severity", "UNKNOWN")
            result.severity_reasoning = args.get("reasoning", "")
        elif name == "generate_unified_alarm":
            result.unified_message = args.get("unified_message", "")
            result.action_required = args.get("action_required", "monitor")
        elif name == "explain_grouping":
            result.explanation = args.get("explanation", "")
            result.clinical_context = args.get("clinical_context", "")
        else:
            logger.warning(f"Tool necunoscut: {name}")

    # procesare un cluster
    def process_cluster(self, cluster: AlarmCluster) -> AgentResult:
        """
            proceseaza un cluster si returneaza AgentResult.
        Fluxul cu Azure OpenAI tool use:
          1. Trimitem prompt + tools
          2. GPT-4o returneaza tool_calls
          3. Executam tool call-urile si trimitem rezultatele inapoi
          4. Repetam pana toate 3 tool-urile au fost apelate
        """
        result = AgentResult(
            cluster_id=cluster.cluster_id,
            cluster_size=cluster.size,
            original_messages=cluster.messages,
        )
        start = time.time()

        try:
            logger.info(
                f"Procesare cluster #{cluster.cluster_id} "
                f"({cluster.size} alarme, max severity: {cluster.max_severity})..."
            )

            # conversatia ca lista de mesaje
            messages = [
                {"role": "system", "content": (
                    "You are SilentICU, an AI assistant for ICU alarm deduplication. "
                    "Always call all three tools: classify_severity, "
                    "generate_unified_alarm, and explain_grouping."
                )},
                {"role": "user", "content": self._build_prompt(cluster)},
            ]

            for _ in range(5):  # max 5 runde
                response = self.client.chat.completions.create(
                    model=self.deployment,
                    messages=messages,
                    tools=TOOLS,
                    tool_choice="auto",
                )

                msg = response.choices[0].message
                tool_calls = msg.tool_calls or []

                if not tool_calls:
                    break  # niciun tool call -> gpt a terminat

                # adaugam raspunsul asistentului in conversatie
                messages.append(msg)

                # procesam fiecare tool call si trimitem rezultatele
                for tc in tool_calls:
                    try:
                        args = json.loads(tc.function.arguments)
                    except json.JSONDecodeError:
                        args = {}

                    self._apply_tool_call(tc.function.name, args, result)
                    logger.debug(f"  Tool: {tc.function.name} -> {args}")

                    # trimitem rezultatul tool call-ului inapoi
                    messages.append({
                        "role": "tool",
                        "tool_call_id": tc.id,
                        "content": json.dumps({"result": "ok"}),
                    })

                # verificam daca toate tool-urile au fost apelate
                missing = (
                    {"classify_severity", "generate_unified_alarm", "explain_grouping"}
                    - set(result.tools_called)
                )
                if not missing:
                    break

            self._apply_fallbacks(cluster, result)

        except Exception as e:
            logger.error(f"Eroare cluster #{cluster.cluster_id}: {e}")
            result.error = str(e)
            self._apply_fallbacks(cluster, result)

        result.processing_time_s = time.time() - start
        logger.info(f"  -> [{result.severity}] {result.unified_message[:70]}...")
        return result

    # fallback-uri
    def _apply_fallbacks(self, cluster: AlarmCluster, result: AgentResult) -> None:
        if not result.severity or result.severity == "UNKNOWN":
            result.severity = cluster.max_severity
            result.severity_reasoning = (
                f"Fallback: severitate maxima din cluster ({cluster.max_severity})."
            )
        if not result.unified_message:
            sources = ", ".join(set(cluster.sources)) if cluster.sources else "multiple sources"
            result.unified_message = (
                f"[{result.severity}] {cluster.representative_message} "
                f"- {cluster.size} similar alarm(s) from {sources}."
            )
            result.action_required = "assess"
        if not result.explanation:
            result.explanation = (
                f"{cluster.size} alarms describing the same clinical event were grouped "
                f"within a {cluster.time_span_minutes:.0f}-minute window. "
                f"Displayed as one consolidated notification to reduce alarm fatigue."
            )

    # batch
    def process_all(
        self,
        clusters: list[AlarmCluster],
        max_clusters: Optional[int] = None,
    ) -> list[AgentResult]:
        """ proceseaza o lista de clustere (de obicei duplicate_clusters())."""
        if max_clusters:
            clusters = clusters[:max_clusters]

        logger.info(f"=== Procesare batch: {len(clusters)} clustere ===")
        results = []

        for i, cluster in enumerate(clusters):
            result = self.process_cluster(cluster)
            results.append(result)
            if i < len(clusters) - 1:
                time.sleep(self.rate_limit_delay)

        n_ok = sum(1 for r in results if r.success)
        logger.info(f"=== Batch complet: {n_ok} OK, {len(results)-n_ok} erori ===")
        return results

    # DataFrame
    @staticmethod
    def results_to_df(results: list[AgentResult]):
        import pandas as pd
        return pd.DataFrame([r.to_dict() for r in results])