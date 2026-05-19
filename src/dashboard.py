import sys
import os
import json
import numpy as np
import pandas as pd
import streamlit as st
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import seaborn as sns
from sklearn.decomposition import PCA

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

# config pagina
st.set_page_config(
    page_title="SilentICU Dashboard",
    page_icon="🏥",
    layout="wide",
    initial_sidebar_state="expanded",
)

#culori consistente cu notebook-urile
COLOR_MAP = {
    'CRITICAL': '#D62728',
    'HIGH': '#FF7F0E',
    'MEDIUM': '#1F77B4',
    'LOW': '#2CA02C',
}
ACTION_COLORS = {
    'none': '#2ecc71',
    'monitor': '#3498db',
    'assess': '#f39c12',
    'intervene': '#e67e22',
    'emergency': '#e74c3c',
}

PLT_STYLE = {
    'figure.facecolor': 'white',
    'axes.facecolor': 'white',
    'text.color': 'black',
    'axes.labelcolor': 'black',
    'xtick.color': 'black',
    'ytick.color': 'black',
    'axes.edgecolor': 'gray',
    'legend.facecolor': 'white',
    'legend.edgecolor': 'gray',
    'savefig.facecolor': 'white',
}
plt.rcParams.update(PLT_STYLE)


# incarcare date din notebook-uri
@st.cache_data
def load_all():
    data = {}
    errors = []

    # date brute
    p = 'data/raw/alarms_combined.csv'
    if os.path.exists(p):
        data['df_raw'] = pd.read_csv(p)
        data['df_raw']['timestamp'] = pd.to_datetime(data['df_raw']['timestamp'])
    else:
        errors.append(f"Lipseste: {p} — ruleaza data_evaluation.ipynb")

    # embeddings
    p = 'data/embeddings/embeddings_minilm.npy'
    if os.path.exists(p):
        data['embeddings'] = np.load(p)
    else:
        errors.append(f"Lipseste: {p} — ruleaza embeddins_notebook.ipynb")

    # date duplicate
    p = 'data/processed/alarms_deduplicated.csv'
    if os.path.exists(p):
        data['df_dedup'] = pd.read_csv(p)
        data['df_dedup']['timestamp'] = pd.to_datetime(data['df_dedup']['timestamp'])
    else:
        errors.append(f"Lipseste: {p} — ruleaza agent.ipynb")

    # rezultate agent json
    p = 'data/results/agent_results.json'
    if os.path.exists(p):
        with open(p, encoding='utf-8') as f:
            data['agent_results'] = json.load(f)
    else:
        errors.append(f"Lipseste: {p} — ruleaza agent.ipynb")

    # raport kpi json
    p = 'data/evaluations/kpi_report.json'
    if os.path.exists(p):
        with open(p, encoding='utf-8') as f:
            data['kpi_report'] = json.load(f)
    else:
        errors.append(f"Lipseste: {p} — ruleaza evaluation.ipynb")

    return data, errors


data, load_errors = load_all()

# sidebar
with st.sidebar:
    st.title("🏥 SilentICU")
    st.caption("Semantic Alarm Deduplication Engine")
    st.divider()

    # Erori de incarcare
    if load_errors:
        st.error("**Date lipsă:**")
        for e in load_errors:
            st.caption(f"⚠ {e}")
        st.divider()

    # filtre
    st.subheader("Filtre")

    severity_options = ['CRITICAL', 'HIGH', 'MEDIUM', 'LOW']
    selected_severity = st.multiselect(
        "Severitate alarme",
        options=severity_options,
        default=severity_options,
    )

    # prag similaritate (doar afisare — nu ruleaza din nou deduplicarea)
    threshold_info = st.slider(
        "Prag similaritate folosit",
        min_value=0.50, max_value=0.99,
        value=0.80, step=0.01,
        disabled=True,
        help="Pragul cu care a rulat deduplicatorul în agent.ipynb"
    )

    st.divider()

    # metrici rapide in sidebar
    if 'df_dedup' in data:
        df_d = data['df_dedup']
        total = len(df_d)
        suppr = int(df_d['is_dedup_duplicate'].sum()) if 'is_dedup_duplicate' in df_d.columns else 0
        rate = suppr / total * 100 if total > 0 else 0
        n_clus = df_d['cluster_id'].nunique() if 'cluster_id' in df_d.columns else 0

        st.metric("Total alarme", total)
        st.metric("Alarme suprimate", suppr)
        st.metric("Reducere zgomot", f"{rate:.1f}%")
        st.metric("Clustere formate", n_clus)

    if 'agent_results' in data:
        results = data['agent_results']
        times   = [r['processing_time_s'] for r in results if r.get('success')]
        if times:
            st.metric("Timp mediu agent", f"{np.mean(times):.2f}s")


# titlu principal
st.title("🏥 SilentICU — Alarm Intelligence Dashboard")
st.caption("Semantic Alarm Deduplication Engine")

if load_errors:
    st.warning(
        f"**{len(load_errors)} fișier(e) lipsesc.** "
        "Rulează mai întâi notebook-urile în ordine: "
        "data_evaluation → embeddins_notebook → agent → evaluation."
    )

# tab-uri principale
tab_kpi, tab_dedup, tab_agent, tab_embed, tab_raw, tab_clusters = st.tabs([
    "📊 KPI",
    "🔁 Deduplicare",
    "🤖 Agent LLM",
    "🔵 Embeddings",
    "📋 Alarme brute",
    "📦 Clustere",
])


# tab 1 - kpi
with tab_kpi:
    st.subheader("Metrici principale")

    if 'kpi_report' not in data:
        st.info("Rulează evaluation.ipynb pentru a genera kpi_report.json")
    else:
        report = data['kpi_report']
        d_met  = report.get('deduplication', {})
        c_met  = report.get('clustering', {})
        a_met  = report.get('agent', {})

        # rand 1: Deduplication KPIs
        st.markdown("#### Deduplicare")
        c1, c2, c3, c4, c5 = st.columns(5)
        c1.metric("Total alarme", d_met.get('total_alarms', '—'))
        c2.metric("Alarme suprimate", d_met.get('alarms_suppressed', '—'))
        c3.metric("Dedup rate",f"{d_met.get('deduplication_rate', 0)*100:.1f}%")
        c4.metric("Precision",f"{d_met.get('precision', 0):.4f}")
        c5.metric("F1 Score",f"{d_met.get('f1', 0):.4f}")

        st.divider()

        # rand 2: Clustering KPIs
        st.markdown("#### Calitate clusterizare (vs. ground truth)")
        c1, c2, c3, c4, c5 = st.columns(5)
        c1.metric("ARI", f"{c_met.get('ari', 0):.4f}", help="Adjusted Rand Index — 1.0 = perfect")
        c2.metric("NMI", f"{c_met.get('nmi', 0):.4f}")
        c3.metric("Homogeneity", f"{c_met.get('homogeneity', 0):.4f}")
        c4.metric("Completeness", f"{c_met.get('completeness', 0):.4f}")
        c5.metric("V-measure", f"{c_met.get('v_measure', 0):.4f}")

        st.divider()

        # rand 3: Agent KPIs
        st.markdown("#### Agent LLM")
        c1, c2, c3 = st.columns(3)
        c1.metric("Clustere procesate",  a_met.get('total_processed', '—'))
        c2.metric("Success rate", f"{a_met.get('success_rate', 0)*100:.1f}%")
        c3.metric("Tool call completeness", f"{a_met.get('tool_call_completeness', 0)*100:.1f}%")

        st.divider()

        # grafic KPI overview
        st.markdown("#### Vizualizare KPI")

        fig, axes = plt.subplots(1, 3, figsize=(16, 5))
        fig.suptitle("SilentICU — KPI Overview", fontsize=14, fontweight='bold', color='black')

        # deduplication bar
        labels_d = ['Precision', 'Recall', 'F1']
        vals_d = [d_met.get('precision', 0),
                    d_met.get('recall', 0),
                    d_met.get('f1', 0)]
        colors_d = ['#4C72B0', '#55A868', '#C44E52']
        bars = axes[0].bar(labels_d, vals_d, color=colors_d, width=0.5, edgecolor='white')
        axes[0].set_ylim(0, 1.15)
        axes[0].set_title("Metrici deduplicare", color='black', fontsize=12)
        axes[0].set_ylabel("Score", color='black')
        axes[0].tick_params(colors='black')
        axes[0].set_xticks(range(len(labels_d)))
        axes[0].set_xticklabels(labels_d, color='black', fontweight='bold')
        axes[0].axhline(1.0, color='gray', linestyle='--', linewidth=0.8)
        for bar, val in zip(bars, vals_d):
            axes[0].text(bar.get_x() + bar.get_width()/2,
                         val + 0.02, f'{val:.4f}',
                         ha='center', va='bottom',
                         fontweight='bold', color='black', fontsize=10)

        # clustering bar
        labels_c = ['ARI', 'NMI', 'Homog.', 'Compl.', 'V-meas.']
        vals_c   = [c_met.get('ari', 0), c_met.get('nmi', 0),
                    c_met.get('homogeneity', 0),
                    c_met.get('completeness', 0),
                    c_met.get('v_measure', 0)]
        bars2 = axes[1].bar(labels_c, vals_c, color='#7F77DD',
                            width=0.6, edgecolor='white')
        axes[1].set_ylim(0, 1.15)
        axes[1].set_title("Metrici clustering", color='black', fontsize=12)
        axes[1].set_ylabel("Score", color='black')
        axes[1].tick_params(colors='black')
        axes[1].set_xticks(range(len(labels_c)))
        axes[1].set_xticklabels(labels_c, color='black',
                                fontweight='bold', fontsize=9)
        for bar, val in zip(bars2, vals_c):
            axes[1].text(bar.get_x() + bar.get_width()/2,
                         val + 0.02, f'{val:.3f}',
                         ha='center', va='bottom',
                         fontweight='bold', color='black', fontsize=9)

        # alarme inainte vs dupa
        total_a = d_met.get('total_alarms', 0)
        suppr_a = d_met.get('alarms_suppressed', 0)
        remain_a = total_a - suppr_a
        axes[2].bar(['Alarme originale', 'Alarme după dedup'],
                    [total_a, remain_a],
                    color=['#E24B4A', '#2CA02C'], edgecolor='white', width=0.5)
        axes[2].set_title("Reducere alarme", color='black', fontsize=12)
        axes[2].set_ylabel("Număr alarme", color='black')
        axes[2].tick_params(colors='black')
        axes[2].set_xticks([0, 1])
        axes[2].set_xticklabels(['Originale', 'După dedup'],
                                color='black', fontweight='bold')
        axes[2].text(0, total_a + 5, str(total_a),
                     ha='center', fontweight='bold', color='black')
        axes[2].text(1, remain_a + 5, str(remain_a),
                     ha='center', fontweight='bold', color='black')
        axes[2].set_ylim(0, total_a * 1.2)

        plt.tight_layout()
        st.pyplot(fig)
        plt.close()

        # sensitivity analysis
        if report.get('threshold_comparison'):
            st.subheader("Sensitivity analysis — comparație praguri")
            df_sens = pd.DataFrame(report['threshold_comparison'])

            fig, axes = plt.subplots(1, 3, figsize=(16, 5))
            for ax in axes:
                ax.tick_params(colors='black')
                ax.set_facecolor('white')

            axes[0].plot(df_sens['threshold'],
                         df_sens['deduplication_rate'] * 100,
                         'b-o', linewidth=2, markersize=5)
            axes[0].set_title("Rată deduplicare vs Prag",
                              color='black', fontsize=12)
            axes[0].set_xlabel("Prag similaritate", color='black')
            axes[0].set_ylabel("Rată deduplicare (%)", color='black')
            axes[0].axvline(0.80, color='gray', linestyle='--', alpha=0.7)
            axes[0].grid(alpha=0.3)

            if 'false_merge_rate' in df_sens.columns:
                axes[1].plot(df_sens['threshold'],
                             df_sens['false_merge_rate'] * 100,
                             'r-s', linewidth=2, markersize=5)
                axes[1].set_title("False merge rate vs Prag",
                                  color='black', fontsize=12)
                axes[1].set_xlabel("Prag similaritate", color='black')
                axes[1].set_ylabel("False merge rate (%)", color='black')
                axes[1].axvline(0.80, color='gray', linestyle='--', alpha=0.7)
                axes[1].grid(alpha=0.3)

            if 'ari' in df_sens.columns:
                axes[2].plot(df_sens['threshold'],
                             df_sens['ari'],
                             'g-^', linewidth=2, markersize=5)
                axes[2].set_title("ARI vs Prag", color='black', fontsize=12)
                axes[2].set_xlabel("Prag similaritate", color='black')
                axes[2].set_ylabel("Adjusted Rand Index", color='black')
                axes[2].axvline(0.80, color='gray', linestyle='--',
                                alpha=0.7, label='Prag ales: 0.80')
                axes[2].legend(fontsize=10)
                axes[2].grid(alpha=0.3)

            fig.suptitle("Sensitivity Analysis — impact prag similaritate",
                         fontsize=13, fontweight='bold', color='black')
            plt.tight_layout()
            st.pyplot(fig)
            plt.close()


# tab 1 — deduplicare
with tab_dedup:
    st.subheader("Analiza deduplicării semantice")

    if 'df_dedup' not in data:
        st.info("Rulează agent.ipynb pentru a genera alarms_deduplicated.csv")
    else:
        df_d = data['df_dedup']

        # filtru severitate
        if selected_severity and 'severity' in df_d.columns:
            df_d = df_d[df_d['severity'].isin(selected_severity)]

        col1, col2 = st.columns(2)

        with col1:
            st.markdown("#### Confusion Matrix")
            if 'kpi_report' in data:
                dm = data['kpi_report'].get('deduplication', {})
                tp = dm.get('true_positive', 0)
                fp = dm.get('false_positive', 0)
                fn = dm.get('false_negative', 0)
                tn = dm.get('true_negative', 0)
                cm = np.array([[tn, fp], [fn, tp]])

                fig, ax = plt.subplots(figsize=(5, 4))
                fig.patch.set_facecolor('white')
                ax.set_facecolor('white')
                sns.heatmap(
                    cm, annot=True, fmt='d', cmap='Blues',
                    xticklabels=['Non-Dup', 'Dup'],
                    yticklabels=['Non-Dup', 'Dup'],
                    ax=ax, linewidths=0.5,
                    annot_kws={'size': 14, 'weight': 'bold', 'color': 'black'},
                )
                ax.set_xlabel('Predicție', color='black', fontsize=12)
                ax.set_ylabel('Adevărat', color='black', fontsize=12)
                ax.set_title('Confusion Matrix — deduplicare',
                             color='black', fontsize=12)
                ax.tick_params(colors='black')
                plt.tight_layout()
                st.pyplot(fig)
                plt.close()
            else:
                st.info("Necesită kpi_report.json")

        with col2:
            st.markdown("#### Precision / Recall / F1")
            if 'kpi_report' in data:
                dm = data['kpi_report'].get('deduplication', {})
                labels_prf = ['Precision', 'Recall', 'F1']
                vals_prf   = [dm.get('precision', 0),
                              dm.get('recall', 0),
                              dm.get('f1', 0)]
                colors_prf = ['#4C72B0', '#55A868', '#C44E52']

                fig, ax = plt.subplots(figsize=(5, 4))
                fig.patch.set_facecolor('white')
                ax.set_facecolor('white')
                bars = ax.bar(labels_prf, vals_prf, color=colors_prf,
                              width=0.5, edgecolor='white')
                ax.set_ylim(0, 1.15)
                ax.set_ylabel('Score', color='black', fontsize=12)
                ax.set_title('Precision / Recall / F1',
                             color='black', fontsize=12)
                ax.tick_params(colors='black')
                ax.set_xticks(range(len(labels_prf)))
                ax.set_xticklabels(labels_prf, color='black',
                                   fontweight='bold', fontsize=12)
                ax.axhline(1.0, color='gray', linestyle='--',
                           linewidth=0.8, alpha=0.5)
                for bar, val in zip(bars, vals_prf):
                    ax.text(bar.get_x() + bar.get_width()/2,
                            val + 0.02, f'{val:.4f}',
                            ha='center', va='bottom',
                            fontweight='bold', color='black', fontsize=11)
                plt.tight_layout()
                st.pyplot(fig)
                plt.close()

        st.divider()

        # distributie dimensiuni clustere
        st.markdown("#### Distribuția dimensiunilor clusterelor")

        if 'cluster_size' in df_d.columns:
            cluster_sizes = df_d.groupby('cluster_id')['cluster_size'].first()

            fig, axes = plt.subplots(1, 2, figsize=(14, 5))
            fig.patch.set_facecolor('white')
            for ax in axes:
                ax.set_facecolor('white')
                ax.tick_params(colors='black')

            axes[0].hist(cluster_sizes.values, bins=20,
                         color='#4E79A7', edgecolor='white')
            axes[0].set_title('Histograma dimensiuni clustere',
                              color='black', fontsize=12)
            axes[0].set_xlabel('Dimensiune cluster', color='black')
            axes[0].set_ylabel('Frecvență', color='black')

            size_counts = cluster_sizes.value_counts().sort_index()
            axes[1].bar(size_counts.index.astype(str), size_counts.values,
                        color='#F28E2B', edgecolor='white')
            axes[1].set_title('Câte clustere au N alarme',
                              color='black', fontsize=12)
            axes[1].set_xlabel('Dimensiune cluster (N alarme)', color='black')
            axes[1].set_ylabel('Număr clustere', color='black')
            for i, (idx, val) in enumerate(size_counts.items()):
                axes[1].text(i, val + 0.3, str(val),
                             ha='center', fontsize=10,
                             fontweight='bold', color='black')

            plt.tight_layout()
            st.pyplot(fig)
            plt.close()

        # timeline alarme
        st.markdown("#### Timeline — alarme per oră")

        if 'timestamp' in df_d.columns:
            timeline_all = df_d.set_index('timestamp').resample('1h').size()
            timeline_dup  = df_d[df_d.get('is_dedup_duplicate', False) == True
                                 if 'is_dedup_duplicate' in df_d.columns
                                 else df_d.index == -1
                                ].set_index('timestamp').resample('1h').size()

            fig, ax = plt.subplots(figsize=(14, 4))
            fig.patch.set_facecolor('white')
            ax.set_facecolor('white')
            ax.tick_params(colors='black')

            ax.plot(timeline_all.index, timeline_all.values,
                    color='#E24B4A', linewidth=2, label='Alarme originale')

            if 'is_dedup_duplicate' in df_d.columns:
                timeline_kept = df_d[df_d['is_dedup_duplicate'] == False
                               ].set_index('timestamp').resample('1h').size()
                ax.plot(timeline_kept.index, timeline_kept.values,
                        color='#2CA02C', linewidth=2,
                        label='Alarme după deduplicare')

            ax.set_title('Alarme per oră — înainte și după deduplicare',
                         color='black', fontsize=12)
            ax.set_xlabel('Timp', color='black')
            ax.set_ylabel('Număr alarme', color='black')
            leg = ax.legend(fontsize=10)
            for t in leg.get_texts(): t.set_color('black')
            ax.grid(alpha=0.3)
            plt.tight_layout()
            st.pyplot(fig)
            plt.close()

# tab 3 — agent llm
with tab_agent:
    st.subheader("Rezultatele agentului LLM — Gemini 1.5 Flash")

    if 'agent_results' not in data:
        st.info("Rulează agent.ipynb pentru a genera agent_results.json")
    else:
        results = data['agent_results']

        # filtrare dupa severitate
        if selected_severity:
            results_f = [r for r in results
                         if r.get('severity', '') in selected_severity]
        else:
            results_f = results

        col1, col2, col3 = st.columns(3)

        with col1:
            st.markdown("#### Distribuție severitate")
            sev_counts = pd.Series(
                [r.get('severity', 'UNKNOWN') for r in results_f]
            ).value_counts()
            fig, ax = plt.subplots(figsize=(5, 4))
            fig.patch.set_facecolor('white')
            ax.set_facecolor('white')
            ax.tick_params(colors='black')
            colors_sev = [COLOR_MAP.get(s, '#888')
                          for s in sev_counts.index]
            bars = ax.bar(sev_counts.index, sev_counts.values,
                          color=colors_sev, edgecolor='white', width=0.6)
            ax.set_title('Distribuție severitate clustere',
                         color='black', fontsize=11)
            ax.set_ylabel('Număr clustere', color='black')
            ax.set_xticks(range(len(sev_counts)))
            ax.set_xticklabels(sev_counts.index, color='black',
                               fontweight='bold', fontsize=10)
            for bar, val in zip(bars, sev_counts.values):
                ax.text(bar.get_x() + bar.get_width()/2,
                        val + 0.1, str(val),
                        ha='center', fontweight='bold', color='black')
            plt.tight_layout()
            st.pyplot(fig)
            plt.close()

        with col2:
            st.markdown("#### Acțiuni recomandate")
            action_counts = pd.Series(
                [r.get('action_required', 'none') for r in results_f]
            ).value_counts()
            fig, ax = plt.subplots(figsize=(5, 4))
            fig.patch.set_facecolor('white')
            colors_act = [ACTION_COLORS.get(a, '#888')
                          for a in action_counts.index]
            wedges, _, ats = ax.pie(
                action_counts.values,
                autopct='%1.0f%%',
                colors=colors_act,
                startangle=90,
                pctdistance=0.65,
                wedgeprops={'edgecolor': 'white', 'linewidth': 1.5},
            )
            for at in ats:
                at.set_fontsize(11)
                at.set_fontweight('bold')
                at.set_color('white')
            ax.set_title('Acțiuni recomandate', color='black', fontsize=11)
            leg = ax.legend(wedges,
                            [f"{a}: {v}" for a, v in action_counts.items()],
                            loc='lower center',
                            bbox_to_anchor=(0.5, -0.15),
                            fontsize=9, frameon=True)
            for t in leg.get_texts(): t.set_color('black')
            plt.tight_layout()
            st.pyplot(fig)
            plt.close()

        with col3:
            st.markdown("#### Timp procesare per cluster")
            times = [r.get('processing_time_s', 0)
                     for r in results_f if r.get('success')]
            if times:
                fig, ax = plt.subplots(figsize=(5, 4))
                fig.patch.set_facecolor('white')
                ax.set_facecolor('white')
                ax.tick_params(colors='black')
                ax.hist(times, bins=20, color='#7F77DD', edgecolor='white')
                ax.axvline(np.mean(times), color='#D62728', linestyle='--',
                           linewidth=2, label=f'Medie: {np.mean(times):.2f}s')
                ax.set_title('Distribuție timp procesare',
                             color='black', fontsize=11)
                ax.set_xlabel('Timp (secunde)', color='black')
                ax.set_ylabel('Frecvență', color='black')
                leg = ax.legend(fontsize=10)
                for t in leg.get_texts(): t.set_color('black')
                plt.tight_layout()
                st.pyplot(fig)
                plt.close()

        st.divider()

        # tool call completeness
        st.markdown("#### Tool call completeness — agentul a apelat toate 3 uneltele?")

        all_tools = {'classify_severity', 'generate_unified_alarm', 'explain_grouping'}
        complete   = sum(1 for r in results_f
                         if all_tools.issubset(set(r.get('tools_called', []))))
        incomplete = len(results_f) - complete
        failed     = sum(1 for r in results_f if not r.get('success', True))

        cc1, cc2, cc3 = st.columns(3)
        cc1.metric("Complete (3/3 tool-uri)", complete,
                   delta=f"{complete/len(results_f)*100:.0f}%" if results_f else "—")
        cc2.metric("Incomplete", incomplete)
        cc3.metric("Erori agent", failed)

        st.divider()

        # tabel rezultate agent
        st.markdown("#### Rezultate agent — detalii per cluster")

        rows_agent = [
            {
                "Cluster": r.get('cluster_id', '—'),
                "Dimensiune": r.get('cluster_size', '—'),
                "Severitate": r.get('severity', '—'),
                "Acțiune": r.get('action_required', '—'),
                "Mesaj unificat": r.get('unified_message', '—'),
                "Tool-uri": ', '.join(r.get('tools_called', [])),
                "Timp (s)": f"{r.get('processing_time_s', 0):.2f}",
                "Success": '✅' if r.get('success') else '❌',
            }
            for r in results_f
        ]

        if rows_agent:
            df_agent_display = pd.DataFrame(rows_agent)
            st.dataframe(df_agent_display, use_container_width=True, height=400)


# tab 4 — embeddings
with tab_embed:
    st.subheader("Spațiul de embeddings — vizualizare PCA")

    if 'embeddings' not in data or 'df_dedup' not in data:
        st.info("Necesită embeddings_minilm.npy și alarms_deduplicated.csv")
    else:
        embeddings = data['embeddings']
        df_e = data['df_dedup'].copy()

        # aliniaza dimensiunile
        n = min(len(embeddings), len(df_e))
        embeddings = embeddings[:n]
        df_e = df_e.iloc[:n]

        # PCA 2D
        pca = PCA(n_components=2, random_state=42)
        reduced = pca.fit_transform(embeddings)

        col1, col2 = st.columns(2)

        with col1:
            st.markdown("#### Clustere în spațiul embedding (PCA 2D)")
            fig, ax = plt.subplots(figsize=(7, 6))
            fig.patch.set_facecolor('white')
            ax.set_facecolor('white')
            ax.tick_params(colors='black')

            if 'cluster_id' in df_e.columns:
                scatter = ax.scatter(
                    reduced[:, 0], reduced[:, 1],
                    c=df_e['cluster_id'],
                    cmap='tab20', s=8, alpha=0.7,
                )
                cbar = plt.colorbar(scatter, ax=ax)
                cbar.set_label('Cluster ID', color='black')
                cbar.ax.tick_params(colors='black')

            ax.set_title('PCA — clustere de alarme similare',
                         color='black', fontsize=12)
            ax.set_xlabel(f'PC1 ({pca.explained_variance_ratio_[0]*100:.1f}%)',
                          color='black')
            ax.set_ylabel(f'PC2 ({pca.explained_variance_ratio_[1]*100:.1f}%)',
                          color='black')
            plt.tight_layout()
            st.pyplot(fig)
            plt.close()

        with col2:
            st.markdown("#### Colorat după severitate")
            fig, ax = plt.subplots(figsize=(7, 6))
            fig.patch.set_facecolor('white')
            ax.set_facecolor('white')
            ax.tick_params(colors='black')

            if 'severity' in df_e.columns:
                for sev in ['CRITICAL', 'HIGH', 'MEDIUM', 'LOW']:
                    mask = df_e['severity'] == sev
                    if mask.any() and sev in selected_severity:
                        ax.scatter(
                            reduced[mask, 0], reduced[mask, 1],
                            c=COLOR_MAP[sev], s=8, alpha=0.7,
                            label=sev,
                        )
            ax.set_title('PCA — colorat după severitate',
                         color='black', fontsize=12)
            ax.set_xlabel(f'PC1 ({pca.explained_variance_ratio_[0]*100:.1f}%)',
                          color='black')
            ax.set_ylabel(f'PC2 ({pca.explained_variance_ratio_[1]*100:.1f}%)',
                          color='black')
            leg = ax.legend(fontsize=10, title='Severitate')
            for t in leg.get_texts(): t.set_color('black')
            leg.get_title().set_color('black')
            plt.tight_layout()
            st.pyplot(fig)
            plt.close()

        st.caption(
            f"PCA explică {sum(pca.explained_variance_ratio_)*100:.1f}% "
            f"din varianța totală a {embeddings.shape[1]} dimensiuni."
        )


# tab 5 — alarme brute
with tab_raw:
    st.subheader("Alarme brute — date combinate (Etapa 1)")

    if 'df_raw' not in data:
        st.info("Rulează data_evaluation.ipynb pentru a genera alarms_combined.csv")
    else:
        df_r = data['df_raw'].copy()

        # filtru severitate
        if selected_severity and 'severity' in df_r.columns:
            df_r = df_r[df_r['severity'].isin(selected_severity)]

        # stats rapide
        c1, c2, c3, c4 = st.columns(4)
        c1.metric("Total alarme", len(df_r))
        c2.metric("Mesaje unice", df_r['message'].nunique()
                  if 'message' in df_r.columns else '—')
        c3.metric("Tipuri clinice", df_r['true_group'].nunique()
                  if 'true_group' in df_r.columns else '—')
        c4.metric("Surse", df_r['source'].nunique()
                  if 'source' in df_r.columns else '—')

        st.divider()

        # search
        search = st.text_input("🔍 Caută în mesaje", placeholder="ex: tachycardia")
        if search:
            df_r = df_r[df_r['message'].str.contains(search, case=False, na=False)]
            st.caption(f"{len(df_r)} rezultate pentru '{search}'")

        # top mesaje
        col1, col2 = st.columns([1, 2])
        with col1:
            st.markdown("#### Top 10 mesaje frecvente")
            if 'message' in df_r.columns:
                top = df_r['message'].value_counts().head(10).reset_index()
                top.columns = ['Mesaj', 'Frecvență']
                st.dataframe(top, use_container_width=True, height=320)

        with col2:
            st.markdown("#### Distribuție surse de date")
            if 'data_source' in df_r.columns:
                src_counts = df_r['data_source'].value_counts()
                fig, ax = plt.subplots(figsize=(6, 4))
                fig.patch.set_facecolor('white')
                ax.set_facecolor('white')
                ax.tick_params(colors='black')
                ax.bar(src_counts.index, src_counts.values,
                       color=['#4E79A7', '#F28E2B'],
                       edgecolor='white', width=0.5)
                ax.set_title('Alarme per sursă de date',
                             color='black', fontsize=12)
                ax.set_ylabel('Număr alarme', color='black')
                ax.set_xticks(range(len(src_counts)))
                ax.set_xticklabels(src_counts.index, color='black',
                                   fontweight='bold')
                for i, val in enumerate(src_counts.values):
                    ax.text(i, val + 2, str(val), ha='center',
                            fontweight='bold', color='black')
                plt.tight_layout()
                st.pyplot(fig)
                plt.close()

        st.divider()
        st.markdown("#### Tabel alarme brute")
        cols_show = [c for c in ['alarm_id', 'timestamp', 'message',
                                  'source', 'true_group', 'severity',
                                  'data_source']
                     if c in df_r.columns]
        st.dataframe(df_r[cols_show].sort_values('timestamp'),
                     use_container_width=True, height=400)

# tab 6 — clustere
with tab_clusters:
    st.subheader("Clustere formate — alarme grupate")

    if 'df_dedup' not in data:
        st.info("Rulează agent.ipynb pentru a genera alarms_deduplicated.csv")
    else:
        df_cl = data['df_dedup'].copy()

        if selected_severity and 'severity' in df_cl.columns:
            df_cl = df_cl[df_cl['severity'].isin(selected_severity)]

        # arata doar clusterele cu duplicate
        show_only_dups = st.checkbox("Arată doar clustere cu duplicate", value=True)

        if 'cluster_id' in df_cl.columns:
            cluster_sizes = df_cl.groupby('cluster_id').size()

            if show_only_dups:
                dup_cluster_ids = cluster_sizes[cluster_sizes > 1].index
                df_cl = df_cl[df_cl['cluster_id'].isin(dup_cluster_ids)]

            st.caption(f"Afișând {df_cl['cluster_id'].nunique()} clustere "
                       f"({len(df_cl)} alarme)")

            # cluster explorer
            st.markdown("#### Explorer clustere")
            selected_cluster = st.selectbox(
                "Selectează cluster",
                options=sorted(df_cl['cluster_id'].unique()),
                format_func=lambda x: f"Cluster #{x} "
                    f"({len(df_cl[df_cl['cluster_id']==x])} alarme)"
            )

            if selected_cluster is not None:
                df_sel = df_cl[df_cl['cluster_id'] == selected_cluster]
                sev_cl = df_sel['severity'].mode()[0] if 'severity' in df_sel.columns else '—'

                st.markdown(f"**Cluster #{selected_cluster}** — "
                            f"{len(df_sel)} alarme — Severitate: "
                            f"**:{sev_cl.lower()}[{sev_cl}]**")

                # mesaje din cluster
                for _, row in df_sel.iterrows():
                    is_rep = row.get('is_representative', False)
                    prefix = "🔹 **[REPREZENTATIV]**" if is_rep else "▪"
                    src = row.get('source', '—')
                    msg = row.get('message', '—')
                    ts  = str(row.get('timestamp', ''))[:19]
                    st.markdown(f"{prefix} `[{src}]` {msg} _{ts}_")

                # rezultat agent pentru acest cluster
                if 'agent_results' in data:
                    agent_r = next(
                        (r for r in data['agent_results']
                         if r.get('cluster_id') == selected_cluster),
                        None
                    )
                    if agent_r:
                        st.markdown("**Rezultat agent:**")
                        c1, c2 = st.columns(2)
                        c1.info(f"**Mesaj unificat:** {agent_r.get('unified_message', '—')}")
                        c2.info(f"**Explicație:** {agent_r.get('explanation', '—')}")

            st.divider()

            # tabel complet clustere
            st.markdown("#### Toate clusterele")
            cols_cl = [c for c in ['cluster_id', 'alarm_id', 'timestamp',
                                    'message', 'source', 'severity',
                                    'cluster_size', 'is_dedup_duplicate',
                                    'is_representative']
                       if c in df_cl.columns]
            st.dataframe(df_cl[cols_cl].sort_values('cluster_id'),
                         use_container_width=True, height=450)