import os
import io
import textwrap

import numpy as np
import pandas as pd
import streamlit as st
import seaborn as sns
import matplotlib.pyplot as plt

from scipy.stats import hypergeom
from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer
from reportlab.lib.styles import getSampleStyleSheet
from reportlab.lib.pagesizes import A4


# ---------------------------------------
# CONFIG & PATHWAYS DE BASE
# ---------------------------------------

# Quelques pathways “classiques” en dur, comme dans PATHWAY_ANALYSIS.py
BASE_PATHWAY_GENES = {
    "MAPK": [
        "HRAS", "KRAS", "NRAS",
        "BRAF", "RAF1",
        "MAP2K1", "MAP2K2",
        "MAPK1", "MAPK3",
        "ELK1", "FOS", "JUN", "JUND", "JUNB",
        "DUSP4", "DUSP6", "RASA1", "SOS1", "SOS2",
        "CBL", "YWHAQ", "YWHAB", "SFN", "NCK1", "NCK2",
    ],
    "PI3K_AKT": [
        "PIK3CA", "PIK3CB", "PIK3CD",
        "PIK3R1", "PIK3R2", "PIK3R3",
        "PTEN",
        "AKT1", "AKT2", "AKT3",
        "PDK1", "PDK2",
        "MTOR", "RPTOR", "RICTOR",
        "TSC1", "TSC2",
        "IRS1", "IRS2",
    ],
    "JAK_STAT": [
        "JAK1", "JAK2", "JAK3", "TYK2",
        "STAT1", "STAT3", "STAT5A", "STAT5B", "STAT2", "STAT4",
        "SOCS1", "SOCS3",
    ],
    "RTK": [
        "PDGFRB", "PDGFRA", "PDGFRL",
        "EGFR", "ERBB2", "ERBB3", "ERBB4",
        "FGFR1", "FGFR2", "FGFR3", "FGFR4",
        "MET", "KIT", "FLT3", "KDR", "VEGFR1", "VEGFR2",
        "RET", "ROS1", "ALK",
    ],
    "CYTOSKELETON_RHO": [
        "RHOA", "RHOB", "RHOC",
        "RAC1", "RAC2", "RAC3",
        "CDC42", "VAV1", "VAV2", "VAV3",
        "WASL",
        "ARPC1B", "ARPC2", "ARPC3", "ARPC4", "ARPC5",
        "ACTB", "ACTN4", "ITGB3", "ITGAV",
    ],
}


# ---------------------------------------
# OUTILS
# ---------------------------------------

def clean_text(val):
    if not isinstance(val, str):
        return ""
    import re
    cleaned = re.sub(r'[^A-Za-z0-9]+', '', val.strip().lower())
    return cleaned


def load_variants(uploaded_file, sep_guess="auto"):
    if uploaded_file is None:
        return None

    # Détection simple du séparateur
    if sep_guess == "auto":
        sample = uploaded_file.read().decode("utf-8", errors="ignore")
        uploaded_file.seek(0)
        if "\t" in sample:
            sep = "\t"
        elif ";" in sample:
            sep = ";"
        else:
            sep = ","
    else:
        sep = sep_guess

    df = pd.read_csv(uploaded_file, sep=sep, dtype=str)
    df = df.replace('"', '', regex=True)
    return df


def apply_filtering_and_scoring(
    df,
    allelic_ratio_min=0.02,
    gnomad_max=0.001,
    use_gnomad_filter=True,
    min_patho_score=4,
    use_msc=False,
    constraint_file=None,
    genes_exclude=None,
    patients_exclude=None,
):
    df = df.copy()

    # Standardisation du gène
    if "Gene_symbol" not in df.columns:
        st.error("La colonne 'Gene_symbol' est absente du fichier.")
        return None

    df["Gene_symbol"] = df["Gene_symbol"].str.upper().str.replace(" ", "")

    # Count initial
    initial_count = len(df)

    # Conversion numérique
    numeric_cols = ["gnomad_exomes_NFE_AF", "CADD_phred", "Allelic_ratio"]
    for col in numeric_cols:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce")

    # Nettoyage texte
    text_cols = ["Putative_impact", "Clinvar_significance", "Variant_effect", "hgvs.p", "hgvs.c"]
    for col in text_cols:
        if col in df.columns:
            df[col] = df[col].apply(clean_text)

    # Filtre Allelic ratio
    if "Allelic_ratio" not in df.columns:
        st.error("La colonne 'Allelic_ratio' est absente du fichier.")
        return None

    df = df[df["Allelic_ratio"] >= allelic_ratio_min]

    # Intégration MSC si demandé
    df["msc_weight"] = 0.0
    if use_msc and constraint_file is not None:
        try:
            constraint_df = pd.read_csv(constraint_file, sep="\t", dtype=str)
            if "gene" not in constraint_df.columns:
                st.warning("Le fichier de contrainte gnomAD ne contient pas la colonne 'gene'. MSC ignoré.")
            else:
                constraint_df["gene"] = constraint_df["gene"].str.upper().str.replace(" ", "")
                if "mis.z_score" in constraint_df.columns:
                    constraint_df = constraint_df.rename(columns={"mis.z_score": "mis_z"})
                elif "mis_z" not in constraint_df.columns:
                    st.warning("Aucune colonne 'mis.z_score' ou 'mis_z' dans le fichier gnomAD. MSC ignoré.")
                else:
                    pass

                if "mis_z" in constraint_df.columns:
                    constraint_df["mis_z"] = pd.to_numeric(constraint_df["mis_z"], errors="coerce")
                    constraint_df = constraint_df.groupby("gene", as_index=False)["mis_z"].max()
                    df = df.merge(constraint_df[["gene", "mis_z"]], left_on="Gene_symbol", right_on="gene", how="left")
                    df["mis_z"] = df["mis_z"].fillna(0)
                    df["msc_weight"] = df["mis_z"] * 0.5
        except Exception as e:
            st.warning(f"Impossible de charger le fichier gnomAD constraint : {e}. MSC ignoré.")

    # Exclusion gènes
    if genes_exclude:
        df = df[~df["Gene_symbol"].isin(genes_exclude)]

    # Exclusion patients
    if patients_exclude and "Pseudo" in df.columns:
        df = df[~df["Pseudo"].isin(patients_exclude)]

    # Filtre gnomAD
    if use_gnomad_filter and "gnomad_exomes_NFE_AF" in df.columns:
        df = df[(df["gnomad_exomes_NFE_AF"].isna()) | (df["gnomad_exomes_NFE_AF"] <= gnomad_max)]

    # Scoring putative
    df["score_putative"] = 0
    if "Putative_impact" in df.columns:
        df.loc[df["Putative_impact"] == "high", "score_putative"] = 3
        df.loc[df["Putative_impact"] == "moderate", "score_putative"] = 2
        df.loc[df["Putative_impact"] == "modifier", "score_putative"] = 2
        df.loc[df["Putative_impact"] == "low", "score_putative"] = 1

    # Scoring CADD
    df["score_cadd"] = 0
    if "CADD_phred" in df.columns:
        df.loc[df["CADD_phred"] >= 30, "score_cadd"] = 3
        df.loc[(df["CADD_phred"] >= 20) & (df["CADD_phred"] < 30), "score_cadd"] = 2
        df.loc[(df["CADD_phred"] >= 10) & (df["CADD_phred"] < 20), "score_cadd"] = 1

    # ClinVar
    df["score_clinvar"] = 0
    if "Clinvar_significance" in df.columns:
        cv = df["Clinvar_significance"]
        df.loc[cv.str.contains("pathogenic") & ~cv.str.contains("likely"), "score_clinvar"] = 5
        df.loc[cv.str.contains("likelypathogenic"), "score_clinvar"] = 4
        df.loc[cv.str.contains("conflict"), "score_clinvar"] = 3
        df.loc[cv.str.contains("uncertain"), "score_clinvar"] = 2
        df.loc[cv.str.contains("vus"), "score_clinvar"] = 2
        df.loc[cv.str.contains("benign"), "score_clinvar"] = 0
        df.loc[cv.str.contains("notprovided"), "score_clinvar"] = 0

    # Score total
    df["patho_score"] = df["score_putative"] + df["score_cadd"] + df["score_clinvar"] + df["msc_weight"]

    # Filtre sur patho_score
    df = df[df["patho_score"] >= min_patho_score]

    final_count = len(df)
    return df, initial_count, final_count


def build_gene_patient_matrix(df):
    if "Gene_symbol" not in df.columns or "Pseudo" not in df.columns:
        return None
    matrix = df.pivot_table(
        index="Gene_symbol",
        columns="Pseudo",
        values="patho_score",
        aggfunc="max",
        fill_value=0.0,
    )
    matrix = matrix.loc[(matrix != 0).any(axis=1)]
    matrix = matrix.loc[:, (matrix != 0).any(axis=0)]
    return matrix


def load_external_pathways(base_dict, pathways_dir="PATHWAYS"):
    """
    Charge des pathways supplémentaires depuis des fichiers GMT/TSV/CSV
    dans un dossier PATHWAYS (à mettre dans ton repo GitHub).
    """
    PATHWAY_GENES = {k: v[:] for k, v in base_dict.items()}
    if not os.path.exists(pathways_dir):
        return PATHWAY_GENES

    added = 0
    for fname in os.listdir(pathways_dir):
        fpath = os.path.join(pathways_dir, fname)
        if not os.path.isfile(fpath):
            continue
        if fname.lower().endswith(".gmt"):
            try:
                with open(fpath, "r") as f:
                    for line in f:
                        parts = line.strip().split("\t")
                        if len(parts) < 3:
                            continue
                        pw = parts[0].strip()
                        genes = [g.strip().upper() for g in parts[2:] if g.strip()]
                        if not pw or not genes:
                            continue
                        if pw not in PATHWAY_GENES:
                            PATHWAY_GENES[pw] = []
                        for g in genes:
                            if g not in PATHWAY_GENES[pw]:
                                PATHWAY_GENES[pw].append(g)
                                added += 1
            except Exception as e:
                st.warning(f"Erreur lecture GMT {fname}: {e}")
        elif fname.lower().endswith(".tsv") or fname.lower().endswith(".csv"):
            try:
                df_ext = pd.read_csv(fpath, sep="\t")
            except Exception:
                try:
                    df_ext = pd.read_csv(fpath)
                except Exception as e:
                    st.warning(f"Erreur lecture {fname}: {e}")
                    continue
            if not {"pathway", "gene"}.issubset(df_ext.columns):
                st.warning(f"{fname} ne contient pas les colonnes 'pathway' et 'gene'. Ignoré.")
                continue
            for _, row in df_ext.iterrows():
                pw = str(row["pathway"]).strip()
                g = str(row["gene"]).strip().upper()
                if not pw or not g:
                    continue
                if pw not in PATHWAY_GENES:
                    PATHWAY_GENES[pw] = []
                if g not in PATHWAY_GENES[pw]:
                    PATHWAY_GENES[pw].append(g)
                    added += 1
    if added > 0:
        st.info(f"Pathways supplémentaires chargés : {added} associations gène-pathway ajoutées.")
    return PATHWAY_GENES


def compute_pathway_scores(df, pathway_genes):
    if "Gene_symbol" not in df.columns or "Pseudo" not in df.columns:
        return pd.DataFrame()
    df_gene_patient = df.groupby(["Pseudo", "Gene_symbol"], as_index=False)["patho_score"].max()
    patients = sorted(df_gene_patient["Pseudo"].unique())
    scores = {}
    for pw, genes in pathway_genes.items():
        subset = df_gene_patient[df_gene_patient["Gene_symbol"].isin(genes)]
        if subset.empty:
            scores[pw] = [0.0] * len(patients)
        else:
            s = subset.groupby("Pseudo")["patho_score"].sum()
            scores[pw] = [s.get(p, 0.0) for p in patients]
    df_scores = pd.DataFrame(scores, index=patients)
    df_scores.index.name = "Pseudo"
    return df_scores


def compute_pathway_enrichment(df, pathway_genes):
    universe_genes = sorted({g for gl in pathway_genes.values() for g in gl})
    universe_set = set(universe_genes)
    mutated_genes = sorted(set(df["Gene_symbol"].unique()) & universe_set)
    N = len(universe_genes)
    n = len(mutated_genes)
    if N == 0 or n == 0:
        return pd.DataFrame(columns=["pathway", "k_overlap", "n_mutated", "M_pathway_genes",
                                     "N_universe", "p_value", "FDR", "overlap_genes"])
    rows = []
    for pw, genes in pathway_genes.items():
        pw_genes = sorted(set(genes))
        M = len(pw_genes)
        overlap = sorted(set(mutated_genes) & set(pw_genes))
        k = len(overlap)
        if M == 0 or k == 0:
            pval = 1.0
        else:
            pval = hypergeom.sf(k - 1, N, M, n)
        rows.append({
            "pathway": pw,
            "k_overlap": k,
            "n_mutated": n,
            "M_pathway_genes": M,
            "N_universe": N,
            "p_value": pval,
            "overlap_genes": ",".join(overlap) if k > 0 else ""
        })
    df_enrich = pd.DataFrame(rows)
    m = len(df_enrich)
    if m > 0:
        df_enrich = df_enrich.sort_values("p_value").reset_index(drop=True)
        ranks = np.arange(1, m + 1)
        bh_values = df_enrich["p_value"] * m / ranks
        bh_corrected = np.minimum.accumulate(bh_values[::-1])[::-1]
        df_enrich["FDR"] = np.clip(bh_corrected, 0, 1)
        df_enrich = df_enrich.sort_values(["FDR", "p_value", "pathway"]).reset_index(drop=True)
    else:
        df_enrich["FDR"] = []
    return df_enrich


def make_pdf_report(df_filtered, df_scores, df_enrich, buffer):
    styles = getSampleStyleSheet()
    story = []

    story.append(Paragraph("<b>NGS Variant Pathway Analysis Report</b>", styles["Title"]))
    story.append(Spacer(1, 12))

    story.append(Paragraph(f"Nombre total de variants après filtrage : {len(df_filtered)}", styles["Normal"]))
    story.append(Spacer(1, 12))

    story.append(Paragraph("<b>Scores de pathways par patient</b>", styles["Heading2"]))
    story.append(Spacer(1, 12))
    if not df_scores.empty:
        story.append(Paragraph(df_scores.to_string().replace("\n", "<br/>"), styles["Code"]))
    else:
        story.append(Paragraph("Aucun score de pathway calculé.", styles["Normal"]))
    story.append(Spacer(1, 12))

    story.append(Paragraph("<b>Enrichissement de pathways</b>", styles["Heading2"]))
    story.append(Spacer(1, 12))
    if df_enrich is not None and not df_enrich.empty:
        story.append(Paragraph(df_enrich.to_string(index=False).replace("\n", "<br/>"), styles["Code"]))
    else:
        story.append(Paragraph("Aucun pathway enrichi détecté ou enrichissement non calculé.", styles["Normal"]))
    story.append(Spacer(1, 12))

    doc = SimpleDocTemplate(buffer, pagesize=A4)
    doc.build(story)
    buffer.seek(0)


# ---------------------------------------
# STREAMLIT APP
# ---------------------------------------

st.set_page_config(page_title="NGS Variant Pathway Explorer", layout="wide")

st.title("🧬 NGS Variant Pathway Explorer")
st.markdown(
    "Cette application permet de **filtrer, scorer et explorer** des variants NGS "
    "ainsi que d'analyser leur **impact sur les voies de signalisation**."
)

# Upload du fichier de variants
st.sidebar.header("1️⃣ Chargement des données")
uploaded_file = st.sidebar.file_uploader("Importer un fichier de variants (CSV/TSV)", type=["csv", "tsv", "txt"])

sep_choice = st.sidebar.selectbox(
    "Séparateur (détection auto recommandée)",
    options=["auto", "tabulation (\\t)", "virgule (,)", "point-virgule (;)"],
    index=0
)
sep_map = {
    "auto": "auto",
    "tabulation (\\t)": "\t",
    "virgule (,)": ",",
    "point-virgule (;)": ";",
}

# Paramètres de filtrage
st.sidebar.header("2️⃣ Paramètres de filtrage")

allelic_ratio_min = st.sidebar.slider("Seuil minimal Allelic_ratio", 0.0, 0.5, 0.02, step=0.005)
use_gnomad_filter = st.sidebar.checkbox("Filtrer sur gnomAD_exomes_NFE_AF", value=True)
gnomad_max = st.sidebar.number_input("Seuil max gnomAD_exomes_NFE_AF", min_value=0.0, max_value=0.05, value=0.001, step=0.0005)
min_patho_score = st.sidebar.slider("Seuil minimal patho_score", 0.0, 30.0, 4.0, step=0.5)

use_msc = st.sidebar.checkbox("Utiliser le score MSC (gnomAD constraint)", value=False)
constraint_file = None
if use_msc:
    constraint_upload = st.sidebar.file_uploader("Fichier gnomAD constraint (TSV)", type=["tsv"])
    if constraint_upload is not None:
        # on sauvegarde dans un buffer temporaire
        constraint_file = io.StringIO(constraint_upload.read().decode("utf-8"))

st.sidebar.header("3️⃣ Exclusions")

genes_exclude_input = st.sidebar.text_area("Gènes à exclure (séparés par des virgules)", value="KMT2C, CHEK2, BCR")
patients_exclude_input = st.sidebar.text_area("Patients à exclure (Pseudo, séparés par des virgules)", value="")

# Boutons de contrôle
st.sidebar.header("4️⃣ Analyse")
run_filter = st.sidebar.button("Lancer le filtrage et l'analyse")


# ---------------------------------------
# MAIN LOGIC
# ---------------------------------------

if uploaded_file is not None and run_filter:
    with st.spinner("Chargement et filtrage des variants..."):
        df_raw = load_variants(uploaded_file, sep_guess=sep_map[sep_choice])

        if df_raw is None:
            st.error("Impossible de charger le fichier.")
        else:
            # Parsing des listes
            genes_exclude = [g.strip().upper() for g in genes_exclude_input.split(",") if g.strip()] or None
            patients_exclude = [p.strip() for p in patients_exclude_input.split(",") if p.strip()] or None

            result = apply_filtering_and_scoring(
                df_raw,
                allelic_ratio_min=allelic_ratio_min,
                gnomad_max=gnomad_max,
                use_gnomad_filter=use_gnomad_filter,
                min_patho_score=min_patho_score,
                use_msc=use_msc,
                constraint_file=constraint_file,
                genes_exclude=genes_exclude,
                patients_exclude=patients_exclude,
            )

            if result is None:
                st.stop()

            df_filtered, initial_count, final_count = result

            st.subheader("📊 Statistiques de filtrage")
            st.write(f"- Variants initiaux : **{initial_count}**")
            st.write(f"- Variants après filtrage : **{final_count}**")
            st.write(f"- Variants supprimés : **{initial_count - final_count}**")

            st.subheader("🧾 Tableau filtré")
            st.dataframe(df_filtered)

            # Téléchargement CSV
            csv_buf = df_filtered.to_csv(sep="\t", index=False).encode("utf-8")
            st.download_button(
                "📥 Télécharger le tableau filtré (TSV)",
                data=csv_buf,
                file_name="Filtered_variants_scored.tsv",
                mime="text/tab-separated-values",
            )

            # Heatmap gène × patient
            st.subheader("🔥 Heatmap brute (patho_score max par gène/patient)")
            matrix_raw = build_gene_patient_matrix(df_filtered)
            if matrix_raw is None or matrix_raw.empty:
                st.info("Pas assez de données pour générer la heatmap gènes × patients.")
            else:
                fig_hm, ax_hm = plt.subplots(figsize=(10, 8))
                sns.heatmap(matrix_raw, cmap="viridis", ax=ax_hm)
                plt.tight_layout()
                st.pyplot(fig_hm)

            # Pathways : load base + externes
            st.subheader("🧬 Analyse par pathways")

            PATHWAY_GENES = load_external_pathways(BASE_PATHWAY_GENES, pathways_dir="PATHWAYS")

            # Scores de pathways
            df_scores = compute_pathway_scores(df_filtered, PATHWAY_GENES)
            if df_scores.empty:
                st.info("Impossible de calculer les scores de pathways (vérifier colonnes Gene_symbol / Pseudo).")
            else:
                st.markdown("**Scores de pathways par patient (somme des patho_scores des gènes du pathway)**")
                st.dataframe(df_scores)

                # Heatmap pathways × patients
                st.subheader("🌐 Heatmap pathways × patients")
                matrix_pw = df_scores.T
                if not matrix_pw.empty:
                    cluster = sns.clustermap(
                        matrix_pw,
                        cmap="viridis",
                        linewidths=0.2,
                        linecolor="grey",
                        figsize=(10, 8),
                        metric="euclidean",
                        method="ward",
                    )
                    cluster.fig.suptitle("Heatmap des scores de pathways par patient", y=1.02)
                    st.pyplot(cluster.fig)
                    plt.close(cluster.fig)
                else:
                    st.info("Pas assez de données pour la heatmap pathways × patients.")

                # Enrichissement
                st.subheader("📈 Enrichissement de pathways (hypergéométrique + FDR BH)")
                df_enrich = compute_pathway_enrichment(df_filtered, PATHWAY_GENES)
                if df_enrich.empty:
                    st.info("Pas de pathways enrichis (aucun overlap entre gènes mutés et univers de pathways).")
                else:
                    st.dataframe(df_enrich)

                    # Barplot -log10(FDR)
                    df_plot = df_enrich.copy()
                    df_plot["FDR"] = df_plot["FDR"].replace(0, 1e-300)
                    df_plot["minus_log10_FDR"] = -np.log10(df_plot["FDR"])
                    df_plot = df_plot.sort_values("minus_log10_FDR", ascending=False).head(15)

                    fig_bp, ax_bp = plt.subplots(figsize=(8, 6))
                    sns.barplot(
                        x="minus_log10_FDR",
                        y="pathway",
                        data=df_plot,
                        orient="h",
                        ax=ax_bp
                    )
                    ax_bp.set_xlabel("-log10(FDR)")
                    ax_bp.set_ylabel("Pathway")
                    plt.tight_layout()
                    st.pyplot(fig_bp)

            # PDF report
            st.subheader("📄 Rapport PDF")

            if st.button("Générer un rapport PDF"):
                pdf_buffer = io.BytesIO()
                enrich_for_pdf = df_enrich if "df_enrich" in locals() else None
                make_pdf_report(df_filtered, df_scores, enrich_for_pdf, pdf_buffer)

                st.download_button(
                    "📥 Télécharger le rapport PDF",
                    data=pdf_buffer,
                    file_name="NGS_Pathway_Report.pdf",
                    mime="application/pdf",
                )

else:
    st.info("Charge un fichier de variants dans la barre latérale, ajuste les paramètres, puis clique sur **Lancer le filtrage et l'analyse**.")