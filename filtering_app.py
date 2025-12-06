import os
import io
import re
import numpy as np
import pandas as pd
import streamlit as st
import plotly.express as px
import plotly.graph_objects as go
from scipy.stats import hypergeom

# ---------------------------------------
# 1. CONFIGURATION & OUTILS
# ---------------------------------------

st.set_page_config(page_title="NGS ATLAS Explorer v7.0", layout="wide", page_icon="🧬")

# --- Fonctions utilitaires ---

def clean_text(val):
    if not isinstance(val, str): return ""
    return re.sub(r'[^A-Za-z0-9]+', '', val.strip().lower())

def extract_ref_alt_chr(df):
    """
    Tente de créer/nettoyer les colonnes Chromosome, Ref, Alt si elles sont absentes
    mais présentes dans une colonne 'Variant' (ex: 1:1234:A>G ou chr1:1234:A:G).
    """
    # 1. Chromosome
    if "Chromosome" not in df.columns and "Chr" in df.columns:
        df["Chromosome"] = df["Chr"]
    
    # Si toujours pas de Chr, tentative extraction depuis Variant
    if "Chromosome" not in df.columns and "Variant" in df.columns:
        # Suppose format "chr1:..." ou "1:..."
        df["Chromosome"] = df["Variant"].astype(str).str.split(r'[:_-]', n=1, expand=True)[0]

    # Nettoyage Chromosome (standardisation 1, 2, ..., X, Y)
    if "Chromosome" in df.columns:
        df["Chromosome"] = df["Chromosome"].astype(str).str.replace("chr", "", case=False).str.strip()

    # 2. Ref / Alt
    if ("Ref" not in df.columns or "Alt" not in df.columns) and "Variant" in df.columns:
        # Essai format "Ref>Alt" (ex: "G>T") à la fin de la string
        # Regex simple qui cherche [ACGT]>[ACGT] ou [ACGT]:[ACGT]
        extracted = df["Variant"].astype(str).str.extract(r'([ACGT]+)[>:/]([ACGT]+)$', flags=re.IGNORECASE)
        if not extracted.isna().all().all():
            if "Ref" not in df.columns: df["Ref"] = extracted[0].str.upper()
            if "Alt" not in df.columns: df["Alt"] = extracted[1].str.upper()

    return df

@st.cache_data
def load_variants(uploaded_file, sep_guess="auto"):
    if uploaded_file is None: return None
    if sep_guess == "auto":
        sample = uploaded_file.read(4096).decode("utf-8", errors="ignore")
        uploaded_file.seek(0)
        sep = ";" if ";" in sample and "," not in sample else ("\t" if "\t" in sample else ",")
    else: sep = sep_guess
    try:
        # on_bad_lines='skip' pour éviter le crash sur une ligne malformée
        df = pd.read_csv(uploaded_file, sep=sep, dtype=str, on_bad_lines='skip')
        df = df.replace('"', '', regex=True)
        # Prétraitement structurel
        df = extract_ref_alt_chr(df)
        return df
    except Exception as e:
        st.error(f"Erreur lecture : {e}")
        return None

# Fonctions pour générer les URLs
def make_varsome_link(variant_str):
    try:
        v = str(variant_str).replace(">", ":")
        return f"https://varsome.com/variant/hg19/{v}"
    except: return ""

def make_gnomad_link(variant_str):
    try:
        v = str(variant_str).replace(":", "-").replace(">", "-")
        return f"https://gnomad.broadinstitute.org/variant/{v}?dataset=gnomad_r2_1"
    except: return ""

# ---------------------------------------
# 2. LOGIQUE DE FILTRAGE
# ---------------------------------------

@st.cache_data
def apply_filtering_and_scoring(
    df, allelic_ratio_min, gnomad_max, use_gnomad_filter, min_patho_score, use_patho_filter,
    min_depth, min_alt_depth, max_cohort_freq, use_msc, constraint_file_content,
    genes_exclude, patients_exclude, min_cadd,
    variant_effect_keep, putative_keep, clinvar_keep, sort_by_column,
    use_acmg
):
    logs = [] 
    df = df.copy()
    initial_total = len(df)
    logs.append({"Etape": "1. Import Brut", "Restants": initial_total, "Perdus": 0})

    if "Gene_symbol" not in df.columns:
        return None, 0, 0, "Colonne 'Gene_symbol' manquante.", []

    df["Gene_symbol"] = df["Gene_symbol"].str.upper().str.replace(" ", "")

    # Remplissage
    for col in ["Variant_effect", "Putative_impact", "Clinvar_significance"]:
        if col in df.columns: df[col] = df[col].fillna("Non Renseigné")

    # Fréquence Interne
    if "Pseudo" in df.columns and "Variant" in df.columns:
        tot = df["Pseudo"].nunique()
        cts = df.groupby("Variant")["Pseudo"].nunique()
        df["internal_freq"] = df["Variant"].map(cts) / tot
        df["Fréquence_Cohorte"] = df["Variant"].map(cts).astype(str) + "/" + str(tot)
    else:
        df["internal_freq"] = 0.0
        df["Fréquence_Cohorte"] = "N/A"

    # Conversions numériques
    cols_num = ["gnomad_exomes_NFE_AF", "CADD_phred", "Allelic_ratio", "Depth", "Alt_depth_total", "patho_score"]
    for c in cols_num:
        if c in df.columns: df[c] = pd.to_numeric(df[c], errors="coerce")
    
    if "Alt_depth_total" not in df.columns and "Alt_depth" in df.columns:
        # Parfois Alt_depth est "12,0", on prend le premier
        df["Alt_depth_total"] = df["Alt_depth"].astype(str).str.split(',').str[0].str.split(' ').str[0]
        df["Alt_depth_total"] = pd.to_numeric(df["Alt_depth_total"], errors='coerce').fillna(0)

    # --- FILTRES ---
    last_count = len(df)

    if "Depth" in df.columns: df = df[df["Depth"] >= min_depth]
    logs.append({"Etape": "2. Profondeur", "Restants": len(df), "Perdus": last_count - len(df)}); last_count = len(df)

    if "Alt_depth_total" in df.columns: df = df[df["Alt_depth_total"] >= min_alt_depth]
    logs.append({"Etape": "3. Reads Mutés", "Restants": len(df), "Perdus": last_count - len(df)}); last_count = len(df)

    if "Allelic_ratio" in df.columns: df = df[df["Allelic_ratio"] >= allelic_ratio_min]
    logs.append({"Etape": "4. VAF", "Restants": len(df), "Perdus": last_count - len(df)}); last_count = len(df)

    if max_cohort_freq < 1.0: df = df[df["internal_freq"] <= max_cohort_freq]
    logs.append({"Etape": "5. Freq Cohorte", "Restants": len(df), "Perdus": last_count - len(df)}); last_count = len(df)

    if genes_exclude: df = df[~df["Gene_symbol"].isin(genes_exclude)]
    if patients_exclude and "Pseudo" in df.columns: df = df[~df["Pseudo"].isin(patients_exclude)]
    logs.append({"Etape": "6. Exclusions", "Restants": len(df), "Perdus": last_count - len(df)}); last_count = len(df)

    if use_gnomad_filter and "gnomad_exomes_NFE_AF" in df.columns:
        df = df[(df["gnomad_exomes_NFE_AF"].isna()) | (df["gnomad_exomes_NFE_AF"] <= gnomad_max)]
    logs.append({"Etape": "7. gnomAD", "Restants": len(df), "Perdus": last_count - len(df)}); last_count = len(df)

    if variant_effect_keep and "Variant_effect" in df.columns: df = df[df["Variant_effect"].isin(variant_effect_keep)]
    if putative_keep and "Putative_impact" in df.columns: df = df[df["Putative_impact"].isin(putative_keep)]
    if clinvar_keep and "Clinvar_significance" in df.columns: df = df[df["Clinvar_significance"].isin(clinvar_keep)]
    logs.append({"Etape": "8. Catégories", "Restants": len(df), "Perdus": last_count - len(df)}); last_count = len(df)

    if min_cadd is not None and min_cadd > 0 and "CADD_phred" in df.columns:
        df = df[df["CADD_phred"] >= min_cadd]
    logs.append({"Etape": "9. CADD", "Restants": len(df), "Perdus": last_count - len(df)}); last_count = len(df)

    # --- GENERATION DES LIENS ---
    if "Variant" in df.columns:
        df["link_varsome"] = df["Variant"].apply(make_varsome_link)
        df["link_gnomad"] = df["Variant"].apply(make_gnomad_link)
    else:
        df["link_varsome"] = ""
        df["link_gnomad"] = ""

    # --- SCORING ---
    df["msc_weight"] = 0.0
    if use_msc and constraint_file_content:
        try:
            c_df = pd.read_csv(io.StringIO(constraint_file_content), sep="\t", dtype=str)
            if "gene" in c_df.columns:
                c_df["gene"] = c_df["gene"].str.upper().str.replace(" ", "")
                if "mis_z" in c_df.columns: c_df["mis_z"] = pd.to_numeric(c_df["mis_z"], errors="coerce")
                elif "mis.z_score" in c_df.columns: c_df["mis_z"] = pd.to_numeric(c_df["mis.z_score"], errors="coerce")
                
                if "mis_z" in c_df.columns:
                    c_df = c_df.groupby("gene", as_index=False)["mis_z"].max()
                    df = df.merge(c_df[["gene", "mis_z"]], left_on="Gene_symbol", right_on="gene", how="left")
                    df["mis_z"] = df["mis_z"].fillna(0)
                    df["msc_weight"] = df["mis_z"] * 0.5
        except: pass

    df["score_putative"] = 0
    if "Putative_impact" in df.columns:
        def get_imp(x):
            s = str(x).lower()
            if "high" in s: return 3
            if "moderate" in s or "modifier" in s: return 2
            if "low" in s: return 1
            return 0
        df["score_putative"] = df["Putative_impact"].apply(get_imp)

    df["score_cadd"] = 0
    if "CADD_phred" in df.columns:
        df["score_cadd"] = np.select([(df["CADD_phred"] >= 30), (df["CADD_phred"] >= 20)], [3, 2], default=0)

    df["score_clinvar"] = 0
    if "Clinvar_significance" in df.columns:
        cv = df["Clinvar_significance"].astype(str).str.lower()
        df["score_clinvar"] = np.select(
            [cv.str.contains("pathogenic") & ~cv.str.contains("likely"), cv.str.contains("uncertain")],
            [5, 2], default=0
        )

    df["patho_score"] = df["score_putative"] + df["score_cadd"] + df["score_clinvar"] + df["msc_weight"]

    # --- ACMG ---
    if use_acmg:
        def compute_acmg_class(row):
            eff = str(row.get("Variant_effect", "")).lower()
            pvs1 = any(x in eff for x in ["stopgained", "frameshift", "splice_acceptor", "splice_donor"])
            af = row.get("gnomad_exomes_NFE_AF", np.nan)
            pm2 = pd.isna(af) or (af < 0.0001)
            cadd = row.get("CADD_phred", 0)
            pp3 = cadd >= 25
            clv = str(row.get("Clinvar_significance", "")).lower()
            pp5 = "pathogenic" in clv and "conflict" not in clv
            ba1 = (af > 0.05) if not pd.isna(af) else False
            
            criteria_met = []
            if ba1: return "Benign", "BA1"
            score = 0
            if pvs1: score += 4; criteria_met.append("PVS1")
            if pm2: score += 2; criteria_met.append("PM2")
            if pp5: score += 2; criteria_met.append("PP5")
            if pp3: score += 1; criteria_met.append("PP3")
            
            final_class = "VUS"
            if score >= 5: final_class = "Pathogenic"
            elif score >= 3: final_class = "Likely Pathogenic"
            return final_class, ", ".join(criteria_met)

        acmg_res = df.apply(compute_acmg_class, axis=1, result_type='expand')
        df["ACMG_Class"] = acmg_res[0]
        df["ACMG_Criteria"] = acmg_res[1]
    else:
        df["ACMG_Class"] = "Non calculé"
        df["ACMG_Criteria"] = ""

    if use_patho_filter:
        df = df[df["patho_score"] >= min_patho_score]
    logs.append({"Etape": "10. Score Final", "Restants": len(df), "Perdus": last_count - len(df)})

    # Tri
    if sort_by_column == "Score Pathogénique (Décroissant)": df = df.sort_values("patho_score", ascending=False)
    elif sort_by_column == "Classification ACMG": df = df.sort_values("ACMG_Class", ascending=True)
    elif sort_by_column == "Fréquence Cohorte (Décroissant)": df = df.sort_values("internal_freq", ascending=False)
    elif sort_by_column == "Patient (A-Z)": 
        if "Pseudo" in df.columns: df = df.sort_values("Pseudo", ascending=True)

    return df, initial_total, len(df), None, logs

# ---------------------------------------
# 3. ANALYSE PATHWAYS
# ---------------------------------------

def parse_pathway_files(uploaded_files):
    pathways = {}
    if not uploaded_files: return pathways
    for up_file in uploaded_files:
        content = up_file.read().decode("utf-8", errors="ignore")
        fname = up_file.name.lower()
        if fname.endswith(".gmt"):
            for line in content.splitlines():
                parts = line.strip().split("\t")
                if len(parts) >= 3:
                    pw = parts[0].strip()
                    genes = [g.strip().upper() for g in parts[2:] if g.strip()]
                    if pw and genes:
                        if pw not in pathways: pathways[pw] = []
                        pathways[pw].extend(genes)
    for k in pathways: pathways[k] = list(set(pathways[k]))
    return pathways

@st.cache_data
def compute_enrichment(df, pathway_genes):
    universe = sorted({g for gl in pathway_genes.values() for g in gl})
    N = len(universe)
    mutated = sorted(set(df["Gene_symbol"].unique()) & set(universe))
    n = len(mutated)
    if N == 0 or n == 0: return pd.DataFrame()

    rows = []
    for pw, genes in pathway_genes.items():
        M = len(genes)
        overlap = set(mutated) & set(genes)
        k = len(overlap)
        if k > 0:
            pval = hypergeom.sf(k - 1, N, M, n)
            patients_str = "N/A"
            if "Pseudo" in df.columns:
                subset = df[df["Gene_symbol"].isin(overlap)]
                affected_patients = sorted(subset["Pseudo"].unique())
                patients_str = ", ".join(affected_patients)

            rows.append({
                "pathway": pw, "p_value": pval, "k_overlap": k, 
                "genes": ",".join(overlap), "patients": patients_str
            })

    df_res = pd.DataFrame(rows)
    if not df_res.empty:
        df_res = df_res.sort_values("p_value")
        m = len(df_res)
        df_res["rank"] = np.arange(1, m + 1)
        df_res["FDR"] = np.clip(np.minimum.accumulate(((df_res["p_value"] * m) / df_res["rank"])[::-1])[::-1], 0, 1)
        df_res["minus_log10_FDR"] = -np.log10(df_res["FDR"] + 1e-300)
    return df_res

# ---------------------------------------
# 4. INTERFACE
# ---------------------------------------

st.title("🧬 NGS ATLAS Explorer v7.0")
st.markdown("---")

if "analysis_done" not in st.session_state:
    st.session_state["analysis_done"] = False
    st.session_state["df_res"] = None
    st.session_state["kpis"] = (0, 0)
    st.session_state["logs"] = []

with st.sidebar:
    st.header("1. Données")
    uploaded_file = st.file_uploader("Fichier Variants", type=["csv", "tsv", "txt"])
    df_raw = None
    v_opts, p_opts, c_opts = [], [], []

    if uploaded_file:
        df_raw = load_variants(uploaded_file)
        if df_raw is not None:
            if "Variant_effect" in df_raw.columns: v_opts = sorted(df_raw["Variant_effect"].fillna("Non Renseigné").unique())
            if "Putative_impact" in df_raw.columns: p_opts = sorted(df_raw["Putative_impact"].fillna("Non Renseigné").unique())
            if "Clinvar_significance" in df_raw.columns: c_opts = sorted(df_raw["Clinvar_significance"].fillna("Non Renseigné").unique())

    with st.form("params"):
        st.header("2. Paramètres")
        sort_choice = st.selectbox("Tri", ["Score Pathogénique (Décroissant)", "Classification ACMG", "Fréquence Cohorte (Décroissant)", "Patient (A-Z)"])
        
        c1, c2 = st.columns(2)
        with c1:
            min_dp = st.number_input("Depth Min", 0, 10000, 50)
            allelic_min = st.number_input("VAF Min", 0.0, 1.0, 0.02)
        with c2:
            min_ad = st.number_input("Alt Depth Min", 0, 1000, 5)
            max_cohort_freq = st.slider("Max Freq Cohorte", 0.0, 1.0, 1.0, 0.05)

        c3, c4 = st.columns(2)
        with c3:
            min_patho = st.number_input("Score Patho Min", 0.0, 50.0, 4.0)
            gnomad_max = st.number_input("gnomAD Max", 0.0, 1.0, 0.001, format="%.4f")
        with c4:
            min_cadd_val = st.number_input("CADD Min (0=all)", 0.0, 60.0, 0.0)
            use_acmg = st.checkbox("Activer classification ACMG (Auto)", value=False)

        with st.expander("Avancé"):
            sel_var = st.multiselect("Effet", v_opts, default=v_opts)
            sel_put = st.multiselect("Impact", p_opts, default=p_opts)
            sel_clin = st.multiselect("ClinVar", c_opts, default=c_opts)
            genes_ex = st.text_area("Exclure Gènes", "KMT2C, CHEK2, TTN, MUC16")
            pseudo_ex = st.text_area("Exclure Patients", "")
            use_gnomad = st.checkbox("Filtre gnomAD", True)
            use_patho = st.checkbox("Filtre Patho", True)
            use_msc = st.checkbox("MSC", False)
            msc_file = st.file_uploader("Fichier MSC", type=["tsv"])

        st.header("3. Pathways")
        path_files = st.file_uploader("Fichiers GMT", accept_multiple_files=True)
        
        submitted = st.form_submit_button("🚀 LANCER L'ANALYSE")

if submitted and df_raw is not None:
    g_list = [x.strip().upper() for x in genes_ex.split(",") if x.strip()]
    p_list = [x.strip() for x in pseudo_ex.split(",") if x.strip()]
    msc_c = msc_file.read().decode("utf-8") if (use_msc and msc_file) else None

    res, ini, fin, err, logs = apply_filtering_and_scoring(
        df_raw, allelic_min, gnomad_max, use_gnomad, min_patho, use_patho, 
        min_dp, min_ad, max_cohort_freq, use_msc, msc_c, g_list, p_list, 
        min_cadd_val, sel_var, sel_put, sel_clin, sort_choice, use_acmg
    )

    if err:
        st.error(err)
    else:
        st.session_state["analysis_done"] = True
        st.session_state["df_res"] = res
        st.session_state["kpis"] = (ini, fin)
        st.session_state["logs"] = logs
        st.session_state["use_acmg"] = use_acmg
        
        user_pathways = parse_pathway_files(path_files)
        if user_pathways:
            df_enr = compute_enrichment(res, user_pathways)
            st.session_state["df_enr"] = df_enr
        else:
            st.session_state["df_enr"] = pd.DataFrame()

# ---------------------------------------
# 5. AFFICHAGE DES RESULTATS
# ---------------------------------------

if st.session_state["analysis_done"]:
    df_res = st.session_state["df_res"]
    n_ini, n_fin = st.session_state["kpis"]
    logs = st.session_state["logs"]
    df_enr = st.session_state.get("df_enr", pd.DataFrame())
    acmg_active = st.session_state.get("use_acmg", False)

    # --- KPI HEADER ---
    k1, k2, k3 = st.columns(3)
    k1.metric("Initial", n_ini)
    k2.metric("Final", n_fin)
    ratio = round(n_fin/n_ini*100, 2) if n_ini > 0 else 0
    k3.metric("Ratio", f"{ratio}%")

    # --- TABS ---
    tabs = st.tabs([
        "📋 Tableau", 
        "🔍 Inspecteur", 
        "🧩 Corrélation", 
        "📊 Spectre Mutationnel", 
        "📍 Lollipops", 
        "📈 QC & Chromosomes", 
        "🧬 Pathways"
    ])

    # TAB 1: Tableau
    with tabs[0]:
        col_config = {
            "Fréquence_Cohorte": st.column_config.TextColumn("Freq. Cohorte"),
            "patho_score": st.column_config.NumberColumn("Score", format="%.1f"),
            "Allelic_ratio": st.column_config.NumberColumn("VAF", format="%.3f"),
            "gnomad_exomes_NFE_AF": st.column_config.NumberColumn("gnomAD", format="%.5f"),
            "link_varsome": st.column_config.LinkColumn("Varsome", display_text="🔗 Lien"),
            "link_gnomad": st.column_config.LinkColumn("gnomAD", display_text="🔗 Lien"),
        }
        if acmg_active:
            col_config["ACMG_Class"] = st.column_config.TextColumn("ACMG")
            
        st.dataframe(df_res, use_container_width=True, column_config=col_config)
        st.download_button("Télécharger CSV", df_res.to_csv(sep="\t", index=False).encode('utf-8'), "variants_filtered.tsv")

    # TAB 2: Inspecteur Patient
    with tabs[1]:
        st.subheader("🔍 Focus Patient")
        if "Pseudo" in df_res.columns:
            patients = sorted(df_res["Pseudo"].unique())
            if patients:
                sel_pat = st.selectbox("Sélectionner un patient", patients)
                if sel_pat:
                    df_pat = df_res[df_res["Pseudo"] == sel_pat].copy()
                    df_pat = df_pat.sort_values("patho_score", ascending=False).head(15)
                    st.caption(f"Top variants pour {sel_pat}")
                    
                    fig_pat = px.bar(
                        df_pat, x="patho_score", y="Gene_symbol", orientation='h', 
                        color="patho_score", title=f"Top Scores - {sel_pat}",
                        hover_data=["Variant_effect"]
                    )
                    fig_pat.update_layout(yaxis={'categoryorder':'total ascending'})
                    st.plotly_chart(fig_pat, use_container_width=True)
            else: st.warning("Pas de colonne Pseudo.")

    # TAB 3: Corrélation & OncoPrint
    with tabs[2]:
        st.subheader("🧩 Corrélation & Co-occurrence")
        if "Pseudo" in df_res.columns and "Gene_symbol" in df_res.columns:
            # Sélecteur de mode
            view_mode = st.radio("Type de visualisation", ["Heatmap Simple", "OncoPrint"], horizontal=True)
            
            top_genes = df_res["Gene_symbol"].value_counts().head(30).index.tolist()
            df_heat = df_res[df_res["Gene_symbol"].isin(top_genes)].copy()
            
            if not df_heat.empty:
                if view_mode == "Heatmap Simple":
                    matrix = df_heat.pivot_table(index="Pseudo", columns="Gene_symbol", aggfunc='size', fill_value=0)
                    matrix[matrix > 0] = 1 # Binarisation
                    # Correlation matrix
                    co_occ = matrix.T.dot(matrix)
                    fig_corr = px.imshow(co_occ, text_auto=True, color_continuous_scale="Viridis", title="Co-occurrence (Top 30 gènes)")
                    st.plotly_chart(fig_corr, use_container_width=True)
                    
                else: # OncoPrint
                    # Mapping des effets pour les couleurs
                    # Priorité: Stop/Frameshift (3) > Splicing (2) > Missense (1) > Autre (0.5)
                    def get_effect_score(eff):
                        e = str(eff).lower()
                        if "stop" in e or "frameshift" in e: return 3
                        if "splice" in e: return 2
                        if "missense" in e: return 1
                        return 0.5
                    
                    df_heat["Effet_Code"] = df_heat["Variant_effect"].apply(get_effect_score)
                    
                    # On garde le pire effet par paire Gene/Patient
                    matrix_onco = df_heat.pivot_table(index="Gene_symbol", columns="Pseudo", values="Effet_Code", aggfunc='max', fill_value=0)
                    
                    # Custom colorscale discrète
                    colors = [
                        [0.0, "white"], [0.2, "white"],     # 0: Rien
                        [0.2, "lightgrey"], [0.4, "lightgrey"], # 0.5: Autre
                        [0.4, "blue"], [0.6, "blue"],       # 1: Missense
                        [0.6, "orange"], [0.8, "orange"],   # 2: Splicing
                        [0.8, "red"], [1.0, "red"]          # 3: Tronquant
                    ]
                    
                    fig_onco = go.Figure(data=go.Heatmap(
                        z=matrix_onco.values,
                        x=matrix_onco.columns,
                        y=matrix_onco.index,
                        colorscale=colors,
                        showscale=False
                    ))
                    fig_onco.update_layout(title="OncoPrint (Rouge=Tronquant, Orange=Splice, Bleu=Missense)")
                    st.plotly_chart(fig_onco, use_container_width=True)
                    st.info("Légende : Rouge = Stop/Frameshift | Orange = Splicing | Bleu = Missense | Gris = Autre")
        else:
            st.warning("Données insuffisantes (Colonnes Pseudo/Gene manquantes).")

    # TAB 4: Spectre Mutationnel (NOUVEAU)
    with tabs[3]:
        st.subheader("📊 Spectre Mutationnel (Transitions / Transversions)")
        if "Ref" in df_res.columns and "Alt" in df_res.columns:
            df_mut = df_res.copy()
            df_mut["mutation"] = df_mut["Ref"] + ">" + df_mut["Alt"]
            
            # Mapping vers les 6 classes canoniques (Pyrimidine C/T)
            trans_map = {
                'G>T': 'C>A', 'G>C': 'C>G', 'G>A': 'C>T',
                'A>T': 'T>A', 'A>G': 'T>C', 'A>C': 'T>G'
            }
            # On applique le mapping, si pas dans la map on garde l'original (pour C>A etc)
            df_mut["canon_mut"] = df_mut["mutation"].apply(lambda x: trans_map.get(x, x))
            
            # Filtre pour ne garder que les variantes SNV classiques
            valid_snvs = ['C>A', 'C>G', 'C>T', 'T>A', 'T>C', 'T>G']
            df_mut = df_mut[df_mut["canon_mut"].isin(valid_snvs)]
            
            if not df_mut.empty:
                counts = df_mut["canon_mut"].value_counts().reset_index()
                counts.columns = ["Mutation", "Count"]
                
                # Couleurs COSMIC standard
                colors = {
                    'C>A': '#1ebff0', # Bleu
                    'C>G': '#050708', # Noir
                    'C>T': '#e62725', # Rouge
                    'T>A': '#cbcacb', # Gris
                    'T>C': '#a1cf64', # Vert
                    'T>G': '#edc8c5'  # Rose
                }
                
                fig_spec = px.bar(
                    counts, x="Mutation", y="Count", color="Mutation",
                    color_discrete_map=colors,
                    title="Distribution des substitutions (SNV)"
                )
                st.plotly_chart(fig_spec, use_container_width=True)
            else:
                st.warning("Aucun SNV standard détecté pour le spectre.")
        else:
            st.warning("Colonnes 'Ref' et 'Alt' introuvables ou impossible à déduire.")

    # TAB 5: Lollipops (NOUVEAU)
    with tabs[4]:
        st.subheader("📍 Lollipop Plot")
        # On a besoin d'une colonne protéine (HGVSp ou Protein_position)
        # On cherche souvent 'HGVSp' ou 'Protein_change'
        prot_col = None
        for c in ["HGVSp", "Protein_change", "AA_change"]:
            if c in df_res.columns:
                prot_col = c
                break
        
        if prot_col:
            genes_avail = sorted(df_res["Gene_symbol"].unique())
            sel_gene_lol = st.selectbox("Choisir un gène :", genes_avail)
            
            if sel_gene_lol:
                df_lol = df_res[df_res["Gene_symbol"] == sel_gene_lol].copy()
                # Extraction position: cherche p.Ala123... -> 123
                df_lol["AA_pos"] = df_lol[prot_col].astype(str).str.extract(r'(\d+)')[0]
                df_lol["AA_pos"] = pd.to_numeric(df_lol["AA_pos"], errors="coerce")
                
                df_lol = df_lol.dropna(subset=["AA_pos"])
                
                if not df_lol.empty:
                    max_pos = df_lol["AA_pos"].max()
                    fig_lol = px.scatter(
                        df_lol, x="AA_pos", y="patho_score",
                        color="Variant_effect",
                        size="patho_score",
                        hover_data=["Pseudo", prot_col],
                        range_x=[0, max_pos*1.1],
                        range_y=[0, df_lol["patho_score"].max()*1.2],
                        title=f"Distribution des mutations sur {sel_gene_lol}"
                    )
                    # Ajout des "tiges"
                    for _, row in df_lol.iterrows():
                        fig_lol.add_shape(
                            type="line",
                            x0=row["AA_pos"], y0=0,
                            x1=row["AA_pos"], y1=row["patho_score"],
                            line=dict(color="grey", width=1)
                        )
                    fig_lol.update_layout(xaxis_title="Position Acide Aminé")
                    st.plotly_chart(fig_lol, use_container_width=True)
                else:
                    st.warning(f"Impossible d'extraire des positions protéiques pour {sel_gene_lol}.")
        else:
            st.warning("Colonne HGVSp manquante, impossible de générer les lollipops.")

    # TAB 6: QC & Chromosomes
    with tabs[5]:
        st.subheader("📈 Contrôle Qualité")
        if not df_res.empty:
            # QC Classique
            c1, c2 = st.columns(2)
            with c1: st.plotly_chart(px.scatter(df_res, x="Depth", y="Allelic_ratio", color="ACMG_Class" if acmg_active else "Putative_impact", log_x=True, title="Depth vs VAF"), use_container_width=True)
            with c2: st.plotly_chart(px.histogram(df_res, x="internal_freq", title="Fréquence Cohorte (Histogramme)"), use_container_width=True)

            st.markdown("---")
            # Distribution Chromosomique
            st.subheader("Distribution Chromosomique")
            if "Chromosome" in df_res.columns:
                # Tri naturel des chromosomes
                chr_list = [str(i) for i in range(1, 23)] + ["X", "Y", "M", "MT"]
                df_res["Chr_Sorted"] = pd.Categorical(df_res["Chromosome"].astype(str), categories=chr_list, ordered=True)
                
                counts_chr = df_res["Chr_Sorted"].value_counts().sort_index()
                fig_chr = px.bar(counts_chr, title="Nombre de variants par Chromosome")
                st.plotly_chart(fig_chr, use_container_width=True)
            else:
                st.warning("Information Chromosome manquante.")

    # TAB 7: Pathways
    with tabs[6]:
        if df_enr.empty:
            st.info("Chargez des fichiers .gmt pour voir les pathways.")
        else:
            top = df_enr.sort_values("FDR").head(15)
            st.plotly_chart(px.bar(top, x="minus_log10_FDR", y="pathway", orientation='h', color="k_overlap"), use_container_width=True)
            st.dataframe(df_enr)

elif not submitted:
    st.info("👈 Chargez fichier + Lancer.")
