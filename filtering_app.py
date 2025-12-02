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

st.set_page_config(page_title="NGS ATLAS Explorer", layout="wide", page_icon="🧬")

# ==========================================
# 🔐 BLOC SÉCURITÉ
# ==========================================
def check_password():
    """Retourne True si l'utilisateur a le bon mot de passe."""
    def password_entered():
        if st.session_state["password"] == st.secrets["PASSWORD"]:
            st.session_state["password_correct"] = True
            del st.session_state["password"]
        else:
            st.session_state["password_correct"] = False

    if "password_correct" not in st.session_state:
        st.text_input("🔒 Mot de passe", type="password", on_change=password_entered, key="password")
        return False
    elif not st.session_state["password_correct"]:
        st.text_input("🔒 Mot de passe", type="password", on_change=password_entered, key="password")
        st.error("😕 Incorrect")
        return False
    else:
        return True

if not check_password():
    st.stop()

# ==========================================

def clean_text(val):
    if not isinstance(val, str): return ""
    return re.sub(r'[^A-Za-z0-9]+', '', val.strip().lower())

@st.cache_data
def load_variants(uploaded_file, sep_guess="auto"):
    if uploaded_file is None: return None
    if sep_guess == "auto":
        sample = uploaded_file.read(4096).decode("utf-8", errors="ignore")
        uploaded_file.seek(0)
        sep = ";" if ";" in sample and "," not in sample else ("\t" if "\t" in sample else ",")
    else: sep = sep_guess
    try:
        df = pd.read_csv(uploaded_file, sep=sep, dtype=str)
        df = df.replace('"', '', regex=True)
        return df
    except Exception as e:
        st.error(f"Erreur lecture : {e}")
        return None

def make_varsome_link(variant_str):
    try:
        v = variant_str.replace(">", ":")
        return f"https://varsome.com/variant/hg19/{v}"
    except: return ""

def make_gnomad_link(variant_str):
    try:
        v = variant_str.replace(":", "-").replace(">", "-")
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

    for col in ["Variant_effect", "Putative_impact", "Clinvar_significance"]:
        if col in df.columns: df[col] = df[col].fillna("Non Renseigné")

    if "Pseudo" in df.columns and "Variant" in df.columns:
        tot = df["Pseudo"].nunique()
        cts = df.groupby("Variant")["Pseudo"].nunique()
        df["internal_freq"] = df["Variant"].map(cts) / tot
        df["Fréquence_Cohorte"] = df["Variant"].map(cts).astype(str) + "/" + str(tot)
    else:
        df["internal_freq"] = 0.0
        df["Fréquence_Cohorte"] = "N/A"

    cols_num = ["gnomad_exomes_NFE_AF", "CADD_phred", "Allelic_ratio", "Depth", "Alt_depth_total", "patho_score"]
    for c in cols_num:
        if c in df.columns: df[c] = pd.to_numeric(df[c], errors="coerce")
    
    if "Alt_depth_total" not in df.columns and "Alt_depth" in df.columns:
        df["Alt_depth_total"] = df["Alt_depth"].astype(str).str.split(' ').str[0].apply(lambda x: int(x) if x.isdigit() else 0)

    # FILTRES
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

    # LIENS
    if "Variant" in df.columns:
        df["link_varsome"] = df["Variant"].apply(make_varsome_link)
        df["link_gnomad"] = df["Variant"].apply(make_gnomad_link)
    else:
        df["link_varsome"] = ""
        df["link_gnomad"] = ""

    # SCORING
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

    # ACMG
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
# 3. ANALYSE PATHWAYS (HYBRIDE LOCAL/UPLOAD)
# ---------------------------------------

# Fonction pour charger le fichier local s'il existe
def load_local_pathways(filepath="pathways.gmt"):
    pathways = {}
    if not os.path.exists(filepath):
        return pathways
    try:
        with open(filepath, "r", encoding="utf-8") as f:
            for line in f:
                parts = line.strip().split("\t")
                if len(parts) >= 3:
                    pw = parts[0].strip()
                    genes = [g.strip().upper() for g in parts[2:] if g.strip()]
                    if pw and genes:
                        if pw not in pathways: pathways[pw] = []
                        pathways[pw].extend(genes)
    except Exception as e:
        st.warning(f"Fichier local trouvé mais erreur de lecture : {e}")
    return pathways

def parse_uploaded_pathways(uploaded_files):
    pathways = {}
    if not uploaded_files: return pathways
    for up_file in uploaded_files:
        content = up_file.read().decode("utf-8", errors="ignore")
        if up_file.name.lower().endswith(".gmt"):
            for line in content.splitlines():
                parts = line.strip().split("\t")
                if len(parts) >= 3:
                    pw = parts[0].strip()
                    genes = [g.strip().upper() for g in parts[2:] if g.strip()]
                    if pw and genes:
                        if pw not in pathways: pathways[pw] = []
                        pathways[pw].extend(genes)
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

st.title("🧬 NGS ATLAS Variants Explorer")
st.markdown("---")

if "analysis_done" not in st.session_state:
    st.session_state["analysis_done"] = False
    st.session_state["df_res"] = None
    st.session_state["kpis"] = (0, 0)
    st.session_state["logs"] = []

# --- CHARGEMENT AUTO DES PATHWAYS LOCAUX ---
# On charge dès le lancement si le fichier existe
local_pws = load_local_pathways("pathways.gmt")
has_local = len(local_pws) > 0

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
        if has_local:
            st.success(f"✅ Fichier 'pathways.gmt' chargé ({len(local_pws)} voies).")
            st.info("Vous pouvez ajouter d'autres fichiers ci-dessous si besoin.")
        
        path_files = st.file_uploader("Fichiers GMT Supplémentaires", accept_multiple_files=True)
        
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
        
        # COMBINAISON DES PATHWAYS (LOCAL + UPLOAD)
        combined_pathways = local_pws.copy()
        uploaded_pws = parse_uploaded_pathways(path_files)
        # Fusion des dictionnaires
        for k, v in uploaded_pws.items():
            if k not in combined_pathways:
                combined_pathways[k] = v
            else:
                # Si le pathway existe déjà, on évite les doublons
                combined_pathways[k] = list(set(combined_pathways[k] + v))

        if combined_pathways:
            df_enr = compute_enrichment(res, combined_pathways)
            st.session_state["df_enr"] = df_enr
        else:
            st.session_state["df_enr"] = pd.DataFrame()

if st.session_state["analysis_done"]:
    df_res = st.session_state["df_res"]
    n_ini, n_fin = st.session_state["kpis"]
    logs = st.session_state["logs"]
    df_enr = st.session_state.get("df_enr", pd.DataFrame())
    acmg_active = st.session_state.get("use_acmg", False)

    k1, k2, k3 = st.columns(3)
    k1.metric("Initial", n_ini)
    k2.metric("Final", n_fin)
    ratio = round(n_fin/n_ini*100, 2) if n_ini > 0 else 0
    k3.metric("Ratio", f"{ratio}%")

    t1, t2, t3, t4, t5, t6 = st.tabs(["📋 Tableau", "🔍 Inspecteur Patient", "🧩 Corrélation", "🕵️‍♂️ Enquête", "📈 QC", "🧬 Pathways"])

    with t1:
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
            col_config["ACMG_Criteria"] = st.column_config.TextColumn("Critères")

        st.dataframe(df_res, use_container_width=True, column_config=col_config)
        st.download_button("Télécharger", df_res.to_csv(sep="\t", index=False).encode('utf-8'), "variants_filtered.tsv")

    with t2:
        st.subheader("🔍 Focus Patient")
        if "Pseudo" in df_res.columns:
            patients = sorted(df_res["Pseudo"].unique())
            if patients:
                sel_pat = st.selectbox("Sélectionner un patient", patients)
                if sel_pat:
                    df_pat = df_res[df_res["Pseudo"] == sel_pat].copy()
                    df_pat = df_pat.sort_values("patho_score", ascending=False).head(10)
                    if df_pat.empty:
                        st.warning("Aucun variant.")
                    else:
                        st.caption(f"Top variants pour {sel_pat}")
                        cols_show = ["Gene_symbol", "ACMG_Class", "hgvs.c", "patho_score", "Variant_effect", "link_varsome"]
                        final_cols = [c for c in cols_show if c in df_pat.columns]
                        st.dataframe(
                            df_pat[final_cols], use_container_width=True,
                            column_config={
                                "patho_score": st.column_config.ProgressColumn("Sévérité", min_value=0, max_value=20, format="%.1f"),
                                "link_varsome": st.column_config.LinkColumn("Lien", display_text="🔗")
                            }
                        )
                        fig_pat = px.bar(df_pat, x="patho_score", y="Gene_symbol", orientation='h', color="patho_score", title=f"Top Scores - {sel_pat}")
                        fig_pat.update_layout(yaxis={'categoryorder':'total ascending'})
                        st.plotly_chart(fig_pat, use_container_width=True)
            else: st.warning("Pas de colonne Pseudo.")

    with t3:
        st.subheader("🧩 Matrice de Co-occurrence (Heatmap)")
        if "Pseudo" in df_res.columns and "Gene_symbol" in df_res.columns:
            top_genes = df_res["Gene_symbol"].value_counts().head(30).index.tolist()
            df_heat = df_res[df_res["Gene_symbol"].isin(top_genes)]
            
            if not df_heat.empty:
                matrix = df_heat.pivot_table(index="Pseudo", columns="Gene_symbol", aggfunc='size', fill_value=0)
                matrix[matrix > 0] = 1
                co_occ = matrix.T.dot(matrix)
                
                fig_corr = px.imshow(co_occ, text_auto=True, color_continuous_scale="Viridis", aspect="auto", title="Co-occurrence des mutations (Top 30 gènes)")
                st.plotly_chart(fig_corr, use_container_width=True)
                
                st.subheader("🏆 Top Gènes Mutés")
                top_counts = df_res["Gene_symbol"].value_counts().head(20).reset_index()
                top_counts.columns = ["Gène", "Nombre de Patients"]
                fig_bar = px.bar(top_counts, x="Nombre de Patients", y="Gène", orientation='h', title="Fréquence des mutations par gène")
                fig_bar.update_layout(yaxis={'categoryorder':'total ascending'})
                st.plotly_chart(fig_bar, use_container_width=True)
            else:
                st.warning("Pas assez de données pour la corrélation.")
        else:
            st.warning("Colonnes 'Pseudo' et 'Gene_symbol' requises.")

    with t4:
        df_log = pd.DataFrame(logs)
        fig_wf = go.Figure(go.Waterfall(
            orientation = "v", measure = ["relative"] * len(df_log),
            x = df_log["Etape"], text = df_log["Restants"],
            y = [df_log["Restants"].iloc[0]] + [-x for x in df_log["Perdus"].iloc[1:]]
        ))
        st.plotly_chart(fig_wf, use_container_width=True)
        st.dataframe(df_log)

    with t5:
        if not df_res.empty:
            c1, c2 = st.columns(2)
            with c1: st.plotly_chart(px.scatter(df_res, x="Depth", y="Allelic_ratio", color="ACMG_Class" if acmg_active else "Putative_impact", log_x=True, title="QC: Depth vs VAF"), use_container_width=True)
            with c2: st.plotly_chart(px.histogram(df_res, x="internal_freq", title="QC: Freq Cohorte"), use_container_width=True)

    with t6:
        if df_enr.empty:
            st.info("Aucun pathway chargé (local ou upload).")
        else:
            top = df_enr.sort_values("FDR").head(15)
            st.plotly_chart(px.bar(top, x="minus_log10_FDR", y="pathway", orientation='h', color="k_overlap"), use_container_width=True)
            st.dataframe(df_enr)

elif not submitted:
    st.info("👈 Chargez fichier + Lancer.")
