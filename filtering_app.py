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

st.set_page_config(page_title="NGS ATLAS Explorer v2", layout="wide", page_icon="🧬")

# CSS pour améliorer l'apparence des tableaux
st.markdown("""
<style>
    .stDataFrame { border: 1px solid #f0f2f6; border-radius: 5px; }
    h1 { color: #2c3e50; }
    h2, h3 { color: #34495e; }
</style>
""", unsafe_allow_html=True)

# ==========================================
# 🔐 BLOC SÉCURITÉ
# ==========================================
def check_password():
    """Gère l'authentification simple par mot de passe."""
    def password_entered():
        if st.session_state["password"] == st.secrets.get("PASSWORD", "admin"): # "admin" par défaut si pas de secrets
            st.session_state["password_correct"] = True
            del st.session_state["password"]
        else:
            st.session_state["password_correct"] = False

    if "password_correct" not in st.session_state:
        st.text_input("🔒 Mot de passe", type="password", on_change=password_entered, key="password")
        st.info("Note: Configurez le mot de passe dans .streamlit/secrets.toml")
        return False
    elif not st.session_state["password_correct"]:
        st.text_input("🔒 Mot de passe", type="password", on_change=password_entered, key="password")
        st.error("😕 Mot de passe incorrect")
        return False
    else:
        return True

if not check_password():
    st.stop()

# ==========================================
# 🛠️ FONCTIONS UTILITAIRES
# ==========================================

@st.cache_data
def load_variants(uploaded_file):
    """Charge le fichier CSV/TSV et nettoie les colonnes."""
    if uploaded_file is None: return None
    try:
        # Lecture d'un échantillon pour deviner le séparateur
        sample = uploaded_file.read(4096).decode("utf-8", errors="ignore")
        uploaded_file.seek(0)
        sep = ";" if ";" in sample and "," not in sample else ("\t" if "\t" in sample else ",")
        
        df = pd.read_csv(uploaded_file, sep=sep, dtype=str)
        df = df.replace('"', '', regex=True)
        
        # Nettoyage des noms de colonnes (enlève les espaces autour)
        df.columns = df.columns.str.strip()
        return df
    except Exception as e:
        st.error(f"Erreur lors de la lecture du fichier : {e}")
        return None

def make_varsome_link(variant_str):
    """Génère un lien vers Varsome."""
    if pd.isna(variant_str): return ""
    try:
        v = variant_str.replace(">", ":") # Format hg19 souvent attendu
        return f"https://varsome.com/variant/hg19/{v}"
    except: return ""

def make_gnomad_link(variant_str):
    """Génère un lien vers gnomAD."""
    if pd.isna(variant_str): return ""
    try:
        # Format gnomAD: CHR-POS-REF-ALT
        v = variant_str.replace(":", "-").replace(">", "-")
        return f"https://gnomad.broadinstitute.org/variant/{v}?dataset=gnomad_r2_1"
    except: return ""

# ---------------------------------------
# 2. LOGIQUE DE FILTRAGE (CŒUR DU REACTEUR)
# ---------------------------------------

@st.cache_data
def apply_filtering_and_scoring(
    df, allelic_ratio_min, gnomad_max, use_gnomad_filter, min_patho_score, use_patho_filter,
    min_depth, min_alt_depth, max_cohort_freq, use_msc, constraint_file_content,
    genes_exclude, patients_exclude, min_cadd,
    variant_effect_keep, putative_keep, clinvar_keep, sort_by_column,
    use_acmg
):
    """
    Applique tous les filtres séquentiellement et calcule les scores.
    Retourne le dataframe filtré et les logs pour le diagramme en entonnoir.
    """
    logs = [] 
    df = df.copy()
    initial_total = len(df)
    
    # On initialise le compteur
    logs.append({"Etape": "Total Brut", "Variants": initial_total})

    # --- Pré-traitement ---
    if "Gene_symbol" not in df.columns:
        return None, 0, 0, "❌ La colonne 'Gene_symbol' est manquante dans votre fichier.", []

    df["Gene_symbol"] = df["Gene_symbol"].str.upper().str.replace(" ", "")
    
    # Remplissage des vides pour éviter les plantages
    for col in ["Variant_effect", "Putative_impact", "Clinvar_significance"]:
        if col in df.columns: df[col] = df[col].fillna("Non Renseigné")

    # Calcul fréquence interne
    if "Pseudo" in df.columns and "Variant" in df.columns:
        tot_patients = df["Pseudo"].nunique()
        cts = df.groupby("Variant")["Pseudo"].nunique()
        df["internal_freq"] = df["Variant"].map(cts) / tot_patients
        df["Fréquence_Cohorte"] = df["Variant"].map(cts).astype(str) + "/" + str(tot_patients)
    else:
        df["internal_freq"] = 0.0
        df["Fréquence_Cohorte"] = "N/A"

    # Conversion numérique
    cols_num = ["gnomad_exomes_NFE_AF", "CADD_phred", "Allelic_ratio", "Depth", "Alt_depth_total"]
    for c in cols_num:
        if c in df.columns: df[c] = pd.to_numeric(df[c], errors="coerce")
    
    # Gestion spécifique Alt_depth (parfois format "10,2")
    if "Alt_depth_total" not in df.columns and "Alt_depth" in df.columns:
        df["Alt_depth_total"] = df["Alt_depth"].astype(str).str.split(' ').str[0].apply(lambda x: int(x) if x.isdigit() else 0)

    # --- FILTRES SÉQUENTIELS ---
    
    if "Depth" in df.columns: 
        df = df[df["Depth"] >= min_depth]
    logs.append({"Etape": "Profondeur (DP)", "Variants": len(df)})

    if "Alt_depth_total" in df.columns: 
        df = df[df["Alt_depth_total"] >= min_alt_depth]
    logs.append({"Etape": "Lectures Mutées (AD)", "Variants": len(df)})

    if "Allelic_ratio" in df.columns: 
        df = df[df["Allelic_ratio"] >= allelic_ratio_min]
    logs.append({"Etape": "Fréquence Allélique (VAF)", "Variants": len(df)})

    if max_cohort_freq < 1.0: 
        df = df[df["internal_freq"] <= max_cohort_freq]
    logs.append({"Etape": "Fréq. dans la Cohorte", "Variants": len(df)})

    # Exclusions (Genes et Patients)
    if genes_exclude: df = df[~df["Gene_symbol"].isin(genes_exclude)]
    if patients_exclude and "Pseudo" in df.columns: df = df[~df["Pseudo"].isin(patients_exclude)]
    logs.append({"Etape": "Exclusions Manuelles", "Variants": len(df)})

    # Filtre gnomAD
    if use_gnomad_filter and "gnomad_exomes_NFE_AF" in df.columns:
        # On garde si NaN (pas dans gnomad = potentiellement rare) OU si fréquence < seuil
        df = df[(df["gnomad_exomes_NFE_AF"].isna()) | (df["gnomad_exomes_NFE_AF"] <= gnomad_max)]
    logs.append({"Etape": "Filtre gnomAD", "Variants": len(df)})

    # Filtres Catégoriels (Effet, Impact, Clinvar)
    if variant_effect_keep and "Variant_effect" in df.columns: df = df[df["Variant_effect"].isin(variant_effect_keep)]
    if putative_keep and "Putative_impact" in df.columns: df = df[df["Putative_impact"].isin(putative_keep)]
    if clinvar_keep and "Clinvar_significance" in df.columns: df = df[df["Clinvar_significance"].isin(clinvar_keep)]
    logs.append({"Etape": "Impact & ClinVar", "Variants": len(df)})

    if min_cadd is not None and min_cadd > 0 and "CADD_phred" in df.columns:
        df = df[df["CADD_phred"] >= min_cadd]
    logs.append({"Etape": "Score CADD", "Variants": len(df)})

    # --- SCORING & ANNOTATION ---
    
    # Liens Web
    if "Variant" in df.columns:
        df["link_varsome"] = df["Variant"].apply(make_varsome_link)
        df["link_gnomad"] = df["Variant"].apply(make_gnomad_link)
    else:
        df["link_varsome"], df["link_gnomad"] = "", ""

    # Score MSC (Mutation Significance Cutoff)
    df["msc_weight"] = 0.0
    if use_msc and constraint_file_content:
        try:
            c_df = pd.read_csv(io.StringIO(constraint_file_content), sep="\t", dtype=str)
            if "gene" in c_df.columns:
                c_df["gene"] = c_df["gene"].str.upper().str.replace(" ", "")
                # Gestion noms de colonnes variables
                col_z = "mis_z" if "mis_z" in c_df.columns else ("mis.z_score" if "mis.z_score" in c_df.columns else None)
                
                if col_z:
                    c_df[col_z] = pd.to_numeric(c_df[col_z], errors="coerce")
                    c_df = c_df.groupby("gene", as_index=False)[col_z].max()
                    df = df.merge(c_df[["gene", col_z]], left_on="Gene_symbol", right_on="gene", how="left")
                    df["mis_z"] = df[col_z].fillna(0)
                    df["msc_weight"] = df["mis_z"] * 0.5 # Poids arbitraire pour le score final
        except Exception as e: 
            st.warning(f"Erreur MSC: {e}")

    # Score Putative (Impact)
    df["score_putative"] = 0
    if "Putative_impact" in df.columns:
        def get_imp(x):
            s = str(x).lower()
            if "high" in s: return 3
            if "moderate" in s or "modifier" in s: return 2
            if "low" in s: return 1
            return 0
        df["score_putative"] = df["Putative_impact"].apply(get_imp)

    # Score CADD simplifié
    df["score_cadd"] = 0
    if "CADD_phred" in df.columns:
        df["score_cadd"] = np.select([(df["CADD_phred"] >= 30), (df["CADD_phred"] >= 20)], [3, 2], default=0)

    # Score ClinVar
    df["score_clinvar"] = 0
    if "Clinvar_significance" in df.columns:
        cv = df["Clinvar_significance"].astype(str).str.lower()
        df["score_clinvar"] = np.select(
            [cv.str.contains("pathogenic") & ~cv.str.contains("likely"), cv.str.contains("uncertain")],
            [5, 2], default=0
        )

    # Score Global "Maison"
    df["patho_score"] = df["score_putative"] + df["score_cadd"] + df["score_clinvar"] + df["msc_weight"]

    # --- CLASSIFICATION ACMG (AUTOMATIQUE & NAÏVE) ---
    if use_acmg:
        def compute_acmg_class(row):
            eff = str(row.get("Variant_effect", "")).lower()
            # PVS1: Null variant (très fort impact)
            pvs1 = any(x in eff for x in ["stopgained", "frameshift", "splice_acceptor", "splice_donor"])
            
            # PM2: Absent from controls (gnomAD)
            af = row.get("gnomad_exomes_NFE_AF", np.nan)
            pm2 = pd.isna(af) or (af < 0.0001)
            
            # PP3: Computational evidence (CADD)
            cadd = row.get("CADD_phred", 0)
            pp3 = cadd >= 25
            
            # PP5: ClinVar pathogenic
            clv = str(row.get("Clinvar_significance", "")).lower()
            pp5 = "pathogenic" in clv and "conflict" not in clv
            
            # BA1: Trop fréquent (>5%)
            ba1 = (af > 0.05) if not pd.isna(af) else False
            
            criteria_met = []
            if ba1: return "Benign", "BA1"
            
            score = 0
            if pvs1: score += 4; criteria_met.append("PVS1 (Null)")
            if pm2: score += 2; criteria_met.append("PM2 (Rare)")
            if pp5: score += 2; criteria_met.append("PP5 (ClinVar)")
            if pp3: score += 1; criteria_met.append("PP3 (InSilico)")
            
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

    # Filtre sur le Score Final
    if use_patho_filter:
        df = df[df["patho_score"] >= min_patho_score]
    logs.append({"Etape": "Score Pathogénique Final", "Variants": len(df)})

    # --- TRI FINAL ---
    if sort_by_column == "Score Pathogénique (Décroissant)": df = df.sort_values("patho_score", ascending=False)
    elif sort_by_column == "Classification ACMG": df = df.sort_values("ACMG_Class", ascending=True)
    elif sort_by_column == "Fréquence Cohorte (Décroissant)": df = df.sort_values("internal_freq", ascending=False)
    elif sort_by_column == "Patient (A-Z)": 
        if "Pseudo" in df.columns: df = df.sort_values("Pseudo", ascending=True)

    return df, initial_total, len(df), None, logs

# ---------------------------------------
# 3. ANALYSE PATHWAYS
# ---------------------------------------

def load_local_pathways(filepath="pathways.gmt"):
    """Charge un fichier GMT local s'il est présent."""
    pathways = {}
    if not os.path.exists(filepath): return pathways
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
    except: pass
    return pathways

def parse_uploaded_pathways(uploaded_files):
    """Charge des fichiers GMT uploadés par l'utilisateur."""
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
    """Calcule l'enrichissement statistique (test hypergéométrique)."""
    # Univers = tous les gènes connus dans les pathways
    universe = sorted({g for gl in pathway_genes.values() for g in gl})
    N = len(universe)
    
    # Gènes mutés dans notre liste filtrée (qui sont aussi dans l'univers)
    mutated = sorted(set(df["Gene_symbol"].unique()) & set(universe))
    n = len(mutated)
    
    if N == 0 or n == 0: return pd.DataFrame()

    rows = []
    for pw, genes in pathway_genes.items():
        M = len(genes) # Taille du pathway
        overlap = set(mutated) & set(genes)
        k = len(overlap) # Nombre de gènes mutés dans ce pathway
        
        if k > 0:
            # P-value (probabilité d'avoir k gènes par hasard)
            pval = hypergeom.sf(k - 1, N, M, n)
            
            # Quels patients ont ces mutations ?
            patients_str = "N/A"
            if "Pseudo" in df.columns:
                subset = df[df["Gene_symbol"].isin(overlap)]
                affected_patients = sorted(subset["Pseudo"].unique())
                patients_str = ", ".join(affected_patients)

            rows.append({
                "pathway": pw, 
                "p_value": pval, 
                "k_overlap": k, 
                "genes_in_overlap": ",".join(overlap), 
                "patients": patients_str,
                "pathway_size": M
            })

    df_res = pd.DataFrame(rows)
    if not df_res.empty:
        df_res = df_res.sort_values("p_value")
        # Correction FDR (Benjamini-Hochberg) simplifiée
        m = len(df_res)
        df_res["rank"] = np.arange(1, m + 1)
        df_res["FDR"] = np.clip(np.minimum.accumulate(((df_res["p_value"] * m) / df_res["rank"])[::-1])[::-1], 0, 1)
        # Score pour le plot (-log10)
        df_res["minus_log10_p"] = -np.log10(df_res["p_value"] + 1e-300)
    return df_res

# ---------------------------------------
# 4. INTERFACE UTILISATEUR (STREAMLIT)
# ---------------------------------------

st.title("🧬 NGS ATLAS Variants Explorer")
st.markdown("---")

# Initialisation de l'état (Session State) pour garder les résultats en mémoire
if "analysis_done" not in st.session_state:
    st.session_state["analysis_done"] = False
    st.session_state["df_res"] = None
    st.session_state["logs"] = []

# Chargement automatique pathways locaux
local_pws = load_local_pathways("pathways.gmt")
has_local = len(local_pws) > 0

# --- BARRE LATÉRALE (INPUTS) ---
with st.sidebar:
    st.header("1. Données")
    uploaded_file = st.file_uploader("📂 Fichier Variants (CSV/TSV)", type=["csv", "tsv", "txt"])
    
    df_raw = None
    v_opts, p_opts, c_opts = [], [], []

    if uploaded_file:
        df_raw = load_variants(uploaded_file)
        if df_raw is not None:
            st.success(f"{len(df_raw)} variants chargés.")
            if "Variant_effect" in df_raw.columns: v_opts = sorted(df_raw["Variant_effect"].fillna("Non Renseigné").unique())
            if "Putative_impact" in df_raw.columns: p_opts = sorted(df_raw["Putative_impact"].fillna("Non Renseigné").unique())
            if "Clinvar_significance" in df_raw.columns: c_opts = sorted(df_raw["Clinvar_significance"].fillna("Non Renseigné").unique())

    with st.form("params"):
        st.header("2. Filtres & Paramètres")
        
        # Onglets dans la sidebar pour ne pas surcharger
        tab_main, tab_adv, tab_path = st.tabs(["Général", "Avancé", "Pathways"])
        
        with tab_main:
            sort_choice = st.selectbox("Trier par", ["Score Pathogénique (Décroissant)", "Classification ACMG", "Fréquence Cohorte (Décroissant)", "Patient (A-Z)"])
            
            c1, c2 = st.columns(2)
            min_dp = c1.number_input("Profondeur Min (DP)", 0, 10000, 30)
            allelic_min = c2.number_input("VAF Min (0-1)", 0.0, 1.0, 0.05)
            
            c3, c4 = st.columns(2)
            min_ad = c3.number_input("Lectures Mutées Min (AD)", 0, 1000, 5)
            gnomad_max = c4.number_input("gnomAD Max (Freq)", 0.0, 1.0, 0.001, format="%.4f")
            
            use_gnomad = st.checkbox("Activer Filtre gnomAD", True)
            use_acmg = st.checkbox("Classification ACMG (Auto)", value=True, help="Classification indicative basée sur 4 critères simples.")

        with tab_adv:
            max_cohort_freq = st.slider("Freq. Max Cohorte", 0.0, 1.0, 1.0, 0.05, help="Éliminer les variants présents chez tout le monde dans ce run.")
            
            min_patho = st.number_input("Score Patho Min", 0.0, 50.0, 4.0)
            use_patho = st.checkbox("Filtrer par score patho", True)
            
            min_cadd_val = st.number_input("CADD Min (0=désactivé)", 0.0, 60.0, 0.0)
            
            st.caption("Filtres Catégoriels (Laisser vide pour tout garder)")
            sel_var = st.multiselect("Effet (Variant Effect)", v_opts, default=v_opts)
            sel_put = st.multiselect("Impact (Putative)", p_opts, default=p_opts)
            sel_clin = st.multiselect("ClinVar", c_opts, default=c_opts)
            
            genes_ex = st.text_area("Exclure Gènes (séparés par ,)", "TTN, MUC16")
            pseudo_ex = st.text_area("Exclure Patients", "")
            
            st.markdown("---")
            use_msc = st.checkbox("Utiliser score MSC", False)
            msc_file = st.file_uploader("Fichier MSC (.tsv)", type=["tsv"])

        with tab_path:
            st.write(f"Pathways locaux: {len(local_pws)}")
            path_files = st.file_uploader("Ajouter .gmt", accept_multiple_files=True)

        submitted = st.form_submit_button("🚀 LANCER L'ANALYSE", type="primary")

# --- EXECUTION ---
if submitted and df_raw is not None:
    g_list = [x.strip().upper() for x in genes_ex.split(",") if x.strip()]
    p_list = [x.strip() for x in pseudo_ex.split(",") if x.strip()]
    msc_c = msc_file.read().decode("utf-8") if (use_msc and msc_file) else None

    with st.spinner("Analyse en cours..."):
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
        
        # Merge Pathways
        combined_pathways = local_pws.copy()
        uploaded_pws = parse_uploaded_pathways(path_files)
        for k, v in uploaded_pws.items():
            if k not in combined_pathways: combined_pathways[k] = v
            else: combined_pathways[k] = list(set(combined_pathways[k] + v))

        if combined_pathways:
            df_enr = compute_enrichment(res, combined_pathways)
            st.session_state["df_enr"] = df_enr
        else:
            st.session_state["df_enr"] = pd.DataFrame()

# --- AFFICHAGE RESULTATS ---
if st.session_state["analysis_done"]:
    df_res = st.session_state["df_res"]
    n_ini, n_fin = st.session_state["kpis"]
    logs = st.session_state["logs"]
    df_enr = st.session_state.get("df_enr", pd.DataFrame())
    acmg_active = st.session_state.get("use_acmg", False)

    # Indicateurs Clés (KPIs)
    k1, k2, k3, k4 = st.columns(4)
    k1.metric("Variants Initiaux", n_ini)
    k2.metric("Variants Retenus", n_fin)
    ratio = round(n_fin/n_ini*100, 2) if n_ini > 0 else 0
    k3.metric("Pourcentage retenu", f"{ratio}%")
    unique_genes = df_res["Gene_symbol"].nunique() if not df_res.empty else 0
    k4.metric("Gènes Uniques", unique_genes)

    # Onglets de visualisation
    t1, t2, t3, t4, t5 = st.tabs(["📋 Liste Variants", "🔍 Inspecteur Patient", "📉 Entonnoir de Filtre", "📊 Stats QC", "🧬 Pathways"])

    with t1:
        st.subheader("📋 Tableau des Variants")
        
        # 1. Option pour tout afficher
        show_all_cols = st.checkbox("👀 Voir toutes les colonnes (Mode expert)", value=False)
        
        st.caption("Tableau interactif. Cliquez sur les en-têtes pour trier. Double-cliquez sur une cellule pour voir tout son contenu.")
        
        # 2. Configuration du formatage (pour que les liens et barres fonctionnent toujours)
        col_config = {
            "Fréquence_Cohorte": st.column_config.TextColumn("Freq. Cohorte", help="Combien de fois ce variant apparait dans le fichier"),
            "patho_score": st.column_config.ProgressColumn("Score", min_value=0, max_value=15, format="%.1f"),
            "Allelic_ratio": st.column_config.NumberColumn("VAF", format="%.2f"),
            "gnomad_exomes_NFE_AF": st.column_config.NumberColumn("gnomAD", format="%.5f"),
            "link_varsome": st.column_config.LinkColumn("Varsome", display_text="🔗 Lien"),
            "link_gnomad": st.column_config.LinkColumn("gnomAD", display_text="🔗 Lien"),
        }
        if acmg_active:
            col_config["ACMG_Class"] = st.column_config.TextColumn("ACMG", width="medium")

        # 3. Logique d'affichage
        if show_all_cols:
            # Si la case est cochée, on envoie TOUT le dataframe 'df_res'
            # On place quand même les colonnes importantes en premier pour le confort
            priority_cols = ["Pseudo", "Gene_symbol", "hgvs.c", "patho_score", "ACMG_Class", "link_varsome"]
            # On cherche quelles colonnes prioritaires existent vraiment
            existing_priority = [c for c in priority_cols if c in df_res.columns]
            # Les autres colonnes sont "le reste"
            other_cols = [c for c in df_res.columns if c not in existing_priority]
            
            # On réorganise : Priorité d'abord + le reste ensuite
            final_df = df_res[existing_priority + other_cols]
            
            st.dataframe(
                final_df, 
                use_container_width=True, 
                column_config=col_config, 
                height=700
            )
        else:
            # Si la case n'est PAS cochée, on garde la vue "épurée"
            cols_to_show = ["Pseudo", "Gene_symbol", "hgvs.c", "ACMG_Class", "patho_score", "Variant_effect", "Allelic_ratio", "Depth", "Fréquence_Cohorte", "link_varsome", "link_gnomad"]
            final_cols = [c for c in cols_to_show if c in df_res.columns]
            
            st.dataframe(
                df_res[final_cols], 
                use_container_width=True, 
                column_config=col_config, 
                height=600
            )
        
        st.download_button(
            "💾 Télécharger les résultats complets (TSV)", 
            df_res.to_csv(sep="\t", index=False).encode('utf-8'), 
            "variants_filtered.tsv",
            mime="text/csv"
        )

    with t2:
        st.subheader("Fiche Patient")
        if "Pseudo" in df_res.columns:
            patients = sorted(df_res["Pseudo"].unique())
            col_sel, col_graph = st.columns([1, 3])
            
            with col_sel:
                sel_pat = st.radio("Choisir un patient", patients)
            
            with col_graph:
                if sel_pat:
                    df_pat = df_res[df_res["Pseudo"] == sel_pat].copy()
                    st.write(f"**{len(df_pat)} variants** trouvés pour {sel_pat}")
                    
                    if not df_pat.empty:
                        # Graphique des scores par gène
                        fig_pat = px.bar(
                            df_pat.sort_values("patho_score"), 
                            x="patho_score", y="Gene_symbol", 
                            orientation='h', 
                            color="ACMG_Class" if acmg_active else "patho_score",
                            title=f"Sévérité des variants chez {sel_pat}",
                            hover_data=["hgvs.c", "Variant_effect"]
                        )
                        st.plotly_chart(fig_pat, use_container_width=True)
                        
                        # Petit tableau résumé
                        st.dataframe(df_pat[["Gene_symbol", "hgvs.c", "ACMG_Class", "patho_score"]], use_container_width=True)
        else:
            st.warning("La colonne 'Pseudo' est absente du fichier.")

    with t3:
        # Funnel Chart (Entonnoir)
        st.subheader("Combien de variants perdus à chaque étape ?")
        df_log = pd.DataFrame(logs)
        
        fig_funnel = go.Figure(go.Funnel(
            y = df_log["Etape"],
            x = df_log["Variants"],
            textinfo = "value+percent initial"
        ))
        fig_funnel.update_layout(title="Entonnoir de filtration")
        st.plotly_chart(fig_funnel, use_container_width=True)
        st.dataframe(df_log, use_container_width=True)

    with t4:
        st.subheader("Contrôle Qualité (QC)")
        if not df_res.empty:
            c1, c2 = st.columns(2)
            with c1:
                # Scatter Depth vs VAF
                fig_qc1 = px.scatter(
                    df_res, x="Depth", y="Allelic_ratio", 
                    color="ACMG_Class" if acmg_active else "Putative_impact",
                    hover_data=["Gene_symbol"],
                    log_x=True, 
                    title="Qualité : Profondeur vs VAF"
                )
                st.plotly_chart(fig_qc1, use_container_width=True)
            with c2:
                # Distribution gnomAD
                if "gnomad_exomes_NFE_AF" in df_res.columns:
                    fig_qc2 = px.histogram(df_res, x="gnomad_exomes_NFE_AF", nbins=50, title="Distribution Fréquence gnomAD")
                    st.plotly_chart(fig_qc2, use_container_width=True)

    with t5:
        st.subheader("Analyse de Voies (Enrichment)")
        if df_enr.empty:
            st.info("Aucun pathway enrichi ou aucun fichier pathway chargé.")
        else:
            # Bubble Plot
            top_pw = df_enr.sort_values("p_value").head(20)
            fig_bubble = px.scatter(
                top_pw, 
                x="minus_log10_p", y="pathway",
                size="k_overlap", color="p_value",
                color_continuous_scale="Bluered",
                hover_data=["genes_in_overlap"],
                title="Top 20 Pathways Enrichis (Taille = nb gènes mutés)",
                labels={"minus_log10_p": "-log10(p-value)", "pathway": "Voie"}
            )
            fig_bubble.update_layout(yaxis={'categoryorder':'total ascending'})
            st.plotly_chart(fig_bubble, use_container_width=True)
            
            st.dataframe(df_enr)

elif not submitted:
    st.info("👈 Veuillez charger un fichier dans la barre latérale et cliquer sur LANCER.")
