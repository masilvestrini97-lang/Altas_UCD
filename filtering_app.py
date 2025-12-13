import os
import io
import re
import glob
import requests
import json
import tempfile
from datetime import datetime

import numpy as np
import pandas as pd
import streamlit as st
import plotly.express as px
import plotly.graph_objects as go
from scipy.stats import hypergeom

# --- IMPORTS POUR VISUALISATION & RAPPORT ---
import matplotlib.pyplot as plt
import seaborn as sns
from st_aggrid import AgGrid, GridOptionsBuilder, JsCode
from fpdf import FPDF
from streamlit_agraph import agraph, Node, Edge, Config
from sklearn.cluster import KMeans

# ---------------------------------------
# 1. CONFIGURATION & DESIGN
# ---------------------------------------

st.set_page_config(page_title="NGS ATLAS Explorer", layout="wide", page_icon="🧬")

MSC_LOCAL_FILENAME = "MSC_CI99_v1.7.txt"
HEADER_IMG_URL = "https://raw.githubusercontent.com/masilvestrini97-lang/Altas_UCD/refs/heads/main/images.jpeg"

def render_custom_header():
    st.markdown(f"""
    <style>
    .header-container {{
        background-image: linear-gradient(rgba(0, 0, 0, 0.6), rgba(0, 0, 0, 0.6)), url('{HEADER_IMG_URL}');
        background-size: cover;
        background-position: center 30%;
        padding: 50px 20px;
        border-radius: 15px;
        text-align: center;
        margin-bottom: 30px;
        box-shadow: 0 4px 10px rgba(0,0,0,0.2);
    }}
    .header-title {{
        color: #8C005F; 
        font-family: 'Helvetica Neue', Helvetica, Arial, sans-serif;
        font-size: 48px;
        font-weight: 800;
        text-transform: uppercase;
        letter-spacing: 2px;
        text-shadow: 2px 2px 4px #000000;
        margin: 0;
    }}
    .header-subtitle {{
        color: #f1f1f1;
        font-size: 18px;
        font-weight: 300;
        margin-top: 10px;
        text-shadow: 1px 1px 2px #000000;
    }}
    </style>
    <div class="header-container">
        <div class="header-title">🧬 NGS ATLAS Explorer</div>
        <div class="header-subtitle">Dev by Castleman Team</div>
    </div>
    """, unsafe_allow_html=True)

# --- FONCTIONS UTILITAIRES ---

def clean_text(val):
    if not isinstance(val, str): return ""
    return re.sub(r'[^A-Za-z0-9]+', '', val.strip().lower())

def extract_ref_alt_chr(df):
    if "Variant" in df.columns:
        extracted = df["Variant"].astype(str).str.extract(r'([ACGT]+)[>:/]([ACGT]+)$', flags=re.IGNORECASE)
        if "Ref" not in df.columns and not extracted[0].isna().all():
            df["Ref"] = extracted[0].str.upper()
        if "Alt" not in df.columns and not extracted[1].isna().all():
            df["Alt"] = extracted[1].str.upper()
        if "Chromosome" not in df.columns:
            df["Chromosome"] = df["Variant"].astype(str).str.split(r'[:_-]', n=1, expand=True)[0]
            df["Chromosome"] = df["Chromosome"].str.replace("chr", "", case=False).str.strip()
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
        df = pd.read_csv(uploaded_file, sep=sep, dtype=str, on_bad_lines='skip')
        df = df.replace('"', '', regex=True)
        df = extract_ref_alt_chr(df)
        return df
    except Exception as e:
        st.error(f"Erreur lecture : {e}")
        return None

def make_varsome_link(variant_str):
    try:
        v = str(variant_str).replace(">", ":")
        return f"https://varsome.com/variant/hg19/{v}"
    except: return ""

@st.cache_data
def get_string_network(gene_symbol, limit=10):
    url = "https://string-db.org/api/json/network"
    params = {"identifiers": gene_symbol, "species": 9606, "limit": limit, "network_type": "functional"}
    try:
        response = requests.get(url, params=params, timeout=5)
        if response.status_code == 200: return response.json()
    except: return []
    return []

# --- RAPPORT PDF ---
def create_pdf_report(patient_id, df_variants, user_comments=""):
    class PDF(FPDF):
        def header(self):
            self.set_font('Arial', 'B', 14)
            self.cell(0, 10, f'Rapport NGS - {patient_id}', 0, 1, 'C')
            self.ln(2)
        def footer(self):
            self.set_y(-15)
            self.set_font('Arial', 'I', 8)
            self.cell(0, 10, f'Page {self.page_no()}', 0, 0, 'C')

    pdf = PDF(orientation='L', unit='mm', format='A4')
    pdf.add_page()
    
    pdf.set_font("Arial", 'B', 10)
    pdf.cell(0, 6, f"Date: {datetime.now().strftime('%Y-%m-%d')} | Variants: {len(df_variants)}", 0, 1)
    if user_comments:
        pdf.set_font("Arial", 'I', 10)
        pdf.multi_cell(0, 6, f"Note: {user_comments}")
    pdf.ln(5)

    prot_candidates = ["hgvs.p", "HGVSp", "Protein_change", "AA_change", "hgvsp", "p."]
    found_prot = next((c for c in prot_candidates if c in df_variants.columns), None)

    columns_config = [
        ("Gene", "Gene_symbol", 25),
        ("Variant", "Variant", 45),
        ("Protein", found_prot, 35) if found_prot else ("Effect", "Variant_effect", 35),
        ("CADD", "CADD_phred", 15),
        ("MSC", "MSC_Ref", 15),
        ("ACMG", "ACMG_Class", 35),
        ("VAF", "Allelic_ratio", 15),
        ("Depth", "Depth", 15),
        ("gnomAD", "gnomad_exomes_NFE_AF", 25)
    ]
    columns_config = [c for c in columns_config if c is not None]

    pdf.set_font("Arial", 'B', 8)
    pdf.set_fill_color(240, 240, 240)
    for label, _, width in columns_config:
        pdf.cell(width, 8, label, 1, 0, 'C', 1)
    pdf.ln()
    
    pdf.set_font("Arial", '', 7)
    for _, row in df_variants.iterrows():
        for label, col_name, width in columns_config:
            raw_val = row.get(col_name, "")
            display_val = str(raw_val)
            
            if col_name in ["Allelic_ratio", "CADD_phred", "MSC_Ref"]:
                try: display_val = str(round(float(raw_val), 2))
                except: pass
            elif col_name == "gnomad_exomes_NFE_AF":
                try:
                    val_float = float(raw_val)
                    display_val = "<1e-4" if 0 < val_float < 0.0001 else f"{val_float:.4f}"
                    if val_float == 0: display_val = "0"
                except: display_val = ""

            max_char = int(width / 1.8)
            if len(display_val) > max_char: display_val = display_val[:max_char-2] + ".."
            pdf.cell(width, 8, display_val, 1, 0, 'C')
        pdf.ln()

    return pdf.output(dest='S').encode('latin-1', 'ignore')

# ---------------------------------------
# 2. LOGIQUE DE FILTRAGE
# ---------------------------------------

@st.cache_data
def apply_filtering_and_scoring(
    df, allelic_ratio_min, gnomad_max, use_gnomad_filter, 
    min_depth, min_alt_depth, max_cohort_freq, 
    msc_file_uploaded_content,
    genes_exclude, patients_exclude, min_cadd,
    variant_effect_keep, putative_keep, clinvar_keep, sort_by_column,
    use_acmg, use_msc_filter_strict,
    acmg_keep_list 
):
    logs = [] 
    df = df.copy()
    initial_total = len(df)
    logs.append({"Etape": "1. Import Brut", "Restants": initial_total, "Perdus": 0})

    if "Gene_symbol" not in df.columns:
        return None, 0, 0, "Colonne 'Gene_symbol' manquante.", []

    df["Gene_symbol"] = df["Gene_symbol"].str.upper().str.replace(" ", "")
    df = extract_ref_alt_chr(df)

    for col in ["Variant_effect", "Putative_impact", "Clinvar_significance"]:
        if col in df.columns: df[col] = df[col].fillna("Non Renseigné")

    if "Pseudo" in df.columns and "Variant" in df.columns:
        tot = df["Pseudo"].nunique()
        cts = df.groupby("Variant")["Pseudo"].nunique()
        df["internal_freq"] = df["Variant"].map(cts) / tot
    else: df["internal_freq"] = 0.0

    cols_num = ["gnomad_exomes_NFE_AF", "CADD_phred", "Allelic_ratio", "Depth", "Alt_depth_total"]
    for c in cols_num:
        if c in df.columns: df[c] = pd.to_numeric(df[c], errors="coerce")
    
    if "Alt_depth_total" not in df.columns and "Alt_depth" in df.columns:
        df["Alt_depth_total"] = df["Alt_depth"].astype(str).str.split(',').str[0].str.split(' ').str[0]
        df["Alt_depth_total"] = pd.to_numeric(df["Alt_depth_total"], errors='coerce').fillna(0)

    # --- FILTRES ---
    last_count = len(df)
    if "Depth" in df.columns: df = df[df["Depth"] >= min_depth]
    if "Alt_depth_total" in df.columns: df = df[df["Alt_depth_total"] >= min_alt_depth]
    if "Allelic_ratio" in df.columns: df = df[df["Allelic_ratio"] >= allelic_ratio_min]
    if max_cohort_freq < 1.0: df = df[df["internal_freq"] <= max_cohort_freq]
    if genes_exclude: df = df[~df["Gene_symbol"].isin(genes_exclude)]
    if patients_exclude and "Pseudo" in df.columns: df = df[~df["Pseudo"].isin(patients_exclude)]
    logs.append({"Etape": "2. Qualité & Cohorte", "Restants": len(df), "Perdus": last_count - len(df)}); last_count = len(df)

    if use_gnomad_filter and "gnomad_exomes_NFE_AF" in df.columns:
        df = df[(df["gnomad_exomes_NFE_AF"].isna()) | (df["gnomad_exomes_NFE_AF"] <= gnomad_max)]
    logs.append({"Etape": "3. gnomAD", "Restants": len(df), "Perdus": last_count - len(df)}); last_count = len(df)

    if variant_effect_keep and "Variant_effect" in df.columns: df = df[df["Variant_effect"].isin(variant_effect_keep)]
    if putative_keep and "Putative_impact" in df.columns: df = df[df["Putative_impact"].isin(putative_keep)]
    if clinvar_keep and "Clinvar_significance" in df.columns: df = df[df["Clinvar_significance"].isin(clinvar_keep)]
    if min_cadd is not None and min_cadd > 0 and "CADD_phred" in df.columns:
        df = df[(df["CADD_phred"].isna()) | (df["CADD_phred"] >= min_cadd)]
    logs.append({"Etape": "4. Catégories & CADD", "Restants": len(df), "Perdus": last_count - len(df)}); last_count = len(df)

    # --- LIENS ---
    if "Variant" in df.columns: df["link_varsome"] = df["Variant"].apply(make_varsome_link)
    else: df["link_varsome"] = ""

    # --- MSC ---
    df["MSC_Ref"] = np.nan
    df["MSC_Status"] = "N/A"

    msc_df = None
    if os.path.exists(MSC_LOCAL_FILENAME):
        try: msc_df = pd.read_csv(MSC_LOCAL_FILENAME, sep="\t", dtype=str)
        except: pass
    if msc_file_uploaded_content is not None:
        try: msc_df = pd.read_csv(io.StringIO(msc_file_uploaded_content), sep="\t", dtype=str)
        except:
             try: msc_df = pd.read_csv(io.StringIO(msc_file_uploaded_content), sep=",", dtype=str)
             except: pass
    
    if msc_df is not None:
        try:
            msc_df.columns = [c.strip() for c in msc_df.columns]
            if "Gene" in msc_df.columns and "MSC" in msc_df.columns:
                msc_clean = msc_df[["Gene", "MSC"]].copy()
                msc_clean["Gene"] = msc_clean["Gene"].str.upper().str.strip()
                msc_clean["MSC"] = pd.to_numeric(msc_clean["MSC"], errors='coerce')
                msc_clean = msc_clean.groupby("Gene")["MSC"].max().reset_index()
                
                df = df.merge(msc_clean, left_on="Gene_symbol", right_on="Gene", how="left")
                df["MSC_Ref"] = df["MSC"]
                
                if "CADD_phred" in df.columns:
                    cond_has_values = (df["CADD_phred"].notna()) & (df["MSC_Ref"].notna())
                    cond_low = cond_has_values & (df["CADD_phred"] < df["MSC_Ref"])
                    df.loc[cond_low, "MSC_Status"] = "Background"
                    df.loc[~cond_low & cond_has_values, "MSC_Status"] = "High Impact"

                    if use_msc_filter_strict:
                        df = df[~cond_low]
                        logs.append({"Etape": "5. Filtre MSC Strict", "Restants": len(df), "Perdus": last_count - len(df)})
                        last_count = len(df)
        except Exception as e:
            logs.append({"Etape": "Erreur MSC", "Info": str(e)})

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
            benign_clinvar = "benign" in clv and "pathogenic" not in clv
            ba1 = (af > 0.05) if not pd.isna(af) else False
            
            if ba1 or benign_clinvar: return "Benign", 0
            score = 0
            if pvs1: score += 4
            if pm2: score += 2
            if pp5: score += 2
            if pp3: score += 1
            
            if score >= 5: return "Pathogenic", 4
            elif score >= 3: return "Likely Pathogenic", 3
            elif score >= 1: return "VUS", 2
            else: return "Likely Benign", 1

        acmg_res = df.apply(compute_acmg_class, axis=1, result_type='expand')
        df["ACMG_Class"] = acmg_res[0]
        df["ACMG_Rank"] = acmg_res[1]
    else:
        df["ACMG_Class"] = "Non calculé"
        df["ACMG_Rank"] = 0

    # --- FILTRE ACMG GLOBAL ---
    if acmg_keep_list:
        last_count = len(df)
        df = df[df["ACMG_Class"].isin(acmg_keep_list)]
        logs.append({"Etape": "6. Filtre ACMG Global", "Restants": len(df), "Perdus": last_count - len(df)})

    # --- TRI ---
    if sort_by_column == "Classification ACMG (Priorité)": 
        df = df.sort_values("ACMG_Rank", ascending=False)
    elif sort_by_column == "Score CADD (Décroissant)": 
        if "CADD_phred" in df.columns: df = df.sort_values("CADD_phred", ascending=False)
    elif sort_by_column == "Patient (A-Z)": 
        if "Pseudo" in df.columns: df = df.sort_values("Pseudo", ascending=True)

    return df, initial_total, len(df), None, logs

# ---------------------------------------
# 3. ANALYSE PATHWAYS
# ---------------------------------------

def load_local_pathways(directory="."):
    pathways = {}
    gmt_files = glob.glob(os.path.join(directory, "*.gmt"))
    for fpath in gmt_files:
        try:
            with open(fpath, "r", encoding="utf-8", errors="ignore") as f:
                for line in f:
                    parts = line.strip().split("\t")
                    if len(parts) >= 3:
                        pw = parts[0].strip()
                        genes = [g.strip().upper() for g in parts[2:] if g.strip()]
                        if pw and genes:
                            if pw not in pathways: pathways[pw] = []
                            pathways[pw].extend(genes)
        except: pass
    
    for k in pathways: pathways[k] = list(set(pathways[k]))
    return pathways, [os.path.basename(f) for f in gmt_files]

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
            rows.append({"pathway": pw, "p_value": pval, "k_overlap": k, "genes": ",".join(overlap)})

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

render_custom_header()

if "analysis_done" not in st.session_state:
    st.session_state["analysis_done"] = False
    st.session_state["df_res"] = None
    st.session_state["kpis"] = (0, 0)
    st.session_state["logs"] = []

# ==========================================================
# PARTIE SIDEBAR - CORRIGÉE POUR LE JSON ET LES FILTRES
# ==========================================================

with st.sidebar:
    st.header("1. Données")
    uploaded_file = st.file_uploader("Fichier Variants", type=["csv", "tsv", "txt"])
    
    df_raw = None
    v_opts, p_opts, c_opts = [], [], []

    # Calcul des options si le fichier est là
    if uploaded_file:
        df_raw = load_variants(uploaded_file)
        if df_raw is not None:
            if "Variant_effect" in df_raw.columns: 
                v_opts = sorted(df_raw["Variant_effect"].fillna("Non Renseigné").unique())
            if "Putative_impact" in df_raw.columns: 
                p_opts = sorted(df_raw["Putative_impact"].fillna("Non Renseigné").unique())
            if "Clinvar_significance" in df_raw.columns: 
                c_opts = sorted(df_raw["Clinvar_significance"].fillna("Non Renseigné").unique())

            # --- CORRECTION 1 : INITIALISATION "TOUT COCHÉ PAR DÉFAUT" ---
            # Si le state n'existe pas, on le met à TOUTES les options.
            # Si le state existe, on nettoie pour enlever les options qui n'existent pas dans ce nouveau fichier
            # (ce qui évite les erreurs de clé invalide quand on change de fichier CSV)

            if "sel_var" not in st.session_state:
                st.session_state["sel_var"] = v_opts
            else:
                # On garde seulement ce qui est valide dans le fichier actuel
                st.session_state["sel_var"] = [x for x in st.session_state["sel_var"] if x in v_opts]
                # Si la liste est vide après nettoyage, on re-sélectionne tout (sécurité)
                if not st.session_state["sel_var"]: st.session_state["sel_var"] = v_opts

            if "sel_put" not in st.session_state:
                st.session_state["sel_put"] = p_opts
            else:
                st.session_state["sel_put"] = [x for x in st.session_state["sel_put"] if x in p_opts]
                if not st.session_state["sel_put"]: st.session_state["sel_put"] = p_opts

            if "sel_clin" not in st.session_state:
                st.session_state["sel_clin"] = c_opts
            else:
                st.session_state["sel_clin"] = [x for x in st.session_state["sel_clin"] if x in c_opts]
                if not st.session_state["sel_clin"]: st.session_state["sel_clin"] = c_opts

    # ------------------------------------------------------------
    # GESTIONNAIRE DE CONFIGURATION (JSON)
    # ------------------------------------------------------------
    st.header("0. Configuration")
    uploaded_config = st.file_uploader("📂 Charger stratégie (JSON)", type=["json"], key="config_uploader")
    
    if uploaded_config is not None:
        try:
            # CORRECTION 2 : APPLICATION FIABLE DU JSON
            if st.button("🔄 APPLIQUER LA CONFIGURATION"):
                data = json.load(uploaded_config)
                
                # On met à jour le session_state DIRECTEMENT
                for key, value in data.items():
                    # Pour les listes, on vérifie que les valeurs existent dans le fichier actuel
                    if key == "sel_var" and isinstance(value, list):
                        valid_vals = [x for x in value if x in v_opts]
                        st.session_state[key] = valid_vals
                    elif key == "sel_put" and isinstance(value, list):
                        valid_vals = [x for x in value if x in p_opts]
                        st.session_state[key] = valid_vals
                    elif key == "sel_clin" and isinstance(value, list):
                        valid_vals = [x for x in value if x in c_opts]
                        st.session_state[key] = valid_vals
                    else:
                        st.session_state[key] = value
                
                st.success("Configuration chargée !")
                st.rerun() # INDISPENSABLE : Recharge l'interface pour afficher les nouvelles valeurs
                
        except Exception as e:
            st.error(f"Erreur config: {e}")

    # Sauvegarde (toujours disponible)
    keys_to_save = ["sort_choice", "min_dp", "allelic_min", "min_ad", "max_cohort_freq",
                    "gnomad_max", "min_cadd_val", "use_acmg", "use_gnomad", 
                    "acmg_to_keep", "genes_ex", "pseudo_ex", 
                    "sel_var", "sel_put", "sel_clin", "use_msc_strict"]
    
    current_config = {k: st.session_state[k] for k in keys_to_save if k in st.session_state}
    if current_config:
        st.download_button("💾 Sauvegarder Config", json.dumps(current_config, indent=4), "ngs_filter.json", "application/json")
    
    st.markdown("---")

    # ------------------------------------------------------------
    # FORMULAIRE
    # ------------------------------------------------------------
    with st.form("params"):
        st.header("2. Paramètres")
        sort_choice = st.selectbox("Tri initial", ["Classification ACMG (Priorité)", "Score CADD (Décroissant)", "Patient (A-Z)"], key="sort_choice")
        
        c1, c2 = st.columns(2)
        with c1:
            min_dp = st.number_input("Depth Min", 0, 10000, 50, key="min_dp")
            allelic_min = st.number_input("VAF Min", 0.0, 1.0, 0.02, key="allelic_min")
        with c2:
            min_ad = st.number_input("Alt Depth Min", 0, 1000, 5, key="min_ad")
            max_cohort_freq = st.slider("Max Freq Cohorte", 0.0, 1.0, 1.0, 0.05, key="max_cohort_freq")

        c3, c4 = st.columns(2)
        with c3:
            gnomad_max = st.number_input("gnomAD Max", 0.0, 1.0, 0.001, format="%.4f", key="gnomad_max")
            min_cadd_val = st.number_input("CADD Min (0=all)", 0.0, 60.0, 0.0, key="min_cadd_val")
        with c4:
            use_acmg = st.checkbox("Calculer ACMG", value=True, key="use_acmg")
            use_gnomad = st.checkbox("Filtre gnomAD", True, key="use_gnomad")
            
            acmg_options = ["Pathogenic", "Likely Pathogenic", "VUS", "Likely Benign", "Benign", "Non calculé"]
            acmg_to_keep = st.multiselect("Filtre Global ACMG", options=acmg_options, default=acmg_options, key="acmg_to_keep")

        with st.expander("Avancé & Filtres MSC"):
            # CORRECTION 3 : Suppression du paramètre 'default' car 'key' est utilisé
            # Streamlit utilisera automatiquement la valeur stockée dans st.session_state[key]
            # qui a été initialisée plus haut.
            
            sel_var = st.multiselect("Effet", options=v_opts, key="sel_var")
            sel_put = st.multiselect("Impact", options=p_opts, key="sel_put")
            sel_clin = st.multiselect("ClinVar", options=c_opts, key="sel_clin")
            
            genes_ex = st.text_area("Exclure Gènes", "KMT2C, CHEK2, TTN, MUC16", key="genes_ex")
            pseudo_ex = st.text_area("Exclure Patients", "", key="pseudo_ex")
            
            st.markdown("---")
            st.markdown("**🛡️ MSC Filter**")
            
            has_local_msc = os.path.exists(MSC_LOCAL_FILENAME)
            msc_file_upload = None
            if has_local_msc:
                st.success(f"MSC local : {MSC_LOCAL_FILENAME}")
            else:
                st.warning("MSC local non trouvé.")
                msc_file_upload = st.file_uploader("Upload MSC", type=["txt", "tsv", "csv"])

            use_msc_filter_strict = st.checkbox("Exclure si CADD < MSC", value=False, key="use_msc_strict")

        st.header("3. Pathways")
        submitted = st.form_submit_button("🚀 LANCER L'ANALYSE")

if submitted and df_raw is not None:
    g_list = [x.strip().upper() for x in genes_ex.split(",") if x.strip()]
    p_list = [x.strip() for x in pseudo_ex.split(",") if x.strip()]
    
    msc_content = None
    if msc_file_upload is not None:
        msc_content = msc_file_upload.read().decode("utf-8")

    res, ini, fin, err, logs = apply_filtering_and_scoring(
        df_raw, allelic_min, gnomad_max, use_gnomad, 
        min_dp, min_ad, max_cohort_freq, 
        msc_content,
        g_list, p_list, 
        min_cadd_val, sel_var, sel_put, sel_clin, sort_choice, use_acmg,
        use_msc_filter_strict,
        acmg_to_keep 
    )

    if err: st.error(err)
    else:
        st.session_state["analysis_done"] = True
        st.session_state["df_res"] = res
        st.session_state["kpis"] = (ini, fin)
        st.session_state["logs"] = logs
        
        user_pathways, file_names = load_local_pathways()
        st.session_state["gmt_files"] = file_names
        if user_pathways:
            st.session_state["df_enr"] = compute_enrichment(res, user_pathways)
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

    # KPI
    k1, k2, k3 = st.columns(3)
    k1.metric("Initial", n_ini)
    k2.metric("Final", n_fin)
    k3.metric("Ratio", f"{round(n_fin/n_ini*100, 2) if n_ini>0 else 0}%")

    # Onglets
    tabs = st.tabs(["📋 Tableau", "🔍 Inspecteur", "🧩 Corrélation", "📊 Spectre", "📍 Lollipops", "📈 QC", "🧬 Pathways", "🕸️ PPI", "🧬 Évolution Clonale"])

    # --- TAB 1: AGGRID ---
    with tabs[0]:
        st.subheader("📋 Liste des variants filtrés")
        
        df_to_show = df_res.copy() 
        
        # Filtre local d'affichage ACMG
        if "ACMG_Class" in df_to_show.columns:
            all_acmg_classes = sorted(df_to_show["ACMG_Class"].astype(str).unique())
            selected_acmg = st.multiselect("Filtrer l'affichage :", options=all_acmg_classes, default=all_acmg_classes)
            df_to_show = df_to_show[df_to_show["ACMG_Class"].isin(selected_acmg)]
            st.caption(f"Affichage de {len(df_to_show)} variants sur {len(df_res)} (Total filtré).")

        if "link_varsome" in df_to_show.columns:
            df_to_show["Varsome_HTML"] = df_to_show["link_varsome"].apply(lambda x: f'<a href="{x}" target="_blank">🔗</a>' if x else "")
        
        cols_base = ["Pseudo", "Gene_symbol", "Variant", "ACMG_Class", "CADD_phred", "MSC_Ref", "MSC_Status", "Allelic_ratio","Varsome_HTML"]
        existing = [c for c in cols_base if c in df_to_show.columns]
        others = [c for c in df_to_show.columns if c not in existing and c not in ["link_varsome", "link_gnomad", "MSC", "ACMG_Rank"]]
        
        df_display = df_to_show[existing + others].copy()

        gb = GridOptionsBuilder.from_dataframe(df_display)
        gb.configure_pagination(paginationPageSize=20)
        gb.configure_selection('multiple', use_checkbox=True)
        gb.configure_default_column(resizable=True, filterable=True, sortable=True, minWidth=150)
        gb.configure_column("Variant", minWidth=200)

        acmg_style = JsCode("""
        function(params) {
            if (params.value == 'Pathogenic') return {'color': 'white', 'backgroundColor': '#d9534f'};
            if (params.value == 'Likely Pathogenic') return {'color': 'black', 'backgroundColor': '#f0ad4e'};
            if (params.value == 'VUS') return {'color': 'black', 'backgroundColor': '#5bc0de'};
            return null;
        };
        """)
        gb.configure_column("ACMG_Class", cellStyle=acmg_style)

        msc_style = JsCode("""
        function(params) {
            if (params.value == 'High Impact') return {'color': '#a94442', 'backgroundColor': '#f2dede', 'fontWeight': 'bold'};
            if (params.value == 'Background') return {'color': '#999', 'fontStyle': 'italic'};
            return null;
        };
        """)
        gb.configure_column("MSC_Status", cellStyle=msc_style)
        
        if "Varsome_HTML" in df_display.columns: gb.configure_column("Varsome_HTML", headerName="Lien", cellRenderer="html")
        if "link_varsome" in df_display.columns: gb.configure_column("link_varsome", hide=True)

        grid_response = AgGrid(df_display, gridOptions=gb.build(), allow_unsafe_jscode=True, height=600, fit_columns_on_grid_load=False)
        
        df_selected = pd.DataFrame(grid_response['selected_rows'])
        if df_selected.empty: df_selected = df_display

        st.markdown("---")
        c_rep1, c_rep2 = st.columns([3, 1])
        with c_rep1:
            st.info(f"**{len(df_selected)} variants** pour le rapport.")
            user_comment = st.text_area("Commentaire clinique :")
        with c_rep2:
            st.write("##")
            pat_id = df_selected["Pseudo"].unique()[0] if "Pseudo" in df_selected.columns and len(df_selected["Pseudo"].unique()) == 1 else "Multi"
            try:
                pdf_bytes = create_pdf_report(pat_id, df_selected, user_comment)
                st.download_button("📥 PDF Report", pdf_bytes, f"Rapport_{pat_id}.pdf", "application/pdf", type="primary")
            except Exception as e: st.error(f"Erreur PDF: {e}")

    # --- TAB 2: INSPECTEUR ---
    with tabs[1]:
        st.subheader("🔍 Inspecteur Clinique (ACMG)")
        if "Pseudo" in df_res.columns:
            sel_pat = st.selectbox("Patient", sorted(df_res["Pseudo"].unique()))
            if sel_pat:
                df_pat = df_res[df_res["Pseudo"] == sel_pat].copy()
                df_pat = df_pat.sort_values(["ACMG_Rank", "CADD_phred"], ascending=False).head(20)
                
                fig_pat = px.bar(
                    df_pat, x="CADD_phred", y="Gene_symbol", orientation='h',
                    color="ACMG_Class", 
                    title=f"Top Variants {sel_pat}",
                    hover_data=["Variant", "MSC_Status"],
                    color_discrete_map={"Pathogenic": "#d9534f", "Likely Pathogenic": "#f0ad4e", "VUS": "#5bc0de", "Likely Benign": "#5cb85c", "Benign": "green"}
                )
                fig_pat.update_layout(yaxis={'categoryorder':'total ascending'})
                st.plotly_chart(fig_pat, use_container_width=True)

    # --- TAB 3: CORRELATION ---
    with tabs[2]:
        st.subheader("🧩 OncoPrint & Heatmap")
        if "Pseudo" in df_res.columns and "Gene_symbol" in df_res.columns and not df_res.empty:
            view_mode = st.radio("Mode", ["Heatmap", "OncoPrint"], horizontal=True)
            top_genes = df_res["Gene_symbol"].value_counts().head(30).index.tolist()
            df_heat = df_res[df_res["Gene_symbol"].isin(top_genes)].copy()
            
            if not df_heat.empty:
                if view_mode == "Heatmap":
                    matrix = df_heat.pivot_table(index="Pseudo", columns="Gene_symbol", aggfunc='size', fill_value=0)
                    matrix[matrix > 0] = 1 
                    co_occ = matrix.T.dot(matrix)
                    st.plotly_chart(px.imshow(co_occ, text_auto=True, color_continuous_scale="Viridis", height=800), use_container_width=True)
                else: 
                    def get_effect_score(eff):
                        e = str(eff).lower()
                        if any(x in e for x in ["stop", "frameshift", "nonsense"]): return 3
                        if "splice" in e: return 2
                        if "missense" in e: return 1
                        return 0.5
                    df_heat["Effet_Code"] = df_heat["Variant_effect"].apply(get_effect_score)
                    matrix_onco = df_heat.pivot_table(index="Gene_symbol", columns="Pseudo", values="Effet_Code", aggfunc='max', fill_value=0)
                    colors = [[0,"white"],[0.05,"white"],[0.05,"lightgrey"],[0.25,"lightgrey"],[0.25,"blue"],[0.5,"blue"],[0.5,"orange"],[0.8,"orange"],[0.8,"red"],[1,"red"]]
                    st.plotly_chart(go.Figure(data=go.Heatmap(z=matrix_onco.values, x=matrix_onco.columns, y=matrix_onco.index, colorscale=colors, showscale=False, zmin=0, zmax=3), layout=dict(height=800)), use_container_width=True)
        else: st.warning("Données insuffisantes.")

    # --- TAB 4: SPECTRE ---
    with tabs[3]:
        st.subheader("📊 Spectre Mutationnel")
        df_mut = extract_ref_alt_chr(df_res.copy())
        if "Ref" in df_mut.columns and "Alt" in df_mut.columns:
            df_mut["mutation"] = df_mut["Ref"] + ">" + df_mut["Alt"]
            trans_map = {'G>T': 'C>A', 'G>C': 'C>G', 'G>A': 'C>T', 'A>T': 'T>A', 'A>G': 'T>C', 'A>C': 'T>G'}
            df_mut["canon_mut"] = df_mut["mutation"].apply(lambda x: trans_map.get(x, x))
            valid_snvs = ['C>A', 'C>G', 'C>T', 'T>A', 'T>C', 'T>G']
            df_mut = df_mut[df_mut["canon_mut"].isin(valid_snvs)]
            if not df_mut.empty:
                counts = df_mut.groupby(["Pseudo", "canon_mut"]).size().reset_index(name="Count")
                colors = {'C>A': '#1ebff0', 'C>G': '#050708', 'C>T': '#e62725', 'T>A': '#cbcacb', 'T>C': '#a1cf64', 'T>G': '#edc8c5'}
                st.plotly_chart(px.bar(counts, x="Pseudo", y="Count", color="canon_mut", color_discrete_map=colors), use_container_width=True)
            else: st.info("Aucun SNV standard.")
        else: st.warning("Impossible d'extraire Ref/Alt.")

    # --- TAB 5: LOLLIPOPS ---
    with tabs[4]:
        st.subheader("📍 Lollipop Plot")
        prot_cols = ["hgvs.p", "HGVSp", "Protein_change", "AA_change", "hgvsp"]
        found_col = next((c for c in prot_cols if c in df_res.columns), None)
        if found_col:
            sel_gene_lol = st.selectbox("Gène", sorted(df_res["Gene_symbol"].unique()))
            if sel_gene_lol:
                df_lol = df_res[df_res["Gene_symbol"] == sel_gene_lol].copy()
                df_lol["AA_pos"] = pd.to_numeric(df_lol[found_col].astype(str).str.extract(r'(\d+)')[0], errors="coerce")
                df_lol = df_lol.dropna(subset=["AA_pos", "CADD_phred"])
                if not df_lol.empty:
                    fig_lol = px.scatter(df_lol, x="AA_pos", y="CADD_phred", color="ACMG_Class", size="CADD_phred", hover_data=["Pseudo", found_col], title=f"{sel_gene_lol}")
                    if "MSC_Ref" in df_lol.columns: fig_lol.add_hline(y=df_lol["MSC_Ref"].iloc[0], line_dash="dash", line_color="red")
                    for _, row in df_lol.iterrows(): fig_lol.add_shape(type="line", x0=row["AA_pos"], y0=0, x1=row["AA_pos"], y1=row["CADD_phred"], line=dict(color="grey", width=1))
                    st.plotly_chart(fig_lol, use_container_width=True)

    # --- TAB 6: QC ---
    with tabs[5]: st.plotly_chart(px.scatter(df_res, x="Depth", y="Allelic_ratio", color="ACMG_Class", log_x=True))

    # --- TAB 7: PATHWAYS ---
    with tabs[6]:
        if not df_enr.empty and "minus_log10_FDR" in df_enr.columns:
            st.plotly_chart(px.bar(df_enr.sort_values("FDR").head(20), x="minus_log10_FDR", y="pathway", orientation='h', color="k_overlap"), use_container_width=True)
            st.dataframe(df_enr)
        else: st.info("Aucun enrichissement significatif.")

    # --- TAB 8: PPI ---
    with tabs[7]:
        st.subheader("🕸️ Réseau STRING")
        all_genes = sorted(df_res["Gene_symbol"].unique())
        if all_genes:
            c_ppi1, c_ppi2 = st.columns([1, 3])
            with c_ppi1:
                selected_gene_ppi = st.selectbox("Gène :", all_genes)
                nb_partners = st.slider("Partenaires", 5, 20, 10)
            with c_ppi2:
                if selected_gene_ppi:
                    network = get_string_network(selected_gene_ppi, limit=nb_partners)
                    if network:
                        nodes, edges, added = [], [], set()
                        nodes.append(Node(id=selected_gene_ppi, label=selected_gene_ppi, size=25, color="#d9534f", shape="dot"))
                        added.add(selected_gene_ppi)
                        for i in network:
                            try:
                                ga, gb, s = i.get("preferredName_A").upper(), i.get("preferredName_B").upper(), i.get("score", 0)
                                if s < 0.4: continue
                                if ga not in added: nodes.append(Node(id=ga, label=ga, size=15, color="#5bc0de")); added.add(ga)
                                if gb not in added: nodes.append(Node(id=gb, label=gb, size=15, color="#5bc0de")); added.add(gb)
                                edges.append(Edge(source=ga, target=gb, width=s*2))
                            except: pass
                        agraph(nodes=nodes, edges=edges, config=Config(width=700, height=500, directed=False, physics=True))
                    else: st.warning("Pas d'interactions.")
    
    # --- TAB 9: EVOLUTION CLONALE (AVEC RAPPORT PDF GLOBAL) ---
    with tabs[8]:
        st.subheader("🧬 Analyse de l'Architecture Clonale")
        
        # Vérification des colonnes nécessaires
        if "Pseudo" in df_res.columns and "Allelic_ratio" in df_res.columns:
            
            # --- SECTION 1 : VISUALISATION INTERACTIVE (Reste inchangée) ---
            c_sel1, c_sel2 = st.columns([1, 3])
            with c_sel1:
                patients_list = sorted(df_res["Pseudo"].astype(str).unique())
                sel_pat_clon = st.selectbox("Sélectionner un Patient :", patients_list)
            
            n_clusters_def = 3 
            
            if sel_pat_clon:
                df_clon = df_res[df_res["Pseudo"] == sel_pat_clon].copy()
                df_clon = df_clon.dropna(subset=["Allelic_ratio"])
                
                col_c1, col_c2 = st.columns([1, 3])
                with col_c1:
                    n_clusters = st.slider("Nombre de clones", 1, 5, n_clusters_def, key="slider_clon_indiv")
                    st.info(f"Variants : {len(df_clon)}")
                
                with col_c2:
                    if len(df_clon) < 3:
                        st.warning("Pas assez de variants (<3).")
                    else:
                        try:
                            # K-Means
                            X = df_clon[["Allelic_ratio"]].values
                            kmeans = KMeans(n_clusters=n_clusters, random_state=42, n_init=10)
                            df_clon["Cluster_ID"] = kmeans.fit_predict(X)
                            
                            centroids = df_clon.groupby("Cluster_ID")["Allelic_ratio"].mean().sort_values().index
                            cluster_map = {old_id: f"C{i+1}" for i, old_id in enumerate(centroids)}
                            df_clon["Cluster_Label"] = df_clon["Cluster_ID"].map(cluster_map)
                            
                            # Plotly (Interactif)
                            fig_clon = px.histogram(
                                df_clon, x="Allelic_ratio", color="Cluster_Label", 
                                nbins=30, marginal="rug", opacity=0.7, barmode="overlay",
                                title=f"Architecture - {sel_pat_clon}",
                                color_discrete_sequence=px.colors.qualitative.G10
                            )
                            fig_clon.update_layout(xaxis_range=[0, 1.05])
                            st.plotly_chart(fig_clon, use_container_width=True)
                            
                        except Exception as e: st.error(f"Erreur : {e}")

            # --- SECTION 2 : EXPORT PDF GLOBAL ---
            st.markdown("---")
            st.subheader("📄 Rapport PDF Global")
            st.info("Génère un PDF unique avec une page par patient (Graphique + Tableau).")
            
            if st.button("Générer le PDF Global"):
                
                class PDFReport(FPDF):
                    def header(self):
                        self.set_font('Arial', 'B', 10)
                        self.cell(0, 10, 'Atlas Clonal Analysis Report', 0, 1, 'R')
                    def footer(self):
                        self.set_y(-15)
                        self.set_font('Arial', 'I', 8)
                        self.cell(0, 10, f'Page {self.page_no()}', 0, 0, 'C')

                pdf = PDFReport()
                progress_bar = st.progress(0)
                
                # Création d'un dossier temporaire pour les images
                with tempfile.TemporaryDirectory() as tmp_dir:
                    total_pats = len(patients_list)
                    
                    for idx, pat in enumerate(patients_list):
                        progress_bar.progress((idx + 1) / total_pats)
                        
                        # Filtrage et Calcul
                        d_temp = df_res[df_res["Pseudo"] == pat].copy().dropna(subset=["Allelic_ratio"])
                        
                        if len(d_temp) >= 3:
                            try:
                                # 1. Calculs Clusters
                                km = KMeans(n_clusters=3, random_state=42, n_init=10) # Force 3 clusters pour standardisation
                                d_temp["Cluster_ID"] = km.fit_predict(d_temp[["Allelic_ratio"]].values)
                                cents = d_temp.groupby("Cluster_ID")["Allelic_ratio"].mean().sort_values().index
                                cmap = {oid: f"C{i+1}" for i, oid in enumerate(cents)}
                                d_temp["Cluster_Label"] = d_temp["Cluster_ID"].map(cmap)
                                
                                # 2. Génération Graphique Matplotlib (Statique pour PDF)
                                plt.figure(figsize=(10, 5))
                                sns.histplot(data=d_temp, x="Allelic_ratio", hue="Cluster_Label", 
                                             bins=30, kde=True, palette="viridis", element="step")
                                plt.title(f"Patient: {pat} - Architecture Clonale")
                                plt.xlim(0, 1.05)
                                plt.xlabel("VAF")
                                plt.ylabel("Count")
                                
                                # Sauvegarde image
                                img_path = os.path.join(tmp_dir, f"{clean_text(pat)}.png")
                                plt.savefig(img_path, dpi=100, bbox_inches='tight')
                                plt.close()

                                # 3. Ajout Page PDF
                                pdf.add_page()
                                pdf.set_font("Arial", 'B', 16)
                                pdf.cell(0, 10, f"Patient : {pat}", 0, 1, 'L')
                                
                                # Image
                                pdf.image(img_path, x=10, y=30, w=190)
                                
                                # Tableau
                                pdf.set_y(130)
                                pdf.set_font("Arial", 'B', 10)
                                pdf.cell(0, 10, "Tableau des Variants (Top 15 par VAF)", 0, 1)
                                
                                # En-têtes tableau
                                cols = [("Gene", 25), ("Variant", 50), ("VAF", 20), ("Clone", 20), ("ACMG", 40)]
                                pdf.set_fill_color(220, 220, 220)
                                for c_name, c_w in cols:
                                    pdf.cell(c_w, 8, c_name, 1, 0, 'C', 1)
                                pdf.ln()
                                
                                # Données tableau (Trié par Clone puis VAF)
                                pdf.set_font("Arial", '', 9)
                                d_table = d_temp.sort_values(["Cluster_Label", "Allelic_ratio"], ascending=[True, False]).head(15)
                                
                                for _, row in d_table.iterrows():
                                    gene = str(row.get("Gene_symbol", ""))[:10]
                                    var = str(row.get("Variant", ""))[:25]
                                    vaf = f"{row.get('Allelic_ratio', 0):.2f}"
                                    clone = str(row.get("Cluster_Label", ""))
                                    acmg = str(row.get("ACMG_Class", ""))[:20]
                                    
                                    pdf.cell(25, 7, gene, 1)
                                    pdf.cell(50, 7, var, 1)
                                    pdf.cell(20, 7, vaf, 1, 0, 'C')
                                    pdf.cell(20, 7, clone, 1, 0, 'C')
                                    pdf.cell(40, 7, acmg, 1)
                                    pdf.ln()
                                    
                            except Exception as e:
                                st.warning(f"Skipped {pat}: {e}")
                
                # Téléchargement
                pdf_bytes = pdf.output(dest='S').encode('latin-1', 'ignore')
                st.download_button(
                    label="📥 Télécharger le Rapport PDF",
                    data=pdf_bytes,
                    file_name=f"Rapport_Clonal_Global_{datetime.now().strftime('%Y%m%d')}.pdf",
                    mime="application/pdf",
                    type="primary"
                )
                progress_bar.empty()

        else:
            st.warning("Données insuffisantes (Pseudo/VAF manquants).")

elif not submitted:
    st.info("👈 Chargez fichier + Lancer.")
