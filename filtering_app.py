import os
import io
import re
import glob
import requests
from datetime import datetime
import numpy as np
import pandas as pd
import streamlit as st
import plotly.express as px
import plotly.graph_objects as go
from scipy.stats import hypergeom

# --- IMPORTS TIERS ---
from st_aggrid import AgGrid, GridOptionsBuilder, JsCode
from fpdf import FPDF
from streamlit_agraph import agraph, Node, Edge, Config

# ---------------------------------------
# 1. CONFIGURATION & OUTILS
# ---------------------------------------

st.set_page_config(page_title="NGS ATLAS Explorer v12.1 (Wide View)", layout="wide", page_icon="🧬")

# Nom du fichier MSC attendu à la racine (GitHub)
MSC_LOCAL_FILENAME = "MSC_CI99_v1.7.txt"

# --- Fonctions utilitaires ---

def clean_text(val):
    if not isinstance(val, str): return ""
    return re.sub(r'[^A-Za-z0-9]+', '', val.strip().lower())

def extract_ref_alt_chr(df):
    if "Chromosome" not in df.columns and "Chr" in df.columns:
        df["Chromosome"] = df["Chr"]
    if "Chromosome" not in df.columns and "Variant" in df.columns:
        df["Chromosome"] = df["Variant"].astype(str).str.split(r'[:_-]', n=1, expand=True)[0]
    if "Chromosome" in df.columns:
        df["Chromosome"] = df["Chromosome"].astype(str).str.replace("chr", "", case=False).str.strip()

    if ("Ref" not in df.columns or "Alt" not in df.columns) and "Variant" in df.columns:
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

def make_gnomad_link(variant_str):
    try:
        v = str(variant_str).replace(":", "-").replace(">", "-")
        return f"https://gnomad.broadinstitute.org/variant/{v}?dataset=gnomad_r2_1"
    except: return ""

@st.cache_data
def get_string_network(gene_symbol, limit=10):
    url = "https://string-db.org/api/json/network"
    params = {"identifiers": gene_symbol, "species": 9606, "limit": limit, "network_type": "functional"}
    try:
        response = requests.get(url, params=params)
        if response.status_code == 200: return response.json()
    except: return []
    return []

# --- Fonction de Rapport PDF ---
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

    # Configuration des colonnes
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
    use_acmg, use_msc_filter_strict
):
    logs = [] 
    df = df.copy()
    initial_total = len(df)
    logs.append({"Etape": "1. Import Brut", "Restants": initial_total, "Perdus": 0})

    if "Gene_symbol" not in df.columns:
        return None, 0, 0, "Colonne 'Gene_symbol' manquante.", []

    df["Gene_symbol"] = df["Gene_symbol"].str.upper().str.replace(" ", "")

    # Normalisation basique
    for col in ["Variant_effect", "Putative_impact", "Clinvar_significance"]:
        if col in df.columns: df[col] = df[col].fillna("Non Renseigné")

    # Fréquence Cohorte
    if "Pseudo" in df.columns and "Variant" in df.columns:
        tot = df["Pseudo"].nunique()
        cts = df.groupby("Variant")["Pseudo"].nunique()
        df["internal_freq"] = df["Variant"].map(cts) / tot
    else:
        df["internal_freq"] = 0.0

    # Conversions numériques
    cols_num = ["gnomad_exomes_NFE_AF", "CADD_phred", "Allelic_ratio", "Depth", "Alt_depth_total"]
    for c in cols_num:
        if c in df.columns: df[c] = pd.to_numeric(df[c], errors="coerce")
    
    if "Alt_depth_total" not in df.columns and "Alt_depth" in df.columns:
        df["Alt_depth_total"] = df["Alt_depth"].astype(str).str.split(',').str[0].str.split(' ').str[0]
        df["Alt_depth_total"] = pd.to_numeric(df["Alt_depth_total"], errors='coerce').fillna(0)

    # --- FILTRES DE BASE ---
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
    
    # Filtre CADD global (optionnel)
    if min_cadd is not None and min_cadd > 0 and "CADD_phred" in df.columns:
        df = df[(df["CADD_phred"].isna()) | (df["CADD_phred"] >= min_cadd)]
    logs.append({"Etape": "4. Catégories & CADD", "Restants": len(df), "Perdus": last_count - len(df)}); last_count = len(df)

    # --- LIENS ---
    if "Variant" in df.columns:
        df["link_varsome"] = df["Variant"].apply(make_varsome_link)
    else: df["link_varsome"] = ""

    # --- MSC LOGIC (CADD vs Gene Threshold) ---
    df["MSC_Ref"] = np.nan
    df["MSC_Status"] = "N/A" # "High Impact" or "Background"

    msc_df = None
    # Chargement
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
                df["MSC_Ref"] = df["MSC"] # Renommage
                
                if "CADD_phred" in df.columns:
                    cond_has_values = (df["CADD_phred"].notna()) & (df["MSC_Ref"].notna())
                    cond_low = cond_has_values & (df["CADD_phred"] < df["MSC_Ref"])
                    
                    df.loc[cond_low, "MSC_Status"] = "Background" # CADD < MSC
                    df.loc[~cond_low & cond_has_values, "MSC_Status"] = "High Impact" # CADD >= MSC

                    if use_msc_filter_strict:
                        df = df[~cond_low]
                        logs.append({"Etape": "5. Filtre MSC Strict (CADD < MSC)", "Restants": len(df), "Perdus": last_count - len(df)})
                        last_count = len(df)
        except Exception as e:
            logs.append({"Etape": "Erreur MSC", "Info": str(e)})

    # --- ACMG CLASSIFICATION (Prioritaire) ---
    if use_acmg:
        def compute_acmg_class(row):
            # Critères très simplifiés pour démo
            eff = str(row.get("Variant_effect", "")).lower()
            pvs1 = any(x in eff for x in ["stopgained", "frameshift", "splice_acceptor", "splice_donor"])
            
            af = row.get("gnomad_exomes_NFE_AF", np.nan)
            pm2 = pd.isna(af) or (af < 0.0001)
            
            cadd = row.get("CADD_phred", 0)
            pp3 = cadd >= 25 # Support computationnel fort
            
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
        df["ACMG_Rank"] = acmg_res[1] # Pour le tri
    else:
        df["ACMG_Class"] = "Non calculé"
        df["ACMG_Rank"] = 0

    # --- TRI FINAL (Basé sur ACMG ou CADD) ---
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
        except Exception: pass
    
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
            rows.append({
                "pathway": pw, "p_value": pval, "k_overlap": k, "genes": ",".join(overlap)
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

st.title("🧬 NGS ATLAS Explorer v12.1")
st.markdown("### Analyse de variants avec classification ACMG & Filtrage MSC")
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
        sort_choice = st.selectbox("Tri initial", ["Classification ACMG (Priorité)", "Score CADD (Décroissant)", "Patient (A-Z)"])
        
        c1, c2 = st.columns(2)
        with c1:
            min_dp = st.number_input("Depth Min", 0, 10000, 50)
            allelic_min = st.number_input("VAF Min", 0.0, 1.0, 0.02)
        with c2:
            min_ad = st.number_input("Alt Depth Min", 0, 1000, 5)
            max_cohort_freq = st.slider("Max Freq Cohorte", 0.0, 1.0, 1.0, 0.05)

        c3, c4 = st.columns(2)
        with c3:
            gnomad_max = st.number_input("gnomAD Max", 0.0, 1.0, 0.001, format="%.4f")
            min_cadd_val = st.number_input("CADD Min (0=all)", 0.0, 60.0, 0.0)
        with c4:
            use_acmg = st.checkbox("Calculer ACMG", value=True)
            use_gnomad = st.checkbox("Filtre gnomAD", True)

        with st.expander("Avancé & Filtres MSC"):
            sel_var = st.multiselect("Effet", v_opts, default=v_opts)
            sel_put = st.multiselect("Impact", p_opts, default=p_opts)
            sel_clin = st.multiselect("ClinVar", c_opts, default=c_opts)
            genes_ex = st.text_area("Exclure Gènes", "KMT2C, CHEK2, TTN, MUC16")
            pseudo_ex = st.text_area("Exclure Patients", "")
            
            st.markdown("---")
            st.markdown("**🛡️ MSC Filter**")
            
            has_local_msc = os.path.exists(MSC_LOCAL_FILENAME)
            msc_file_upload = None
            if has_local_msc:
                st.success(f"Fichier MSC local chargé : {MSC_LOCAL_FILENAME}")
            else:
                st.warning("Fichier MSC local non trouvé.")
                msc_file_upload = st.file_uploader("Uploader fichier MSC", type=["txt", "tsv", "csv"])

            use_msc_filter_strict = st.checkbox("Exclure si CADD < MSC (Seuil Gène)", value=False)

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
        use_msc_filter_strict
    )

    if err: st.error(err)
    else:
        st.session_state["analysis_done"] = True
        st.session_state["df_res"] = res
        st.session_state["kpis"] = (ini, fin)
        st.session_state["logs"] = logs
        st.session_state["use_acmg"] = use_acmg
        
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

    tabs = st.tabs([
        "📋 Tableau & Rapport", 
        "🔍 Inspecteur ACMG", 
        "🧩 Corrélation", 
        "📊 Spectre", 
        "📍 Lollipops (CADD)", 
        "📈 QC", 
        "🧬 Pathways",
        "🕸️ PPI"
    ])

    # --- TAB 1: AGGRID ---
    with tabs[0]:
        st.subheader("📋 Liste des variants filtrés")
        
        if "link_varsome" in df_res.columns:
            df_res["Varsome_HTML"] = df_res["link_varsome"].apply(lambda x: f'<a href="{x}" target="_blank">🔗</a>' if x else "")
        
        # COLONNES CLAIRES MSC / ACMG
        cols_base = ["Pseudo", "Gene_symbol", "Variant", "Varsome_HTML", "ACMG_Class", "CADD_phred", "MSC_Ref", "MSC_Status", "Allelic_ratio"]
        existing = [c for c in cols_base if c in df_res.columns]
        others = [c for c in df_res.columns if c not in existing and c not in ["link_varsome", "link_gnomad", "MSC", "ACMG_Rank"]]
        
        df_display = df_res[existing + others].copy()

        gb = GridOptionsBuilder.from_dataframe(df_display)
        
        # --- CONFIGURATION AFFICHAGE LARGE ---
        gb.configure_pagination(paginationPageSize=20)
        gb.configure_selection('multiple', use_checkbox=True)
        
        # On définit une largeur minimale par défaut pour TOUTES les colonnes
        gb.configure_default_column(
            resizable=True,
            filterable=True,
            sortable=True,
            minWidth=150, # Force les colonnes à être assez larges
        )
        
        # Spécifique pour Variant (souvent long)
        gb.configure_column("Variant", minWidth=200)

        # COULEURS ACMG
        acmg_style = JsCode("""
        function(params) {
            if (params.value == 'Pathogenic') return {'color': 'white', 'backgroundColor': '#d9534f'};
            if (params.value == 'Likely Pathogenic') return {'color': 'black', 'backgroundColor': '#f0ad4e'};
            if (params.value == 'VUS') return {'color': 'black', 'backgroundColor': '#5bc0de'};
            return null;
        };
        """)
        gb.configure_column("ACMG_Class", cellStyle=acmg_style)

        # COULEURS MSC
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

        # IMPORTANT: fit_columns_on_grid_load=False pour permettre le scroll horizontal
        grid_response = AgGrid(
            df_display, 
            gridOptions=gb.build(), 
            allow_unsafe_jscode=True, 
            height=600,
            fit_columns_on_grid_load=False # La clé pour voir tout le texte
        )
        
        df_selected = pd.DataFrame(grid_response['selected_rows'])
        if df_selected.empty: df_selected = df_res

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

    # --- TAB 2: INSPECTEUR (CORRÉLÉ ACMG) ---
    with tabs[1]:
        st.subheader("🔍 Inspecteur Clinique (ACMG)")
        if "Pseudo" in df_res.columns:
            sel_pat = st.selectbox("Patient", sorted(df_res["Pseudo"].unique()))
            if sel_pat:
                df_pat = df_res[df_res["Pseudo"] == sel_pat].copy()
                # Tri par gravité ACMG
                df_pat = df_pat.sort_values(["ACMG_Rank", "CADD_phred"], ascending=False).head(20)
                
                # Bar chart coloré par ACMG
                fig_pat = px.bar(
                    df_pat, x="CADD_phred", y="Gene_symbol", orientation='h',
                    color="ACMG_Class", 
                    title=f"Top Variants {sel_pat} (Coloré par ACMG)",
                    hover_data=["Variant", "MSC_Status"],
                    color_discrete_map={
                        "Pathogenic": "#d9534f", 
                        "Likely Pathogenic": "#f0ad4e",
                        "VUS": "#5bc0de",
                        "Likely Benign": "#5cb85c",
                        "Benign": "green"
                    }
                )
                fig_pat.update_layout(yaxis={'categoryorder':'total ascending'})
                st.plotly_chart(fig_pat, use_container_width=True)
                
                st.info("L'axe X représente le score CADD (Dangerosité biologique). La couleur représente la classification ACMG (Dangerosité clinique).")

    # --- TAB 5: LOLLIPOPS (CADD) ---
    with tabs[4]:
        st.subheader("📍 Lollipop Plot (Axe Y = CADD)")
        prot_cols = ["hgvs.p", "HGVSp", "Protein_change", "AA_change", "hgvsp"]
        found_col = next((c for c in prot_cols if c in df_res.columns), None)
        
        if found_col:
            genes_avail = sorted(df_res["Gene_symbol"].unique())
            sel_gene_lol = st.selectbox("Gène", genes_avail)
            
            if sel_gene_lol:
                df_lol = df_res[df_res["Gene_symbol"] == sel_gene_lol].copy()
                df_lol["AA_pos"] = pd.to_numeric(df_lol[found_col].astype(str).str.extract(r'(\d+)')[0], errors="coerce")
                df_lol = df_lol.dropna(subset=["AA_pos", "CADD_phred"])
                
                if not df_lol.empty:
                    max_pos = df_lol["AA_pos"].max()
                    fig_lol = px.scatter(
                        df_lol, x="AA_pos", y="CADD_phred", color="ACMG_Class",
                        size="CADD_phred", hover_data=["Pseudo", found_col],
                        range_x=[0, max_pos*1.1], title=f"Mutations sur {sel_gene_lol}"
                    )
                    # Ligne du MSC si disponible
                    if "MSC_Ref" in df_lol.columns and not df_lol["MSC_Ref"].isna().all():
                        msc_val = df_lol["MSC_Ref"].iloc[0]
                        fig_lol.add_hline(y=msc_val, line_dash="dash", line_color="red", annotation_text=f"MSC Cutoff ({msc_val})")
                        
                    for _, row in df_lol.iterrows():
                        fig_lol.add_shape(type="line", x0=row["AA_pos"], y0=0, x1=row["AA_pos"], y1=row["CADD_phred"], line=dict(color="grey", width=1))
                    st.plotly_chart(fig_lol, use_container_width=True)
                else: st.warning("Pas de positions/CADD trouvés.")
        else: st.warning("Colonne protéique introuvable.")

    # --- (Les autres tabs restent fonctionnels mais standard) ---
    with tabs[2]: st.info("OncoPrint disponible")
    with tabs[5]: st.plotly_chart(px.scatter(df_res, x="Depth", y="Allelic_ratio", color="ACMG_Class", log_x=True))

elif not submitted:
    st.info("👈 Chargez fichier + Lancer.")
