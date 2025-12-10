import os
import io
import re
import glob
import requests  # Pour l'API STRING
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
from streamlit_agraph import agraph, Node, Edge, Config  # Pour le graphe PPI

# ---------------------------------------
# 1. CONFIGURATION & OUTILS
# ---------------------------------------

st.set_page_config(page_title="NGS ATLAS Explorer v10.2", layout="wide", page_icon="🧬")

# --- Fonctions utilitaires ---

def clean_text(val):
    if not isinstance(val, str): return ""
    return re.sub(r'[^A-Za-z0-9]+', '', val.strip().lower())

def extract_ref_alt_chr(df):
    """Tente de créer/nettoyer les colonnes Chromosome, Ref, Alt si elles sont absentes."""
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
    except Exception: return []
    return []

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
    pdf.cell(0, 6, f"Date: {datetime.now().strftime('%Y-%m-%d')} | Variants sélectionnés: {len(df_variants)}", 0, 1)
    if user_comments:
        pdf.set_font("Arial", 'I', 10)
        pdf.multi_cell(0, 6, f"Note clinique : {user_comments}")
    pdf.ln(5)

    prot_candidates = ["hgvs.p", "HGVSp", "Protein_change", "AA_change", "hgvsp", "p."]
    found_prot = next((c for c in prot_candidates if c in df_variants.columns), None)

    if found_prot:
        columns_config = [
            ("Gene", "Gene_symbol", 25), ("Variant", "Variant", 45), ("Protein", found_prot, 35),
            ("Effect", "Variant_effect", 45), ("VAF", "Allelic_ratio", 17), ("Dp", "Depth", 15),
            ("gnomAD", "gnomad_exomes_NFE_AF", 25), ("ACMG", "ACMG_Class", 35)
        ]
    else:
        columns_config = [
            ("Gene", "Gene_symbol", 30), ("Variant", "Variant", 55), ("Effect", "Variant_effect", 50),
            ("VAF", "Allelic_ratio", 20), ("Depth", "Depth", 20), ("gnomAD", "gnomad_exomes_NFE_AF", 30),
            ("ACMG", "ACMG_Class", 40)
        ]

    pdf.set_font("Arial", 'B', 8)
    pdf.set_fill_color(240, 240, 240)
    for label, _, width in columns_config: pdf.cell(width, 8, label, 1, 0, 'C', 1)
    pdf.ln()
    
    pdf.set_font("Arial", '', 7)
    for _, row in df_variants.iterrows():
        for label, col_name, width in columns_config:
            raw_val = row.get(col_name, "")
            display_val = str(raw_val)
            if col_name == "Allelic_ratio":
                try: display_val = str(round(float(raw_val), 2))
                except: pass
            elif col_name == "gnomad_exomes_NFE_AF":
                try:
                    val_float = float(raw_val)
                    if val_float == 0: display_val = "0"
                    elif val_float < 0.0001: display_val = "<1e-4"
                    else: display_val = f"{val_float:.4f}"
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
        df["Alt_depth_total"] = df["Alt_depth"].astype(str).str.split(',').str[0].str.split(' ').str[0]
        df["Alt_depth_total"] = pd.to_numeric(df["Alt_depth_total"], errors='coerce').fillna(0)

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

    # --- MODIFICATION LOGIQUE CADD ---
    # On garde si (CADD est vide OU CADD >= min)
    if min_cadd is not None and min_cadd > 0 and "CADD_phred" in df.columns:
        df = df[(df["CADD_phred"].isna()) | (df["CADD_phred"] >= min_cadd)]
    logs.append({"Etape": "9. CADD", "Restants": len(df), "Perdus": last_count - len(df)}); last_count = len(df)

    if "Variant" in df.columns:
        df["link_varsome"] = df["Variant"].apply(make_varsome_link)
        df["link_gnomad"] = df["Variant"].apply(make_gnomad_link)
    else:
        df["link_varsome"] = ""; df["link_gnomad"] = ""

    df["msc_weight"] = 0.0
    if use_msc and constraint_file_content:
        try:
            c_df = pd.read_csv(io.StringIO(constraint_file_content), sep="\t", dtype=str)
            if "gene" in c_df.columns:
                c_df["gene"] = c_df["gene"].str.upper().str.replace(" ", "")
                z_col = "mis_z" if "mis_z" in c_df.columns else ("mis.z_score" if "mis.z_score" in c_df.columns else None)
                if z_col:
                    c_df[z_col] = pd.to_numeric(c_df[z_col], errors="coerce")
                    c_df = c_df.groupby("gene", as_index=False)[z_col].max()
                    df = df.merge(c_df[["gene", z_col]], left_on="Gene_symbol", right_on="gene", how="left")
                    df["msc_weight"] = df[z_col].fillna(0) * 0.5
        except: pass

    def get_imp(x):
        s = str(x).lower()
        if "high" in s: return 3
        if "moderate" in s or "modifier" in s: return 2
        if "low" in s: return 1
        return 0
    df["score_putative"] = df["Putative_impact"].apply(get_imp) if "Putative_impact" in df.columns else 0
    
    df["score_cadd"] = 0
    if "CADD_phred" in df.columns:
        df["score_cadd"] = np.select([(df["CADD_phred"] >= 30), (df["CADD_phred"] >= 20)], [3, 2], default=0)

    df["score_clinvar"] = 0
    if "Clinvar_significance" in df.columns:
        cv = df["Clinvar_significance"].astype(str).str.lower()
        df["score_clinvar"] = np.select([cv.str.contains("pathogenic") & ~cv.str.contains("likely"), cv.str.contains("uncertain")], [5, 2], default=0)

    df["patho_score"] = df["score_putative"] + df["score_cadd"] + df["score_clinvar"] + df["msc_weight"]

    if use_acmg:
        def compute_acmg_class(row):
            eff = str(row.get("Variant_effect", "")).lower()
            pvs1 = any(x in eff for x in ["stopgained", "frameshift", "splice_acceptor", "splice_donor", "startlost", "stoplost"])
            af = row.get("gnomad_exomes_NFE_AF", np.nan)
            pm2 = pd.isna(af) or (af < 0.0001)
            pp3 = row.get("CADD_phred", 0) >= 25
            clv = str(row.get("Clinvar_significance", "")).lower()
            pp5 = "pathogenic" in clv and "conflict" not in clv
            ba1 = (af > 0.05) if not pd.isna(af) else False
            if ba1: return "Benign", "BA1"
            score = 0
            criteria_met = []
            if pvs1: score += 4; criteria_met.append("PVS1")
            if pm2: score += 2; criteria_met.append("PM2")
            if pp5: score += 2; criteria_met.append("PP5")
            if pp3: score += 1; criteria_met.append("PP3")
            final_class = "VUS"
            if score >= 5: final_class = "Pathogenic"
            elif score >= 3: final_class = "Likely Pathogenic"
            return final_class, ", ".join(criteria_met)
        acmg_res = df.apply(compute_acmg_class, axis=1, result_type='expand')
        df["ACMG_Class"], df["ACMG_Criteria"] = acmg_res[0], acmg_res[1]
    else:
        df["ACMG_Class"], df["ACMG_Criteria"] = "Non calculé", ""

    if use_patho_filter: df = df[df["patho_score"] >= min_patho_score]
    logs.append({"Etape": "10. Score Final", "Restants": len(df), "Perdus": last_count - len(df)})

    if sort_by_column == "Score Pathogénique (Décroissant)": df = df.sort_values("patho_score", ascending=False)
    elif sort_by_column == "Classification ACMG": df = df.sort_values("ACMG_Class", ascending=True)
    elif sort_by_column == "Fréquence Cohorte (Décroissant)": df = df.sort_values("internal_freq", ascending=False)
    elif sort_by_column == "Patient (A-Z)" and "Pseudo" in df.columns: df = df.sort_values("Pseudo", ascending=True)

    return df, initial_total, len(df), None, logs

# --- Reste du code inchangé (Analyse Pathways, UI, Tabs...) ---
# (Note: Les fonctions compute_enrichment, load_local_pathways et le bloc Main Streamlit sont identiques à ton original)

def load_local_pathways(directory="."):
    pathways = {}
    gmt_files = glob.glob(os.path.join(directory, "*.gmt"))
    for fpath in gmt_files:
        try:
            with open(fpath, "r", encoding="utf-8", errors="ignore") as f:
                for line in f:
                    parts = line.strip().split("\t")
                    if len(parts) >= 3:
                        pw, genes = parts[0].strip(), [g.strip().upper() for g in parts[2:] if g.strip()]
                        if pw and genes:
                            if pw not in pathways: pathways[pw] = []
                            pathways[pw].extend(genes)
        except: pass
    for k in pathways: pathways[k] = list(set(pathways[k]))
    return pathways, [os.path.basename(f) for f in gmt_files]

@st.cache_data
def compute_enrichment(df, pathway_genes):
    universe = sorted({g for gl in pathway_genes.values() for g in gl})
    N, mutated = len(universe), sorted(set(df["Gene_symbol"].unique()) & set(universe))
    n = len(mutated)
    if N == 0 or n == 0: return pd.DataFrame()
    rows = []
    for pw, genes in pathway_genes.items():
        M, overlap = len(genes), set(mutated) & set(genes)
        k = len(overlap)
        if k > 0:
            pval = hypergeom.sf(k - 1, N, M, n)
            pats = ", ".join(sorted(df[df["Gene_symbol"].isin(overlap)]["Pseudo"].unique())) if "Pseudo" in df.columns else "N/A"
            rows.append({"pathway": pw, "p_value": pval, "k_overlap": k, "genes": ",".join(overlap), "patients": pats})
    df_res = pd.DataFrame(rows)
    if not df_res.empty:
        df_res = df_res.sort_values("p_value")
        m = len(df_res)
        df_res["rank"] = np.arange(1, m + 1)
        df_res["FDR"] = np.clip(np.minimum.accumulate(((df_res["p_value"] * m) / df_res["rank"])[::-1])[::-1], 0, 1)
        df_res["minus_log10_FDR"] = -np.log10(df_res["FDR"] + 1e-300)
    return df_res

# ---------------------------------------
# INTERFACE UTILISATEUR (STREAMLIT)
# ---------------------------------------

st.title("🧬 NGS ATLAS Explorer v10.2")
st.markdown("---")

if "analysis_done" not in st.session_state:
    st.session_state.update({"analysis_done": False, "df_res": None, "kpis": (0, 0), "logs": []})

with st.sidebar:
    st.header("1. Données")
    uploaded_file = st.file_uploader("Fichier Variants", type=["csv", "tsv", "txt"])
    df_raw, v_opts, p_opts, c_opts = None, [], [], []
    if uploaded_file:
        df_raw = load_variants(uploaded_file)
        if df_raw is not None:
            if "Variant_effect" in df_raw.columns: v_opts = sorted(df_raw["Variant_effect"].fillna("Non Renseigné").unique())
            if "Putative_impact" in df_raw.columns: p_opts = sorted(df_raw["Putative_impact"].fillna("Non Renseigné").unique())
            if "Clinvar_significance" in df_raw.columns: c_opts = sorted(df_raw["Clinvar_significance"].fillna("Non Renseigné").unique())

    with st.form("params"):
        st.header("2. Paramètres")
        sort_choice = st.selectbox("Tri initial", ["Score Pathogénique (Décroissant)", "Classification ACMG", "Fréquence Cohorte (Décroissant)", "Patient (A-Z)"])
        c1, c2 = st.columns(2)
        min_dp = c1.number_input("Depth Min", 0, 10000, 50)
        allelic_min = c1.number_input("VAF Min", 0.0, 1.0, 0.02)
        min_ad = c2.number_input("Alt Depth Min", 0, 1000, 5)
        max_cohort_freq = c2.slider("Max Freq Cohorte", 0.0, 1.0, 1.0, 0.05)
        c3, c4 = st.columns(2)
        min_patho = c3.number_input("Score Patho Min", 0.0, 50.0, 4.0)
        gnomad_max = c3.number_input("gnomAD Max", 0.0, 1.0, 0.001, format="%.4f")
        min_cadd_val = c4.number_input("CADD Min (0=all)", 0.0, 60.0, 0.0)
        use_acmg = c4.checkbox("Activer ACMG (Auto)", value=False)
        with st.expander("Avancé"):
            sel_var = st.multiselect("Effet", v_opts, default=v_opts)
            sel_put = st.multiselect("Impact", p_opts, default=p_opts)
            sel_clin = st.multiselect("ClinVar", c_opts, default=c_opts)
            genes_ex = st.text_area("Exclure Gènes", "KMT2C, CHEK2, TTN, MUC16")
            pseudo_ex = st.text_area("Exclure Patients", "")
            use_gnomad, use_patho, use_msc = st.checkbox("Filtre gnomAD", True), st.checkbox("Filtre Patho", True), st.checkbox("MSC", False)
            msc_file = st.file_uploader("Fichier MSC", type=["tsv"])
        submitted = st.form_submit_button("🚀 LANCER L'ANALYSE")

if submitted and df_raw is not None:
    g_list, p_list = [x.strip().upper() for x in genes_ex.split(",") if x.strip()], [x.strip() for x in pseudo_ex.split(",") if x.strip()]
    msc_c = msc_file.read().decode("utf-8") if (use_msc and msc_file) else None
    res, ini, fin, err, logs = apply_filtering_and_scoring(df_raw, allelic_min, gnomad_max, use_gnomad, min_patho, use_patho, min_dp, min_ad, max_cohort_freq, use_msc, msc_c, g_list, p_list, min_cadd_val, sel_var, sel_put, sel_clin, sort_choice, use_acmg)
    if err: st.error(err)
    else:
        st.session_state.update({"analysis_done": True, "df_res": res, "kpis": (ini, fin), "logs": logs, "use_acmg": use_acmg})
        user_pathways, file_names = load_local_pathways()
        st.session_state["df_enr"] = compute_enrichment(res, user_pathways) if user_pathways else pd.DataFrame()
        st.session_state["gmt_files"] = file_names

if st.session_state["analysis_done"]:
    df_res, n_ini, n_fin = st.session_state["df_res"], st.session_state["kpis"][0], st.session_state["kpis"][1]
    df_enr, gmt_names = st.session_state.get("df_enr", pd.DataFrame()), st.session_state.get("gmt_files", [])
    
    k1, k2, k3 = st.columns(3)
    k1.metric("Initial", n_ini); k2.metric("Final", n_fin); k3.metric("Ratio", f"{round(n_fin/n_ini*100, 2) if n_ini > 0 else 0}%")
    
    tabs = st.tabs(["📋 Tableau", "🔍 Inspecteur", "🧩 OncoPrint", "📊 Spectre", "📍 Lollipops", "📈 QC", "🧬 Pathways", "🕸️ PPI"])
    
    with tabs[0]:
        st.subheader("📋 Explorateur")
        if "link_varsome" in df_res.columns: df_res["Varsome_HTML"] = df_res["link_varsome"].apply(lambda x: f'<a href="{x}" target="_blank">🔗</a>' if x else "")
        df_display = df_res[["Pseudo", "Gene_symbol", "Variant", "Varsome_HTML", "patho_score", "ACMG_Class", "Allelic_ratio"] + [c for c in df_res.columns if c not in ["Pseudo", "Gene_symbol", "Variant", "Varsome_HTML", "patho_score", "ACMG_Class", "Allelic_ratio", "link_varsome", "link_gnomad"]]]
        gb = GridOptionsBuilder.from_dataframe(df_display)
        gb.configure_pagination(paginationPageSize=20); gb.configure_selection('multiple', use_checkbox=True)
        gb.configure_column("patho_score", cellStyle=JsCode("function(params) { return params.value >= 10 ? {'backgroundColor': '#d9534f', 'color': 'white'} : (params.value >= 5 ? {'backgroundColor': '#f0ad4e'} : null); }"))
        gb.configure_column("Varsome_HTML", cellRenderer="html")
        grid_res = AgGrid(df_display, gridOptions=gb.build(), allow_unsafe_jscode=True, height=500, theme='streamlit')
        sel_df = pd.DataFrame(grid_res['selected_rows']) if grid_res['selected_rows'] is not None else df_res
        
        st.subheader("📄 Rapport PDF")
        comment = st.text_area("Note clinique :", "Compatible phénotype...")
        if not sel_df.empty:
            pdf_b = create_pdf_report(sel_df["Pseudo"].unique()[0] if len(sel_df["Pseudo"].unique())==1 else "Multi", sel_df, comment)
            st.download_button("📥 PDF", pdf_b, "NGS_Report.pdf", "application/pdf")

    with tabs[2]: # OncoPrint simplified
        if "Pseudo" in df_res.columns:
            top_genes = df_res["Gene_symbol"].value_counts().head(30).index
            df_h = df_res[df_res["Gene_symbol"].isin(top_genes)]
            matrix = df_h.pivot_table(index="Gene_symbol", columns="Pseudo", values="patho_score", aggfunc='max', fill_value=0)
            st.plotly_chart(px.imshow(matrix, color_continuous_scale="Viridis", title="Heatmap Top Genes"), use_container_width=True)

    with tabs[6]: # Pathways
        if not df_enr.empty: st.plotly_chart(px.bar(df_enr.head(15), x="minus_log10_FDR", y="pathway", orientation='h', title="Top Pathways"), use_container_width=True)
        else: st.info("GMT non trouvé.")

elif not submitted:
    st.info("👈 Chargez un fichier et lancez l'analyse.")
