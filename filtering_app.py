import os
import io
import re
import glob
import requests
import json
import tempfile
from datetime import datetime

# Imports scientifiques
import numpy as np
import pandas as pd
from scipy.stats import fisher_exact, hypergeom
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA

# Imports Streamlit & Visu
import streamlit as st
import plotly.express as px
import plotly.graph_objects as go
import matplotlib.pyplot as plt
import seaborn as sns
from st_aggrid import AgGrid, GridOptionsBuilder, JsCode
from fpdf import FPDF
from streamlit_agraph import agraph, Node, Edge, Config

# ---------------------------------------
# 1. CONFIGURATION
# ---------------------------------------
st.set_page_config(page_title="NGS ATLAS Explorer", layout="wide", page_icon="🧬")

MSC_LOCAL_FILENAME = "MSC_CI99_v1.7.txt"
HEADER_IMG_URL = "https://raw.githubusercontent.com/masilvestrini97-lang/Altas_UCD/refs/heads/main/images.jpeg"

def render_custom_header():
    st.markdown(f"""
    <style>
    .header-container {{
        background-image: linear-gradient(rgba(0, 0, 0, 0.6), rgba(0, 0, 0, 0.6)), url('{HEADER_IMG_URL}');
        background-size: cover; background-position: center 30%; padding: 50px 20px;
        border-radius: 15px; text-align: center; margin-bottom: 30px; box-shadow: 0 4px 10px rgba(0,0,0,0.2);
    }}
    .header-title {{ color: #8C005F; font-family: sans-serif; font-size: 48px; font-weight: 800; text-transform: uppercase; margin: 0; text-shadow: 2px 2px 4px #000; }}
    .header-subtitle {{ color: #f1f1f1; font-size: 18px; margin-top: 10px; text-shadow: 1px 1px 2px #000; }}
    </style>
    <div class="header-container">
        <div class="header-title">🧬 NGS ATLAS Explorer</div>
        <div class="header-subtitle">Castleman Team Analysis Tool</div>
    </div>
    """, unsafe_allow_html=True)

# ---------------------------------------
# 2. FONCTIONS UTILITAIRES & CHARGEMENT
# ---------------------------------------
def clean_text(val):
    return re.sub(r'[^A-Za-z0-9]+', '', str(val).strip().lower()) if val else ""

def extract_ref_alt_chr(df):
    if "Variant" in df.columns:
        extracted = df["Variant"].astype(str).str.extract(r'([ACGT]+)[>:/]([ACGT]+)$', flags=re.IGNORECASE)
        if "Ref" not in df.columns and not extracted[0].isna().all(): df["Ref"] = extracted[0].str.upper()
        if "Alt" not in df.columns and not extracted[1].isna().all(): df["Alt"] = extracted[1].str.upper()
        if "Chromosome" not in df.columns:
            df["Chromosome"] = df["Variant"].astype(str).str.split(r'[:_-]', n=1, expand=True)[0]
            df["Chromosome"] = df["Chromosome"].str.replace("chr", "", case=False).str.strip()
    return df

@st.cache_data
def load_variants(uploaded_file):
    if uploaded_file is None: return None
    sample = uploaded_file.read(4096).decode("utf-8", errors="ignore")
    uploaded_file.seek(0)
    sep = ";" if ";" in sample and "," not in sample else ("\t" if "\t" in sample else ",")
    try:
        df = pd.read_csv(uploaded_file, sep=sep, dtype=str, on_bad_lines='skip').replace('"', '', regex=True)
        df.columns = df.columns.str.strip() # Nettoyage titres colonnes
        if "Pseudo" in df.columns: df["Pseudo"] = df["Pseudo"].str.strip() # Nettoyage IDs patients
        return extract_ref_alt_chr(df)
    except Exception as e:
        st.error(f"Erreur lecture : {e}")
        return None

def make_varsome_link(variant_str):
    return f"https://varsome.com/variant/hg19/{str(variant_str).replace('>', ':')}" if variant_str else ""

# --- RAPPORT PDF ---
def create_pdf_report(patient_id, df_variants, user_comments=""):
    class PDF(FPDF):
        def header(self):
            self.set_font('Arial', 'B', 14)
            self.cell(0, 10, f'Rapport NGS - {patient_id}', 0, 1, 'C'); self.ln(2)
        def footer(self):
            self.set_y(-15); self.set_font('Arial', 'I', 8)
            self.cell(0, 10, f'Page {self.page_no()}', 0, 0, 'C')
    
    pdf = PDF(orientation='L', unit='mm', format='A4'); pdf.add_page()
    pdf.set_font("Arial", 'B', 10)
    pdf.cell(0, 6, f"Date: {datetime.now().strftime('%Y-%m-%d')} | Variants: {len(df_variants)}", 0, 1)
    if user_comments: pdf.set_font("Arial", 'I', 10); pdf.multi_cell(0, 6, f"Note: {user_comments}")
    pdf.ln(5)
    
    # Colonnes du rapport
    cols_config = [("Gene", "Gene_symbol", 25), ("Variant", "Variant", 45), ("ACMG", "ACMG_Class", 35),
                   ("CADD", "CADD_phred", 15), ("VAF", "Allelic_ratio", 15), ("gnomAD", "gnomad_exomes_NFE_AF", 25)]
    
    pdf.set_font("Arial", 'B', 8); pdf.set_fill_color(240, 240, 240)
    for l, _, w in cols_config: pdf.cell(w, 8, l, 1, 0, 'C', 1)
    pdf.ln()
    
    pdf.set_font("Arial", '', 7)
    for _, row in df_variants.iterrows():
        for _, col, w in cols_config:
            val = str(row.get(col, ""))
            if col in ["Allelic_ratio", "CADD_phred"]: 
                try: val = str(round(float(val), 2))
                except: pass
            pdf.cell(w, 8, val[:int(w/1.8)], 1, 0, 'C')
        pdf.ln()
    return pdf.output(dest='S').encode('latin-1', 'ignore')

# ---------------------------------------
# 3. FILTRAGE ET SCORING
# ---------------------------------------
@st.cache_data
def apply_filtering_and_scoring(df, vaf_min, gnomad_max, use_gnomad, min_dp, min_alt, max_freq, msc_content, genes_ex, pats_ex, min_cadd, v_eff, p_imp, clin_sig, sort_col, use_acmg, use_msc_strict, acmg_keep):
    logs = []; initial = len(df); df = df.copy()
    if "Gene_symbol" not in df.columns: return None, 0, 0, "No Gene_symbol column", []
    
    df["Gene_symbol"] = df["Gene_symbol"].str.upper().str.replace(" ", "")
    
    # Conversion numérique
    for c in ["gnomad_exomes_NFE_AF", "CADD_phred", "Allelic_ratio", "Depth"]:
        if c in df.columns: df[c] = pd.to_numeric(df[c], errors="coerce")
    
    # Calcul Fréquence Interne
    if "Pseudo" in df.columns and "Variant" in df.columns:
        df["Pseudo"] = df["Pseudo"].astype(str).str.strip()
        cts = df.groupby("Variant")["Pseudo"].nunique()
        df["internal_freq"] = df["Variant"].map(cts) / df["Pseudo"].nunique()
    else: df["internal_freq"] = 0.0

    # Filtres Numériques
    if "Depth" in df.columns: df = df[df["Depth"] >= min_dp]
    if "Allelic_ratio" in df.columns: df = df[df["Allelic_ratio"] >= vaf_min]
    if max_freq < 1.0: df = df[df["internal_freq"] <= max_freq]
    if use_gnomad and "gnomad_exomes_NFE_AF" in df.columns:
        df = df[(df["gnomad_exomes_NFE_AF"].isna()) | (df["gnomad_exomes_NFE_AF"] <= gnomad_max)]
    if min_cadd > 0 and "CADD_phred" in df.columns:
        df = df[(df["CADD_phred"].isna()) | (df["CADD_phred"] >= min_cadd)]

    # Filtres Listes
    if genes_ex: df = df[~df["Gene_symbol"].isin(genes_ex)]
    if pats_ex and "Pseudo" in df.columns: df = df[~df["Pseudo"].isin(pats_ex)]
    if v_eff and "Variant_effect" in df.columns: df = df[df["Variant_effect"].isin(v_eff)]
    if clin_sig and "Clinvar_significance" in df.columns: df = df[df["Clinvar_significance"].isin(clin_sig)]

    # MSC Scoring
    df["MSC_Ref"] = np.nan; df["MSC_Status"] = "N/A"
    msc_df = None
    if msc_content:
        try: msc_df = pd.read_csv(io.StringIO(msc_content), sep="\t", dtype=str)
        except: pass
    elif os.path.exists(MSC_LOCAL_FILENAME):
         try: msc_df = pd.read_csv(MSC_LOCAL_FILENAME, sep="\t", dtype=str)
         except: pass
         
    if msc_df is not None:
        try:
            msc_df.columns = [c.strip() for c in msc_df.columns]
            msc_clean = msc_df.groupby("Gene")["MSC"].max().reset_index()
            msc_clean["MSC"] = pd.to_numeric(msc_clean["MSC"], errors='coerce')
            df = df.merge(msc_clean, left_on="Gene_symbol", right_on="Gene", how="left")
            df["MSC_Ref"] = df["MSC"]
            if "CADD_phred" in df.columns:
                cond = (df["CADD_phred"].notna()) & (df["MSC_Ref"].notna())
                df.loc[cond & (df["CADD_phred"] < df["MSC_Ref"]), "MSC_Status"] = "Background"
                df.loc[cond & (df["CADD_phred"] >= df["MSC_Ref"]), "MSC_Status"] = "High Impact"
                if use_msc_strict: df = df[~(cond & (df["CADD_phred"] < df["MSC_Ref"]))]
        except: pass

    # ACMG Scoring
    if use_acmg:
        def get_acmg(row):
            score = 0
            eff = str(row.get("Variant_effect", "")).lower()
            if any(x in eff for x in ["stop", "frame", "splice"]): score += 4
            af = row.get("gnomad_exomes_NFE_AF", np.nan)
            if pd.isna(af) or af < 0.0001: score += 2
            cadd = row.get("CADD_phred", 0)
            if cadd >= 25: score += 1
            clin = str(row.get("Clinvar_significance", "")).lower()
            if "pathogenic" in clin and "conflict" not in clin: score += 2
            
            if score >= 5: return "Pathogenic", 4
            if score >= 3: return "Likely Pathogenic", 3
            if score >= 1: return "VUS", 2
            return "Likely Benign", 1
            
        res_acmg = df.apply(get_acmg, axis=1, result_type='expand')
        df["ACMG_Class"] = res_acmg[0]; df["ACMG_Rank"] = res_acmg[1]
    else:
        df["ACMG_Class"] = "Non calculé"; df["ACMG_Rank"] = 0

    if acmg_keep: df = df[df["ACMG_Class"].isin(acmg_keep)]
    
    # Tri
    if sort_col == "Classification ACMG (Priorité)": df = df.sort_values("ACMG_Rank", ascending=False)
    elif sort_col == "Score CADD (Décroissant)": 
        if "CADD_phred" in df.columns: df = df.sort_values("CADD_phred", ascending=False)
    
    if "Variant" in df.columns: df["link_varsome"] = df["Variant"].apply(make_varsome_link)
    
    return df, initial, len(df), None, logs

# ---------------------------------------
# 4. INTERFACE PRINCIPALE
# ---------------------------------------
render_custom_header()

if "analysis_done" not in st.session_state: 
    st.session_state["analysis_done"] = False
    st.session_state["df_res"] = None

with st.sidebar:
    st.header("1. Données")
    uploaded_file = st.file_uploader("Fichier Variants (CSV/TSV)", type=["csv", "tsv", "txt"])
    
    # Options dynamiques
    v_opts, p_opts, c_opts = [], [], []
    df_raw = None
    if uploaded_file:
        df_raw = load_variants(uploaded_file)
        if df_raw is not None:
             if "Variant_effect" in df_raw.columns: v_opts = sorted(df_raw["Variant_effect"].dropna().unique())
             if "Clinvar_significance" in df_raw.columns: c_opts = sorted(df_raw["Clinvar_significance"].dropna().unique())

    # Formulaire de filtres
    with st.form("params_form"):
        st.header("2. Filtres")
        sort_choice = st.selectbox("Tri", ["Classification ACMG (Priorité)", "Score CADD (Décroissant)"])
        
        c1, c2 = st.columns(2)
        min_dp = c1.number_input("Prof. Min (Depth)", 0, 1000, 50)
        vaf_min = c2.number_input("VAF Min (0-1)", 0.0, 1.0, 0.02)
        gnomad_max = st.number_input("gnomAD Max AF", 0.0, 1.0, 0.001, format="%.4f")
        min_cadd = st.number_input("CADD Phred Min", 0.0, 60.0, 0.0)
        
        acmg_keep = st.multiselect("ACMG Keep", ["Pathogenic", "Likely Pathogenic", "VUS"], default=["Pathogenic", "Likely Pathogenic", "VUS"])
        
        with st.expander("Filtres Avancés"):
            sel_var = st.multiselect("Effet Variant", v_opts)
            sel_clin = st.multiselect("ClinVar", c_opts)
            genes_ex = st.text_area("Exclure Gènes", "TTN, MUC16, KMT2C")
            pats_ex = st.text_area("Exclure Patients", "")
            
            msc_file = st.file_uploader("Fichier MSC (Optionnel)", type=["txt", "tsv"])
            use_msc_strict = st.checkbox("Exclure si CADD < MSC", False)

        submitted = st.form_submit_button("🚀 LANCER L'ANALYSE")

if submitted and df_raw is not None:
    g_ex_list = [x.strip().upper() for x in genes_ex.split(",") if x.strip()]
    p_ex_list = [x.strip() for x in pats_ex.split(",") if x.strip()]
    msc_txt = msc_file.read().decode("utf-8") if msc_file else None
    
    res, ini, fin, err, logs = apply_filtering_and_scoring(
        df_raw, vaf_min, gnomad_max, True, min_dp, 5, 1.0, msc_txt, 
        g_ex_list, p_ex_list, min_cadd, sel_var, [], sel_clin, sort_choice, True, use_msc_strict, acmg_keep
    )
    
    if err: st.error(err)
    else:
        st.session_state["analysis_done"] = True
        st.session_state["df_res"] = res
        st.session_state["kpi"] = (ini, fin)

# ---------------------------------------
# 5. RÉSULTATS ET ONGLETS
# ---------------------------------------
if st.session_state["analysis_done"]:
    df_res = st.session_state["df_res"]
    ini, fin = st.session_state["kpi"]
    
    # KPI
    k1, k2, k3 = st.columns(3)
    k1.metric("Total", ini); k2.metric("Retenus", fin); k3.metric("Filtre %", f"{round(fin/ini*100,1)}%")
    
    tabs = st.tabs(["📋 Tableau", "🧩 Corrélation", "🧬 Clonale", "🔥 Matrice", "🏙️ Manhattan", "📊 TMB", "🏥 Clinique"])

    # --- TAB 1: TABLEAU ---
    with tabs[0]:
        gb = GridOptionsBuilder.from_dataframe(df_res[["Pseudo", "Gene_symbol", "Variant", "ACMG_Class", "CADD_phred", "Allelic_ratio", "MSC_Status"]])
        gb.configure_selection('multiple', use_checkbox=True)
        gb.configure_pagination(paginationPageSize=20)
        grid = AgGrid(df_res, gridOptions=gb.build(), height=500, fit_columns_on_grid_load=False)
        
        sel_rows = pd.DataFrame(grid['selected_rows'])
        if not sel_rows.empty:
            st.write("---")
            if st.button("📄 Générer PDF Rapport"):
                pdf_val = create_pdf_report("Selection", sel_rows)
                st.download_button("📥 Télécharger PDF", pdf_val, "rapport.pdf", "application/pdf")

    # --- TAB 3: CORRELATION (Index 2) ---
    with tabs[2]:
        st.subheader("🧩 OncoPrint & Analyse Statistique")
        
        # Nettoyage préventif
        if "Pseudo" in df_res.columns: df_res["Pseudo"] = df_res["Pseudo"].astype(str)
        if "Gene_symbol" in df_res.columns: df_res["Gene_symbol"] = df_res["Gene_symbol"].astype(str)

        if "Pseudo" in df_res.columns and "Gene_symbol" in df_res.columns and not df_res.empty:
            
            # --- A. VISUALISATION HEATMAP ---
            top_genes_list = df_res["Gene_symbol"].value_counts().head(30).index.tolist()
            df_heat = df_res[df_res["Gene_symbol"].isin(top_genes_list)].copy()
            
            # Matrice simple pour affichage
            matrix_viz = df_heat.pivot_table(index="Pseudo", columns="Gene_symbol", aggfunc='size', fill_value=0)
            matrix_viz[matrix_viz > 0] = 1 
            
            if not matrix_viz.empty:
                co_occ = matrix_viz.T.dot(matrix_viz)
                st.plotly_chart(px.imshow(co_occ, text_auto=True, color_continuous_scale="Viridis", height=700), use_container_width=True)
            else:
                st.info("Pas assez de données pour la Heatmap.")

            # --- B. STATISTIQUES (FISHER) ---
            st.markdown("---")
            st.subheader("🧪 Tableau des Significativités (Fisher)")

            # 1. Préparation Matrice Complète
            matrix_bin = df_res.pivot_table(index="Pseudo", columns="Gene_symbol", aggfunc='size', fill_value=0)
            matrix_bin[matrix_bin > 0] = 1
            
            col_param1, col_param2 = st.columns(2)
            with col_param1:
                # On met 1 par défaut pour forcer des résultats
                min_patients = st.number_input("Min. patients par gène", 1, 100, 1, key="fisher_min_pat")
            with col_param2:
                # On met 1.0 par défaut pour TOUT voir, même le non-significatif
                p_val_cut = st.number_input("Seuil P-value (1.0 = tout)", 0.0, 1.0, 1.0, step=0.01, key="fisher_pval")

            # Filtrage des gènes
            genes_to_test = [g for g in matrix_bin.columns if matrix_bin[g].sum() >= min_patients]
            
            st.info(f"ℹ️ **Diagnostic :** {len(genes_to_test)} gènes qualifiés pour le test (sur {len(matrix_bin.columns)} total).")

            # Initialisation Session State pour garder le tableau affiché
            if "fisher_df" not in st.session_state:
                st.session_state["fisher_df"] = None

            # BOUTON D'ANALYSE
            if len(genes_to_test) >= 2:
                if st.button("▶️ LANCER LE CALCUL MAINTENANT", key="btn_fisher", type="primary"):
                    
                    # Limite de sécurité
                    if len(genes_to_test) > 60:
                        st.warning("Trop de gènes (>60). Analyse restreinte aux 60 plus fréquents pour éviter le crash.")
                        genes_to_test = matrix_bin[genes_to_test].sum().sort_values(ascending=False).head(60).index.tolist()

                    results = []
                    prog_bar = st.progress(0)
                    total_comb = (len(genes_to_test) * (len(genes_to_test) - 1)) // 2
                    curr = 0
                    
                    for i in range(len(genes_to_test)):
                        for j in range(i + 1, len(genes_to_test)):
                            g1, g2 = genes_to_test[i], genes_to_test[j]
                            
                            # Calcul Contingence
                            # Both (1,1) | G1 (1,0)
                            # G2   (0,1) | None (0,0)
                            both = ((matrix_bin[g1] == 1) & (matrix_bin[g2] == 1)).sum()
                            g1_only = ((matrix_bin[g1] == 1) & (matrix_bin[g2] == 0)).sum()
                            g2_only = ((matrix_bin[g1] == 0) & (matrix_bin[g2] == 1)).sum()
                            none = ((matrix_bin[g1] == 0) & (matrix_bin[g2] == 0)).sum()
                            
                            try:
                                odds, p_val = fisher_exact([[both, g1_only], [g2_only, none]])
                                
                                if p_val <= p_val_cut:
                                    rel_type = "Co-occurrence" if odds > 1 else "Exclusion"
                                    if both == 0: rel_type = "Exclusion"
                                    
                                    results.append({
                                        "Paire": f"{g1} - {g2}",
                                        "Type": rel_type,
                                        "Commun": int(both),
                                        "G1 seul": int(g1_only),
                                        "G2 seul": int(g2_only),
                                        "P-value": round(p_val, 5),
                                        "Odds Ratio": round(odds, 2)
                                    })
                            except: pass
                            
                            curr += 1
                            if total_comb > 0: prog_bar.progress(min(curr / total_comb, 1.0))
                    
                    prog_bar.empty()
                    
                    if results:
                        # On sauvegarde dans la session pour que ça reste affiché
                        st.session_state["fisher_df"] = pd.DataFrame(results).sort_values("P-value")
                    else:
                        st.session_state["fisher_df"] = pd.DataFrame() # Vide mais existe
                        st.warning("Aucun résultat trouvé avec ces critères.")

            # AFFICHAGE DU RÉSULTAT (En dehors du if button pour persister)
            if st.session_state["fisher_df"] is not None and not st.session_state["fisher_df"].empty:
                st.success(f"✅ {len(st.session_state['fisher_df'])} paires analysées.")
                
                # Fonction de style simple
                def color_p(val):
                    return 'background-color: #d4edda; color: black' if val < 0.05 else ''
                
                st.dataframe(
                    st.session_state["fisher_df"].style.applymap(color_p, subset=['P-value']),
                    use_container_width=True,
                    height=500
                )
                
                # Export CSV
                csv = st.session_state["fisher_df"].to_csv(index=False).encode('utf-8')
                st.download_button("📥 Télécharger CSV", csv, "stats_fisher.csv", "text/csv")
            
            elif st.session_state["fisher_df"] is not None and st.session_state["fisher_df"].empty:
                st.warning("Le calcul a été fait mais le tableau est vide (aucun résultat sous le seuil P-value).")

        else:
            st.error("Données insuffisantes pour l'onglet Corrélation.")
    # --- TAB 3: CLONALE ---
    with tabs[2]:
        st.subheader("🧬 Architecture Clonale")
        pat_list = sorted(df_res["Pseudo"].unique())
        sel_pat = st.selectbox("Patient", pat_list)
        d_clon = df_res[df_res["Pseudo"] == sel_pat].dropna(subset=["Allelic_ratio"])
        
        if len(d_clon) >= 3:
            try:
                km = KMeans(n_clusters=3, n_init=10).fit(d_clon[["Allelic_ratio"]])
                d_clon["Cluster"] = km.labels_.astype(str)
                st.plotly_chart(px.histogram(d_clon, x="Allelic_ratio", color="Cluster", nbins=40, title=f"VAF Distribution - {sel_pat}"))
            except: st.error("Erreur K-Means")
        else: st.warning("Pas assez de variants (<3) pour ce patient.")

    # --- TAB 4: MATRICE ---
    with tabs[3]:
        st.subheader("🔥 OncoPrint Simulé")
        top_g = df_res["Gene_symbol"].value_counts().head(25).index
        d_viz = df_res[df_res["Gene_symbol"].isin(top_g)]
        
        # Scoring sévérité pour couleur
        d_viz["Score"] = d_viz["ACMG_Class"].map({"Pathogenic":3, "Likely Pathogenic":2, "VUS":1}).fillna(0)
        mat_viz = d_viz.pivot_table(index="Gene_symbol", columns="Pseudo", values="Score", aggfunc='max', fill_value=0)
        
        st.plotly_chart(px.imshow(mat_viz, color_continuous_scale="Reds", height=700))

    # --- TAB 5: MANHATTAN ---
    with tabs[4]:
        df_man = df_res.dropna(subset=["CADD_phred", "Chromosome"])
        if not df_man.empty:
            st.plotly_chart(px.scatter(df_man, x="Chromosome", y="CADD_phred", color="ACMG_Class", hover_name="Gene_symbol"))

    # --- TAB 6: TMB ---
    with tabs[5]:
        panel_size = st.number_input("Taille Panel (Mb)", 0.1, 100.0, 38.0)
        tmb = df_res.groupby("Pseudo")["Variant"].count() / panel_size
        st.plotly_chart(px.bar(tmb, title="TMB (Mutations / Mb)"))

    # --- TAB 7: CLINIQUE ---
    with tabs[6]:
        st.subheader("🏥 PCA Clinique & Génomique")
        tech_cols = ["Pseudo", "Gene_symbol", "Variant", "CADD_phred", "Allelic_ratio", "ACMG_Class", "ACMG_Rank", "Depth"]
        clin_cols = [c for c in df_res.columns if c not in tech_cols]
        
        if clin_cols:
            sel_clin = st.multiselect("Variables Cliniques", clin_cols, default=clin_cols[:2] if len(clin_cols)>2 else clin_cols)
            
            if st.button("Lancer PCA Intégrée") and sel_clin:
                # Alignement Strict
                all_pats = sorted(df_res["Pseudo"].unique())
                
                # 1. Genomique (Top 30 gènes)
                top30 = df_res["Gene_symbol"].value_counts().head(30).index
                df_g = df_res[df_res["Gene_symbol"].isin(top30)]
                mat_g = df_g.pivot_table(index="Pseudo", columns="Gene_symbol", aggfunc='size', fill_value=0).reindex(all_pats, fill_value=0)
                mat_g[mat_g>0]=1 # Binaire
                
                # 2. Clinique
                df_c = df_res.groupby("Pseudo")[sel_clin].first().reindex(all_pats)
                # Encodage (One-Hot pour texte, 0 pour NaN chiffres)
                df_num = df_c.select_dtypes(include=np.number).fillna(0)
                df_str = df_c.select_dtypes(exclude=np.number).fillna("NA")
                df_c_enc = pd.concat([df_num, pd.get_dummies(df_str, drop_first=True)], axis=1)
                
                # 3. Fusion & Clean Variance Nulle
                full = pd.concat([mat_g, df_c_enc], axis=1)
                full = full.loc[:, (full != full.iloc[0]).any()] # Suppr colonnes constantes
                
                if full.shape[1] > 1:
                    X = StandardScaler().fit_transform(full)
                    coords = PCA(n_components=2).fit_transform(X)
                    
                    df_pca = pd.DataFrame(coords, columns=["PC1", "PC2"], index=all_pats)
                    # Cluster rapide pour couleur
                    df_pca["Cluster"] = KMeans(n_clusters=3, n_init=10).fit_predict(X).astype(str)
                    
                    st.plotly_chart(px.scatter(df_pca, x="PC1", y="PC2", color="Cluster", hover_name=df_pca.index, title="PCA Mixte"))
                else: st.error("Pas assez de variance pour PCA.")
        else: st.warning("Pas de colonnes cliniques trouvées.")
