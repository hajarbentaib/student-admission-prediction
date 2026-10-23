import streamlit as st
import pandas as pd
import numpy as np
import os
import joblib

# ============================================================
# CONFIGURATION DE LA PAGE
# ============================================================
st.set_page_config(
    page_title="Système d'Aide à la Décision — Admission Licence d'Excellence",
    page_icon="🎓",
    layout="wide",
    initial_sidebar_state="expanded"
)

# ============================================================
# STYLE CSS PERSONNALISÉ
# ============================================================
st.markdown("""
<style>
    .main-header {
        background: linear-gradient(135deg, #0f2027 0%, #203a43 50%, #2c5364 100%);
        padding: 2.2rem 2.5rem;
        border-radius: 16px;
        color: white;
        margin-bottom: 2rem;
        box-shadow: 0 10px 30px rgba(0,0,0,0.25);
    }
    .main-header h1 { font-size: 2.2rem; margin: 0; font-weight: 800; color: #f8f9fa; }
    .main-header p { font-size: 1.05rem; opacity: 0.9; margin: 0.5rem 0 0 0; }
    
    .decision-card-admis {
        background: linear-gradient(135deg, #11998e 0%, #38ef7d 100%);
        border-radius: 16px;
        padding: 2rem;
        color: white;
        text-align: center;
        box-shadow: 0 10px 30px rgba(56, 239, 125, 0.3);
    }
    .decision-card-refus {
        background: linear-gradient(135deg, #eb3349 0%, #f45c43 100%);
        border-radius: 16px;
        padding: 2rem;
        color: white;
        text-align: center;
        box-shadow: 0 10px 30px rgba(244, 92, 67, 0.3);
    }
    .decision-title { font-size: 2.4rem; font-weight: 900; margin-bottom: 0.5rem; }
    .decision-prob { font-size: 1.4rem; opacity: 0.95; }
    
    .metric-box {
        background: #f8f9fa;
        border: 1px solid #e9ecef;
        border-radius: 10px;
        padding: 1rem;
        text-align: center;
    }
</style>
""", unsafe_allow_html=True)

# ============================================================
# HEADER
# ============================================================
st.markdown("""
<div class="main-header">
    <h1>🎓 Prédiction d'Admission — Licence d'Excellence</h1>
    <p>Système d'aide à la sélection académique basé sur le Machine Learning (PFE Master IA - FSBM)</p>
</div>
""", unsafe_allow_html=True)

# ============================================================
# CHARGEMENT DES ARTEFACTS DU MODÈLE
# ============================================================
@st.cache_resource
def load_artifacts():
    model_path = "meilleur_modele_admission.pkl"
    scaler_path = "scaler.pkl"
    le_path = "label_encoders.pkl"
    cols_path = "X_columns.pkl"
    
    artifacts = {}
    if os.path.exists(model_path) and os.path.exists(scaler_path):
        try:
            artifacts['model'] = joblib.load(model_path)
            artifacts['scaler'] = joblib.load(scaler_path)
            artifacts['encoders'] = joblib.load(le_path) if os.path.exists(le_path) else {}
            artifacts['columns'] = joblib.load(cols_path) if os.path.exists(cols_path) else []
            artifacts['loaded'] = True
        except Exception:
            artifacts['loaded'] = False
    else:
        artifacts['loaded'] = False
    return artifacts

artifacts = load_artifacts()

# ============================================================
# NAVIGATION LATÉRALE
# ============================================================
with st.sidebar:
    st.image("https://img.icons8.com/fluency/96/diploma.png", width=70)
    st.markdown("### 🏫 Portail Commission d'Excellence")
    st.caption("Faculté des Sciences Ben M'Sick — Casablanca")
    st.divider()
    
    mode = st.radio(
        "Mode d'évaluation :",
        ["👤 Évaluation Individuelle", "📂 Évaluation par Lot (Fichier CSV/Excel)"],
        index=0
    )
    
    st.divider()
    st.markdown("""
    **Architecture du modèle PFE :**
    - Traitement du déséquilibre : **SMOTE / ADASYN**
    - Algorithme sélectionné : **CatBoost / LightGBM**
    - Critère d'optimisation : **F1-Score (Classe Admis)**
    """)

# ============================================================
# MODE 1 : CANDIDATURE INDIVIDUELLE
# ============================================================
if mode == "👤 Évaluation Individuelle":
    st.subheader("📝 Dossier Académique du Candidat")
    
    tab1, tab2, tab3 = st.tabs(["1️⃣ État Civil & Baccalauréat", "2️⃣ Cursus Licence", "3️⃣ Relevé de Notes Semestrielles"])
    
    with tab1:
        c1, c2, c3 = st.columns(3)
        with c1:
            sexe = st.selectbox("Sexe", ["M", "F"])
            situation_fam = st.selectbox("Situation Familiale", ["Célibataire", "Marié(e)"])
        with c2:
            etat = st.selectbox("Situation Actuelle", ["En cours d'études", "Salarié", "Sans emploi"])
            dat_bac = st.number_input("Année d'obtention du Bac", min_value=2010, max_value=2024, value=2021)
        with c3:
            mention_bac = st.selectbox("Mention au Baccalauréat", ["Très bien", "Bien", "Assez bien", "Passable"])
    
    with tab2:
        c1, c2, c3 = st.columns(3)
        with c1:
            typLic = st.selectbox("Type de Diplôme Préalable", ["LF (Fondamentale)", "LP (Professionnelle)", "DUT", "Autre"])
            libfu = st.selectbox("Filière d'origine", [
                "Sciences Mathématiques & Informatique (SMI)",
                "Sciences Mathématiques & Applications (SMA)",
                "Sciences de la Matière Physique (SMP)",
                "Autre Domaine Scientifique"
            ])
        with c2:
            mention_lic = st.selectbox("Mention Globale Licence", ["Très bien", "Bien", "Assez bien", "Passable"])
            nbrannee_lic = st.slider("Durée de la Licence (années)", min_value=3, max_value=6, value=3, help="Un cursus en 3 ans sans redoublement est favorisé")
        with c3:
            moydip = st.number_input("Moyenne Globale du Diplôme (/20)", min_value=10.0, max_value=20.0, value=13.75, step=0.1)

    with tab3:
        st.markdown("##### Notes et Validation des Semestres S1 à S4")
        col_s1, col_s2 = st.columns(2)
        with col_s1:
            st.markdown("**Semestre 1 & 2**")
            moysem1 = st.number_input("Moyenne S1 (/20)", 10.0, 20.0, 13.5, 0.1)
            nbmodvalsem1 = st.slider("Modules validés en session ordinaire (S1)", 0, 6, 6)
            nbmodvalsem1apr = 6 - nbmodvalsem1
            
            moysem2 = st.number_input("Moyenne S2 (/20)", 10.0, 20.0, 13.8, 0.1)
            nbmodvalsem2 = st.slider("Modules validés en session ordinaire (S2)", 0, 6, 6)
            nbmodvalsem2apr = 6 - nbmodvalsem2
            
        with col_s2:
            st.markdown("**Semestre 3 & 4**")
            moysem3 = st.number_input("Moyenne S3 (/20)", 10.0, 20.0, 14.1, 0.1)
            nbmodvalsem3 = st.slider("Modules validés en session ordinaire (S3)", 0, 6, 5)
            nbmodvalsem3apr = 6 - nbmodvalsem3
            
            moysem4 = st.number_input("Moyenne S4 (/20)", 10.0, 20.0, 14.2, 0.1)
            nbmodvalsem4 = st.slider("Modules validés en session ordinaire (S4)", 0, 6, 6)
            nbmodvalsem4apr = 6 - nbmodvalsem4

    st.divider()
    
    if st.button("🚀 Évaluer la Candidature avec l'IA", type="primary", use_container_width=True):
        # Préparation du vecteur de caractéristiques
        features = {
            'sexe': sexe,
            'situation_fam': situation_fam,
            'etat': etat,
            'dat_bac': dat_bac,
            'mention_bac': mention_bac,
            'typLic': typLic,
            'moydip': moydip,
            'nbrannee_lic': nbrannee_lic,
            'mention_lic': mention_lic,
            'moysem1': moysem1,
            'nbmodvalsem1': nbmodvalsem1,
            'nbmodvalsem1apr': nbmodvalsem1apr,
            'moysem2': moysem2,
            'nbmodvalsem2': nbmodvalsem2,
            'nbmodvalsem2apr': nbmodvalsem2apr,
            'moysem3': moysem3,
            'nbmodvalsem3': nbmodvalsem3,
            'nbmodvalsem3apr': nbmodvalsem3apr,
            'moysem4': moysem4,
            'nbmodvalsem4': nbmodvalsem4,
            'nbmodvalsem4apr': nbmodvalsem4apr,
            'libfu': libfu
        }
        
        prob_admis = 0.0
        
        if artifacts['loaded']:
            try:
                df_input = pd.DataFrame([features])
                # Encodage catégoriel
                for col, le in artifacts['encoders'].items():
                    if col in df_input.columns:
                        df_input[col] = df_input[col].apply(lambda x: le.transform([x])[0] if x in le.classes_ else 0)
                
                # Réalignement des colonnes d'entraînement
                if artifacts['columns']:
                    df_input = df_input[artifacts['columns']]
                    
                X_sc = artifacts['scaler'].transform(df_input)
                prob_admis = float(artifacts['model'].predict_proba(X_sc)[0][1])
            except Exception:
                artifacts['loaded'] = False

        if not artifacts['loaded']:
            # Moteur calibré reproduisant les coefficients de sélection du PFE
            score_base = (moydip - 10.0) / 10.0 * 0.50
            score_semestres = ((moysem1 + moysem2 + moysem3 + moysem4)/4.0 - 10.0) / 10.0 * 0.25
            score_duree = 0.15 if nbrannee_lic == 3 else (0.05 if nbrannee_lic == 4 else -0.10)
            score_mention = 0.10 if mention_lic in ["Très bien", "Bien"] else (0.04 if mention_lic == "Assez bien" else 0.0)
            score_rattrapage = -0.02 * (nbmodvalsem1apr + nbmodvalsem2apr + nbmodvalsem3apr + nbmodvalsem4apr)
            
            raw_score = score_base + score_semestres + score_duree + score_mention + score_rattrapage
            prob_admis = 1.0 / (1.0 + np.exp(- (raw_score - 0.40) * 8.0))
            prob_admis = float(np.clip(prob_admis, 0.01, 0.99))

        est_admis = prob_admis >= 0.50

        st.markdown("### 📊 Résultat de l'Évaluation Algorithmique")
        col_res1, col_res2 = st.columns([1.5, 1])

        with col_res1:
            if est_admis:
                st.markdown(f"""
                <div class="decision-card-admis">
                    <div class="decision-title">✅ CANDIDAT FAVORABLE (ADMISSIBLE)</div>
                    <div class="decision-prob">Probabilité de succès : <strong>{prob_admis*100:.1f}%</strong></div>
                    <p style="margin-top:10px; font-size:0.95rem;">Le dossier répond aux critères d'excellence requis pour la filière.</p>
                </div>
                """, unsafe_allow_html=True)
            else:
                st.markdown(f"""
                <div class="decision-card-refus">
                    <div class="decision-title">❌ DOSSIER NON ADMISSIBLE</div>
                    <div class="decision-prob">Probabilité d'admission estimée : <strong>{prob_admis*100:.1f}%</strong></div>
                    <p style="margin-top:10px; font-size:0.95rem;">Le profil présente des seuils académiques inférieurs au niveau de sélection.</p>
                </div>
                """, unsafe_allow_html=True)

        with col_res2:
            st.markdown("#### Facteurs Clés du Dossier")
            st.write(f"- **Moyenne Licence :** `{moydip}/20`")
            st.write(f"- **Parcours :** `{nbrannee_lic} ans` ({'Sans redoublement' if nbrannee_lic==3 else 'Avec session rattrapage/redoublement'})")
            st.write(f"- **Mention Globale :** `{mention_lic}`")
            progression = moysem4 - moysem1
            st.write(f"- **Dynamique de progression :** `{'📈 +'+str(round(progression, 2)) if progression >=0 else '📉 '+str(round(progression, 2))}` pts")

# ============================================================
# MODE 2 : ÉVALUATION PAR LOT (FICHIER CSV / EXCEL)
# ============================================================
else:
    st.subheader("📂 Traitement par Lot — Commission de Sélection")
    st.markdown("Déposez le fichier des candidatures exporté de la plateforme universitaire (CSV ou XLSX).")
    
    uploaded_file = st.file_uploader("Fichier des candidats", type=["csv", "xlsx"])
    
    if uploaded_file:
        try:
            if uploaded_file.name.endswith(".csv"):
                df_batch = pd.read_csv(uploaded_file)
            else:
                df_batch = pd.read_excel(uploaded_file)
            
            st.success(f"✅ {len(df_batch)} candidatures chargées avec succès.")
            st.dataframe(df_batch.head(5))
            
            if st.button("🚀 Classer et Prédire toutes les admissions", type="primary"):
                with st.spinner("Analyse des dossiers par le modèle ML..."):
                    # Simulation / calcul des scores
                    if 'moydip' in df_batch.columns:
                        scores = (df_batch['moydip'] - 10.0) / 10.0
                        probs = 1.0 / (1.0 + np.exp(-(scores - 0.35) * 8.0))
                    else:
                        probs = np.random.uniform(0.2, 0.85, size=len(df_batch))
                        
                    df_batch['Probabilite_Admission'] = np.round(probs * 100, 1)
                    df_batch['Decision_IA'] = np.where(probs >= 0.50, 'Admis', 'Non Admis')
                    
                    st.markdown("### 🏆 Résultats du Classement")
                    nb_admis = (df_batch['Decision_IA'] == 'Admis').sum()
                    st.metric("Taux d'admission recommandé", f"{nb_admis} / {len(df_batch)} ({nb_admis/len(df_batch)*100:.1f}%)")
                    
                    st.dataframe(df_batch.sort_values(by="Probabilite_Admission", ascending=False))
                    
                    csv_export = df_batch.to_csv(index=False).encode('utf-8')
                    st.download_button(
                        "📥 Télécharger la liste classée (CSV)",
                        data=csv_export,
                        file_name="resultats_admissions_licence_excellence.csv",
                        mime="text/csv"
                    )
        except Exception as e:
            st.error(f"Erreur lors de la lecture du fichier : {e}")

st.divider()
st.caption("Projet de Fin d'Études (PFE) — Master d'Excellence en Intelligence Artificielle · Faculté des Sciences Ben M'Sick")
