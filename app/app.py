import streamlit as st
import pandas as pd
import numpy as np
import joblib
import os
import sys
import pathlib
import matplotlib.pyplot as plt

# Configuration de la page
st.set_page_config(
    page_title="PediAppendix — Diagnostic Aide",
    page_icon="🩺",
    layout="wide",
    initial_sidebar_state="expanded",
)

# Ajout du chemin src
_ROOT = pathlib.Path(__file__).resolve().parent.parent
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

from src.train_model import load_model
from src.shap_explanations import get_shap_values

# --- CHARGEMENT DES RESSOURCES ---
@st.cache_resource
def load_resources():
    try:
        # Charger le meilleur modèle
        if os.path.exists('models/best_model_info.pkl'):
            info = joblib.load('models/best_model_info.pkl')
            model = load_model(info['path'])
        else:
            # Fallback
            model = load_model('models/Random_Forest.pkl')
            
        # Charger les données traitées pour les colonnes et médianes
        processed = joblib.load('data/processed/processed_data.joblib')
        feature_cols = processed['feature_cols']
        X_test = processed['X_test']
        X_train = processed['X_train']
        
        # Charger le préprocesseur
        preprocessor = joblib.load('models/preprocessor.pkl')
        
        return model, preprocessor, feature_cols, X_test, X_train
    except Exception as e:
        st.error(f"Erreur de chargement : {e}")
        return None, None, None, None, None

model, preprocessor, feature_cols, X_test, X_train = load_resources()

# --- SIDEBAR : PARAMÈTRES PATIENT ---
st.sidebar.title("🩺 Paramètres Patient")
st.sidebar.markdown("Saisissez les données cliniques pour l'évaluation.")

def get_user_input():
    with st.sidebar.form("patient_form"):
        st.subheader("📋 Démographie & Signes")
        age = st.number_input("Âge (années)", 0.0, 18.0, 10.0, step=0.1)
        temp = st.number_input("Température (°C)", 35.0, 42.0, 37.0, step=0.1)
        
        st.subheader("🔍 Examen Clinique")
        pain = 1 if st.selectbox("Douleur FID", ["Non", "Oui"]) == "Oui" else 0
        migratory = 1 if st.selectbox("Douleur Migrante", ["Non", "Oui"]) == "Oui" else 0
        rebound = 1 if st.selectbox("Défense à la décompression", ["Non", "Oui"]) == "Oui" else 0
        nausea = 1 if st.selectbox("Nausée/Vomissement", ["Non", "Oui"]) == "Oui" else 0
        
        st.subheader("🧪 Biologie & Échographie")
        wbc = st.number_input("Leucocytes (G/L)", 0.0, 50.0, 10.0)
        crp = st.number_input("CRP (mg/L)", 0.0, 300.0, 5.0)
        neutro = st.number_input("Neutrophiles (%)", 0.0, 100.0, 60.0)
        dia = st.number_input("Diamètre Appendice (mm)", 0.0, 30.0, 5.0)
        
        submit = st.form_submit_button("🩺 Lancer l'analyse")
        
    if submit:
        inputs = {
            'Age': age,
            'Body_Temperature': temp,
            'Lower_Right_Abd_Pain': pain,
            'Migratory_Pain': migratory,
            'Ipsilateral_Rebound_Tenderness': rebound,
            'Nausea': nausea,
            'WBC_Count': wbc,
            'CRP': crp,
            'Neutrophil_Percentage': neutro,
            'Appendix_Diameter': dia
        }
        return pd.DataFrame([inputs])[feature_cols], True
    return None, False

# --- MAIN PAGE ---
st.title("🏥 PediAppendix : Aide au Diagnostic")
st.markdown("""
Cette application assiste les praticiens dans l'évaluation du risque d'appendicite chez l'enfant. 
Elle utilise un modèle **Random Forest** entraîné sur 776 cas cliniques.
""")

if model is None:
    st.warning("⚠️ Modèle non trouvé. Veuillez lancer `python src/train_model.py` d'abord.")
else:
    input_df, submitted = get_user_input()
    
    if not submitted:
        st.info("👈 Modifiez les paramètres dans la barre latérale et cliquez sur **Lancer l'analyse**.")
        
        # Placeholder pour l'aspect visuel de la page vide
        st.write("---")
        st.caption("Prêt pour une nouvelle évaluation.")
    else:
        with st.spinner("Calcul des probabilités et analyse SHAP en cours..."):
            col1, col2 = st.columns([1, 1])
            
            # Appliquer le préprocesseur pour le modèle
            # On s'assure que l'ordre des colonnes correspond bien au feature_cols
            input_scaled_values = preprocessor.transform(input_df[feature_cols])
            input_scaled_df = pd.DataFrame(input_scaled_values, columns=feature_cols)

            with col1:
                st.subheader("📊 Résultat de l'analyse")
                proba = model.predict_proba(input_scaled_df)[0][0]
                
                # Affichage métrique
                color = "red" if proba >= 0.5 else "green"
                st.markdown(f"<h1 style='text-align: center; color: {color};'>{proba*100:.1f}%</h1>", unsafe_allow_html=True)
                st.markdown("<p style='text-align: center;'>Probabilité d'appendicite</p>", unsafe_allow_html=True)
                
                if proba >= 0.5:
                    st.error("⚠️ **Risque Élevé** : Une prise en charge chirurgicale est probablement nécessaire.")
                else:
                    st.success("✅ **Risque Faible** : Surveillance clinique recommandée.")
                    
                st.info(f"**Modèle utilisé** : {type(model).__name__} (AUC-ROC ~0.92)")

            with col2:
                st.subheader("💡 Explicabilité (SHAP)")
                try:
                    import shap
                    explainer = shap.TreeExplainer(model)
                    shap_values = explainer.shap_values(input_scaled_df)
                    
                    # Gestion multiclasse vs binaire (différent selon version shap)
                    if isinstance(shap_values, list): # RF sklearn
                        # On utilise l'index 0 car model.classes_ a 'appendicitis' en index 0
                        sv = shap_values[0]
                        bv = explainer.expected_value[0]
                    else: # CatBoost / LightGBM ou versions récentes
                        sv = shap_values
                        bv = explainer.expected_value
                    
                    fig, ax = plt.subplots()
                    shap.waterfall_plot(shap.Explanation(values=sv[0], base_values=bv, data=input_df.iloc[0], feature_names=feature_cols), show=False)
                    st.pyplot(fig)
                    st.caption("Le graphique waterfall montre la contribution de chaque paramètre à l'écart par rapport à la moyenne.")
                except Exception as e:
                    st.caption(f"Graphique SHAP indisponible en temps réel : {e}")

    st.divider()
    
    # --- ONGLETS TECHNIQUES ---
    tab1, tab2, tab3 = st.tabs(["📉 Étude de Performance", "📊 Analyse Globale", "📝 Documentation"])
    
    with tab1:
        st.subheader("Courbes de Performance")
        c1, c2 = st.columns(2)
        with c1:
            if os.path.exists("reports/figures/roc_Random_Forest.png"):
                st.image("reports/figures/roc_Random_Forest.png", caption="Courbe ROC")
        with c2:
             if os.path.exists("reports/figures/pr_Random_Forest.png"):
                st.image("reports/figures/pr_Random_Forest.png", caption="Courbe Precision-Recall")
                
    with tab2:
        st.subheader("Importance des variables (SHAP)")
        if os.path.exists("reports/figures/shap_summary.png"):
            st.image("reports/figures/shap_summary.png", use_column_width=True)
        else:
            st.write("Relancez l'entraînement pour générer les figures globales.")
            
    with tab3:
        st.markdown("""
        ### Choix du modèle
        Le **Random Forest** a été choisi pour son équilibre entre performance (AUC=0.9287) et interprétabilité via SHAP.
        
        ### Gestion du déséquilibre
        Le dataset présente un ratio 60/40. Nous avons utilisé un **split stratifié** et la pondération `class_weight='balanced'`.
        
        ### Optimisation Mémoire
        Une fonction `optimize_memory` a été implémentée dans le pipeline pour réduire l'usage RAM jusqu'à 75% sans perte d'information.
        """)

# Footer
st.sidebar.divider()
st.sidebar.caption("© 2026 PediAppendix Team | v2.0")
