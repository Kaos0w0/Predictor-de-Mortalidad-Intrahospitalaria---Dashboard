import streamlit as st
import pandas as pd
import numpy as np
import joblib
import shap
import matplotlib.pyplot as plt
import warnings
import json
from pathlib import Path

warnings.filterwarnings("ignore")

st.set_page_config(
    page_title="Predictor de Mortalidad - Octogenarios",
    page_icon="🏥",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Estilo personalizado básico
st.markdown("""
    <style>
    .main { background-color: #f8fafc; }
    h1, h2, h3 { color: #0f172a; }
    .stAlert { border-radius: 8px; }
    .metric-card { 
        background-color: white; 
        padding: 20px; 
        border-radius: 10px; 
        box-shadow: 0 4px 6px -1px rgba(0, 0, 0, 0.1);
        text-align: center;
    }
    </style>
""", unsafe_allow_html=True)

BASE_DIR = Path(__file__).resolve().parent
MODEL_PATH = BASE_DIR / "artifacts_xgb" / "best_xgb_no_smote.joblib"
METADATA_PATH = BASE_DIR / "artifacts_modelos" / "metadata_experimento.json"

TOP_INPUT_FEATURES = [
    "1. ¿cuántas actividades realiza sólo?",
    "disfagia orofaringea",
    "puntaje cam",
    "barthel basal - alimentación",
    "diagnóstico geriátrico infeccioso (choice=neumonía)",
    "frail - fatiga",
    "otros antecedentes",
    "4. ¿cuántas actividades no realiza?",
    "infección por covid-19 confirmado?",
    "puntaje manual lawton y brody",
    "diagnóstico otra enfermedad urológica o nefrológica",
    "diagnóstico principal agrupado",
    "puntaje frail calculado de manera manual",
    "cuidado de la casa",
    "preparación de la comida",
    "(mna) mini nutritional assessment",
]


@st.cache_resource
def load_model_and_explainer():
    #Carga el pipeline XGBoost entrenado sin SMOTE.
    if not MODEL_PATH.exists():
        raise FileNotFoundError(
            f"No se encontró el modelo XGBoost en: {MODEL_PATH}"
        )
    if not METADATA_PATH.exists():
        raise FileNotFoundError(
            f"No se encontraron los metadatos del modelo en: {METADATA_PATH}"
        )

    model = joblib.load(MODEL_PATH)
    metadata = json.loads(METADATA_PATH.read_text(encoding="utf-8"))
    expected_features = metadata["features"]

    if not hasattr(model, "named_steps") or "prep" not in model.named_steps:
        raise ValueError("El artefacto XGBoost no contiene el pipeline 'prep'.")

    explainer = None
    try:
        explainer = shap.TreeExplainer(model.named_steps["model"])
    except Exception:
        pass

    return model, explainer, expected_features


def get_model_feature_groups(model):
    preprocessor = model.named_steps["prep"]
    numeric_features = set(preprocessor.transformers_[0][2])
    categorical_features = list(preprocessor.transformers_[1][2])
    categories_by_feature = {}
    categorical_pipe = preprocessor.named_transformers_["cat"]
    encoder = categorical_pipe.named_steps["onehot"]
    for feature, categories in zip(categorical_features, encoder.categories_):
        categories_by_feature[feature] = [str(category) for category in categories]
    return numeric_features, categorical_features, categories_by_feature


try:
    model, explainer, expected_features = load_model_and_explainer()
    numeric_features, categorical_features, categories_by_feature = get_model_feature_groups(model)
    load_error = None
except (FileNotFoundError, ValueError, OSError) as error:
    model = explainer = expected_features = None
    numeric_features = categorical_features = categories_by_feature = None
    load_error = str(error)

st.title("🏥 Predictor de Mortalidad Intrahospitalaria")
st.markdown("### Modelo de Machine Learning para Pacientes Octogenarios (XGBoost sin SMOTE)")

if load_error:
    st.error(f"No se pudo cargar el modelo XGBoost: {load_error}")
    st.stop()

# Módulo 4: Documentación en la barra lateral
with st.sidebar:
    st.header("Información del Modelo")
    st.info("""
    **Propósito:**
    Estimar el riesgo de mortalidad intrahospitalaria en pacientes ≥ 80 años utilizando datos clínicos y escalas geriátricas al ingreso.
    
    **Métricas del Modelo (Test):**
    - **Algoritmo:** XGBoost sin SMOTE
    - **Variables capturadas:** 16 variables de mayor prioridad
    - **Variables restantes:** imputadas por el pipeline original
    
    **⚠️ Aviso Clínico:**
    Esta es una herramienta de apoyo a la decisión clínica. *No sustituye el juicio médico profesional.* Las predicciones deben interpretarse en conjunto con la valoración geriátrica integral.
    """)
    st.markdown("---")
    st.markdown("Desarrollado por: **Brayan Andres Sanchez Lozano**")

st.header("1. Ingreso de Datos Clínicos del Paciente")

with st.form("patient_data_form"):
    input_values = {}
    default_values = {
        "1. ¿cuántas actividades realiza sólo?": 4.0,
        "puntaje cam": 0.0,
        "4. ¿cuántas actividades no realiza?": 2.0,
        "puntaje manual lawton y brody": 4.0,
        "puntaje frail calculado de manera manual": 2.0,
        "(mna) mini nutritional assessment": 18.5,
    }
    columns = st.columns(3)

    for index, feature in enumerate(TOP_INPUT_FEATURES):
        with columns[index % 3]:
            if feature in numeric_features:
                input_values[feature] = st.number_input(
                    feature,
                    value=float(default_values.get(feature, 0.0)),
                    key=f"input_{index}",
                )
            else:
                options = ["No informado"] + categories_by_feature.get(feature, [])
                selected = st.selectbox(feature, options, key=f"input_{index}")
                input_values[feature] = np.nan if selected == "No informado" else selected

    submitted = st.form_submit_button("Analizar Riesgo y Explicar", type="primary")

if submitted:
    st.markdown("---")
    st.header("2. Resultados del Análisis")
    
    # Solo se capturan las variables prioritarias, el resto conserva la forma
    # original y queda en NaN para que el pipeline aplique sus imputadores
    input_data = pd.DataFrame(
        [{feature: input_values.get(feature, np.nan) for feature in expected_features}]
    )
    
    # 1. PREDICCIÓN
    with st.spinner('Calculando riesgo y generando explicaciones clínicas...'):
        prob_mortality = model.predict_proba(input_data)[0, 1]
        
        # Tarjeta de métrica principal
        col_res1, col_res2 = st.columns([1, 2])
        
        with col_res1:
            color = "#10b981" if prob_mortality < 0.15 else ("#f59e0b" if prob_mortality < 0.40 else "#ef4444")
            riesgo_texto = "Bajo" if prob_mortality < 0.15 else ("Moderado" if prob_mortality < 0.40 else "Alto")
            
            st.markdown(f"""
                <div class="metric-card" style="border-top: 5px solid {color};">
                    <h3 style="margin-bottom: 0;">Probabilidad de Mortalidad</h3>
                    <h1 style="color: {color}; font-size: 4rem; margin: 10px 0;">{prob_mortality*100:.1f}%</h1>
                    <h4 style="color: #64748b;">Riesgo {riesgo_texto}</h4>
                </div>
            """, unsafe_allow_html=True)
            
        # 2. EXPLICABILIDAD LOCAL (SHAP)
        # Temporalmente desactivada; se habilitará en una iteración posterior.