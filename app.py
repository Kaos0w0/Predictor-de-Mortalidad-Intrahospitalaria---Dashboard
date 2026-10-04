import streamlit as st
import pandas as pd
import numpy as np
import joblib
import shap
import matplotlib.pyplot as plt
from lime.lime_tabular import LimeTabularExplainer
import warnings
import json
from pathlib import Path
import sys
import sklearn.compose._column_transformer as ct

# Parche temporal de compatibilidad para evitar el error de _RemainderColsList
if not hasattr(ct, "_RemainderColsList"):
    class _RemainderColsList(list):
        pass
    ct._RemainderColsList = _RemainderColsList

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
MODEL_PATHS = {
    "XGBoost": BASE_DIR / "artifacts_xgb" / "best_xgb_no_smote.joblib",
    "Gradient Boosting": BASE_DIR / "artifacts_gb" / "best_gb.joblib",
    "Random Forest": BASE_DIR / "artifacts_modelos" / "best_rf.joblib",
}
METADATA_PATH = BASE_DIR / "artifacts_modelos" / "metadata_experimento.json"
CALIBRATED_MODEL_DIR = BASE_DIR / "artifacts_calibrados"
METRICS_PATH = BASE_DIR / "comparacion_modelos_test.csv"
WEIGHT_METRIC = "youden"

IMPORTANCE_COVERAGE_TARGET = 0.75

NUMERIC_RANGES = {
    "1. ¿cuántas actividades realiza sólo?": (0.0, 23.0, 1.0),
    "puntaje cam": (0.0, 4.0, 1.0),
    "4. ¿cuántas actividades no realiza?": (0.0, 14.0, 1.0),
    "puntaje manual lawton y brody": (0.0, 8.0, 1.0),
    "puntaje frail calculado de manera manual": (0.0, 5.0, 1.0),
    "(mna) mini nutritional assessment": (0.0, 26.0, 1.0),
    "barthel actual calculado manual": (0.0, 100.0, 1.0),
    "mini-mental state examination (mmse)": (0.0, 30.0, 1.0),
}

FEATURE_GUIDANCE = {
    "1. ¿cuántas actividades realiza sólo?": "0-23 actividades. Más alto = mayor independencia funcional.",
    "4. ¿cuántas actividades no realiza?": "0-14 actividades. Más alto = mayor dependencia funcional.",
    "puntaje cam": "0-4 criterios CAM. 0 = ningún criterio registrado; no es un porcentaje ni una escala lineal de gravedad.",
    "puntaje manual lawton y brody": "0-8 puntos. Más alto = mayor independencia en actividades instrumentales.",
    "puntaje frail calculado de manera manual": "0-5 puntos. Más alto = mayor fragilidad.",
    "(mna) mini nutritional assessment": "0-26 puntos en este formulario. Más bajo = mayor riesgo nutricional; más alto = mejor estado nutricional.",
    "barthel actual calculado manual": "0-100 puntos. Más bajo = mayor dependencia; más alto = mayor independencia funcional.",
    "mini-mental state examination (mmse)": "0-30 puntos. Más bajo = menor desempeño cognitivo; más alto = mejor desempeño. Dejar sin informar si no se aplicó.",
}

NUMERIC_BINARY_FEATURES = {
    "diagnóstico geriátrico infeccioso (choice=neumonía)",
    "infección por covid-19 confirmado?",
}

TEXT_FEATURES = {
    "otros antecedentes",
    "diagnóstico otra enfermedad urológica o nefrológica",
}

DISPLAY_TO_MODEL_BINARY = {"No": 0, "Si": 1}


def get_feature_guidance(feature):
    if feature in FEATURE_GUIDANCE:
        return FEATURE_GUIDANCE[feature]
    if feature in NUMERIC_BINARY_FEATURES:
        return "Seleccione No o Si. Se codifica como 0/1; no representa una escala de severidad."
    if feature in TEXT_FEATURES:
        return "Seleccione una categoría conocida por el modelo o No informado; no tiene un rango numérico."
    return "Variable categórica. Seleccione la categoría registrada; no se interpreta como alto o bajo."


def to_string_df(data):
    return data.astype(str)


def to_string_df_for_calibration(data):
    return to_string_df(data)


setattr(sys.modules["__main__"], "to_string_df", to_string_df)
setattr(sys.modules["__main__"], "to_string_df_for_calibration", to_string_df_for_calibration)


@st.cache_data
def load_consensus_weights():
    if not METRICS_PATH.exists():
        raise FileNotFoundError(f"No se encontraron las métricas en: {METRICS_PATH}")

    metrics = pd.read_csv(METRICS_PATH)
    score_by_model = {}
    model_aliases = {
        "XGBoost": {"XGBoost", "XGBoost sin SMOTE"},
        "Gradient Boosting": {"Gradient Boosting"},
        "Random Forest": {"Random Forest"},
    }
    for model_name, aliases in model_aliases.items():
        matches = metrics[metrics["modelo"].isin(aliases)]
        if matches.empty or WEIGHT_METRIC not in matches.columns:
            raise ValueError(
                f"No se encontró {WEIGHT_METRIC} para el modelo {model_name}."
            )
        score_by_model[model_name] = max(float(matches.iloc[0][WEIGHT_METRIC]), 0.0)

    score_total = sum(score_by_model.values())
    if score_total <= 0:
        raise ValueError("La suma de las métricas para ponderar debe ser positiva.")
    weights = {
        model_name: score / score_total
        for model_name, score in score_by_model.items()
    }
    return weights, score_by_model


@st.cache_resource
def load_models_and_explainers():
    if not METADATA_PATH.exists():
        raise FileNotFoundError(
            f"No se encontraron los metadatos del modelo en: {METADATA_PATH}"
        )

    metadata = json.loads(METADATA_PATH.read_text(encoding="utf-8"))
    models = {}
    calibrated_models = {}
    explainers = {}
    expected_features = {}

    for model_name, model_path in MODEL_PATHS.items():
        if not model_path.exists():
            raise FileNotFoundError(
                f"No se encontró el modelo {model_name} en: {model_path}"
            )

        model = joblib.load(model_path)
        if not hasattr(model, "named_steps") or "prep" not in model.named_steps:
            raise ValueError(
                f"El artefacto {model_name} no contiene el pipeline 'prep'."
            )

        models[model_name] = model
        calibrated_path = CALIBRATED_MODEL_DIR / f"{model_name.lower().replace(' ', '_')}_platt.joblib"
        if not calibrated_path.exists():
            raise FileNotFoundError(
                "No se encontró el calibrador Platt para "
                f"{model_name}: {calibrated_path}. Ejecuta la celda de calibración de arboles.ipynb."
            )
        calibrated_models[model_name] = joblib.load(calibrated_path)
        expected_features[model_name] = metadata["features"] if model_name == "XGBoost" else list(
            model.named_steps["prep"].feature_names_in_
        )

        try:
            explainers[model_name] = shap.TreeExplainer(model.named_steps["model"])
        except Exception:
            explainers[model_name] = None

    return models, calibrated_models, explainers, expected_features


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


def get_original_feature_importances(model):
    preprocessor = model.named_steps["prep"]
    model_importances = np.asarray(model.named_steps["model"].feature_importances_)
    importances = {}

    numeric_features = list(preprocessor.transformers_[0][2])
    numeric_transformer = preprocessor.transformers_[0][1]
    numeric_output_features = list(
        numeric_transformer.get_feature_names_out(numeric_features)
    )
    numeric_slice = preprocessor.output_indices_["num"]
    for offset, feature in enumerate(numeric_output_features):
        importances[feature] = model_importances[numeric_slice.start + offset]

    categorical_features = list(preprocessor.transformers_[1][2])
    encoder = preprocessor.named_transformers_["cat"].named_steps["onehot"]
    categorical_slice = preprocessor.output_indices_["cat"]
    importance_index = categorical_slice.start
    for feature, categories in zip(categorical_features, encoder.categories_):
        category_count = len(categories)
        importances[feature] = model_importances[
            importance_index: importance_index + category_count
        ].sum()
        importance_index += category_count

    if categorical_slice.stop != len(model_importances):
        raise ValueError(
            "No se pudo mapear toda la importancia del modelo a sus variables originales."
        )

    total_importance = sum(importances.values())
    if total_importance <= 0:
        raise ValueError("El modelo no contiene importancias de variables válidas.")
    return pd.Series(importances, dtype=float) / total_importance


def select_consensus_features(models):
    model_importances = pd.concat(
        {
            model_name: get_original_feature_importances(model)
            for model_name, model in models.items()
        },
        axis=1,
    ).fillna(0.0)
    average_importance = model_importances.mean(axis=1).sort_values(ascending=False)
    cumulative_importance = average_importance.cumsum()
    cutoff = int(
        np.searchsorted(
            cumulative_importance.to_numpy(), IMPORTANCE_COVERAGE_TARGET,
        )
    )
    selected_features = average_importance.iloc[: cutoff + 1].index.tolist()
    achieved_coverage = float(cumulative_importance.iloc[cutoff])
    return selected_features, average_importance, achieved_coverage


@st.cache_resource
def load_lime_context():
    X_train = joblib.load(BASE_DIR / "artifacts_modelos" / "X_train.joblib")
    training_columns = []
    training_matrix = []
    categorical_features = []
    categorical_names = {}
    encoders = {}
    numeric_medians = {}

    for feature_index, feature in enumerate(selected_features):
        if feature in numeric_features:
            values = pd.to_numeric(X_train[feature], errors="coerce")
            median = float(values.median()) if values.notna().any() else 0.0
            numeric_medians[feature] = median
            training_columns.append(values.fillna(median).to_numpy(dtype=float))
        else:
            values = X_train[feature].fillna("__MISSING__").astype(str)
            top_values = values.value_counts().head(50).index.tolist()
            categories = [value for value in top_values if value != "__MISSING__"]
            categories = ["__MISSING__", *categories, "__OTHER__"]
            mapping = {value: index for index, value in enumerate(categories)}
            encoded_values = values.map(lambda value: mapping.get(value, mapping["__OTHER__"]))
            categorical_features.append(feature_index)
            categorical_names[feature_index] = categories
            encoders[feature] = mapping
            training_columns.append(encoded_values.to_numpy(dtype=float))

    training_matrix = np.column_stack(training_columns)
    explainer = LimeTabularExplainer(
        training_matrix,
        feature_names=selected_features,
        class_names=["No mortalidad", "Mortalidad"],
        categorical_features=categorical_features,
        categorical_names=categorical_names,
        discretize_continuous=True,
        random_state=42,
    )
    return explainer, encoders, numeric_medians, categorical_names


def decode_lime_rows(matrix, encoders, numeric_medians, categorical_names):
    decoded = pd.DataFrame(matrix, columns=selected_features)
    for feature_index, feature in enumerate(selected_features):
        if feature in numeric_features:
            min_value, max_value, _ = NUMERIC_RANGES.get(feature, (None, None, None))
            decoded[feature] = pd.to_numeric(decoded[feature], errors="coerce").fillna(
                numeric_medians[feature]
            )
            if min_value is not None:
                decoded[feature] = decoded[feature].clip(min_value, max_value)
        else:
            categories = categorical_names[feature_index]
            category_indices = np.rint(decoded[feature]).astype(int).clip(0, len(categories) - 1)
            decoded[feature] = [categories[index] for index in category_indices]
    return decoded


def encode_lime_instance(values, encoders, numeric_medians, categorical_names):
    encoded = []
    for feature_index, feature in enumerate(selected_features):
        value = values.get(feature, np.nan)
        if feature in numeric_features:
            numeric_value = pd.to_numeric(pd.Series([value]), errors="coerce").iloc[0]
            encoded.append(
                numeric_medians[feature] if pd.isna(numeric_value) else float(numeric_value)
            )
        else:
            category = "__MISSING__" if pd.isna(value) else str(value)
            categories = categorical_names[feature_index]
            encoded.append(encoders[feature].get(category, encoders[feature]["__OTHER__"]))
    return np.asarray(encoded, dtype=float)


def predict_consensus_for_lime(matrix, encoders, numeric_medians, categorical_names):
    decoded_rows = decode_lime_rows(matrix, encoders, numeric_medians, categorical_names)
    probabilities = []
    for _, values in decoded_rows.iterrows():
        model_probabilities = {}
        for model_name in models:
            model_input = pd.DataFrame([
                {
                    feature: values.get(feature, np.nan)
                    for feature in expected_features[model_name]
                }
            ])
            model_probabilities[model_name] = calibrated_models[model_name].predict_proba(
                model_input
            )[0, 1]
        probabilities.append(sum(
            consensus_weights[model_name] * probability
            for model_name, probability in model_probabilities.items()
        ))
    probabilities = np.asarray(probabilities, dtype=float)
    return np.column_stack([1.0 - probabilities, probabilities])


def render_local_shap(xgb_input):
    xgb_model = models["XGBoost"]
    preprocessor = xgb_model.named_steps["prep"]
    transformed_input = preprocessor.transform(xgb_input)
    if hasattr(transformed_input, "toarray"):
        transformed_input = transformed_input.toarray()
    else:
        transformed_input = np.asarray(transformed_input)
    if transformed_input.ndim != 2:
        raise ValueError(
            f"La matriz transformada para SHAP debe ser 2-D, no {transformed_input.shape}."
        )
    explainer = explainers.get("XGBoost") or shap.TreeExplainer(
        xgb_model.named_steps["model"]
    )
    shap_values = explainer.shap_values(transformed_input)
    if isinstance(shap_values, list):
        shap_values = shap_values[-1]
    expected_value = explainer.expected_value
    if isinstance(expected_value, (list, np.ndarray)):
        expected_value = expected_value[-1]
    feature_names = preprocessor.get_feature_names_out()
    explanation = shap.Explanation(
        values=np.asarray(shap_values)[0],
        base_values=float(expected_value),
        data=transformed_input[0],
        feature_names=feature_names,
    )
    figure = plt.figure(figsize=(10, 7))
    shap.plots.waterfall(explanation, max_display=15, show=False)
    st.pyplot(figure, clear_figure=True)
    plt.close(figure)


try:
    models, calibrated_models, explainers, expected_features = load_models_and_explainers()
    consensus_weights, consensus_scores = load_consensus_weights()
    model = models["XGBoost"]
    numeric_features, categorical_features, categories_by_feature = get_model_feature_groups(model)
    selected_features, consensus_importance, achieved_coverage = select_consensus_features(models)
    load_error = None
except (FileNotFoundError, ValueError, OSError) as error:
    models = calibrated_models = explainers = expected_features = None
    consensus_weights = consensus_scores = None
    model = None
    numeric_features = categorical_features = categories_by_feature = None
    selected_features = consensus_importance = achieved_coverage = None
    load_error = str(error)

st.title("🏥 Predictor de Mortalidad Intrahospitalaria")
st.markdown("### Consenso de modelos para pacientes octogenarios")

if load_error:
    st.error(f"No se pudieron cargar los modelos del consenso: {load_error}")
    st.stop()

# Módulo 4: Documentación en la barra lateral
with st.sidebar:
    st.header("Información del Modelo")
    st.info(f"""
    **Propósito:**
    Estimar el riesgo de mortalidad intrahospitalaria en pacientes ≥ 80 años utilizando datos clínicos y escalas geriátricas al ingreso.
    
    **Modelos del consenso:**
    - **XGBoost sin SMOTE**
    - **Gradient Boosting**
    - **Random Forest**
    - **Ponderación:** Índice de Youden normalizado
    - **Pesos:** {", ".join(f"{name} {weight * 100:.1f}%" for name, weight in consensus_weights.items())}
    - **Variables capturadas:** {len(selected_features)} variables
    - **Cobertura de importancia agregada:** {achieved_coverage * 100:.1f}%
    - **Calibración:** Platt Scaling con validación cruzada de 5 folds
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

    for index, feature in enumerate(selected_features):
        with columns[index % 3]:
            st.caption(get_feature_guidance(feature))
            if feature in NUMERIC_BINARY_FEATURES:
                selected = st.selectbox(
                    feature,
                    ["No", "Si"],
                    format_func=lambda value: value,
                    key=f"input_{index}",
                )
                input_values[feature] = DISPLAY_TO_MODEL_BINARY[selected]
            elif feature in numeric_features:
                min_value, max_value, step = NUMERIC_RANGES.get(
                    feature,
                    (0.0, 100.0, 1.0),
                )
                data_available = st.checkbox(
                    f"{feature}: dato disponible",
                    value=False,
                    help="Actívela únicamente si el valor numérico está disponible. Si queda desactivada, se usará NaN y el modelo imputará el dato.",
                    key=f"available_{index}",
                )
                numeric_value = st.number_input(
                    feature,
                    min_value=min_value,
                    max_value=max_value,
                    value=min(max(float(default_values.get(feature, min_value)), min_value), max_value),
                    step=step,
                    key=f"input_{index}",
                )
                input_values[feature] = numeric_value if data_available else np.nan
            elif feature in TEXT_FEATURES:
                options = [
                    "No informado",
                    *[
                        value
                        for value in categories_by_feature.get(feature, [])
                        if value.lower() != "nan"
                    ],
                ]
                selected = st.selectbox(
                    feature,
                    options,
                    key=f"input_{index}",
                    help="Busque y seleccione una categoría registrada por el modelo.",
                )
                input_values[feature] = np.nan if selected == "No informado" else selected
            else:
                options = [
                    "No informado",
                    *[
                        value
                        for value in categories_by_feature.get(feature, [])
                        if value.lower() != "nan"
                    ],
                ]
                if feature == "disfagia orofaringea":
                    options = ["No informado", "No", "Si", "No sabe"]
                selected = st.selectbox(feature, options, key=f"input_{index}")
                if feature == "disfagia orofaringea":
                    input_values[feature] = {
                        "No informado": np.nan,
                        "No": "0",
                        "Si": "si",
                        "No sabe": "No sabe",
                    }[selected]
                else:
                    input_values[feature] = np.nan if selected == "No informado" else selected

    submitted = st.form_submit_button("Analizar Riesgo y Explicar", type="primary")

if submitted:
    st.markdown("---")
    st.header("2. Resultados del Análisis")
    
    # Solo se capturan las variables prioritarias, el resto conserva la forma
    # original y queda en NaN para que el pipeline aplique sus imputadores
    # 1. PREDICCIÓN
    with st.spinner('Calculando riesgo y generando explicaciones clínicas...'):
        model_probabilities = {}
        model_inputs = {}
        for model_name, current_model in models.items():
            model_input = pd.DataFrame([
                {
                    feature: input_values.get(feature, np.nan)
                    for feature in expected_features[model_name]
                }
            ])
            model_inputs[model_name] = model_input
            model_probabilities[model_name] = calibrated_models[model_name].predict_proba(
                model_input
            )[0, 1]

        prob_mortality = float(sum(
            consensus_weights[model_name] * probability
            for model_name, probability in model_probabilities.items()
        ))
        probs_array = np.array(list(model_probabilities.values()), dtype=float)
        sigma = float(np.std(probs_array))
        
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
            
        st.subheader("Probabilidad estimada por modelo")
        probability_columns = st.columns(len(model_probabilities))
        for column, (model_name, probability) in zip(
            probability_columns, model_probabilities.items()
        ):
            with column:
                st.metric(
                    f"{model_name} ({consensus_weights[model_name] * 100:.1f}%)",
                    f"{probability * 100:.1f}%",
                )

        st.caption(
            "La probabilidad final es una combinación ponderada de las probabilidades "
            "calibradas mediante Platt Scaling; los pesos se normalizan desde el Índice de Youden."
        )

        if sigma < 0.05:
            st.success(
                f"**Consenso sólido (desviación: {sigma:.3f}):** "
                "los modelos convergen en un riesgo similar."
            )
        elif sigma >= 0.15:
            st.error(
                f"**Alerta de incertidumbre (desviación: {sigma:.3f}):** "
                "los modelos discrepan significativamente; se recomienda revisión clínica detallada."
            )
        else:
            st.info(
                f"**Acuerdo moderado (desviación: {sigma:.3f}):** "
                "existen variaciones entre los algoritmos."
            )

        with st.expander("Explicabilidad local: LIME + SHAP"):
            st.markdown(
                "LIME explica la probabilidad final del consenso ponderado. "
                "SHAP muestra la contribución detallada del XGBoost base en escala de log-odds."
            )
            try:
                lime_explainer, lime_encoders, lime_medians, lime_categories = load_lime_context()
                lime_instance = encode_lime_instance(
                    input_values,
                    lime_encoders,
                    lime_medians,
                    lime_categories,
                )
                lime_explanation = lime_explainer.explain_instance(
                    lime_instance,
                    lambda matrix: predict_consensus_for_lime(
                        matrix,
                        lime_encoders,
                        lime_medians,
                        lime_categories,
                    ),
                    num_features=min(10, len(selected_features)),
                    num_samples=1000,
                )
                st.markdown("**LIME: factores que explican el consenso final**")
                st.pyplot(lime_explanation.as_pyplot_figure(), clear_figure=True)

                st.markdown("**SHAP: contribución local del XGBoost base**")
                render_local_shap(model_inputs["XGBoost"])
            except Exception as error:
                st.warning(f"No fue posible generar la explicabilidad local: {error}")