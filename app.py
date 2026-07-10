from pathlib import Path

import joblib
import pandas as pd
import streamlit as st


APP_DIR = Path(__file__).parent
MODEL_PATH = APP_DIR / "modelo_crediticio.pkl"
COLUMNS_PATH = APP_DIR / "columnas_modelo.pkl"
TEMPLATE_PATH = APP_DIR / "data" / "plantilla_clientes.csv"

CATEGORICAL_VALUES = {
    "estado_cuenta": ["A11", "A12", "A13", "A14"],
    "historial_credito": ["A30", "A31", "A32", "A33", "A34"],
    "proposito": ["A40", "A41", "A410", "A42", "A43", "A44", "A45", "A46", "A48", "A49"],
    "cuenta_ahorro": ["A61", "A62", "A63", "A64", "A65"],
    "empleo": ["A71", "A72", "A73", "A74", "A75"],
    "estado_personal": ["A91", "A92", "A93", "A94", "A95"],
    "otros_deudores": ["A101", "A102", "A103"],
    "propiedad": ["A121", "A122", "A123", "A124"],
    "otros_planes": ["A141", "A142", "A143"],
    "vivienda": ["A151", "A152", "A153"],
    "trabajo": ["A171", "A172", "A173", "A174"],
    "telefono": ["A191", "A192"],
    "trabajador_extranjero": ["A201", "A202"],
}

FIELD_HELP = {
    "estado_cuenta": "Estado de la cuenta corriente.",
    "historial_credito": "Historial crediticio registrado.",
    "proposito": "Destino del credito solicitado.",
    "cuenta_ahorro": "Nivel de ahorro disponible.",
    "empleo": "Antiguedad laboral.",
    "estado_personal": "Estado personal y sexo segun codificacion German Credit.",
    "otros_deudores": "Garantias o deudores adicionales.",
    "propiedad": "Tipo de propiedad del solicitante.",
    "otros_planes": "Planes de pago externos.",
    "vivienda": "Tipo de vivienda.",
    "trabajo": "Nivel/tipo de ocupacion.",
    "telefono": "Disponibilidad de telefono.",
    "trabajador_extranjero": "Condicion de trabajador extranjero.",
}


@st.cache_resource
def load_artifacts():
    model = joblib.load(MODEL_PATH)
    columns = joblib.load(COLUMNS_PATH)
    return model, list(columns)


def label_encode_like_training(df: pd.DataFrame) -> pd.DataFrame:
    encoded = df.copy()
    for column, values in CATEGORICAL_VALUES.items():
        if column not in encoded.columns:
            continue

        mapping = {value: idx for idx, value in enumerate(sorted(set(values)))}
        normalized = encoded[column].astype(str).str.strip()
        unknown = sorted(set(normalized.dropna()) - set(mapping))
        if unknown:
            raise ValueError(
                f"La columna '{column}' contiene codigos no reconocidos: {', '.join(unknown)}"
            )
        encoded[column] = normalized.map(mapping).astype(int)

    return encoded


def prepare_input(df: pd.DataFrame, expected_columns: list[str]) -> pd.DataFrame:
    df = df.copy()
    df.columns = [str(col).strip() for col in df.columns]

    missing = [column for column in expected_columns if column not in df.columns]
    extra = [column for column in df.columns if column not in expected_columns]

    if missing:
        raise ValueError("Faltan columnas obligatorias: " + ", ".join(missing))

    if extra:
        st.info("Se ignoraran columnas adicionales: " + ", ".join(extra))

    df = df[expected_columns]
    df = label_encode_like_training(df)

    numeric_columns = [col for col in expected_columns if col not in CATEGORICAL_VALUES]
    for column in numeric_columns:
        df[column] = pd.to_numeric(df[column], errors="coerce")

    if df.isna().any().any():
        bad_columns = df.columns[df.isna().any()].tolist()
        raise ValueError("Hay valores vacios o invalidos en: " + ", ".join(bad_columns))

    return df


def classify(df: pd.DataFrame, model, expected_columns: list[str]) -> pd.DataFrame:
    prepared = prepare_input(df, expected_columns)
    predictions = model.predict(prepared)
    probabilities = model.predict_proba(prepared)[:, 1]

    result = df.copy()
    result["Prediccion"] = [
        "MOROSO" if prediction == 1 else "NO MOROSO" for prediction in predictions
    ]
    result["Probabilidad de morosidad"] = [f"{probability:.1%}" for probability in probabilities]
    result["Nivel de riesgo"] = [
        "Alto" if probability >= 0.65 else "Medio" if probability >= 0.35 else "Bajo"
        for probability in probabilities
    ]
    return result


def render_summary(result: pd.DataFrame):
    total = len(result)
    morosos = int((result["Prediccion"] == "MOROSO").sum())
    no_morosos = total - morosos

    metric_cols = st.columns(3)
    metric_cols[0].metric("Total clientes", total)
    metric_cols[1].metric("No morosos", no_morosos)
    metric_cols[2].metric("Morosos", morosos)


def manual_form(expected_columns: list[str]) -> pd.DataFrame:
    defaults = {
        "estado_cuenta": "A14",
        "duracion": 12,
        "historial_credito": "A32",
        "proposito": "A43",
        "monto_credito": 2500,
        "cuenta_ahorro": "A61",
        "empleo": "A73",
        "tasa_cuota": 2,
        "estado_personal": "A93",
        "otros_deudores": "A101",
        "residencia": 2,
        "propiedad": "A121",
        "edad": 35,
        "otros_planes": "A143",
        "vivienda": "A152",
        "creditos_existentes": 1,
        "trabajo": "A173",
        "personas_cargo": 1,
        "telefono": "A191",
        "trabajador_extranjero": "A201",
    }

    values = {}
    left, right = st.columns(2)
    for index, column in enumerate(expected_columns):
        container = left if index % 2 == 0 else right
        default = defaults.get(column)
        with container:
            if column in CATEGORICAL_VALUES:
                options = sorted(set(CATEGORICAL_VALUES[column]))
                values[column] = st.selectbox(
                    column,
                    options,
                    index=options.index(default) if default in options else 0,
                    help=FIELD_HELP.get(column),
                )
            else:
                min_value = 1 if column != "edad" else 18
                max_value = 100000 if column == "monto_credito" else 100
                values[column] = st.number_input(
                    column,
                    min_value=min_value,
                    max_value=max_value,
                    value=int(default or min_value),
                    step=1,
                )

    return pd.DataFrame([values], columns=expected_columns)


st.set_page_config(page_title="Clasificador de Riesgo Crediticio", page_icon=":credit_card:")
st.title("Sistema de Evaluacion de Riesgo Crediticio")
st.caption("Dataset German Credit | Modelo Random Forest + SMOTE")

try:
    modelo, columnas = load_artifacts()
except Exception as exc:
    st.error(f"No se pudieron cargar los artefactos del modelo: {exc}")
    st.stop()

st.sidebar.header("Cargar CSV")
archivo = st.sidebar.file_uploader("Sube un archivo con las columnas del modelo", type=["csv"])
if TEMPLATE_PATH.exists():
    st.sidebar.download_button(
        "Descargar plantilla CSV",
        TEMPLATE_PATH.read_text(encoding="utf-8"),
        file_name="plantilla_clientes.csv",
        mime="text/csv",
    )

tab_csv, tab_manual = st.tabs(["Prediccion por CSV", "Prediccion manual"])

with tab_csv:
    if archivo is None:
        st.info("Sube un CSV o descarga la plantilla desde la barra lateral.")
    else:
        try:
            uploaded_df = pd.read_csv(archivo)
            st.subheader("Vista previa")
            st.dataframe(uploaded_df.head(), use_container_width=True)

            result_df = classify(uploaded_df, modelo, columnas)
            st.subheader("Resultados")
            st.dataframe(result_df, use_container_width=True)
            render_summary(result_df)
        except Exception as exc:
            st.error(f"No se pudo procesar el CSV: {exc}")

with tab_manual:
    st.subheader("Datos del solicitante")
    manual_df = manual_form(columnas)
    if st.button("Clasificar cliente", type="primary"):
        try:
            result_df = classify(manual_df, modelo, columnas)
            st.subheader("Resultado")
            st.dataframe(
                result_df[["Prediccion", "Probabilidad de morosidad", "Nivel de riesgo"]],
                use_container_width=True,
            )
        except Exception as exc:
            st.error(f"No se pudo clasificar el cliente: {exc}")
