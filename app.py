from pathlib import Path

import joblib
import pandas as pd
import streamlit as st


APP_DIR = Path(__file__).parent
MODEL_PATH = APP_DIR / "modelo_crediticio.pkl"
COLUMNS_PATH = APP_DIR / "columnas_modelo.pkl"
EXAMPLE_CSV_PATH = APP_DIR / "data" / "clientes_1000.csv"

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

CODE_LABELS = {
    "estado_cuenta": {
        "A11": "Cuenta corriente < 0 DM",
        "A12": "Cuenta corriente entre 0 y 200 DM",
        "A13": "Cuenta corriente >= 200 DM",
        "A14": "Sin cuenta corriente",
    },
    "historial_credito": {
        "A30": "Sin creditos o todos pagados",
        "A31": "Creditos en este banco pagados",
        "A32": "Creditos existentes pagados hasta ahora",
        "A33": "Retrasos en pagos anteriores",
        "A34": "Cuenta critica u otros creditos",
    },
    "proposito": {
        "A40": "Auto nuevo",
        "A41": "Auto usado",
        "A410": "Otros",
        "A42": "Muebles/equipamiento",
        "A43": "Radio/television",
        "A44": "Electrodomesticos",
        "A45": "Reparaciones",
        "A46": "Educacion",
        "A48": "Reentrenamiento",
        "A49": "Negocio",
    },
    "cuenta_ahorro": {
        "A61": "Ahorros < 100 DM",
        "A62": "Ahorros entre 100 y 500 DM",
        "A63": "Ahorros entre 500 y 1000 DM",
        "A64": "Ahorros >= 1000 DM",
        "A65": "Sin cuenta/desconocido",
    },
    "empleo": {
        "A71": "Desempleado",
        "A72": "Empleado menos de 1 anio",
        "A73": "Empleado entre 1 y 4 anios",
        "A74": "Empleado entre 4 y 7 anios",
        "A75": "Empleado 7 anios o mas",
    },
    "estado_personal": {
        "A91": "Hombre divorciado/separado",
        "A92": "Mujer divorciada/separada/casada",
        "A93": "Hombre soltero",
        "A94": "Hombre casado/viudo",
        "A95": "Mujer soltera",
    },
    "otros_deudores": {"A101": "Ninguno", "A102": "Codeudor", "A103": "Garante"},
    "propiedad": {
        "A121": "Bienes raices",
        "A122": "Seguro de vida/ahorro",
        "A123": "Auto u otros bienes",
        "A124": "Sin propiedad conocida",
    },
    "otros_planes": {"A141": "Banco", "A142": "Tiendas", "A143": "Ninguno"},
    "vivienda": {"A151": "Alquila", "A152": "Propia", "A153": "Gratis"},
    "trabajo": {
        "A171": "Desempleado/no residente",
        "A172": "No calificado residente",
        "A173": "Calificado/empleado oficial",
        "A174": "Gerente/autonomo/alta calificacion",
    },
    "telefono": {"A191": "Sin telefono", "A192": "Con telefono"},
    "trabajador_extranjero": {"A201": "Si", "A202": "No"},
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


def risk_color(level: str) -> str:
    return {"Bajo": "#1f9d55", "Medio": "#b7791f", "Alto": "#c53030"}.get(level, "#4a5568")


def render_code_guide():
    guide_rows = []
    for field, mapping in CODE_LABELS.items():
        for code, description in mapping.items():
            guide_rows.append({"Campo": field, "Codigo": code, "Significado": description})
    st.dataframe(pd.DataFrame(guide_rows), use_container_width=True, hide_index=True)


def render_summary(result: pd.DataFrame):
    total = len(result)
    morosos = int((result["Prediccion"] == "MOROSO").sum())
    no_morosos = total - morosos
    risk_counts = result["Nivel de riesgo"].value_counts().reindex(["Bajo", "Medio", "Alto"], fill_value=0)

    metric_cols = st.columns(4)
    metric_cols[0].metric("Total clientes", total)
    metric_cols[1].metric("No morosos", no_morosos)
    metric_cols[2].metric("Morosos", morosos)
    metric_cols[3].metric("% morosos", f"{morosos / total:.1%}" if total else "0.0%")

    st.subheader("Distribucion de riesgo")
    st.bar_chart(risk_counts)

    st.subheader("Comparativa")
    risk_table = pd.DataFrame(
        {
            "Nivel": risk_counts.index,
            "Clientes": risk_counts.values,
            "Porcentaje": [f"{value / total:.1%}" if total else "0.0%" for value in risk_counts.values],
        }
    )
    st.dataframe(risk_table, use_container_width=True, hide_index=True)


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
                labels = CODE_LABELS.get(column, {})
                display_options = [f"{option} - {labels.get(option, option)}" for option in options]
                selected = st.selectbox(
                    column,
                    display_options,
                    index=options.index(default) if default in options else 0,
                    help=FIELD_HELP.get(column),
                )
                values[column] = selected.split(" - ", 1)[0]
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
st.markdown(
    """
    <style>
    .stApp { background: #f7fafc; }
    [data-testid="stMetric"] {
        background: #ffffff;
        border: 1px solid #e2e8f0;
        border-radius: 8px;
        padding: 14px 16px;
        box-shadow: 0 1px 3px rgba(15, 23, 42, 0.06);
    }
    div[data-testid="stDataFrame"] {
        border: 1px solid #e2e8f0;
        border-radius: 8px;
        overflow: hidden;
    }
    .risk-card {
        background: #ffffff;
        border: 1px solid #e2e8f0;
        border-left: 6px solid var(--risk-color);
        border-radius: 8px;
        padding: 16px;
        margin: 8px 0 16px;
    }
    .risk-card strong { color: #1a202c; }
    </style>
    """,
    unsafe_allow_html=True,
)
st.title("Sistema de Evaluacion de Riesgo Crediticio")
st.caption("Dataset German Credit | Modelo Random Forest + SMOTE | Despliegue en Streamlit")

try:
    modelo, columnas = load_artifacts()
except Exception as exc:
    st.error(f"No se pudieron cargar los artefactos del modelo: {exc}")
    st.stop()

st.sidebar.header("Cargar CSV")
archivo = st.sidebar.file_uploader("Sube un archivo con las columnas del modelo", type=["csv"])
if EXAMPLE_CSV_PATH.exists():
    st.sidebar.download_button(
        "Descargar clientes_1000.csv",
        EXAMPLE_CSV_PATH.read_text(encoding="utf-8"),
        file_name="clientes_1000.csv",
        mime="text/csv",
    )

tab_csv, tab_manual, tab_codigos = st.tabs(
    ["Prediccion por CSV", "Prediccion manual", "Guia de codigos"]
)

with tab_csv:
    if archivo is None:
        st.info("Sube un CSV o descarga clientes_1000.csv desde la barra lateral.")
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
    with st.expander("Ver significado de los codigos"):
        render_code_guide()
    manual_df = manual_form(columnas)
    if st.button("Clasificar cliente", type="primary"):
        try:
            result_df = classify(manual_df, modelo, columnas)
            prediction = result_df.loc[0, "Prediccion"]
            probability = result_df.loc[0, "Probabilidad de morosidad"]
            risk = result_df.loc[0, "Nivel de riesgo"]
            st.markdown(
                f"""
                <div class="risk-card" style="--risk-color: {risk_color(risk)};">
                    <strong>Resultado:</strong> {prediction}<br>
                    <strong>Probabilidad de morosidad:</strong> {probability}<br>
                    <strong>Nivel de riesgo:</strong> {risk}
                </div>
                """,
                unsafe_allow_html=True,
            )
            st.subheader("Resultado")
            st.dataframe(
                result_df[["Prediccion", "Probabilidad de morosidad", "Nivel de riesgo"]],
                use_container_width=True,
            )
        except Exception as exc:
            st.error(f"No se pudo clasificar el cliente: {exc}")

with tab_codigos:
    st.subheader("Guia rapida de codigos German Credit")
    st.caption("Usa esta tabla para interpretar opciones como A11, A12, A73 o A201.")
    render_code_guide()
