import pandas as pd
import streamlit as st
from pathlib import Path


APP_DIR = Path(__file__).parent
EXAMPLE_CSV_PATH = APP_DIR / "data" / "clientes_1000.csv"

EXPECTED_COLUMNS = [
    "estado_cuenta",
    "duracion",
    "historial_credito",
    "proposito",
    "monto_credito",
    "cuenta_ahorro",
    "empleo",
    "tasa_cuota",
    "estado_personal",
    "otros_deudores",
    "residencia_actual",
    "propiedad",
    "edad",
    "otros_planes",
    "vivienda",
    "creditos_existentes",
    "trabajo",
    "personas_mantenimiento",
    "telefono",
    "trabajador_extranjero",
]

COLUMN_ALIASES = {
    "residencia": "residencia_actual",
    "personas_cargo": "personas_mantenimiento",
}

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
    return None, EXPECTED_COLUMNS


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
    df = df.rename(columns=COLUMN_ALIASES)

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


def score_credit_risk(df: pd.DataFrame) -> pd.Series:
    work = df.copy()
    for column in work.columns:
        if column in CATEGORICAL_VALUES:
            work[column] = work[column].astype(str).str.strip()
        else:
            work[column] = pd.to_numeric(work[column], errors="coerce")

    score = pd.Series(0.30, index=work.index, dtype="float64")
    score += work["estado_cuenta"].map({"A11": 0.24, "A12": 0.10, "A13": -0.04, "A14": -0.10}).fillna(0)
    score += work["historial_credito"].map({"A30": 0.18, "A31": 0.08, "A32": 0.00, "A33": 0.14, "A34": -0.06}).fillna(0)
    score += work["cuenta_ahorro"].map({"A61": 0.12, "A62": 0.06, "A63": 0.00, "A64": -0.08, "A65": 0.03}).fillna(0)
    score += work["empleo"].map({"A71": 0.12, "A72": 0.08, "A73": 0.03, "A74": -0.03, "A75": -0.06}).fillna(0)
    score += work["propiedad"].map({"A121": -0.06, "A122": -0.02, "A123": 0.03, "A124": 0.10}).fillna(0)
    score += work["otros_planes"].map({"A141": 0.06, "A142": 0.04, "A143": -0.03}).fillna(0)
    score += work["vivienda"].map({"A151": 0.05, "A152": -0.04, "A153": 0.02}).fillna(0)
    score += work["trabajo"].map({"A171": 0.08, "A172": 0.04, "A173": 0.00, "A174": -0.03}).fillna(0)
    score += work["telefono"].map({"A191": 0.03, "A192": -0.02}).fillna(0)
    score += work["trabajador_extranjero"].map({"A201": 0.02, "A202": -0.03}).fillna(0)

    score += ((work["duracion"] - 24).clip(lower=0) / 120).fillna(0)
    score += ((work["monto_credito"] - 3000).clip(lower=0) / 30000).fillna(0)
    score += ((work["tasa_cuota"] - 2).clip(lower=0) * 0.04).fillna(0)
    score += ((25 - work["edad"]).clip(lower=0) * 0.01).fillna(0)
    score += ((work["creditos_existentes"] - 1).clip(lower=0) * 0.03).fillna(0)
    score += ((work["personas_mantenimiento"] - 1).clip(lower=0) * 0.03).fillna(0)

    return score.clip(lower=0.02, upper=0.95)


def classify(df: pd.DataFrame, model, expected_columns: list[str]) -> pd.DataFrame:
    normalized_df = df.copy()
    normalized_df.columns = [str(col).strip() for col in normalized_df.columns]
    normalized_df = normalized_df.rename(columns=COLUMN_ALIASES)
    prepare_input(normalized_df, expected_columns)
    probabilities = score_credit_risk(normalized_df[expected_columns])
    predictions = (probabilities >= 0.50).astype(int)

    result = normalized_df.copy()
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


def render_bar_group(title: str, values: pd.Series, colors: dict[str, str]):
    st.subheader(title)
    total = int(values.sum())
    for label, value in values.items():
        percent = (int(value) / total) if total else 0
        color = colors.get(str(label), "#2563eb")
        st.markdown(
            f"""
            <div class="bar-row">
                <div class="bar-label">
                    <strong>{label}</strong>
                    <span>{int(value)} ({percent:.1%})</span>
                </div>
                <div class="bar-track">
                    <div class="bar-fill" style="width: {percent * 100:.1f}%; background: {color};"></div>
                </div>
            </div>
            """,
            unsafe_allow_html=True,
        )


def render_summary(result: pd.DataFrame):
    total = len(result)
    morosos = int((result["Prediccion"] == "MOROSO").sum())
    no_morosos = total - morosos
    risk_counts = result["Nivel de riesgo"].value_counts().reindex(["Bajo", "Medio", "Alto"], fill_value=0)
    class_counts = result["Prediccion"].value_counts().reindex(["NO MOROSO", "MOROSO"], fill_value=0)

    metric_cols = st.columns(4)
    metric_cols[0].metric("Total clientes", total)
    metric_cols[1].metric("No morosos", no_morosos)
    metric_cols[2].metric("Morosos", morosos)
    metric_cols[3].metric("% morosos", f"{morosos / total:.1%}" if total else "0.0%")

    chart_left, chart_right = st.columns([1, 1])
    with chart_left:
        render_bar_group(
            "Distribucion por riesgo",
            risk_counts,
            {"Bajo": "#16a34a", "Medio": "#d97706", "Alto": "#dc2626"},
        )
    with chart_right:
        render_bar_group(
            "Comparativa de decision",
            class_counts,
            {"NO MOROSO": "#0f766e", "MOROSO": "#dc2626"},
        )

    st.subheader("Resumen comparativo")
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


st.set_page_config(
    page_title="Clasificador de Riesgo Crediticio",
    page_icon=":credit_card:",
    layout="wide",
)
st.markdown(
    """
    <style>
    :root {
        --ink: #172033;
        --muted: #667085;
        --line: #d9e2ec;
        --panel: #ffffff;
        --accent: #2563eb;
    }
    .stApp {
        background: #eef3f8;
        color: var(--ink);
    }
    .block-container {
        max-width: 1180px;
        padding-top: 2rem;
        padding-bottom: 3rem;
    }
    h1, h2, h3, h4, h5, h6, p, label, span, div {
        color: var(--ink);
    }
    [data-testid="stMarkdownContainer"] p {
        color: var(--muted);
    }
    .hero {
        background: linear-gradient(135deg, #123a6f 0%, #0f766e 100%);
        border-radius: 8px;
        padding: 24px 28px;
        margin-bottom: 18px;
        box-shadow: 0 12px 30px rgba(15, 23, 42, 0.12);
    }
    .hero h1 {
        color: #ffffff !important;
        margin: 0 0 8px 0;
        font-size: 2.1rem;
        line-height: 1.15;
    }
    .hero p {
        color: #e6f0ff !important;
        margin: 0;
        max-width: 860px;
    }
    .quick-card {
        background: var(--panel);
        border: 1px solid var(--line);
        border-radius: 8px;
        padding: 14px 16px;
        min-height: 88px;
    }
    .quick-card strong {
        display: block;
        color: var(--ink);
        margin-bottom: 4px;
    }
    .quick-card span {
        color: var(--muted);
        font-size: 0.92rem;
    }
    [data-testid="stMetric"] {
        background: #ffffff;
        border: 1px solid var(--line);
        border-radius: 8px;
        padding: 14px 16px;
        box-shadow: 0 1px 3px rgba(15, 23, 42, 0.06);
    }
    [data-testid="stMetricLabel"] p,
    [data-testid="stMetricValue"] {
        color: var(--ink) !important;
    }
    div[data-testid="stDataFrame"] {
        border: 1px solid var(--line);
        border-radius: 8px;
        overflow: hidden;
    }
    .stTabs [data-baseweb="tab-list"] {
        gap: 6px;
        background: transparent;
    }
    .stTabs [data-baseweb="tab"] {
        background: #ffffff;
        border: 1px solid var(--line);
        border-radius: 8px 8px 0 0;
        padding: 10px 14px;
    }
    .stTabs [aria-selected="true"] {
        border-color: var(--accent);
        color: var(--accent) !important;
    }
    section[data-testid="stSidebar"] {
        background: #ffffff;
        border-right: 1px solid var(--line);
    }
    .risk-card {
        background: #ffffff;
        border: 1px solid var(--line);
        border-left: 6px solid var(--risk-color);
        border-radius: 8px;
        padding: 16px;
        margin: 8px 0 16px;
    }
    .risk-card strong { color: #1a202c; }
    .bar-row {
        background: #ffffff;
        border: 1px solid var(--line);
        border-radius: 8px;
        padding: 12px 14px;
        margin: 10px 0;
    }
    .bar-label {
        display: flex;
        justify-content: space-between;
        gap: 12px;
        color: var(--ink);
        font-size: 0.94rem;
        margin-bottom: 8px;
    }
    .bar-label span {
        color: var(--muted);
        white-space: nowrap;
    }
    .bar-track {
        height: 12px;
        background: #e6edf5;
        border-radius: 999px;
        overflow: hidden;
    }
    .bar-fill {
        height: 100%;
        border-radius: 999px;
        min-width: 3px;
    }
    </style>
    """,
    unsafe_allow_html=True,
)
st.markdown(
    """
    <div class="hero">
        <h1>Sistema de Evaluacion de Riesgo Crediticio</h1>
        <p>Clasificacion de clientes con variables German Credit y scoring de riesgo. Sube una cartera CSV, revisa metricas comparativas o simula un solicitante manualmente.</p>
    </div>
    """,
    unsafe_allow_html=True,
)

info_a, info_b, info_c = st.columns(3)
info_a.markdown(
    '<div class="quick-card"><strong>CSV de prueba</strong><span>Descarga clientes_1000.csv desde la barra lateral.</span></div>',
    unsafe_allow_html=True,
)
info_b.markdown(
    '<div class="quick-card"><strong>Entrada manual clara</strong><span>Los codigos A11, A73 o A201 muestran su significado.</span></div>',
    unsafe_allow_html=True,
)
info_c.markdown(
    '<div class="quick-card"><strong>Lectura ejecutiva</strong><span>Metricas, riesgo y comparativa visual en una sola vista.</span></div>',
    unsafe_allow_html=True,
)
st.write("")

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
            st.caption(f"{len(uploaded_df):,} filas cargadas. Se muestran las primeras 20 para revisar formato.")
            st.dataframe(uploaded_df.head(20), use_container_width=True)

            result_df = classify(uploaded_df, modelo, columnas)
            st.subheader("Resultados")
            st.caption("Vista previa de resultados. Descarga el CSV para revisar toda la cartera.")
            st.dataframe(result_df.head(50), use_container_width=True)
            st.download_button(
                "Descargar resultados completos",
                data=result_df.to_csv(index=False).encode("utf-8"),
                file_name="resultados_riesgo_crediticio.csv",
                mime="text/csv",
            )
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
