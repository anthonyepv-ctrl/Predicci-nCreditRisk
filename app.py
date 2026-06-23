import streamlit as st
import pandas as pd
import numpy as np
import joblib

# Cargar modelo
modelo = joblib.load('modelo_crediticio.pkl')
columnas = joblib.load('columnas_modelo.pkl')

st.set_page_config(page_title="Clasificador de Riesgo Crediticio", page_icon="🏦")
st.title("🏦 Sistema de Evaluación de Riesgo Crediticio")
st.markdown("**Dataset:** German Credit | **Modelo:** Random Forest + SMOTE")

st.sidebar.header("📂 Opción 1: Subir archivo CSV")
archivo = st.sidebar.file_uploader("Sube tu archivo con datos del cliente", type=["csv"])

st.header("✏️ Opción 2: Ingresar datos manualmente")

col1, col2 = st.columns(2)
with col1:
    duracion = st.number_input("Duración del crédito (meses)", 1, 72, 12)
    monto = st.number_input("Monto del crédito", 100, 20000, 5000)
    edad = st.number_input("Edad del cliente", 18, 80, 35)
    tasa = st.slider("Tasa de cuota (%)", 1, 4, 2)
with col2:
    empleo = st.selectbox("Tiempo de empleo", ["A73","A74","A75","A71","A72"])
    historial = st.selectbox("Historial crediticio", ["A30","A31","A32","A33","A34"])
    proposito = st.selectbox("Propósito", ["A40","A41","A42","A43","A44","A45","A46","A48","A49","A410"])

if st.button("🔍 Clasificar Cliente", type="primary"):
    # Predicción manual (ejemplo simplificado)
    st.warning("⚠️ Para predicción completa, sube un CSV con todas las columnas del modelo.")

# Predicción por CSV
if archivo is not None:
    df_nuevo = pd.read_csv(archivo)
    st.subheader("Vista previa de los datos:")
    st.dataframe(df_nuevo.head())

    # Codificar igual que en el entrenamiento
    from sklearn.preprocessing import LabelEncoder
    le = LabelEncoder()
    for col in df_nuevo.select_dtypes(include='object').columns:
        df_nuevo[col] = le.fit_transform(df_nuevo[col].astype(str))

    # Asegurar columnas correctas
    df_nuevo = df_nuevo.reindex(columns=columnas, fill_value=0)

    predicciones = modelo.predict(df_nuevo)
    probabilidades = modelo.predict_proba(df_nuevo)[:, 1]

    df_nuevo['Predicción'] = ['🔴 MOROSO' if p == 1 else '🟢 NO MOROSO' for p in predicciones]
    df_nuevo['Probabilidad de Morosidad'] = [f"{p:.1%}" for p in probabilidades]

    st.subheader("📊 Resultados:")
    st.dataframe(df_nuevo[['Predicción', 'Probabilidad de Morosidad']])

    # Resumen
    total = len(predicciones)
    morosos = sum(predicciones)
    st.metric("Total clientes", total)
    col1, col2 = st.columns(2)
    col1.metric("🟢 No Morosos", total - morosos)
    col2.metric("🔴 Morosos", morosos)