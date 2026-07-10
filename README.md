# Sistema de Clasificacion de Riesgo Crediticio

Aplicacion web en Streamlit para clasificar solicitantes de credito como `MOROSO` o `NO MOROSO` usando un modelo Random Forest entrenado con el dataset German Credit.

## Dataset

El entrenamiento usa German Credit de UCI, con 1000 registros, 20 variables predictoras y una variable objetivo binaria:

- `0`: no moroso / buen pagador.
- `1`: moroso / mal pagador.

Las variables conservan la codificacion original del dataset (`A11`, `A12`, etc.) para mantener trazabilidad con el informe del curso.

## Metodologia

El notebook de Colab revisado sigue este flujo:

1. Carga del dataset German Credit desde UCI.
2. Asignacion de nombres a las 20 variables predictoras y a `clase`.
3. Conversion de la clase original: `1 -> 0` y `2 -> 1`.
4. Codificacion de variables categoricas con `LabelEncoder`.
5. Division estratificada 70% entrenamiento y 30% prueba.
6. Aplicacion de SMOTE solo al conjunto de entrenamiento.
7. Entrenamiento de `RandomForestClassifier`.
8. Exportacion de `modelo_crediticio.pkl` y `columnas_modelo.pkl`.

## Estructura del repositorio

```text
.
|-- app.py
|-- modelo_crediticio.pkl
|-- columnas_modelo.pkl
|-- requirements.txt
|-- notebooks/
|   `-- AE_Proyecto.ipynb
|-- data/
|   `-- plantilla_clientes.csv
|-- docs/
|   |-- Proyecto_Aprendizaje_Estadistico.pdf
|   |-- diccionario_datos.md
|   `-- revision_coherencia.md
`-- results/
    `-- metricas_modelo.md
```

## Instalacion local

```bash
pip install -r requirements.txt
streamlit run app.py
```

## Uso

La aplicacion permite dos flujos:

- **Prediccion por CSV:** subir un archivo con las 20 columnas del modelo. Se puede descargar una plantilla desde la barra lateral.
- **Prediccion manual:** ingresar los valores de un solicitante y clasificarlo directamente desde la interfaz.

La app valida que no falten columnas obligatorias y rechaza codigos categoricos desconocidos para evitar predicciones con datos mal codificados.

## Resultados

El informe del proyecto reporta que Random Forest fue elegido por su mejor capacidad discriminativa:

- Accuracy: 81.19%.
- AUC: 0.895.
- Gini: 79.00%.
- Falsos negativos: 47.

Ver `results/metricas_modelo.md` para el resumen y las brechas detectadas frente al informe.

## Pendientes academicos recomendados

- Guardar un pipeline completo con codificadores + modelo para reproducibilidad total.
- Agregar evidencia reproducible de Regresion Logistica y LogitBoost si el informe mantiene la comparacion de tres modelos.
