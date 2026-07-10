# Revision de coherencia informe-repositorio

## Estado previo detectado

El informe indicaba que el repositorio incluia README completo, notebook, carpeta `docs/`, plantilla CSV y resultados. La version revisada del repositorio contenia solo:

- `app.py`
- `modelo_crediticio.pkl`
- `columnas_modelo.pkl`
- `README.md`
- `requirements.txt`

## Correcciones aplicadas

- Se completo la app de Streamlit con prediccion manual real.
- Se agrego validacion estricta de columnas para CSV.
- Se agrego plantilla compatible con el modelo.
- Se agrego el notebook de entrenamiento en `notebooks/AE_Proyecto.ipynb`.
- Se agrego el informe PDF en `docs/Proyecto_Aprendizaje_Estadistico.pdf`.
- Se agrego diccionario de datos.
- Se agrego archivo de metricas y brecha metodologica.
- Se actualizo la documentacion para reflejar el pipeline observado en Colab.

## Pendientes recomendados

- Guardar un pipeline completo de preprocesamiento + modelo.
- Agregar evidencia reproducible de Regresion Logistica y LogitBoost si el informe mantiene esa comparacion.
