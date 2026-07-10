# Metricas del modelo

## Fuente

Las metricas deben mantenerse alineadas con el notebook de entrenamiento `AE_Proyecto.ipynb` y con el informe del curso.

## Pipeline confirmado desde el notebook

- Dataset: German Credit de UCI.
- Registros: 1000.
- Variables predictoras: 20.
- Variable objetivo: `clase`, convertida a `0` para no moroso y `1` para moroso.
- Division: 70% entrenamiento y 30% prueba, estratificada.
- Balanceo: SMOTE aplicado solo al conjunto de entrenamiento.
- Modelo exportado: `RandomForestClassifier`.
- Artefactos: `modelo_crediticio.pkl` y `columnas_modelo.pkl`.

## Resultados reportados en el informe

| Modelo | Accuracy | F1 clase bad | AUC | Gini | FN |
| --- | ---: | ---: | ---: | ---: | ---: |
| Regresion Logistica | 80.95% | 0.812 | 0.863 | 72.60% | 42 |
| Random Forest | 81.19% | 0.810 | 0.895 | 79.00% | 47 |
| LogitBoost | 80.00% | 0.801 | 0.871 | 74.20% | 46 |

## Brecha detectada

El informe compara tres modelos, pero el repositorio y el notebook visible exportan solo Random Forest. Si se mantiene esa comparacion en el informe, deben agregarse los notebooks, capturas o scripts que reproduzcan Regresion Logistica y LogitBoost. Si no se agregan, el informe debe aclarar que el despliegue final usa unicamente Random Forest.
