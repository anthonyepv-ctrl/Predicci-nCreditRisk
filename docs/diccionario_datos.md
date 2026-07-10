# Diccionario de datos

El proyecto usa el dataset German Credit de UCI. La variable objetivo original se transforma asi:

- `0`: no moroso / buen pagador.
- `1`: moroso / mal pagador.

## Variables predictoras

| Columna | Tipo | Descripcion |
| --- | --- | --- |
| `estado_cuenta` | Categorica | Estado de la cuenta corriente del solicitante. |
| `duracion` | Numerica | Duracion del credito en meses. |
| `historial_credito` | Categorica | Historial crediticio previo. |
| `proposito` | Categorica | Proposito del credito solicitado. |
| `monto_credito` | Numerica | Monto total del credito. |
| `cuenta_ahorro` | Categorica | Nivel de ahorro disponible. |
| `empleo` | Categorica | Antiguedad laboral. |
| `tasa_cuota` | Numerica | Tasa de cuota respecto al ingreso disponible. |
| `estado_personal` | Categorica | Estado personal y sexo segun codificacion German Credit. |
| `otros_deudores` | Categorica | Existencia de garante o codeudor. |
| `residencia` | Numerica | Tiempo de residencia. |
| `propiedad` | Categorica | Tipo de propiedad del solicitante. |
| `edad` | Numerica | Edad del solicitante. |
| `otros_planes` | Categorica | Otros planes de pago existentes. |
| `vivienda` | Categorica | Tipo de vivienda. |
| `creditos_existentes` | Numerica | Numero de creditos existentes. |
| `trabajo` | Categorica | Tipo o nivel de ocupacion. |
| `personas_cargo` | Numerica | Personas a cargo del solicitante. |
| `telefono` | Categorica | Disponibilidad de telefono. |
| `trabajador_extranjero` | Categorica | Condicion de trabajador extranjero. |

## Codificacion

Las variables categoricas conservan los codigos originales `Axx` del dataset German Credit. En la app se convierten a numeros siguiendo el orden alfabetico usado por `LabelEncoder` durante el entrenamiento.

## Consideracion importante

Para una reproducibilidad mas fuerte, lo ideal es guardar un pipeline completo con los codificadores y el modelo en un solo artefacto. La version actual replica la codificacion esperada a partir de los codigos conocidos del dataset.
