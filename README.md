# Proyecto 2: Análisis Inicial y Selección de Problema - Datos Financieros

## Descripción

Este proyecto realiza un análisis exploratorio de datos (EDA) inicial para cuatro conjuntos de datos financieros con el objetivo de desarrollar un sistema inteligente de recomendación de inversiones. El análisis se centra en datos históricos de acciones de diferentes mercados para identificar oportunidades de inversión basadas en patrones históricos y análisis de riesgo-retorno.

## Conjuntos de Datos Analizados

### 1. S&P 500
- **Fuente**: Datos históricos del índice S&P 500
- **Período**: Desde diciembre de 1980 hasta la actualidad
- **Tamaño**: ~400 acciones con más de 10,000 registros históricos cada una
- **Características**: Incluye las 500 empresas más grandes de EE.UU. por capitalización de mercado

### 2. NASDAQ
- **Fuente**: Datos históricos del mercado NASDAQ
- **Período**: Desde diciembre de 1980 hasta la actualidad
- **Tamaño**: ~1,500 acciones con datos históricos extensos
- **Características**: Foco en empresas tecnológicas y de crecimiento

### 3. NYSE (New York Stock Exchange)
- **Fuente**: Datos históricos de la Bolsa de Nueva York
- **Período**: Desde noviembre de 2001 hasta la actualidad
- **Tamaño**: ~1,100 acciones con aproximadamente 5,000 registros cada una
- **Características**: Empresas tradicionales y establecidas

### 4. Forbes Global 2000
- **Fuente**: Empresas de la lista Forbes Global 2000
- **Período**: Desde diciembre de 1980 hasta la actualidad
- **Tamaño**: ~600 empresas globales más grandes del mundo
- **Características**: Empresas clasificadas por ventas, beneficios, activos y valor de mercado

## Variables en los Datasets

Todos los conjuntos de datos comparten la misma estructura:

- **Date**: Fecha de la transacción
- **Open**: Precio de apertura
- **High**: Precio máximo del día
- **Low**: Precio mínimo del día
- **Close**: Precio de cierre
- **Volume**: Volumen de transacciones
- **Adjusted Close**: Precio de cierre ajustado por dividendos y splits

## Resumen del EDA Inicial

### Hallazgos Principales

#### Calidad de los Datos:
- ✅ **Sin valores nulos significativos** en ninguno de los datasets
- ✅ **Consistencia temporal** en todos los mercados
- ✅ **Formato uniforme** entre todas las fuentes

#### Correlaciones Identificadas:
- **Alta correlación (>0.99)** entre variables de precio (Open, High, Low, Close)
- **Baja correlación (<0.3)** entre volumen y precios
- **Patrones estacionales** en ciertos sectores

#### Volatilidad y Riesgo:
- **Tecnológicas (NASDAQ)**: Mayor volatilidad pero mayores retornos potenciales
- **Tradicionales (NYSE)**: Menor volatilidad, retornos más estables
- **Diversificadas (S&P 500)**: Balance entre riesgo y retorno
- **Globales (Forbes 2000)**: Diversificación geográfica y sectorial

#### Outliers Detectados:
- **Volúmenes extremos** durante eventos de mercado significativos
- **Movimientos de precio** durante crisis financieras y anuncios corporativos
- **Gaps de precio** en días de baja liquidez

## Problema Seleccionado: Sistema Inteligente de Recomendación de Inversiones

### Justificación de la Elección

Hemos seleccionado desarrollar un **Sistema Inteligente de Recomendación de Inversiones** porque:

1. **Relevancia Práctica**: Problema real que afecta a millones de inversores
2. **Complejidad Técnica**: Combina múltiples técnicas de ML (regresión, clasificación, clustering)
3. **Datos Ricos**: Contamos con 40+ años de datos históricos de múltiples mercados
4. **Impacto Económico**: Las decisiones de inversión tienen consecuencias financieras reales

### Objetivos Específicos

#### 1. Predicción de Precios (Regresión)
- Desarrollar modelos para predecir precios futuros de acciones
- Técnicas: LSTM, ARIMA, Random Forest, SVR
- Métricas: RMSE, MAE, MAPE

#### 2. Clasificación de Tendencias
- Clasificar acciones como "Compra", "Mantener" o "Vender"
- Basado en análisis técnico y fundamental
- Técnicas: SVM, Random Forest, XGBoost

#### 3. Optimización de Portafolios (Clustering)
- Agrupar acciones por características similares
- Diversificación inteligente de riesgo
- Técnicas: K-Means, Hierarchical Clustering

#### 4. Detección de Anomalías
- Identificar comportamientos inusuales del mercado
- Alertas tempranas de volatilidad
- Técnicas: Isolation Forest, One-Class SVM

### Metodología Propuesta

1. **Ingeniería de Características**:
   - Indicadores técnicos (RSI, MACD, Bollinger Bands)
   - Métricas fundamentales (P/E ratio, ROE, etc.)
   - Análisis de sentimiento de noticias

2. **Modelos de Machine Learning**:
   - Regresión para predicción de precios
   - Clasificación para señales de trading
   - Clustering para diversificación de portafolios

3. **Validación y Backtesting**:
   - Validación cruzada temporal
   - Backtesting con datos históricos
   - Métricas de rendimiento ajustadas por riesgo

## Estructura del Repositorio

```
/Proyecto2
├── /datasets
│   ├── sample_sp500.csv
│   ├── sample_nasdaq.csv
│   ├── sample_nyse.csv
│   └── sample_forbes2000.csv
├── /EDA
│   ├── EDA_sp500.ipynb
│   ├── EDA_nasdaq.ipynb
│   ├── EDA_nyse.ipynb
│   └── EDA_forbes2000.ipynb
├── /selected_dataset
│   ├── financial_data_combined.csv
│   └── EDA_selected_dataset.ipynb
├── /stock_market_data (datos originales)
└── README.md
```

## Instrucciones para Ejecutar

### Prerrequisitos
```bash
pip install pandas numpy matplotlib seaborn scikit-learn jupyter
```

### Ejecución de Notebooks
1. Clone el repositorio
2. Navegue a la carpeta EDA
3. Ejecute cada notebook individualmente:
   ```bash
   jupyter notebook EDA_sp500.ipynb
   jupyter notebook EDA_nasdaq.ipynb
   jupyter notebook EDA_nyse.ipynb
   jupyter notebook EDA_forbes2000.ipynb
   ```

### Reproducibilidad
- Todos los notebooks incluyen semillas aleatorias para reproducibilidad
- Los datos están en formato CSV estándar
- No se requieren APIs externas para la ejecución básica

## Parte II: Preprocesamiento y Optimización ✅ COMPLETADA

### Pipeline de Machine Learning Implementado

**Notebook Principal**: `Parte2_Preprocessing_and_Optimization.ipynb`

#### 🔧 Preprocesamiento Avanzado:
- **Feature Engineering**: 20+ indicadores técnicos (RSI, MACD, Bollinger Bands, Stochastic, Williams %R, MFI, ATR)
- **Pipeline Automatizado**: ColumnTransformer con StandardScaler y OneHotEncoder
- **Limpieza de Datos**: Manejo de valores nulos y outliers
- **División Temporal**: 80% entrenamiento, 20% prueba (validación realista)

#### 🤖 Modelos Evaluados:
- Random Forest, XGBoost, LightGBM
- Support Vector Machine, Logistic Regression
- Gradient Boosting, Decision Tree, K-Nearest Neighbors

#### 🔍 Optimización de Hiperparámetros:
- **GridSearchCV**: Búsqueda exhaustiva
- **RandomizedSearchCV**: Búsqueda aleatoria eficiente  
- **Optuna**: Optimización bayesiana avanzada

#### 📊 Evaluación Rigurosa:
- Validación cruzada temporal
- Métricas multiclase (Accuracy, Precision, Recall, F1)
- Matriz de confusión y curvas ROC
- Análisis de importancia de features

### 🎯 Problema Resuelto: Clasificación de Señales de Trading

**Classes**:
- **BUY (0)**: Condiciones favorables para comprar
- **HOLD (1)**: Mantener posición actual
- **SELL (2)**: Condiciones favorables para vender

**Criterios de Señales**:
- Combinación de indicadores técnicos (RSI, Bollinger Bands, MACD, Stochastic)
- Validación con retornos futuros reales
- Balance entre precisión y recall

## Próximos Pasos

1. ✅ **Desarrollo del Modelo Predictivo** (Parte II) - **COMPLETADO**
2. **Backtesting Histórico**: Simulación de estrategias de trading
3. **Dashboard Interactivo**: Visualización en tiempo real con Streamlit/Dash
4. **API de Trading**: Conexión con brokers para ejecución automática
5. **Deployment en Producción**: Containerización y monitoreo continuo

## Autores

- **Analista Principal**: Desarrollo del EDA y selección de problemática
- **Especialista en Datos Financieros**: Interpretación de métricas y validación
- **Ingeniero de Machine Learning**: Diseño de arquitectura de solución

## Licencia

Este proyecto está bajo la Licencia MIT - vea el archivo LICENSE para más detalles.

---

*"El análisis de datos financieros no solo es sobre predecir el futuro, sino sobre entender el presente y aprender del pasado para tomar decisiones informadas."* 