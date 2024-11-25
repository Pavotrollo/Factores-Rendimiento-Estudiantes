import streamlit as st
import pandas as pd
import plotly.express as px
import numpy as np
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split

# Título del Dashboard
st.title("Dashboard de Factores de Rendimiento Académico")

# Sidebar para Filtros
st.sidebar.title("Filtros")
st.sidebar.write("Ajusta los valores para personalizar el análisis.")

# Filtro de características principales
selected_feature = st.sidebar.selectbox(
    "Selecciona una característica para analizar:",
    ['Asistencia', 'Horas de Estudio', 'Sesiones de Tutoría', 'Puntaje Examen']
)

# Filtro para rango de horas de estudio
study_hours = st.sidebar.slider("Horas de Estudio", min_value=0, max_value=40, value=(10, 20))

# Filtro para rango de asistencia
attendance = st.sidebar.slider("Porcentaje de Asistencia", min_value=0, max_value=100, value=(50, 90))

# Datos ficticios para ejemplo
data = {
    'Asistencia': np.random.randint(50, 100, size=100),
    'Horas de Estudio': np.random.randint(5, 40, size=100),
    'Sesiones de Tutoría': np.random.randint(0, 10, size=100),
    'Puntaje Examen': np.random.randint(50, 100, size=100),
}

df = pd.DataFrame(data)

# Filtrar los datos basados en los sliders
filtered_data = df[
    (df['Horas de Estudio'] >= study_hours[0]) & (df['Horas de Estudio'] <= study_hours[1]) &
    (df['Asistencia'] >= attendance[0]) & (df['Asistencia'] <= attendance[1])
]

# Sección de Análisis Exploratorio
st.header("Análisis Exploratorio")

# Gráfico interactivo para la característica seleccionada
st.subheader(f"Distribución de {selected_feature}")
fig = px.histogram(filtered_data, x=selected_feature, nbins=20, title=f"Histograma de {selected_feature}")
st.plotly_chart(fig)

# Relación entre variables
st.subheader("Relación entre Asistencia y Puntaje de Examen")
fig_scatter = px.scatter(
    filtered_data,
    x='Asistencia',
    y='Puntaje Examen',
    title="Relación Asistencia vs Puntaje",
    trendline="ols"
)
st.plotly_chart(fig_scatter)

# Importancia de Características
st.header("Importancia de las Características en el Modelo")

# Definir y entrenar el modelo RandomForestRegressor
X = df[['Asistencia', 'Horas de Estudio', 'Sesiones de Tutoría']]
y = df['Puntaje Examen']

# Dividir en datos de entrenamiento y prueba
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Crear el modelo Random Forest
rf_model = RandomForestRegressor(n_estimators=300, max_depth=20, random_state=42)
rf_model.fit(X_train, y_train)  # Entrenamos el modelo

# Guardamos el modelo entrenado en el estado de la sesión
if 'rf_model' not in st.session_state:
    st.session_state.rf_model = rf_model

# Importancia de las características calculada por el modelo entrenado
importances = st.session_state.rf_model.feature_importances_
features = X.columns

importance_df = pd.DataFrame({'Característica': features, 'Importancia': importances})
fig_importance = px.bar(importance_df, x='Importancia', y='Característica', orientation='h', title="Importancia de Características")
st.plotly_chart(fig_importance)

# Predicción de Puntaje
st.header("Predicción de Puntaje Final")

# Entrada de datos para un estudiante
study_input = st.number_input("Horas de Estudio:", min_value=0, max_value=40, value=10)
attendance_input = st.number_input("Porcentaje de Asistencia:", min_value=0, max_value=100, value=75)
tutoring_input = st.number_input("Sesiones de Tutoría:", min_value=0, max_value=10, value=2)

# Crear DataFrame con la entrada del usuario
new_data = pd.DataFrame({
    'Asistencia': [attendance_input],
    'Horas de Estudio': [study_input],
    'Sesiones de Tutoría': [tutoring_input]
})

# Asegurar que las columnas coincidan con las usadas en el entrenamiento
new_data = new_data[['Asistencia', 'Horas de Estudio', 'Sesiones de Tutoría']]

# Botón para predecir
if st.button("Predecir Puntaje"):
    # Predecir puntaje con el modelo entrenado
    predicted_score = st.session_state.rf_model.predict(new_data)[0]
    st.write(f"El puntaje predicho para el estudiante es: **{predicted_score:.2f}**")

# Footer
st.sidebar.info("Dashboard creado con Streamlit.")
