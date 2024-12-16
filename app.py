import streamlit as st
import pandas as pd
import joblib

# Cargar el modelo y los encoders
model = joblib.load("modelo_gastos.pkl")
encoder_destino = joblib.load("encoder_destino.pkl")
encoder_empleado = joblib.load("encoder_empleado.pkl")

# Título de la aplicación
st.title("Predicción de Gastos de Viaje")

# Entradas del usuario
destino = st.selectbox("Destino del viaje:", ["Lima", "Cusco", "Arequipa", "Otra"])
fecha_salida = st.date_input("Fecha de salida:")
fecha_regreso = st.date_input("Fecha de regreso:")
empleado = st.text_input("Empleado:")
total_dias = (fecha_regreso - fecha_salida).days

if total_dias <= 0:
    st.error("La fecha de regreso debe ser posterior a la fecha de salida.")
else:
    st.success(f"Total de días calculados: {total_dias}")

# Botón para predecir
if st.button("Predecir Gastos"):
    try:
        # Convertir las entradas en formato numérico
        destino_encoded = encoder_destino.transform([destino])[0]
        empleado_encoded = encoder_empleado.transform([empleado])[0]

        # Crear la entrada para el modelo
        entrada = pd.DataFrame([[destino_encoded, total_dias, empleado_encoded]],
                               columns=["Destino", "Total_días", "Empleado"])

        # Realizar la predicción
        prediccion = model.predict(entrada)
        st.subheader("Gasto estimado:")
        st.write(f"S/{prediccion[0]:,.2f}")
    except Exception as e:
        st.error(f"Error al realizar la predicción: {e}")
