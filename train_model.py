import pandas as pd
import joblib
from sklearn.ensemble import RandomForestRegressor
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split

# Cargar el dataset
df = pd.read_csv("dataset_gastos_viaje.csv")
df["Total_días"] = pd.to_datetime(df["Fecha_regreso"]) - pd.to_datetime(df["Fecha_salida"])
df["Total_días"] = df["Total_días"].dt.days

# Preprocesar las columnas categóricas
encoder_destino = LabelEncoder()
encoder_empleado = LabelEncoder()

df["Destino"] = encoder_destino.fit_transform(df["Destino"])
df["Empleado"] = encoder_empleado.fit_transform(df["Empleado"])

# Dividir en características (X) y variable objetivo (y)
X = df[["Destino", "Total_días", "Empleado"]]
y = df["Gasto_total"]

# Dividir los datos en entrenamiento y prueba
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Entrenar el modelo
model = RandomForestRegressor(random_state=42)
model.fit(X_train, y_train)

# Guardar el modelo y los encoders
joblib.dump(model, "modelo_gastos.pkl")
joblib.dump(encoder_destino, "encoder_destino.pkl")
joblib.dump(encoder_empleado, "encoder_empleado.pkl")
