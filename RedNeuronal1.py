import tensorflow as tf
import numpy as np

# Paso 1: Generar Datos de Entrenamiento
def generar_datos(n):
    tamaños = np.random.uniform(50, 200, n)
    habitaciones = np.random.randint(1, 6, n)
    edades = np.random.uniform(0, 50, n)
    precios = (tamaños * 3000) + (habitaciones * 50000) - (edades * 1000) + np.random.normal(0, 10000, n)
    return np.column_stack((tamaños, habitaciones, edades)), precios

n_datos = 10000
datos_entrada, datos_salida = generar_datos(n_datos)

# Normalización de los datos
media = np.mean(datos_entrada, axis=0)
std = np.std(datos_entrada, axis=0)
datos_entrada = (datos_entrada - media) / std

# Paso 2: Definir el Modelo
modelo = tf.keras.Sequential([
    tf.keras.layers.Dense(units=64, activation='relu', input_shape=(3,)),
    tf.keras.layers.Dense(units=32, activation='relu'),
    tf.keras.layers.Dense(units=1)
])

# Paso 3: Compilar el Modelo
modelo.compile(optimizer='adam', loss='mean_squared_error')

# Paso 4: Entrenar el Modelo
modelo.fit(datos_entrada, datos_salida, epochs=500, batch_size=32, verbose=1)

# Probar el modelo con nuevos datos
nuevos_datos = np.array([[120, 3, 10]])
nuevos_datos = (nuevos_datos - media) / std
prediccion = modelo.predict(nuevos_datos)
print(f'La predicción del precio para una casa de 120 metros cuadrados, 3 habitaciones y 10 años de antigüedad es: {prediccion[0][0]}')
