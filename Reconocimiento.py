import cv2
import numpy as np
import os

from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score

ruta = "banco_imagenes"

imagenes = []
etiquetas = []

print("Cargando banco de imagenes...")

for categoria in os.listdir(ruta):
    carpeta = os.path.join(ruta, categoria)

    if os.path.isdir(carpeta):
        for archivo in os.listdir(carpeta):
            camino = os.path.join(carpeta, archivo)

            imagen = cv2.imread(camino)

            if imagen is not None:
                imagen = cv2.resize(imagen, (64, 64))
                gris = cv2.cvtColor(imagen, cv2.COLOR_BGR2GRAY)

                caracteristicas = gris.flatten()

                imagenes.append(caracteristicas)
                etiquetas.append(categoria)

if len(imagenes) < 3:
    print("No hay suficientes imagenes para entrenar.")
    exit()

X = np.array(imagenes)
y = np.array(etiquetas)

X_entrenamiento, X_prueba, y_entrenamiento, y_prueba = train_test_split(
    X, y, test_size=0.2, random_state=42
)

algoritmos = {
    "KNN": KNeighborsClassifier(n_neighbors=3),
    "SVM": SVC(kernel="linear"),
    "Random Forest": RandomForestClassifier(
        n_estimators=100, random_state=42
    )
}

print("\nEntrenando algoritmos...")

for nombre, modelo in algoritmos.items():
    modelo.fit(X_entrenamiento, y_entrenamiento)

    predicciones = modelo.predict(X_prueba)
    precision = accuracy_score(y_prueba, predicciones)

    print(nombre, "Precision:", round(precision * 100, 2), "%")

print("\nSelecciona una imagen para reconocer.")

ruta_imagen = input("Escribe la ruta de la imagen: ")

imagen = cv2.imread(ruta_imagen)

if imagen is None:
    print("No se pudo abrir la imagen. Revisa la ruta.")
    exit()

imagen_procesada = cv2.resize(imagen, (64, 64))
gris = cv2.cvtColor(imagen_procesada, cv2.COLOR_BGR2GRAY)

caracteristicas = gris.flatten().reshape(1, -1)

resultados = []

for nombre, modelo in algoritmos.items():
    resultado = modelo.predict(caracteristicas)[0]

    resultados.append(nombre + ": " + resultado)

    print(nombre, "reconoce:", resultado)

# Preparar imagen para mostrar

alto, ancho = imagen.shape[:2]

escala = min(900 / ancho, 650 / alto, 1)

imagen_mostrar = cv2.resize(
    imagen,
    (int(ancho * escala), int(alto * escala))
)

# Agregar resultados sobre la imagen

cv2.rectangle(
    imagen_mostrar,
    (0, 0),
    (imagen_mostrar.shape[1], 115),
    (0, 0, 0),
    -1
)

for i, texto in enumerate(resultados):
    cv2.putText(
        imagen_mostrar,
        texto,
        (15, 30 + i * 35),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.7,
        (0, 255, 0),
        2
    )

cv2.imshow("RECONOCIMIENTO DE IMAGENES", imagen_mostrar)

print("\nPresiona cualquier tecla sobre la ventana para cerrar.")

cv2.waitKey(0)
cv2.destroyAllWindows()
