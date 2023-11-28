import numpy as np
from flask import Flask, request, jsonify, render_template
import pickle
import pandas as pd
import os

# Inicialización de la aplicación Flask
app = Flask(__name__)

# Carga del modelo preentrenado desde un archivo pickle
model = pickle.load(open('model.pkl', 'rb'))


# Ruta para la página principal
@app.route('/')
def home():
    # Renderiza la plantilla 'index.html' al acceder a la ruta principal
    return render_template('index.html')

# Ruta para realizar predicciones
@app.route('/predict', methods=["POST"])
def predict():
    # Obtención de datos del formulario web
    features = [
        int(request.form['Age']),
        request.form['Sex'],
        int(request.form['Class']),
        int(request.form["Number of brothers"]),
        int(request.form['Number of parents/children']),
        float(request.form["Fee paid"])
    ]

    # Preprocesamiento de datos antes de la predicción
    data = preprocess_data(features)
    
    print(data)
    print(data.shape)
    # Make prediction using model loaded from disk as per the data.
    prediction = model.predict(data)


    print(prediction)
    # Take the first value of prediction
    output = prediction[0]

    if output == 1:
        res = "You would survive to the Titanic"
    else:
        res = "You would die at the Titanic (soorry)"

    return render_template("index.html", prediction_text=res)

# Función para preprocesar los datos antes de realizar la predicción
def preprocess_data(features):
    # Definición de columnas y valores iniciales para el DataFrame
    columns = ['SibSp', 'Parch', 'Fare_Alta', 'Fare_Baja', 'Fare_Media',
               'Fare_Muy alta', 'age__adult', 'age__kid', 'age__senior',
               'age__teenager', 'age__young_adult', 'Sex_female', 'Sex_male',
               'class__1', 'class__2', 'class__3']
    values = [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]
    
    # Creación de un DataFrame con columnas y valores iniciales
    df = pd.DataFrame([values], columns=columns)

    # Codificación de la edad
    if features[0] >= 0 and features[0] <= 12:
        df['age__kid'] = 1
    elif features[0] >= 13 and features[0] <= 17:
        df['age__teenager'] = 1
    elif features[0] >= 18 and features[0] <= 35:
        df['age__young_adult'] = 1
    elif features[0] >= 36 and features[0] <= 64:
        df['age__adult'] = 1
    elif features[0] >= 65:
        df['age__senior'] = 1

    # Codificación del sexo
    if features[1] == 'male':
        df['Sex_male'] = 1
    else:
        df['Sex_female'] = 1

    # Codificación de la clase
    if features[2] == 1:
        df['class__1'] = 1
    elif features[2] == 2:
        df['class__2'] = 1
    elif features[2] == 3:
        df['class__3'] = 1

    # Número de hermanos
    df['SibSp'] = features[3]

    # Número de padres/hijos
    df['Parch'] = features[4]

    # Codificación de la tarifa del ticket
    if features[5] <= 50:
        df['Fare_Baja'] = 1
    elif features[5] <= 100:
        df['Fare_Media'] = 1
    elif features[5] <= 150:
        df['Fare_Alta'] = 1
    else:
        df['Fare_Muy alta'] = 1

    return df

    
    
    




# Inicia la aplicación Flask en modo de depuración
if __name__ == "__main__":
    app.run(debug=True)
    
