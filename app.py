from flask import Flask, request, jsonify
import pickle
import pandas as pd

modelo = None

app = Flask(__name__)

# Cargar modelo guardado
with open("pipeline.pkl", 'rb') as archivo_modelo:
    modelo = pickle.load(archivo_modelo)


@app.route('/predecir', methods=['POST'])
def predecir():
    # Obtener los datos de la solicitud
    data = request.get_json()

    # Crear un Dataframe de pandas a partir del json
    input_data = pd.DataFrame([data])

    # Hacer la prediccion usando el modelo que tiene el piepline que hara la transformacion
    prediccion = modelo.predict(input_data)

    # Devolver la prediccion como json
    output = {'Survived': int(prediccion[0])}

    return jsonify(output)

if __name__ == '__main__':
    app.run(debug=True)
