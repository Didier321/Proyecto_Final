from flask import Flask, render_template, request
import joblib
import numpy as np
import os

model_path = os.path.join(os.path.dirname(__file__), 'models', 'modelo_regresion_arbol_proyectofinal2.pkl')
model = joblib.load(model_path)

app = Flask(__name__)

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/predict',methods=['POST'])
def predict():
    Fiebre= int(request.form['Fiebre'])
    Tos= int(request.form['Tos'])
    Dolor_de_Garganta= int(request.form['Dolor_de_Garganta'])
    Congestión_Nasal= int(request.form['Congestión_Nasal'])
    Dificultad_Respiratoria	= int(request.form['Dificultad_Respiratoria'])


    
    pred_probabilities = np.array([[Fiebre,Tos,Dolor_de_Garganta, Congestión_Nasal,Dificultad_Respiratoria]])
    
    prediccion = model.predict(pred_probabilities)
    
    mensaje = "La clasificacion es : "
    mensaje+= prediccion[0]
    
    return render_template('result.html', pred=mensaje)
    

if __name__ == '__main__':
    app.run()