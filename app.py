import pickle
from flask import Flask,  render_template, request
import numpy as np

app = Flask(__name__ , template_folder='template')
model = pickle.load(open('model.pkl', 'rb'))

@app.route('/')
def page():
    return render_template('home.html')

@app.route('/form')
def home():
    return render_template('index.html')


@app.route('/predict', methods=['POST'])
def predict():
    ele = ['Pregnancies', 'Glucose', 'BloodPressure', 'SkinThickness', 'Insulin', 'BMI', 'DiabetesPedigreeFunction', 'Age']
    fe = []
    for x in ele:
        ans = request.form[x]
        fe.append(ans)

    final_features = [np.array(fe)]
    prediction = model.predict(final_features)
    if prediction == [0]:
        return render_template('result.html', prediction="You are Non-Diabetic")
    else:
        return render_template('result.html', prediction="You are Diabetic")








if __name__ == '__main__':
    app.run(debug=True)
