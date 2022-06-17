from flask import Flask,jsonify,request,render_template
import pickle
import numpy as np

app = Flask(__name__)
model=pickle.load(open('knn.pkl','rb'))


@app.route('/')
def home():
    return render_template("index.html")


@app.route('/predict',methods=['POST'])
def predict():
    int_features=[int(x) for x in request.form.values()]
    final=[np.array(int_features)]
    prediction=model.predict(final)
    return render_template('index.html',prediction_text ='PREDICTED CLASS : {}'.format(float(prediction)))

@app.route('/predict_api',methods=['POST'])
def predict_api():
    '''
    For direct API calls trought request
    '''
    data = request.get_json(force=True)
    prediction = model.predict([int(np.array(list(data.values())))])

    # output = prediction[0]
    return jsonify(prediction)

if __name__ == "__main__":
    app.run(debug=True)