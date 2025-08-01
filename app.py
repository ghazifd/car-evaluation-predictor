from flask import Flask, render_template, request
import pickle

app = Flask(__name__)

# Charger le modèle
model = pickle.load(open("model/model.pkl", "rb"))

# Mapping pour encodage
mappings = {
    'buying': ['low', 'med', 'high', 'vhigh'],
    'maint': ['low', 'med', 'high', 'vhigh'],
    #'doors': ['2', '3', '4', '5more'],
    'persons': ['2', '4', 'more'],
    #'lug_boot': ['small', 'med', 'big'],
    'safety': ['low', 'med', 'high']
}

def encode_input(data):
    return [mappings[col].index(data[col]) for col in mappings]

@app.route('/', methods=['GET', 'POST'])
def index():
    prediction = None
    if request.method == 'POST':
        input_data = {
            'buying': request.form['buying'],
            'maint': request.form['maint'],
            #'doors': request.form['doors'],
            'persons': request.form['persons'],
            #'lug_boot': request.form['lug_boot'],
            'safety': request.form['safety']
        }
        encoded = encode_input(input_data)
        prediction = model.predict([encoded])[0]
    return render_template('index.html', prediction=prediction)

if __name__ == '__main__':
    app.run(debug=True)
