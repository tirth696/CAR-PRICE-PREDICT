

from flask import Flask, render_template, request, jsonify
import pandas as pd
import joblib

app = Flask(__name__)


model = joblib.load('car_price_model.pkl')
encoders = joblib.load('encoders.pkl')


df = pd.read_csv("Car details v3.csv")
df.dropna(subset=['engine'], inplace=True)
df['engine'] = df['engine'].str.replace(' CC', '', regex=False).astype(float)


car_names = sorted(df['name'].unique())
fuel_types = sorted(df['fuel'].unique())
seller_types = sorted(df['seller_type'].unique())
transmissions = sorted(df['transmission'].unique())
owners = sorted(df['owner'].unique())


engine_lookup = df.groupby('name')['engine'].first().to_dict()

@app.route('/')
def index():
    return render_template('index.html',
                           car_names=car_names,
                           fuel_types=fuel_types,
                           seller_types=seller_types,
                           transmissions=transmissions,
                           owners=owners,
                           prediction_text=None)

@app.route('/get_engine', methods=['POST'])
def get_engine():
    name = request.json['name']
    engine = engine_lookup.get(name, 0)
    return jsonify({'engine': engine})

@app.route('/predict', methods=['POST'])
def predict():
    try:
        # Collect form inputs
        name = request.form['name']
        year = int(request.form['year'])
        km_driven = int(request.form['km_driven'])
        fuel = request.form['fuel']
        seller_type = request.form['seller_type']
        transmission = request.form['transmission']
        owner = request.form['owner']
        engine = float(request.form['engine'])

        
        input_data = {
            'name': encoders['name'].transform([name])[0] if name in encoders['name'].classes_ else 0,
            'year': year,
            'km_driven': km_driven,
            'fuel': encoders['fuel'].transform([fuel])[0] if fuel in encoders['fuel'].classes_ else 0,
            'seller_type': encoders['seller_type'].transform([seller_type])[0] if seller_type in encoders['seller_type'].classes_ else 0,
            'transmission': encoders['transmission'].transform([transmission])[0] if transmission in encoders['transmission'].classes_ else 0,
            'owner': encoders['owner'].transform([owner])[0] if owner in encoders['owner'].classes_ else 0,
            'engine': engine
        }

        input_df = pd.DataFrame([input_data])

        # Predict price
        prediction = model.predict(input_df)[0]
        result = f" Predicted Selling Price: ₹{int(prediction):,}"

    except Exception as e:
        result = f"❌ Error: {str(e)}"

    return render_template('index.html',
                           car_names=car_names,
                           fuel_types=fuel_types,
                           seller_types=seller_types,
                           transmissions=transmissions,
                           owners=owners,
                           prediction_text=result)

if __name__ == '__main__':
    app.run(debug=True)
