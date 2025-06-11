

 Car Price Prediction Web App

This is a Machine Learning-powered Flask web application that predicts the selling price of a used car based on user input. It features a clean UI, automatic engine CC detection from car name, and a smart model trained on real-world car data.

 Features:
- Predicts used car prices using a trained ML model (Random Forest Regressor)
- Searchable dropdown for car names (Select2)
- Auto-fills Engine CC when you select the car name
- Clean, responsive, and easy-to-use UI
- Trained on “Car details v3.csv”
- Includes fancy Tableau dashboard (optional)

 Input Fields:
- Car Name (searchable dropdown)
- Year
- Kilometers Driven
- Fuel Type (Petrol, Diesel, CNG, LPG)
- Seller Type (Dealer, Individual)
- Transmission (Manual/Automatic)
- Owner (First Owner, Second Owner, etc.)
- Engine (auto-filled)

 Tech Stack:
- Python
- Flask (Web Framework)
- HTML + CSS + JS + Select2 (Frontend UI)
- Pandas, Scikit-learn (Model Training)
- Joblib (Model Saving)

 Project Structure:
car-price-predictor/
├── app.py                → Main Flask application
├── model/
│   ├── car_price_model.pkl
│   └── encoders.pkl
├── templates/
│   └── index.html        → Web form for input and prediction
├── static/
│   └── style.css         → Optional custom styling
├── Car details v3.csv    → Dataset
├── requirements.txt      → Python dependencies
└── README.txt

 How to Run:
1. Clone this project:
   git clone https://github.com/yourusername/car-price-predictor.git
   cd car-price-predictor

2. Install the required Python libraries:
   pip install -r requirements.txt

3. Run model training script or make sure these files exist:
   - car_price_model.pkl
   - encoders.pkl

4. Start the Flask app:
   python app.py

5. Open the browser:
   http://127.0.0.1:5000/

 Model Info:
- Algorithm: Random Forest Regressor
- Preprocessing: Label Encoding for categorical features
- Model Accuracy: R² Score (Shown using bar chart comparison)
- Feature Columns: name, year, km_driven, fuel, seller_type, transmission, owner

 Future Enhancements:
- OCR Number Plate Reader to auto-fetch car details
- Deployment on Render, Railway, or Heroku
- More advanced ML models (XGBoost, etc.)
- Export predicted price as downloadable report (PDF/Excel)
- Add user login and history tracking

 Dependencies:
- Flask
- Pandas
- Scikit-learn
- Joblib
- Select2 (JS library)

 Contact:
Tirthraj Bhatt  

linkedin:https://www.linkedin.com/in/tirth-bhatt-270187265?utm_source=share&utm_campaign=share_via&utm_content=profile&utm_medium=android_app
