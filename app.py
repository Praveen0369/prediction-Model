from flask import Flask, request, render_template, jsonify
import pandas as pd
import pickle

app = Flask(__name__)

# Load your scikit-learn RandomForest model
forest=open('flight_randf.pkl','rb')
model=pickle.load(forest)

larger_dataset = pd.read_excel('ides.xlsx')  # Replace with the actual path to your dataset
#--------------------------------------------
value_to_id=dict(zip(larger_dataset['value'],larger_dataset['id']))
@app.route('/')
def home():
    return render_template('index.html')

@app.route('/predict_api', methods=['POST'])
def predict_api():
    try:
        # Get user inputs from the form
        airline = request.form.get('airline')
        source_city = request.form.get('source_city')
        departure_time = request.form.get('departure_time')
        stops = request.form['stops']
        arrival_time = request.form['arrival_time']
        destination_city = request.form['destination_city']
        classu = request.form['classu']
        duration = float(request.form['duration'])
        days_left = int(request.form['days_left'])

        # Make prediction
        input_data = predict_fare(airline, source_city, departure_time, stops, arrival_time, destination_city, classu, duration, days_left)
        prediction = model.predict(input_data)
        prediction_list = prediction.tolist()

        # Return the prediction as JSON response
        return jsonify({'prediction': prediction_list})

    except Exception as e:
        return jsonify({'error': str(e)})

def map_value_to_id(input_value):
    return value_to_id.get(input_value, -1)  # Return -1 if the value is not found

def label_encode_input(input_data):
    encoded_data = input_data.copy()
    for column in input_data.columns:
        if input_data[column].dtype == 'object':
            encoded_data[column] = input_data[column].map(map_value_to_id)
    return encoded_data

def predict_fare(airline, source_city, departure_time, stops, arrival_time, destination_city, class_type, duration, days_left):
    # Preprocess input using label encoding
    input_data = label_encode_input(pd.DataFrame({
        'airline': [airline],
        'source_city': [source_city],
        'departure_time': [departure_time],
        'stops': [stops],
        'arrival_time': [arrival_time],
        'destination_city': [destination_city],
        'class': [class_type],
        'duration': [duration],
        'days_left': [days_left],
    }))
    return input_data

if __name__ == "__main__":
    app.run(debug=True)
