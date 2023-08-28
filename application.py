from flask import Flask, request, render_template, jsonify
import pickle
import pandas as pd
from src.pipeline.prediction_pipeline import CustomData, PredictPipeline

application = Flask(__name__)
app = application

# Load the model from the .pkl file
with open('artifacts\model.pkl', 'rb') as f:
    model = pickle.load(f)

@app.route('/')
def home_page():
    return render_template('index.html')

@app.route('/predict', methods=['GET', 'POST'])
def predict_datapoint():
    if request.method == 'GET':
        return render_template('form.html')
    else:
        data = CustomData(
            age = float(request.form.get('age')), 
            children = int(request.form.get('children')),
            bmi = float(request.form.get('bmi')),            
            sex = request.form.get('sex'), 
            region = request.form.get('region'),
            smoker = request.form.get('smoker')
        )

        # Preprocess 'sex' field to convert it into a numeric representation
        sex_mapping = {'female': 0, 'male': 1}
        data.sex = sex_mapping.get(data.sex.lower(), -1)  # -1 if sex is neither 'female' nor 'male'

        # Preprocess 'smoker' field to convert it into a numeric representation
        smoker_mapping = {'no': 0, 'yes': 1}
        data.smoker = smoker_mapping.get(data.smoker.lower(), -1)  # -1 if smoker is neither 'no' nor 'yes'

        # Preprocess 'region' field using one-hot encoding
        region_mapping = {'northeast': 1, 'southeast': 2, 'southwest': 3, 'northwest': 4}
        data.region = region_mapping.get(data.region.lower(), -1)  # -1 if region is not one of the four regions

        input_data = pd.DataFrame(data.get_data_as_dataframe(), index=[0])
        print(input_data)
        predictions = model.predict(input_data)
        expenses = predictions[0]
        return render_template('results.html', final_result=expenses)


if __name__ == "__main__":
    app.run(host='0.0.0.0', port=8080)
