from flask import Flask, render_template, request, redirect, url_for
import pandas as pd
import pickle

app = Flask(__name__)

# Load model and preprocessor
try:
    with open('artifacts/model.pkl', 'rb') as f:
        model = pickle.load(f)
    with open('artifacts/preprocessor.pkl', 'rb') as f:
        preprocessor = pickle.load(f)
except Exception as e:
    print(f"Error loading model files: {str(e)}")
    # Handle model loading failure appropriately

@app.route('/')
def home():
    return redirect(url_for('show_form'))

@app.route('/form')
def show_form():
    return render_template('form.html')

@app.route('/predict', methods=['POST'])
def predict():
    try:
        # Validate required fields
        required_fields = ['carat', 'depth', 'table', 'x', 'y', 'z', 'cut', 'color', 'clarity']
        data = {}
        for field in required_fields:
            value = request.form.get(field)
            if not value:
                return render_template('form.html', error=f"Missing required field: {field}")
            data[field] = value

        # Convert numeric fields
        numeric_fields = ['carat', 'depth', 'table', 'x', 'y', 'z']
        for field in numeric_fields:
            try:
                data[field] = float(data[field])
            except ValueError:
                return render_template('form.html', error=f"Invalid number for {field}")

        # Create DataFrame for prediction
        input_data = pd.DataFrame([[
            data['carat'],
            data['depth'],
            data['table'],
            data['x'],
            data['y'],
            data['z'],
            data['cut'],
            data['color'],
            data['clarity']
        ]], columns=required_fields)

        # Preprocess and predict
        transformed_data = preprocessor.transform(input_data)
        predicted_price = model.predict(transformed_data)[0]

        return render_template('results.html', 
                            final_result=round(predicted_price, 2),
                            carat=data['carat'],
                            color=data['color'],
                            clarity=data['clarity'])

    except Exception as e:
        error_msg = f"Error processing your request: {str(e)}"
        return render_template('form.html', error=error_msg)

# Handle 404 errors by redirecting to form
@app.errorhandler(404)
def page_not_found(e):
    return redirect(url_for('show_form'))

if __name__ == '__main__':
    app.run(debug=True)