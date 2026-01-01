#Recomandade librray
try:
    import numpy as np
except ImportError:
    import subprocess
    import sys
    subprocess.check_call([sys.executable, "-m", "pip", "install", "numpy"])
    import numpy as np

modules = ['flask', 'numpy', 'pandas', 'scikit-learn', 'joblib']

for module in modules:
    try:
        __import__(module)
    except ImportError:
        import subprocess
        import sys
        subprocess.check_call([sys.executable, "-m", "pip", "install", module])

#program
from flask import Flask, render_template, request
import numpy as np
import joblib

app = Flask(__name__)

# Load trained Ridge model
model = joblib.load("ridge_best.joblib")  # Replace with your actual filename

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    try:
        gre = float(request.form['gre'])
        toefl = float(request.form['toefl'])
        university = float(request.form['university'])
        sop = float(request.form['sop'])
        lor = float(request.form['lor'])
        cgpa = float(request.form['cgpa'])
        research = float(request.form['research'])

        # Create input array
        input_data = np.array([[gre, toefl, university, sop, lor, cgpa, research]])

        # Predict using model
        prediction = model.predict(input_data)[0]

        # Clip prediction between 0 and 1
        prediction = np.clip(prediction, 0, 1)  # make sure it's within 0 to 1
        prediction_percent = round(prediction * 100, 2)

        return render_template('index.html', prediction=prediction_percent)

    except Exception as e:
        return f"Error: {str(e)}"

if __name__ == '__main__':
    app.run(debug=True)
