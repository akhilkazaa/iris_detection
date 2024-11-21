from flask import Flask, request, render_template
import torch
import torch.nn as nn
import torch.nn.functional as F

# Define the model
class Model(nn.Module):
    def __init__(self, in_features=4, h1=8, h2=9, out_features=3):
        super().__init__()
        self.fc1 = nn.Linear(in_features, h1)
        self.fc2 = nn.Linear(h1, h2)
        self.out = nn.Linear(h2, out_features)

    def forward(self, x):
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.out(x)
        return x

# Load the model
torch.manual_seed(41)
model = Model()

# Flask app
app = Flask(__name__)

# Route for homepage
@app.route('/')
def home():
    return render_template('interface.html', prediction=None, error=None)

# Route for prediction
@app.route('/predict', methods=['POST'])
def predict():
    try:
        # Get user input
        sepal_length = float(request.form['sepal_length'])
        sepal_width = float(request.form['sepal_width'])
        petal_length = float(request.form['petal_length'])
        petal_width = float(request.form['petal_width'])

        # Create tensor for the input
        new_iris = torch.tensor([sepal_length, sepal_width, petal_length, petal_width], dtype=torch.float32)

        # Predict
        with torch.no_grad():
            pred_tensor = model(new_iris)
            pred_index = torch.argmax(pred_tensor)

        # Decode prediction
        if pred_index == 0:
            pred_iris = "Setosa"
        elif pred_index == 1:
            pred_iris = "Versicolor"
        else:
            pred_iris = "Virginica"

        # Render the prediction in the interface
        return render_template('interface.html', prediction=pred_iris, error=None)

    except Exception as e:
        # Render error message in the interface
        return render_template('interface.html', prediction=None, error=str(e))

# Run the app
if __name__ == "__main__":
    app.run(debug=True)


