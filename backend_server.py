from flask import Flask, render_template, request
import torch
import pandas as pd
import torch.nn as nn
import torch.nn.functional as F
from sklearn.model_selection import train_test_split

# Initialize Flask app
app = Flask(__name__)

# Define your model class
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

# Load Iris dataset and do some preprocessing
url = 'https://gist.githubusercontent.com/netj/8836201/raw/6f9306ad21398ea43cba4f7d537619d0e07d5ae3/iris.csv'
df = pd.read_csv(url)
df['variety'] = df['variety'].replace({'Setosa': 0.0, 'Versicolor': 1.0, 'Virginica': 2.0})
x = df.drop('variety', axis=1).values
y = df['variety'].values
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state=41)

# Convert to tensors
x_train = torch.FloatTensor(x_train)
x_test = torch.FloatTensor(x_test)
y_train = torch.LongTensor(y_train)
y_test = torch.LongTensor(y_test)

# Initialize model, loss function, and optimizer
model = Model()
criterion = nn.CrossEntropyLoss()
optimiser = torch.optim.Adam(model.parameters(), lr=0.015)

# Train the model
epochs = 100
for i in range(epochs):
    y_pred = model(x_train)
    loss = criterion(y_pred, y_train)
    optimiser.zero_grad()
    loss.backward()
    optimiser.step()

# Define the prediction function
def predict_iris(sepal_length, sepal_width, petal_length, petal_width):
    new_data = torch.tensor([sepal_length, sepal_width, petal_length, petal_width]).float()
    model.eval()  # Set the model to evaluation mode
    with torch.no_grad():
        prediction = model(new_data)
        predicted_class = prediction.argmax().item()
        if predicted_class == 0:
            return "Setosa"
        elif predicted_class == 1:
            return "Versicolor"
        else:
            return "Virginica"

# Define the routes
@app.route('/')
def index():
    return render_template('interface.html')

@app.route('/predict', methods=['POST'])
def predict():
    try:
        sepal_length = float(request.form['sepal_length'])
        sepal_width = float(request.form['sepal_width'])
        petal_length = float(request.form['petal_length'])
        petal_width = float(request.form['petal_width'])
        prediction = predict_iris(sepal_length, sepal_width, petal_length, petal_width)
        return render_template('interface.html', prediction=prediction)
    except Exception as e:
        error_message = f"Error: {str(e)}"
        return render_template('interface.html', error=error_message)

if __name__ == '__main__':
    app.run(debug=True)




