# Developing a Neural Network Regression Model

## AIM

To develop a neural network regression model to predict a student's final marks based on the number of hours they study

## THEORY

In educational institutions, understanding the direct impact of study habits on academic performance is crucial. The goal of this project is to build a predictive system that takes a single continuous feature—the number of hours a student spends studying—and outputs a continuous numerical prediction of their expected final exam score. Because the target variable (marks) is a continuous numerical value rather than a discrete category, this challenge is classified as a regression problem

## Neural Network Model

<img width="1536" height="1024" alt="MODEL DIAGRAM" src="https://github.com/user-attachments/assets/3ed6816b-176a-400c-8edb-9c38951df275" />

## DESIGN STEPS
### STEP 1: 

Create your dataset in a Google sheet with one numeric input and one numeric output.

### STEP 2: 

Split the dataset into training and testing

### STEP 3: 

Create MinMaxScalar objects ,fit the model and transform the data.

### STEP 4: 

Build the Neural Network Model and compile the model.

### STEP 5: 

Train the model with the training data.

### STEP 6: 

Plot the performance plot

### STEP 7: 

Evaluate the model with the testing data.

### STEP 8: 

Use the trained model to predict  for a new input value .

## PROGRAM

### Name: MITHUN S

### Register Number: 212224240088

```
import torch
import torch.nn as nn
import torch.optim as optim
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler
dataset1 = pd.read_csv('/content/student_study_hours_regression.csv')
X = dataset1[['Study_Hours']].values
y = dataset1[['Exam_Marks']].values
print(dataset1)
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.33, random_state=33
)
print("TRAINING SAMPLES:",len(X_train))
print("TESTING SAMPLES:",len(X_test))
scaler = MinMaxScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)
X_train_tensor = torch.tensor(X_train, dtype=torch.float32)
y_train_tensor = torch.tensor(y_train, dtype=torch.float32).view(-1, 1)
X_test_tensor = torch.tensor(X_test, dtype=torch.float32)
y_test_tensor = torch.tensor(y_test, dtype=torch.float32).view(-1, 1)
class NeuralNet(nn.Module):
    def __init__(self):
        super().__init__()
        self.fc1 = nn.Linear(1, 10)
        self.relu = nn.ReLU()
        self.fc2 = nn.Linear(10, 1)
        self.history = {'loss': []}
    def forward(self, x):
        x = self.fc1(x)
        x = self.relu(x)
        x = self.fc2(x)
        return x
ai_brain = NeuralNet()
criterion = nn.MSELoss()
optimizer = optim.Adam(ai_brain.parameters(), lr=0.01)
def train_model(ai_brain, X_train, y_train, criterion, optimizer, epochs=2000):
    ai_brain.train()
    for epoch in range(1, epochs + 1):
        optimizer.zero_grad()
        outputs = ai_brain(X_train)
        loss = criterion(outputs, y_train)
        loss.backward()
        optimizer.step()
        ai_brain.history['loss'].append(loss.item())
        if epoch % 200 == 0:
            print(f'Epoch [{epoch}/{epochs}], Loss: {loss.item():.6f}')
train_model(ai_brain, X_train_tensor, y_train_tensor, criterion, optimizer)
ai_brain.eval()
with torch.no_grad():
    test_loss = criterion(ai_brain(X_test_tensor), y_test_tensor)
    print(f'Test Loss: {test_loss.item():.6f}')
    loss_df = pd.DataFrame(ai_brain.history)
import matplotlib.pyplot as plt
loss_df.plot()
plt.xlabel("Epochs")
plt.ylabel("Loss")
plt.title("Loss during Training")
plt.show()
X_n1_1 = torch.tensor([[9]], dtype=torch.float32)
prediction = ai_brain(
    torch.tensor(
        scaler.transform(X_n1_1.numpy()),
        dtype=torch.float32
    )
).item()
print("Sample Input (Study Hours): 9")
print(f"Sample Output (Predicted Exam Marks): {prediction:.2f}")

```

### Dataset Information

<img width="733" height="507" alt="image" src="https://github.com/user-attachments/assets/94fc1c46-2ea9-4855-94f3-e31b655205ee" />

### OUTPUT

### Training Loss Vs Iteration Plot

<img width="723" height="571" alt="image" src="https://github.com/user-attachments/assets/aa109f1f-191d-4ae6-8d30-8a82b039e6bb" />


### New Sample Data Prediction

<img width="720" height="351" alt="image" src="https://github.com/user-attachments/assets/b1cdc983-805f-4ac3-8675-345f706189a0" />

## RESULT
Thus, a neural network regression model was successfully developed and trained using PyTorch.
