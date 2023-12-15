import tkinter as tk
from tkinter import Label, Button
import torch
import torch.nn.functional as F
from keras.models import Sequential
from keras.layers import Dense
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score

class NeuralNetwork(torch.nn.Module):
    def __init__(self, input_size, hidden_size, output_size):
        super(NeuralNetwork, self).__init__()
        self.layer1 = torch.nn.Linear(input_size, hidden_size)
        self.relu = torch.nn.ReLU()
        self.layer2 = torch.nn.Linear(hidden_size, output_size)

    def forward(self, x):
        x = self.layer1(x)
        x = self.relu(x)
        x = self.layer2(x)
        return x

class Scorer:

    def __init__(self, verbose=True):
        self.verbose = verbose
        self.reset()

    def reset(self):
        self.y_true = []
        self.y_pred = []

    def update(self, yh, y):
        _, I = yh.max(dim=1)
        self.y_true.extend(y.cpu().numpy())
        self.y_pred.extend(I.cpu().numpy())

    def peek(self):
        sens = self.calc_sens()
        spec = self.calc_spec()
        acc = self.calc_acc()
        if self.verbose:
            print("Sens/Spec/Acc: %.5f/%.5f/%.5f" % (sens, spec, acc))
        return (sens + spec) / 2.0, acc

    def calc_precision(self):
        return precision_score(self.y_true, self.y_pred, pos_label=1)

    def calc_f1(self):
        return f1_score(self.y_true, self.y_pred, pos_label=1)

    def calc_acc(self):
        return accuracy_score(self.y_true, self.y_pred)

    def calc_sens(self):
        return recall_score(self.y_true, self.y_pred, pos_label=1)

    def calc_spec(self):
        return recall_score(self.y_true, self.y_pred, pos_label=0)

def train_neural_network(X_train, y_train, input_size, hidden_size, output_size, epochs=10):
    model = NeuralNetwork(input_size, hidden_size, output_size)
    criterion = torch.nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)

    for epoch in range(epochs):
        inputs = torch.Tensor(X_train).float()
        labels = torch.LongTensor(y_train)

        optimizer.zero_grad()
        outputs = model(inputs)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()

    return model

def create_keras_model(input_size, hidden_size, output_size):
    model = Sequential()
    model.add(Dense(hidden_size, input_dim=input_size, activation='relu'))
    model.add(Dense(output_size, activation='softmax'))
    model.compile(loss='sparse_categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
    return model

def main():
    # Assuming you have data X and labels y
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    # Train PyTorch model
    pytorch_model = train_neural_network(X_train, y_train, input_size=X_train.shape[1], hidden_size=128, output_size=len(set(y)), epochs=10)
    
    # Train Keras model
    keras_model = create_keras_model(input_size=X_train.shape[1], hidden_size=128, output_size=len(set(y)))
    keras_model.fit(X_train, y_train, epochs=10, batch_size=32, verbose=0)

    # Initialize Scorers
    pytorch_scorer = Scorer()
    keras_scorer = Scorer()

    # Evaluate PyTorch model
    with torch.no_grad():
        pytorch_outputs = F.softmax(pytorch_model(torch.Tensor(X_test).float()), dim=1)
        pytorch_scorer.update(pytorch_outputs, torch.LongTensor(y_test))
        pytorch_scorer.peek()


    keras_outputs = keras_model.predict(X_test)
    keras_scorer.update(torch.Tensor(keras_outputs), torch.LongTensor(y_test))
    keras_scorer.peek()

  
    root = tk.Tk()
    root.title("Model Metrics")

    tk.Label(root, text="PyTorch Metrics").pack()
    tk.Label(root, text=f"Precision: {pytorch_scorer.calc_precision():.5f}").pack()
    tk.Label(root, text=f"F1 Score: {pytorch_scorer.calc_f1():.5f}").pack()

    tk.Label(root, text="Keras Metrics").pack()
    tk.Label(root, text=f"Precision: {keras_scorer.calc_precision():.5f}").pack()
    tk.Label(root, text=f"F1 Score: {keras_scorer.calc_f1():.5f}").pack()

    root.mainloop()

if __name__ == "__main__":
    main()
