import numpy as np  # Numpy kütüphanesini import etme
import pandas as pd  # Pandas kütüphanesini import etme
import seaborn as sns  # Seaborn kütüphanesini import etme
import matplotlib.pyplot as plt  # Matplotlib kütüphanesini import etme
from sklearn.model_selection import train_test_split  # Veri setini eğitim ve test setlerine bölmek için sklearn'den train_test_split fonksiyonunu import etme
from sklearn.preprocessing import StandardScaler  # Verileri ölçeklendirmek için sklearn'den StandardScaler'ı import etme

from sklearn.ensemble import RandomForestClassifier  # Random Forest modelini import etme
import torch  # PyTorch kütüphanesini import etme
import torch.nn as nn  # PyTorch'tan nn modülünü import etme
import torch.optim as optim  # PyTorch'tan optim modülünü import etme
import os # işletim sistemin kütüphanesini import etme
df=pd.read_csv('/content/drive/MyDrive/AIDS_Classification (1).csv') # dataseti  CSV dosyasını okumak ve içeriğini 'df' adında bir DataFrame nesnesine yüklemek

from google.colab import drive
drive.mount('/content/drive')

Drive already mounted at /content/drive; to attempt to forcibly remount, call drive.mount("/content/drive", force_remount=True).

df # dataseti yazıdrmak

df.info()

fig = plt.figure(figsize=(25, 8))

gs = fig.add_gridspec(1, 1)

gs.update(wspace=0.3, hspace=0.15)

ax = fig.add_subplot(gs[0, 0])

ax.set_title("Correlation Matrix", fontsize=28, fontweight='bold', fontfamily='serif', color="#fff")

sns.heatmap(df.corr().transpose(), mask=np.triu(np.ones_like(df.corr().transpose())), fmt=".1f", annot=True, cmap='Blues')

plt.show()

x=df.iloc[:,:-1].values
y=df.iloc[:,-1].values
from sklearn.preprocessing import MinMaxScaler
scaler = MinMaxScaler()
model=scaler.fit(x)
x_scaled=model.transform(x)

# Veriyi %20 test setine ve %80 eğitim setine bölelim
X_train, X_test, y_train, y_test = train_test_split(x_scaled, y, test_size=0.2, random_state=42)


import torch
import torch.nn as nn
import torch.optim as optim
import time
from sklearn.datasets import make_classification
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler



# ANN modellerini tanımla
class ANN(nn.Module):
    def __init__(self, input_dim, hidden_dims, activation_function):
        super(ANN, self).__init__()
        self.layers = nn.ModuleList()
        self.layers.append(nn.Linear(input_dim, hidden_dims[0]))
        for i in range(len(hidden_dims) - 1):
            self.layers.append(activation_function())
            self.layers.append(nn.Linear(hidden_dims[i], hidden_dims[i+1]))
        self.layers.append(activation_function())
        self.layers.append(nn.Linear(hidden_dims[-1], 1))
        self.layers.append(nn.Sigmoid()) # İki sınıflı sınıflandırma için sigmoid

    def forward(self, x):
        for layer in self.layers:
            x = layer(x)
        return x

# Parametre kombinasyonları
learning_rates = [0.1, 0.01, 0.001]
epochs_list = [1000, 500, 100]
hidden_dims_list = [[64, 32, 16], [128, 64, 32], [256, 128, 64]]

activation_functions = [nn.ReLU, nn.Tanh, nn.LeakyReLU]

# Tüm kombinasyonları dene
results = []
for lr in learning_rates:
    for epochs in epochs_list:
        for hidden_dims in hidden_dims_list:

              for activation_function in activation_functions:
                  # Modeli oluştur
                  ann_model = ANN(X_train.shape[1], hidden_dims, activation_function)
                  criterion = nn.BCELoss()  # İki sınıflı sınıflandırma için binary cross entropy
                  optimizer = optim.Adam(ann_model.parameters(), lr=lr)

                  # Eğitim döngüsü
                  start_time = time.time()
                  for epoch in range(epochs):
                      inputs = torch.tensor(X_train, dtype=torch.float32)
                      targets = torch.tensor(y_train, dtype=torch.float32).view(-1, 1)

                      optimizer.zero_grad()
                      outputs = ann_model(inputs)
                      loss = criterion(outputs, targets)
                      loss.backward()
                      optimizer.step()
                  end_time = time.time()

                  # Test verileri üzerinde değerlendirme
                  inputs = torch.tensor(X_test, dtype=torch.float32)
                  targets = torch.tensor(y_test, dtype=torch.float32).view(-1, 1)
                  ann_outputs = ann_model(inputs)
                  ann_score = (ann_outputs.round() == targets).float().mean().item()  # Doğruluk oranı
                  training_time = end_time - start_time
                  print(10*"*")
                  # Sonuçları kaydet
                  results.append({
                      "learning_rate": lr,
                      "epochs": epochs,
                      "hidden_dims": hidden_dims,
                      "neurons":"" ,
                      "activation_function": activation_function.__name__,
                      "accuracy": ann_score,
                      "training_time": training_time
                  })


# Sonuçları yazdır

results_df=[]
for result in results:

    result_data = {
        "Epok Sayısı": result["epochs"],
        "Test %": 20,
        "Katman Sayısı": len(result["hidden_dims"]),
        "Nöron Sayısı": sum(result["hidden_dims"]),
        "Aktivasyon Fonksiyonu": result["activation_function"],
        "Learning rate": result["learning_rate"],
        "Accuracy (ACC)": result["accuracy"]
    }

    results_df.append(result_data)


results_df = pd.DataFrame(results_df)


print(results_df)

results_df


# Random Forest modelini eğitelim
rf_model = RandomForestClassifier(n_estimators=100, random_state=42)
rf_model.fit(X_train, y_train)

rf_score = rf_model.score(X_test, y_test)
print("Random Forest Score:", rf_score)

df = pd.DataFrame(results)
best_model_index = df["accuracy"].idxmax()
best_model = df.iloc[best_model_index].to_dict()



print("En İyi Model:")
print("Learning Rate:", best_model["learning_rate"])
print("Epochs:", best_model["epochs"])
print("Hidden Dims:", best_model["hidden_dims"])
print("Neurons:", best_model["neurons"])
print("Activation Function:", best_model["activation_function"])
print("Accuracy:", best_model["accuracy"])
print("Training Time:", best_model["training_time"])


results_df.to_csv('model_results.csv', index=False)

import torch
import torch.nn as nn
import torch.optim as optim
import time
from sklearn.datasets import make_classification
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler



# ANN modellerini tanımla
class ANN(nn.Module):
    def __init__(self, input_dim, hidden_dims, activation_function):
        super(ANN, self).__init__()
        self.layers = nn.ModuleList()
        self.layers.append(nn.Linear(input_dim, hidden_dims[0]))
        for i in range(len(hidden_dims) - 1):
            self.layers.append(activation_function())
            self.layers.append(nn.Linear(hidden_dims[i], hidden_dims[i+1]))
        self.layers.append(activation_function())
        self.layers.append(nn.Linear(hidden_dims[-1], 1))
        self.layers.append(nn.Sigmoid()) # İki sınıflı sınıflandırma için sigmoid

    def forward(self, x):
        for layer in self.layers:
            x = layer(x)
        return x

# Parametre kombinasyonları

epochs_list = [100, 500, 1000]



# Activation functions
activation_functions = [nn.ReLU, nn.Tanh, nn.LeakyReLU]

# Tüm kombinasyonları dene
results = []

# Create and train models without using for loop for combinations
def train_and_evaluate_model(hidden_dims, lr, epochs, activation_function):
    ann_model = ANN(X_train.shape[1], hidden_dims, activation_function)
    criterion = nn.BCELoss()  # İki sınıflı sınıflandırma için binary cross entropy
    optimizer = optim.Adam(ann_model.parameters(), lr=lr)

    # Eğitim döngüsü
    start_time = time.time()
    for epoch in range(epochs):
        inputs = torch.tensor(X_train, dtype=torch.float32)
        targets = torch.tensor(y_train, dtype=torch.float32).view(-1, 1)

        optimizer.zero_grad()
        outputs = ann_model(inputs)
        loss = criterion(outputs, targets)
        loss.backward()
        optimizer.step()
    end_time = time.time()

    # Test verileri üzerinde değerlendirme
    inputs = torch.tensor(X_test, dtype=torch.float32)
    targets = torch.tensor(y_test, dtype=torch.float32).view(-1, 1)
    ann_outputs = ann_model(inputs)
    ann_score = (ann_outputs.round() == targets).float().mean().item()  # Doğruluk oranı
    training_time = end_time - start_time

    # Sonuçları kaydet
    results.append({
        "learning_rate": lr,
        "epochs": epochs,
        "hidden_dims": hidden_dims,
        "activation_function": activation_function.__name__,
        "accuracy": ann_score,
        "training_time": training_time
    })

# Train models with different configurations
# Model 1: 2 layers
train_and_evaluate_model([64, 32], 0.1, 100, nn.ReLU)
train_and_evaluate_model([64, 32], 0.1, 500, nn.ReLU)
train_and_evaluate_model([64, 32], 0.1, 1000, nn.ReLU)
train_and_evaluate_model([64, 32], 0.01, 100, nn.ReLU)
train_and_evaluate_model([64, 32], 0.01, 500, nn.ReLU)
train_and_evaluate_model([64, 32], 0.01, 1000, nn.ReLU)
train_and_evaluate_model([64, 32],  0.001, 100, nn.ReLU)
train_and_evaluate_model([64, 32],  0.001, 500, nn.ReLU)
train_and_evaluate_model([64, 32],  0.001, 1000, nn.ReLU)

# Model 2: 3 layers
train_and_evaluate_model([128, 64, 32], 0.1, 100, nn.Tanh)
train_and_evaluate_model([128, 64, 32], 0.1, 500, nn.Tanh)
train_and_evaluate_model([128, 64, 32], 0.1, 1000, nn.Tanh)
train_and_evaluate_model([128, 64, 32], 0.01, 100, nn.Tanh)
train_and_evaluate_model([128, 64, 32], 0.01, 500, nn.Tanh)
train_and_evaluate_model([128, 64, 32], 0.01, 1000, nn.Tanh)
train_and_evaluate_model([128, 64, 32],  0.001, 100, nn.Tanh)
train_and_evaluate_model([128, 64, 32],  0.001, 500, nn.Tanh)
train_and_evaluate_model([128, 64, 32],  0.001, 1000, nn.Tanh)

# Model 3: 4 layers
train_and_evaluate_model([256, 128, 64, 32] , 0.1, 100, nn.LeakyReLU)
train_and_evaluate_model([256, 128, 64, 32] , 0.1, 500, nn.LeakyReLU)
train_and_evaluate_model([256, 128, 64, 32] , 0.1, 1000, nn.LeakyReLU)
train_and_evaluate_model([256, 128, 64, 32] , 0.01, 100, nn.LeakyReLU)
train_and_evaluate_model([256, 128, 64, 32] , 0.01, 500, nn.LeakyReLU)
train_and_evaluate_model([256, 128, 64, 32] , 0.01, 1000, nn.LeakyReLU)
train_and_evaluate_model([256, 128, 64, 32] ,  0.001, 100, nn.LeakyReLU)
train_and_evaluate_model([256, 128, 64, 32] ,  0.001, 500, nn.LeakyReLU)
train_and_evaluate_model([256, 128, 64, 32] ,  0.001, 1000, nn.LeakyReLU)

# Print results
for result in results:
    print(result)

    results_df = pd.DataFrame(results)
results_xlsx = "results.xlsx"
results_df.to_excel(results_xlsx)

print("Results saved to", results_xlsx)

#en iyi model :
print(results_df.iloc[np.argmax(results_df.iloc[:,-2])])