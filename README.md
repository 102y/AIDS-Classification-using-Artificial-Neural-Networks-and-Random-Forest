# AIDS-Classification-using-Artificial-Neural-Networks-and-Random-Forest
This project focuses on classifying data related to AIDS using Artificial Neural Networks (ANNs) built with PyTorch and Random Forest classifiers. The main aim is to explore different configurations for training ANNs and compare their performance to traditional machine learning models like Random Forest.
Key Features:
Dataset: A classification dataset used for identifying patterns related to AIDS, loaded and preprocessed with pandas and numpy.
Data Preprocessing: The dataset is scaled using MinMaxScaler, and split into training and testing sets using train_test_split.
ANN Models: Multiple ANN architectures with varying hidden layers, neurons, and activation functions (ReLU, Tanh, LeakyReLU) were created. Each configuration is trained with different learning rates and epochs.
Random Forest Classifier: A Random Forest model is trained for comparison with the ANN models.
Training Process: The ANN models are trained using binary cross-entropy loss (BCELoss) and the Adam optimizer. The performance is evaluated based on accuracy.
Results: The best performing models based on accuracy are saved in an Excel file, with details about the learning rate, epochs, hidden layers, and activation functions.
Project Workflow:
Data Loading and Preprocessing: The dataset is loaded, preprocessed, and a correlation heatmap is generated for feature analysis.
Model Training: Several ANN models are trained with different combinations of hyperparameters, and their accuracy and training times are recorded.
Comparison with Random Forest: A Random Forest classifier is trained for benchmarking the results.
Results Evaluation: The results are saved to an Excel file, and the best performing ANN model is highlighted.
Technologies Used:
Python
PyTorch
Scikit-learn
Pandas
Seaborn & Matplotlib (for data visualization)
How to Run:
Clone the repository.
Install the required dependencies using pip install -r requirements.txt.
Run the Python script to start training the models.
Review the results saved in the output Excel file.
