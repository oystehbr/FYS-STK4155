from FF_Neural_Network import Neural_Network
from sklearn.datasets import load_breast_cancer
from sklearn.metrics import accuracy_score
from sklearn.decomposition import PCA
import pandas as pd
import numpy as np
# TODO: We need to use the accuracy score when looking at the classification problem


cancer = load_breast_cancer()

# Parameter labels
labels = cancer.feature_names[0:30]

# 569 rows (sample data), 30 columns (parameters)
X_cancer = cancer.data
# 569 rows (0 for benign and 1 for malignant)
y_cancer = cancer.target
y_cancer = y_cancer.reshape(-1, 1)

n_components = 2
pca = PCA(n_components=n_components)
X_cancer_2D = pca.fit_transform(X_cancer)

X_scalar = 1/np.max(X_cancer_2D)
X_cancer_2D_scaled = X_cancer_2D*X_scalar
no_hidden_nodes = 8
no_hidden_layers = 1

FFNN = Neural_Network(
    no_input_nodes=n_components,
    no_output_nodes=1,
    no_hidden_nodes=no_hidden_nodes,
    no_hidden_layers=no_hidden_layers
)
# ?? scaling
FFNN.set_SGD_values(
    n_epochs=10,
    batch_size=100,
    gamma=0.8,
    eta=0.01)

for i in range(1, 30):
    FFNN.set_activation_function_output_layer('sigmoid')
    FFNN.train_model(X_cancer_2D_scaled, y_cancer)
    # FFNN.plot_accuracy_score_last_training()
    FFNN.set_activation_function_output_layer('sigmoid_classification')
    print(
        f'Accuracy (after {i} training): = {accuracy_score(FFNN.feed_forward(X_cancer_2D_scaled),  y_cancer)}')


exit()
FFNN.set_activation_function_output_layer("sigmoid_classification")


exit()
print(pca.explained_variance_ratio_)
print(pca.components_)
print(pd.DataFrame(pca.components_,
      columns=X_cancer[0, :], index=['PC-1', 'PC-2']))


exit()
print("Eigenvector of largest eigenvalue")
print(pca.components_.T[:, 0])
