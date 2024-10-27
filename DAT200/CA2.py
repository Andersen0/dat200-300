# -*- coding: utf-8 -*-
"""
Created on Wed Mar 1 12:20:13 2023
@author: Tage Andersen
"""

#%% Imports
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from perceptron import Perceptron as Perceptron
from adaline import AdalineGD as Adaline

#%% Loading dataset
raw_df = pd.read_csv("wine.csv", sep=",") # Loading the dataframe from wine.csv

# Separating features and target variable
X = raw_df.iloc[:, :-1]
y = raw_df.iloc[:, -1]

# Splitting dataset into train and test sets
X_train = X[:400]
y_train = y[:400]
X_test = X[400:]
y_test = y[400:]

#%% Normalizing (scaling) the data:
X_train_mean = np.mean(X_train, axis=0)
X_train_std = np.std(X_train, axis=0, ddof=1)
X_train_sc = (X_train - X_train_mean) / X_train_std
X_test_sc = (X_test - X_train_mean) / X_train_std
# Repairing the scaled 'class' column:
y_train.loc[y_train > 0] = 1
y_train.loc[y_train < 0] = -1
y_test.loc[y_test > 0] = 1
y_test.loc[y_test < 0] = -1

#%% Visualizing the scaled data:
plt.figure(figsize=(14, 5))
sns.violinplot(data=X_train_sc,
rotation=45,
dodge=True, # separate plots of different colors
width=0.5, # width of plots
palette = "bright"
)
plt.title("Violin plot")
plt.show()

#%% Training with sets of different sample sizes
ada_accuracies = np.zeros((8, 50))
ppn_accuracies = np.zeros((8, 50))
for i in range(50, 401, 50):
    # Making subsets from training sets
    subset_x_train = X_train_sc[:i]
    subset_y_train = y_train[:i]

for j in range(50):
    # Create Adaline and Perceptron classes
    ada = Adaline(eta=0.0001, n_iter=j, random_state=1)
    ppn = Perceptron(eta=0.0001, n_iter=j)
    ada.fit(subset_x_train.values, subset_y_train.values)
    ppn.fit(subset_x_train.values, subset_y_train.values)
    # Making predictions on the test data
    y_pred_ppn = ppn.predict(X_test_sc.values)
    y_pred_ada = ada.predict(X_test_sc.values)

# The accuracy of the models on the test data
ada_accuracy = (y_test.values == y_pred_ada).sum() / len(y_test)
ppn_accuracy = (y_test.values == y_pred_ppn).sum() / len(y_test)
ada_accuracies[int(i/50)-1, j] = ada_accuracy
ppn_accuracies[int(i/50)-1, j] = ppn_accuracy

#%% Plot heatmap for Perceptron
plt.figure(figsize=(15, 7.5))
ax = sns.heatmap(ppn_accuracies)
ax.set_xticklabels(np.arange(1, 51, 1))
ax.set_yticklabels(np.arange(50, 401, 50)[::-1])
plt.xlabel("Epochs")
plt.ylabel("Number of samples")
plt.title("Perceptron Classification Accuracy")

# Plot heatmap for Adaline
plt.figure(figsize=(15, 7.5))
ax = sns.heatmap(ada_accuracies)
ax.set_xticklabels(np.arange(1, 51, 1))
ax.set_yticklabels(np.arange(50, 401, 50)[::-1])
plt.xlabel("Epochs")
plt.ylabel("Number of samples")
plt.title("Adaline Classification Accuracy")

#%% Task 4
ppn_max_acc = np.amax(ppn_accuracies)
ada_max_acc = np.amax(ada_accuracies)
ppn_best_sample_epoch = np.unravel_index(ppn_accuracies.argmax(),
ppn_accuracies.shape)
ppn_sample_size = (ppn_best_sample_epoch[0] + 1) * 50
ppn_epochs = ppn_best_sample_epoch[1]
ada_best_sample_epoch = np.unravel_index(ada_accuracies.argmax(),
ada_accuracies.shape)
ada_sample_size = (ada_best_sample_epoch[0] + 1) * 50
ada_epochs = ada_best_sample_epoch[1]

print("Perceptron classifier:")
print("Highest test set classification accuracy: {:.2f}%".format(ppn_max_acc * 100))
print("Best combination of sample size and epochs: {} samples, {} epochs".format(ppn_sample_size, ppn_epochs))
print("\nAdaline classifier:")
print("Highest test set classification accuracy: {:.2f}%".format(ada_max_acc *
100))
print("Best combination of sample size and epochs: {} samples, {} epochs".format(ada_sample_size, ada_epochs))
