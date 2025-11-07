import numpy as np
from sklearn.naive_bayes import GaussianNB
from sklearn.model_selection import train_test_split, cross_val_score
import matplotlib.pyplot as plt

# Utility to visualize classifier (create this file or function)
def visualize_classifier(classifier, X, y):
    x_min, x_max = X[:, 0].min() - 1, X[:, 0].max() + 1
    y_min, y_max = X[:, 1].min() - 1, X[:, 1].max() + 1
    xx, yy = np.meshgrid(np.arange(x_min, x_max, 0.1),
                         np.arange(y_min, y_max, 0.1))
    Z = classifier.predict(np.c_[xx.ravel(), yy.ravel()])
    Z = Z.reshape(xx.shape)
    plt.contourf(xx, yy, Z, alpha=0.3, cmap='rainbow')
    markers = ['o', 's', '^', 'x']
    for idx, cl in enumerate(np.unique(y)):
        plt.scatter(X[y == cl, 0], X[y == cl, 1],
                    marker=markers[idx], label=f'Class {cl}')
    plt.xlabel('Feature 1')
    plt.ylabel('Feature 2')
    plt.legend()
    plt.show()

# Load data (replace with your file)
input_file = 'data_multivar_nb.txt'  # CSV file with features + last column = labels
data = np.loadtxt(input_file, delimiter=',')

X, y = data[:, :-1], data[:, -1]

# 1️⃣ Train a simple Naïve Bayes classifier
classifier = GaussianNB()
classifier.fit(X, y)

# Predict on the training data
y_pred = classifier.predict(X)

# Accuracy
accuracy = 100.0 * (y == y_pred).sum() / X.shape[0]
print("Accuracy of Naïve Bayes classifier =", round(accuracy, 2), "%")

# Visualize
visualize_classifier(classifier, X, y)

# 2️⃣ Split data for training/testing
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=3)
classifier_new = GaussianNB()
classifier_new.fit(X_train, y_train)
y_test_pred = classifier_new.predict(X_test)

accuracy = 100.0 * (y_test == y_test_pred).sum() / X_test.shape[0]
print("Accuracy of the new classifier =", round(accuracy, 2), "%")
visualize_classifier(classifier_new, X_test, y_test)

# 3️⃣ Cross-validation
num_folds = 3
accuracy_values = cross_val_score(classifier, X, y, scoring='accuracy', cv=num_folds)
precision_values = cross_val_score(classifier, X, y, scoring='precision_weighted', cv=num_folds)
recall_values = cross_val_score(classifier, X, y, scoring='recall_weighted', cv=num_folds)
f1_values = cross_val_score(classifier, X, y, scoring='f1_weighted', cv=num_folds)

print(f"Cross-validated Accuracy: {round(100*accuracy_values.mean(),2)}%")
print(f"Cross-validated Precision: {round(100*precision_values.mean(),2)}%")
print(f"Cross-validated Recall: {round(100*recall_values.mean(),2)}%")
print(f"Cross-validated F1: {round(100*f1_values.mean(),2)}%")
