# confusion_matrix_demo.py

import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix, classification_report

# -----------------------------
# Step 1: Define sample labels
# -----------------------------
true_labels = [2, 0, 0, 2, 4, 4, 1, 0, 3, 3, 3]
pred_labels = [2, 1, 0, 2, 4, 3, 1, 0, 1, 3, 3]

# -----------------------------
# Step 2: Create confusion matrix
# -----------------------------
confusion_mat = confusion_matrix(true_labels, pred_labels)

# -----------------------------
# Step 3: Visualize confusion matrix
# -----------------------------
plt.figure(figsize=(6, 5))
plt.imshow(confusion_mat, interpolation='nearest', cmap=plt.cm.gray)
plt.title('Confusion Matrix')
plt.colorbar()

num_classes = len(np.unique(true_labels))
ticks = np.arange(num_classes)
plt.xticks(ticks, ticks)
plt.yticks(ticks, ticks)

plt.ylabel('True Labels')
plt.xlabel('Predicted Labels')
plt.show()

# -----------------------------
# Step 4: Print classification report
# -----------------------------
target_names = ['Class-0', 'Class-1', 'Class-2', 'Class-3', 'Class-4']
print('\nClassification Report:\n')
print(classification_report(true_labels, pred_labels, target_names=target_names))
