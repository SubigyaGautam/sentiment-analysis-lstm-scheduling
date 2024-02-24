import matplotlib.pyplot as plt
import numpy as np

# Provided data
factors = [0.50, 0.51, 0.52, 0.53, 0.54, 0.55]
TP = [10000, 9992, 9897, 9460, 7450, 2522]
FP = [9973, 9750, 8852, 6240, 2337, 389]
FN = [0, 8, 103, 540, 2250, 7478]
TN = [27, 250, 1148, 3760, 7663, 9611]
FPR = [0.99, 0.97, 0.88, 0.62, 0.23, 0.03]
Accuracy = [0.50, 0.51, 0.55, 0.66, 0.75, 0.60]
Precision = [0.50, 0.50, 0.52, 0.60, 0.76, 0.86]
Recall = [1.0, 0.99, 0.98, 0.94, 0.74, 0.25]
F1_Score = [0.66, 0.67, 0.68, 0.73, 0.75, 0.39]

# Line Graphs
plt.figure(figsize=(15, 15))

# True Positive (TP)
plt.subplot(3, 3, 1)
plt.plot(factors, TP, marker='o', linestyle='-', color='green')
plt.ylabel('TP')

# False Positive (FP)
plt.subplot(3, 3, 2)
plt.plot(factors, FP, marker='o', linestyle='-', color='red')
plt.ylabel('FP')

# False Negative (FN)
plt.subplot(3, 3, 3)
plt.plot(factors, FN, marker='o', linestyle='-', color='purple')
plt.ylabel('FN')

# True Negative (TN)
plt.subplot(3, 3, 4)
plt.plot(factors, TN, marker='o', linestyle='-', color='orange')
plt.ylabel('TN')


# False Positive Rate (FPR)
plt.subplot(3, 3, 5)
plt.plot(factors, FPR, marker='o', linestyle='-', color='red')
plt.ylabel('FPR')

# Accuracy
plt.subplot(3, 3, 6)
plt.plot(factors, Accuracy, marker='o', linestyle='-', color='blue')
plt.xlabel('Factors')
plt.ylabel('Accuracy')

# Precision
plt.subplot(3, 3, 7)
plt.plot(factors, Precision, marker='o', linestyle='-', color='purple')
plt.xlabel('Factors')
plt.ylabel('Precision')

# Recall
plt.subplot(3, 3, 8)
plt.plot(factors, Recall, marker='o', linestyle='-', color='orange')
plt.xlabel('Factors')
plt.ylabel('Recall')

# Recall
plt.subplot(3, 3, 9)
plt.plot(factors, F1_Score, marker='o', linestyle='-', color='black')
plt.xlabel('Factors')
plt.ylabel('F1_Score')


plt.tight_layout()
plt.show()
plt.savefig('All_metrics.png')