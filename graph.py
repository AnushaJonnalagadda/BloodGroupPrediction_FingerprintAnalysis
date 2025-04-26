import matplotlib.pyplot as plt

# Training data percentages
data_sizes = [10, 30, 50, 70, 100]

# Accuracy values
accuracy_svm = [45, 55, 60, 63, 65]
accuracy_rf = [50, 60, 65, 68, 70]
accuracy_knn = [40, 50, 58, 60, 62]
accuracy_resnet50 = [60, 63, 67, 69, 70]
accuracy_densenet121 = [70, 76, 82, 86, 90]

# Plot
plt.figure(figsize=(10, 6))

# Plot each algorithm
def plot_with_labels(data_sizes, accuracies, label, marker):
    plt.plot(data_sizes, accuracies, marker=marker, label=label)

plot_with_labels(data_sizes, accuracy_svm, 'SVM', 'o')
plot_with_labels(data_sizes, accuracy_rf, 'Random Forest', 's')
plot_with_labels(data_sizes, accuracy_knn, 'KNN', '^')
plot_with_labels(data_sizes, accuracy_resnet50, 'ResNet50', 'd')
plot_with_labels(data_sizes, accuracy_densenet121, 'DenseNet121', 'v')

plt.title('Accuracy Trend Comparison of Algorithms')
plt.xlabel('Training Data Size (%)')
plt.ylabel('Accuracy (%)')
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.show()
