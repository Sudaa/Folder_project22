# Support Vector Machine
# Importing the libraries

import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns  # For Confusion Matrix visualization
from sklearn.metrics import confusion_matrix, accuracy_score, classification_report
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC
from matplotlib.colors import ListedColormap

# Importing the dataset

datasets = pd.read_csv('Social_Network_Ads.csv')
X = datasets.iloc[:, [2, 3]].values
Y = datasets.iloc[:, 4].values

# Splitting the dataset into the Training set and Test set

X_Train, X_Test, Y_Train, Y_Test = train_test_split(X, Y, test_size=0.25, random_state=0)

# Feature Scaling

sc_X = StandardScaler()
X_Train = sc_X.fit_transform(X_Train)
X_Test = sc_X.transform(X_Test)

# Fitting the classifier into the Training set

classifier = SVC(kernel='linear', random_state=0)
classifier.fit(X_Train, Y_Train)

# Predicting the test set results

Y_Pred = classifier.predict(X_Test)

# Making the Confusion Matrix 

cm = confusion_matrix(Y_Test, Y_Pred)

# Displaying the Classification Report
classification_rep = classification_report(Y_Test, Y_Pred)
accuracy = accuracy_score(Y_Test, Y_Pred)

# 🔹 **Plotting All Visualizations in One Window**
fig, axes = plt.subplots(1, 3, figsize=(18, 5))  # Create a figure with 3 subplots in one row

# **Plot 1: Confusion Matrix**
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=['Not Purchased', 'Purchased'], 
            yticklabels=['Not Purchased', 'Purchased'], ax=axes[0])
axes[0].set_title('Confusion Matrix')
axes[0].set_xlabel('Predicted')
axes[0].set_ylabel('True')

# **Plot 2: Training Set Visualization**
X_Set, Y_Set = X_Train, Y_Train
X1, X2 = np.meshgrid(np.arange(start=X_Set[:, 0].min() - 1, stop=X_Set[:, 0].max() + 1, step=0.01),
                     np.arange(start=X_Set[:, 1].min() - 1, stop=X_Set[:, 1].max() + 1, step=0.01))
axes[1].contourf(X1, X2, classifier.predict(np.array([X1.ravel(), X2.ravel()]).T).reshape(X1.shape),
                 alpha=0.75, cmap=ListedColormap(('red', 'green')))
axes[1].set_xlim(X1.min(), X1.max())
axes[1].set_ylim(X2.min(), X2.max())
for i, j in enumerate(np.unique(Y_Set)):
    axes[1].scatter(X_Set[Y_Set == j, 0], X_Set[Y_Set == j, 1],
                    color=ListedColormap(('red', 'green'))(i), label=j, edgecolors='k')
axes[1].set_title('SVM (Training Set)')
axes[1].set_xlabel('Age')
axes[1].set_ylabel('Estimated Salary')
axes[1].legend()

# **Plot 3: Test Set Visualization**
X_Set, Y_Set = X_Test, Y_Test
X1, X2 = np.meshgrid(np.arange(start=X_Set[:, 0].min() - 1, stop=X_Set[:, 0].max() + 1, step=0.01),
                     np.arange(start=X_Set[:, 1].min() - 1, stop=X_Set[:, 1].max() + 1, step=0.01))
axes[2].contourf(X1, X2, classifier.predict(np.array([X1.ravel(), X2.ravel()]).T).reshape(X1.shape),
                 alpha=0.75, cmap=ListedColormap(('red', 'green')))
axes[2].set_xlim(X1.min(), X1.max())
axes[2].set_ylim(X2.min(), X2.max())
for i, j in enumerate(np.unique(Y_Set)):
    axes[2].scatter(X_Set[Y_Set == j, 0], X_Set[Y_Set == j, 1],
                    color=ListedColormap(('red', 'green'))(i), label=j, edgecolors='k')
axes[2].set_title('SVM (Test Set)')
axes[2].set_xlabel('Age')
axes[2].set_ylabel('Estimated Salary')
axes[2].legend()

# Adjust layout for clarity
plt.tight_layout()

# Show all plots at once without blocking execution
# plt.show(block=False)  # **Prevents blocking further execution**
# plt.pause(2)  # **Wait 2 seconds to ensure all plots render**
# plt.close()  # **Close the plot so the console output is visible**

plt.show()

# 🔹 **Print Classification Report & Accuracy**
print("\n🔹 Classification Report:\n", classification_rep)
print(f'\n🔹 Accuracy of the model: {accuracy * 100:.2f}%')
