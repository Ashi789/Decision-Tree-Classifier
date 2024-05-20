import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score, ConfusionMatrixDisplay
from sklearn.tree import plot_tree
import matplotlib.pyplot as plt


# Load the data
df = pd.read_excel(r'C:\Users\Main Profile\PORDEGY\OnlineRetail.xlsx')

# Data Cleaning
df.dropna(subset=['CustomerID'], inplace=True)
df.drop(['InvoiceNo', 'StockCode', 'Description', 'InvoiceDate'], axis=1, inplace=True)

# Feature Engineering
df['CustomerID'] = df['CustomerID'].astype('int')
customer_df = df.groupby('CustomerID').agg({'Quantity': 'sum', 'UnitPrice': 'mean'}).reset_index()
threshold_quantity = 10
customer_df['Purchase'] = np.where(customer_df['Quantity'] > threshold_quantity, 1, 0)

# Split Data
X = customer_df[['Quantity', 'UnitPrice']]
y = customer_df['Purchase']
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Train the Model
clf = DecisionTreeClassifier(random_state=42)
clf.fit(X_train, y_train)
y_pred = clf.predict(X_test)

# Evaluate the Model
conf_matrix = confusion_matrix(y_test, y_pred)
print("Confusion Matrix:")
print(conf_matrix)
disp = ConfusionMatrixDisplay(confusion_matrix=conf_matrix, display_labels=clf.classes_)
disp.plot(cmap=plt.cm.Blues)
plt.show()

class_report = classification_report(y_test, y_pred)
print("\nClassification Report:")
print(class_report)

accuracy = accuracy_score(y_test, y_pred)
print("\nAccuracy:", accuracy)

# Visualize the Decision Tree
plt.figure(figsize=(10,10))
plot_tree(clf, feature_names=['Quantity', 'UnitPrice'], class_names=['Not Purchase', 'Purchase'], filled=True)
plt.show()
