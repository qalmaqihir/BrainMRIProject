import numpy as np
import openpyxl
import pandas as pd

# Loading the data
import xlsxwriter
from matplotlib import pyplot

df = pd.read_excel("RedChannel.xlsx")
df.to_csv("Red_data.csv")

# Importing the Red dataset
red_data_features = pd.read_csv('Red_data.csv')
labels_file = pd.read_csv('Labels1.csv')
X = red_data_features.iloc[:, 0:9].values
y = labels_file.iloc[:, 0].values

# print(len(X))
# print(len(y))
# Training and Testing Data (divide the data into two part)
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test =train_test_split(X,y,train_size=0.75, random_state=0)

# Normalize the values
from sklearn.preprocessing import StandardScaler
sc = StandardScaler()
X_train = sc.fit_transform(X_train)
X_test = sc.transform(X_test)

# Classifier and Training of the model
from sklearn.tree import DecisionTreeClassifier
classifier=DecisionTreeClassifier(criterion='entropy', random_state=0)
classifier.fit(X_train,y_train)

# Prediction of the model on test data
y_predR = classifier.predict(X_test)
print(len(y_predR))
print(y_predR)

#################################################################################
df = pd.read_excel("GreenChannel.xlsx")
df.to_csv("Green_data.csv")

# Importing the Green dataset
red_data_features = pd.read_csv('Green_data.csv')
labels_file = pd.read_csv('Labels1.csv')
X = red_data_features.iloc[:, 0:9].values
y = labels_file.iloc[:, 0].values

# print(len(X))
# print(len(y))
# Training and Testing Data (divide the data into two part)
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test =train_test_split(X,y,train_size=0.75, random_state=0)

# Normalize the values
from sklearn.preprocessing import StandardScaler
sc = StandardScaler()
X_train = sc.fit_transform(X_train)
X_test = sc.transform(X_test)

# Classifier and Training of the model
from sklearn.tree import DecisionTreeClassifier
classifier=DecisionTreeClassifier(criterion='entropy', random_state=0)
classifier.fit(X_train,y_train)

# Prediction of the model on test data
y_predG = classifier.predict(X_test)
print(len(y_predG))
print(y_predG)

#########################################################################
df = pd.read_excel("BlueChannel.xlsx")
df.to_csv("Blue_data.csv")

# Importing the  Blue dataset
red_data_features = pd.read_csv('Blue_data.csv')
labels_file = pd.read_csv('Labels1.csv')
X = red_data_features.iloc[:, 0:9].values
y = labels_file.iloc[:, 0].values

# print(len(X))
# print(len(y))
# Training and Testing Data (divide the data into two part)
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test =train_test_split(X,y,train_size=0.75, random_state=0)

# Normalize the values
from sklearn.preprocessing import StandardScaler
sc = StandardScaler()
X_train = sc.fit_transform(X_train)
X_test = sc.transform(X_test)

# Classifier and Training of the model
from sklearn.tree import DecisionTreeClassifier
classifier=DecisionTreeClassifier(criterion='entropy', random_state=0)
classifier.fit(X_train,y_train)

# Prediction of the model on test data
y_predB = classifier.predict(X_test)
print(len(y_predB))
print(y_predB)


majority_votes=np.zeros(y_predR.shape,dtype=int)
print(len(majority_votes))

for i in range(len(y_predR)):
    if y_predR[i]==y_predG[i]:
        majority_votes[i]=y_predR[i]
    elif y_predR[i]==y_predB[i]:
        majority_votes[i]=y_predR[i]
    else:
        majority_votes[i]=y_predG[i]

print(f"After majority voting the final predictions: \n{majority_votes}")


# Performance evaluation of the model
from sklearn.metrics import confusion_matrix
cm = confusion_matrix(y_test,majority_votes)

from sklearn.metrics import accuracy_score
accuracy_score(y_test,majority_votes)

# Making the Confusion Matrix
import seaborn as sns
cm = confusion_matrix(majority_votes, y_test)
sns.heatmap(cm,annot=True)
pyplot.savefig('DT.png')
print(cm)

from sklearn.metrics import accuracy_score
accuracy_score(y_test,majority_votes)

from sklearn.metrics import classification_report
print(classification_report(y_test,majority_votes))

# Saving the predictions to file
# Create Red File
outWorkbook = xlsxwriter.Workbook("Models_Predictions.xlsx")
outSheet = outWorkbook.add_worksheet()
# Write Headers
outSheet.write("A1", "DT Algorithm")
outSheet.write("A2", "Red")
outSheet.write("B2", "Green")
outSheet.write("C2", "Blue")
outSheet.write("D2", "Majority Voting")
outWorkbook.close()

# r=3
# c=1
# for i in range(len(y_predR)):
#     y_r=y_predR[i]
#     y_g=y_predG[i]
#     y_b=y_predB[i]
#     m_v=majority_votes[i]
#     predics=[y_r,y_g,y_b,m_v]
#
#     # Write data to file
#     outWorkbook = openpyxl.load_workbook("Models_Predictions.xlsx")
#     outSheet = outWorkbook.active
#     outSheet.append(predics)
#     # outSheet.cell(r,c+1,y_r)
#     # outSheet.cell(r, c+2, y_g)
#     # outSheet.cell(r, c+3, y_b)
#     # outSheet.cell(r, c+4,majority_votes)
#     # r+=1
#
