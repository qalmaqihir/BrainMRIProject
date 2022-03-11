import numpy as np
import openpyxl
import pandas as pd

# Loading the data
import xlsxwriter
from matplotlib import pyplot
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier



df = pd.read_excel("RedChannel.xlsx")
df.to_csv("Red_data.csv")

# Importing the Red dataset
red_data_features = pd.read_csv('Red_data.csv')
labels_file = pd.read_csv('Labels1.csv')
XR = red_data_features.iloc[:, 0:9].values
yR = labels_file.iloc[:, 0].values

# print(len(X))
# print(len(y))
# Training and Testing Data (divide the data into two part)

X_trainR, X_testR, y_trainR, y_testR =train_test_split(XR,yR,train_size=0.75, random_state=0)

# Normalize the values
from sklearn.preprocessing import StandardScaler
sc = StandardScaler()
X_trainR = sc.fit_transform(X_trainR)
X_testR = sc.transform(X_testR)

# Classifier and Training of the model
classifierR=DecisionTreeClassifier(criterion='entropy', random_state=0)
classifierR.fit(X_trainR,y_trainR)
# prediction of model on training data
y_predR_training = classifierR.predict(X_trainR)

# Prediction of the model on test data
y_predR_testing = classifierR.predict(X_testR)
# print("Testing Red predictions")
# print(len(y_predR_testing))
# print(y_predR_testing)


#################################################################################
df = pd.read_excel("GreenChannel.xlsx")
df.to_csv("Green_data.csv")

# Importing the Green dataset
green_data_features = pd.read_csv('Green_data.csv')
labels_file = pd.read_csv('Labels1.csv')
XG = green_data_features.iloc[:, 0:9].values
yG = labels_file.iloc[:, 0].values

# print(len(X))
# print(len(y))
# Training and Testing Data (divide the data into two part)
from sklearn.model_selection import train_test_split
X_trainG, X_testG, y_trainG, y_testG =train_test_split(XG,yG,train_size=0.75, random_state=0)

# Normalize the values
from sklearn.preprocessing import StandardScaler
sc = StandardScaler()
X_trainG = sc.fit_transform(X_trainG)
X_testG = sc.transform(X_testG)

# Classifier and Training of the model
from sklearn.tree import DecisionTreeClassifier
classifierG=DecisionTreeClassifier(criterion='entropy', random_state=0)
classifierG.fit(X_trainG,y_trainG)

# Prediction of the model on test data
y_predG_testing = classifierG.predict(X_testG)
# print("Testing Prediction Green")
# print(len(y_predG_testing))
# print(y_predG_testing)
#



#Prediction on training data
y_predG_training = classifierG.predict(X_trainG)
# print("Training Prediction Green")
# print(len(y_predG_training))
# print(y_predG_training)



#########################################################################
df = pd.read_excel("BlueChannel.xlsx")
df.to_csv("Blue_data.csv")

# Importing the  Blue dataset
blue_data_features = pd.read_csv('Blue_data.csv')
labels_file = pd.read_csv('Labels1.csv')
XB = blue_data_features.iloc[:, 0:9].values
yB = labels_file.iloc[:, 0].values

# print(len(X))
# print(len(y))
# Training and Testing Data (divide the data into two part)
from sklearn.model_selection import train_test_split
X_trainB, X_testB, y_trainB, y_testB =train_test_split(XB,yB,train_size=0.75, random_state=0)

# Normalize the values
from sklearn.preprocessing import StandardScaler
sc = StandardScaler()
X_trainB = sc.fit_transform(X_trainB)
X_testB = sc.transform(X_testB)

# Classifier and Training of the model
from sklearn.tree import DecisionTreeClassifier
classifierB=DecisionTreeClassifier(criterion='entropy', random_state=0)
classifierB.fit(X_trainB,y_trainB)

# Prediction of the model on test data
y_predB_testing = classifierB.predict(X_testB)
# print("Testing Prediction Blue")
# print(len(y_predB_testing))
# print(y_predB_testing)

#Prediction on training data
y_predB_training = classifierB.predict(X_trainB)
# print("Training Prediction Blue")
# print(len(y_predB_training))
# print(y_predB_training)


# from sklearn import metrics
#
# print("#########################################################################\n Testing Accuracies")
# print("Testing Blue Accuracy: ", metrics.accuracy_score(y_testB, y_predB_testing))
# print("Testing Blue  Precision: ", metrics.precision_score(y_testB,y_predB_testing))
# print("Testing Blue Recall: ",metrics.recall_score(y_testB,y_predB_testing))
#
# print("Testing Green Accuracy: ", metrics.accuracy_score(y_testG, y_predG_testing))
# print("Testing Green  Precision: ", metrics.precision_score(y_testG,y_predG_testing))
# print("Testing Green Recall: ",metrics.recall_score(y_testG,y_predG_testing))
#
# print("Testing Red Accuracy: ", metrics.accuracy_score(y_testR, y_predR_testing))
# print("Testing Red Precision: ", metrics.precision_score(y_testR,y_predR_testing))
# print("Testing Red  Recall: ",metrics.recall_score(y_testR,y_predR_testing))
# print("#########################################################################\n Training Accuracies")
# print("Training Green Accuracy: ", metrics.accuracy_score(y_trainG, y_predG_training))
# print("Training Green Precision: ", metrics.precision_score(y_trainG,y_predG_training))
# print("Training Green  Recall: ",metrics.recall_score(y_trainG,y_predG_training))
#
# print("Training Blue Accuracy: ", metrics.accuracy_score(y_trainB, y_predB_training))
# print("Training Blue Precision: ", metrics.precision_score(y_trainB,y_predB_training))
# print("Training Blue  Recall: ",metrics.recall_score(y_trainB,y_predB_training))
#
# print("Training Red Accuracy: ", metrics.accuracy_score(y_trainR, y_predR_training))
# print("Training Red Precision: ", metrics.precision_score(y_trainR,y_predR_training))
# print("Training Red  Recall: ",metrics.recall_score(y_trainR,y_predR_training))
# print("####################################################################################")
# Majority voting for Testing predictions


majority_votes_testing=np.zeros(y_predR_testing.shape,dtype=int)
print(len(majority_votes_testing))

for i in range(len(y_predR_testing)):
    if y_predR_testing[i]==y_predG_testing[i]:
        majority_votes_testing[i]=y_predR_testing[i]
    elif y_predR_testing[i]==y_predB_testing[i]:
        majority_votes_testing[i]=y_predR_testing[i]
    else:
        majority_votes_testing[i]=y_predG_testing[i]

print(f"After majority voting the final predictions for testing: \n{majority_votes_testing}")

# Majority voting for Testing predictions
majority_votes_training=np.zeros(y_predR_training.shape,dtype=int)
print(len(majority_votes_training))

for i in range(len(y_predR_training)):
    if y_predR_training[i]==y_predG_training[i]:
        majority_votes_training[i]=y_predR_training[i]
    elif y_predR_training[i]==y_predB_training[i]:
        majority_votes_training[i]=y_predR_training[i]
    else:
        majority_votes_training[i]=y_predG_training[i]

print(f"After majority voting the final predictions for training: \n{majority_votes_training}")

# Performance evaluation of the model for testing prediction
from sklearn.metrics import confusion_matrix
from sklearn.metrics import accuracy_score
accuracy_score(y_testR,majority_votes_testing)

# Making the Confusion Matrix
# import seaborn as sns
# cmtest = confusion_matrix(majority_votes_testing, y_testR)
# sns.heatmap(cmtest,annot=True)
# pyplot.savefig('DT_testing.png')
# accuracy_score(y_testR,majority_votes_testing)

from sklearn.metrics import classification_report
print(classification_report(y_testR,majority_votes_testing))

#Model evaluation
from sklearn import metrics
# precision=metrics.precision_score(y_testR,majority_votes_testing)
# accuracy=metrics.accuracy_score(y_testR, majority_votes_testing)
# recall=metrics.recall_score(y_testR,majority_votes_testing)
print("Testing Accuracy: ", metrics.accuracy_score(y_testR, majority_votes_testing))
print("Testing Precision: ", metrics.precision_score(y_testR,majority_votes_testing))
print("Testing Recall: ",metrics.recall_score(y_testR,majority_votes_testing))

# Performance evaluation of the model for training prediction
# Making the Confusion Matrix
import seaborn as sns
cmtrain = confusion_matrix(majority_votes_training, y_trainR)
sns.heatmap(cmtrain,annot=True)
pyplot.savefig('DT_training.png')

accuracy_score(y_trainR,majority_votes_training)

print(classification_report(y_trainR,majority_votes_training))

#Model evaluation
# precision=metrics.precision_score(y_trainR,majority_votes_training)
# accuracy=metrics.accuracy_score(y_trainR, majority_votes_training)
# recall=metrics.recall_score(y_trainR,majority_votes_training)
print("Training Accuracy: ", metrics.accuracy_score(y_trainR, majority_votes_training))
print("Training Precision: ", metrics.precision_score(y_trainR,majority_votes_training))
print("Training Recall: ",metrics.recall_score(y_trainR,majority_votes_training))



# values=[accuracy,precision,recall]
# # Saving the predictions to file
# # Create Red File
# outWorkbook = xlsxwriter.Workbook("Models_Predictions.xlsx")
# outSheet = outWorkbook.add_worksheet()
# # Write Headers
# outSheet.write("B1","Decision Tree")
# outSheet.write("A2","Accuracy")
# outSheet.write("A3","Precision")
# outSheet.write("A4","Recall")
# r=1
# c=1
# for item in values:
#     outSheet.write(r,c,item)
#     r+=1
#
#
# outWorkbook.close()


import pickle

pkl_file_nameR="model_dtRed.pkl"
with open(pkl_file_nameR,'wb') as fileR:
    pickle.dump(classifierR, fileR)

pkl_file_nameG="model_dtGreen.pkl"
with open(pkl_file_nameG,'wb') as fileG:
    pickle.dump(classifierG, fileG)

pkl_file_nameB="model_dtBlue.pkl"
with open(pkl_file_nameB,'wb') as fileB:
    pickle.dump(classifierB, fileB)

