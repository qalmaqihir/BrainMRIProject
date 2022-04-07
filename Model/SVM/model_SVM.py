import numpy as np
import openpyxl
import pandas as pd

# Loading the data
import xlsxwriter
from matplotlib import pyplot

#
##############################################################################################
############################# Red Channel model ##############################################
##############################################################################################
# Importing the Red dataset
red_data_features = pd.read_csv('../../Red_data.csv')
labels_file = pd.read_csv('../../Labels1.csv')
XR = red_data_features.iloc[:, 0:9].values
yR = labels_file.iloc[:, 0].values

# print(len(X))
# print(len(y))
# Training and Testing Data (divide the data into two part)
from sklearn.model_selection import train_test_split
X_trainR, X_testR, y_trainR, y_testR =train_test_split(XR,yR,test_size=0.3, random_state=109)

# Import svm model
from sklearn import svm

clfR = svm.SVC(kernel='linear')

# train the model
clfR.fit(X_trainR,y_trainR)

# predicting the reponse for test dataset
y_predR_testing=clfR.predict(X_testR)
print("Testing Red predictions")
print(len(y_predR_testing))
print(y_predR_testing)

y_predR_training=clfR.predict(X_trainR)
print("Traning Red predictions")
print(len(y_predR_training))
print(y_predR_training)


#
# #Model evaluation
# from sklearn import metrics
# print("Accuracy: ", metrics.accuracy_score(y_testR, y_predR))
#
# print("Precision: ", metrics.precision_score(y_testR,y_predR))
#
# print("Recall: ",metrics.recall_score(y_testR,y_predR))


##############################################################################################
############################ Green Channel model ############################################
##############################################################################################
# Importing the Green dataset
green_data_features = pd.read_csv('../../Green_data.csv')
labels_file = pd.read_csv('../../Labels1.csv')
XG = green_data_features.iloc[:, 0:9].values
yG = labels_file.iloc[:, 0].values

# print(len(X))
# print(len(y))
# Training and Testing Data (divide the data into two part)
from sklearn.model_selection import train_test_split
X_trainG, X_testG, y_trainG, y_testG =train_test_split(XG,yG,test_size=0.3, random_state=109)

# Import svm model
from sklearn import svm

clfG = svm.SVC(kernel='linear')

# train the model
clfG.fit(X_trainG,y_trainG)

# predicting the reponse for test dataset
y_predG_testing=clfG.predict(X_testG)
print("Testing Green Prediction")
print(len(y_predG_testing))
print(y_predG_testing)

y_predG_training=clfG.predict(X_trainG)
print("Traning Green predictions")
print(len(y_predG_training))
print(y_predG_training)

# #Model evaluation
# from sklearn import metrics
# print("Accuracy: ", metrics.accuracy_score(y_testG, y_predG))
#
# print("Precision: ", metrics.precision_score(y_testG,y_predG))
#
# print("Recall: ",metrics.recall_score(y_testG,y_predG))



##############################################################################################
############################ Blue Channel model ############################################
##############################################################################################
# Importing the Blue dataset
blue_data_features = pd.read_csv('../../Blue_data.csv')
labels_file = pd.read_csv('../../Labels1.csv')
XB = blue_data_features.iloc[:, 0:9].values
yB = labels_file.iloc[:, 0].values

# print(len(X))
# print(len(y))
# Training and Testing Data (divide the data into two part)
from sklearn.model_selection import train_test_split
X_trainB, X_testB, y_trainB, y_testB =train_test_split(XB,yB,test_size=0.3, random_state=109)

# Import svm model
from sklearn import svm

clfB = svm.SVC(kernel='linear')

# train the model
clfB.fit(X_trainB,y_trainB)

# predicting the reponse for test dataset
y_predB_testing=clfB.predict(X_testB)
print("Testing Blue prediction")
print(len(y_predB_testing))
print(y_predB_testing)

y_predB_training=clfR.predict(X_trainB)
print("Traning Blue predictions")
print(len(y_predB_training))
print(y_predB_training)

#Model evaluation
# from sklearn import metrics
# print("Accuracy: ", metrics.accuracy_score(y_testB, y_predB))
#
# print("Precision: ", metrics.precision_score(y_testB,y_predB))
#
# print("Recall: ",metrics.recall_score(y_testB,y_predB))



# Majority voting for testing prediction
majority_votes_testing=np.zeros(y_predR_testing.shape,dtype=int)
print(len(majority_votes_testing))

for i in range(len(y_predR_testing)):
    if y_predR_testing[i]==y_predG_testing[i]:
        majority_votes_testing[i]=y_predR_testing[i]
    elif y_predR_testing[i]==y_predB_testing[i]:
        majority_votes_testing[i]=y_predR_testing[i]
    else:
        majority_votes_testing[i]=y_predG_testing[i]

print(f"After majority voting the final testing predictions: \n{majority_votes_testing}")


# Majority voting for testing prediction
majority_votes_training=np.zeros(y_predR_training.shape,dtype=int)
print(len(majority_votes_training))

for i in range(len(y_predR_training)):
    if y_predR_training[i]==y_predG_training[i]:
        majority_votes_training[i]=y_predR_training[i]
    elif y_predR_training[i]==y_predB_training[i]:
        majority_votes_training[i]=y_predR_training[i]
    else:
        majority_votes_training[i]=y_predG_training[i]

print(f"After majority voting the final training predictions: \n{majority_votes_training}")


# Performance evaluation of the model testing
from sklearn.metrics import confusion_matrix
cm = confusion_matrix(y_testR,majority_votes_testing)

from sklearn.metrics import accuracy_score
accuracy_score(y_testR,majority_votes_testing)

# Making the Confusion Matrix
import seaborn as sns
cm = confusion_matrix(majority_votes_testing, y_testR)
sns.heatmap(cm,annot=True)
pyplot.savefig('SVM_testing.png')
print(cm)

from sklearn.metrics import accuracy_score
accuracy_score(y_testR,majority_votes_testing)

from sklearn.metrics import classification_report
print(classification_report(y_testR,majority_votes_testing))

#Model evaluation
from sklearn import metrics
precision=metrics.precision_score(y_testR,majority_votes_testing)
accuracy=metrics.accuracy_score(y_testR, majority_votes_testing)
recall=metrics.recall_score(y_testR,majority_votes_testing)

print("Testing Accuracy: ", metrics.accuracy_score(y_testR, majority_votes_testing))
print("Testing Precision: ", metrics.precision_score(y_testR,majority_votes_testing))
print("Testing Recall: ",metrics.recall_score(y_testR,majority_votes_testing))


# Performance evaluation of the model training
from sklearn.metrics import confusion_matrix
cm = confusion_matrix(y_trainR,majority_votes_training)

from sklearn.metrics import accuracy_score
accuracy_score(y_trainR,majority_votes_training)

# Making the Confusion Matrix
import seaborn as sns
cm = confusion_matrix(majority_votes_training, y_trainR)
sns.heatmap(cm,annot=True)
pyplot.savefig('SVM_training.png')
print(cm)

from sklearn.metrics import accuracy_score
accuracy_score(y_trainR,majority_votes_training)

from sklearn.metrics import classification_report
print(classification_report(y_trainR,majority_votes_training))

#Model evaluation
from sklearn import metrics
precision=metrics.precision_score(y_trainR,majority_votes_training)
accuracy=metrics.accuracy_score(y_trainR, majority_votes_training)
recall=metrics.recall_score(y_trainR,majority_votes_training)

print("Training Accuracy: ", metrics.accuracy_score(y_trainR, majority_votes_training))
print("Training Precision: ", metrics.precision_score(y_trainR,majority_votes_training))
print("Training Recall: ",metrics.recall_score(y_trainR,majority_votes_training))


# values=[accuracy,precision,recall]
# # Saving the predictions to file
# outWorkbook = openpyxl.load_workbook("Models_Predictions.xlsx")
# outSheet = outWorkbook.active
# # Write Headers
# outSheet.write("C1","SVM")
#
# r=1
# c=3
# for item in values:
#     outSheet.write(r,c,item)
#     r+=1
#
# outWorkbook.close()
# print("Green")
# print("Accuracy: ", metrics.accuracy_score(y_testG, majority_votes))
# print("Precision: ", metrics.precision_score(y_testG,majority_votes))
# print("Recall: ",metrics.recall_score(y_testG,majority_votes))
#
# print("Blue")
# print("Accuracy: ", metrics.accuracy_score(y_testB, majority_votes))
# print("Precision: ", metrics.precision_score(y_testB,majority_votes))
# print("Recall: ",metrics.recall_score(y_testB,majority_votes))


# Link www.datacamp.com/community/tutorials/svm-classification-scikit-learn-python

# Saving the model for future use
import pickle

pkl_file_nameR="model_svmRed.pkl"
with open(pkl_file_nameR,'wb') as fileR:
    pickle.dump(clfR, fileR)

pkl_file_nameG="model_svmfGreen.pkl"
with open(pkl_file_nameG,'wb') as fileG:
    pickle.dump(clfG, fileG)

pkl_file_nameB="model_svmBlue.pkl"
with open(pkl_file_nameB,'wb') as fileB:
    pickle.dump(clfB, fileB)