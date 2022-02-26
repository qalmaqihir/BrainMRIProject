import numpy as np
import pandas as pd
from matplotlib import pyplot
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn import metrics
##############################################################################################
############################# Red Channel model ##############################################
##############################################################################################
# Importing the Red dataset

red_data_features = pd.read_csv('Red_data.csv')
labels_file = pd.read_csv('Labels1.csv')
XR = red_data_features.iloc[:, 0:9].values
yR = labels_file.iloc[:, 0].values


X_trainR, X_testR, y_trainR, y_testR =train_test_split(XR,yR,test_size=0.3, random_state=109)


clfR=RandomForestClassifier(n_estimators=100,max_depth=10)
clfR.fit(X_trainR,y_trainR)
y_predR_testing = clfR.predict(X_testR)
y_predR_training=clfR.predict(X_trainR)

# printing the score report testing data
#Model evaluation
#
# print("Testing Red Accuracy: ", metrics.accuracy_score(y_testR, y_predR_testing))
# print("Testing Red Precision: ", metrics.precision_score(y_testR,y_predR_testing))
# print("Testing Red Recall: ",metrics.recall_score(y_testR,y_predR_testing))

# printing the score report traning data
#Model evaluation
# print("Training Red Accuracy: ", metrics.accuracy_score(y_trainR, y_predR_training))
# print("Training Red Precision: ", metrics.precision_score(y_trainR,y_predR_training))
# print("Training Red Recall: ",metrics.recall_score(y_trainR,y_predR_training))


##############################################################################################
############################# Green Channel model ##############################################
##############################################################################################
# Importing the Green dataset

green_data_features = pd.read_csv('Green_data.csv')
labels_file = pd.read_csv('Labels1.csv')
XG = green_data_features.iloc[:, 0:9].values
yG = labels_file.iloc[:, 0].values


X_trainG, X_testG, y_trainG, y_testG =train_test_split(XG,yG,test_size=0.3, random_state=109)


clfG=RandomForestClassifier(n_estimators=100,max_depth=10)
clfG.fit(X_trainG,y_trainG)
y_predG_testing = clfG.predict(X_testG)
y_predG_training=clfG.predict(X_trainG)


# printing the score report testing data
#Model evaluation

# print("Testing Green Accuracy: ", metrics.accuracy_score(y_testG, y_predG_testing))
# print("Testing Green Precision: ", metrics.precision_score(y_testG,y_predG_testing))
# print("Testing Green  Recall: ",metrics.recall_score(y_testG,y_predG_testing))
#
# # printing the score report traning data
# #Model evaluation
# print("Training Green  Accuracy: ", metrics.accuracy_score(y_trainG, y_predG_training))
# print("Training Green Precision: ", metrics.precision_score(y_trainG,y_predG_training))
# print("Training Green Recall: ",metrics.recall_score(y_trainG,y_predG_training))

##############################################################################################
############################# Blue Channel model ##############################################
##############################################################################################
# Importing the Green dataset

blue_data_features = pd.read_csv('Blue_data.csv')
labels_file = pd.read_csv('Labels1.csv')
XB = green_data_features.iloc[:, 0:9].values
yB = labels_file.iloc[:, 0].values


X_trainB, X_testB, y_trainB, y_testB =train_test_split(XB,yB,test_size=0.3, random_state=109)


clfB=RandomForestClassifier(n_estimators=100,max_depth=10)
clfB.fit(X_trainB,y_trainB)
y_predB_testing = clfB.predict(X_testB)
y_predB_training=clfB.predict(X_trainB)


# printing the score report testing data
#Model evaluation

# print("Testing Blue Accuracy: ", metrics.accuracy_score(y_testB, y_predB_testing))
# print("Testing Blue Precision: ", metrics.precision_score(y_testB,y_predB_testing))
# print("Testing Green  Recall: ",metrics.recall_score(y_testB,y_predB_testing))
#
# # printing the score report traning data
# #Model evaluation
# print("Training Blue   Accuracy: ", metrics.accuracy_score(y_trainB, y_predB_training))
# print("Training Blue  Precision: ", metrics.precision_score(y_trainB,y_predB_training))
# print("Training Blue Recall: ",metrics.recall_score(y_trainB,y_predB_training))

# Majority voting for Training predictions
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


# Performance evaluation of the model for testing prediction
from sklearn.metrics import confusion_matrix
from sklearn.metrics import accuracy_score

# Making the Confusion Matrix
import seaborn as sns
cm = confusion_matrix(majority_votes_testing, y_testR)
sns.heatmap(cm,annot=True)
pyplot.savefig('RandomForest_testing.png')
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

# Performance evaluation of the model for training prediction
# Making the Confusion Matrix
cm_training = confusion_matrix(majority_votes_training, y_trainR)
sns.heatmap(cm_training,annot=True)
pyplot.savefig('RandomForest_training.png')
print(cm_training)

accuracy_score(y_trainR,majority_votes_training)

print(classification_report(y_trainR,majority_votes_training))

#Model evaluation
precision=metrics.precision_score(y_trainR,majority_votes_training)
accuracy=metrics.accuracy_score(y_trainR, majority_votes_training)
recall=metrics.recall_score(y_trainR,majority_votes_training)
print("Training Accuracy: ", metrics.accuracy_score(y_trainR, majority_votes_training))
print("Training Precision: ", metrics.precision_score(y_trainR,majority_votes_training))
print("Training Recall: ",metrics.recall_score(y_trainR,majority_votes_training))

