from matplotlib import pyplot
from sklearn.naive_bayes import BernoulliNB
import pandas as pd

##############################################################################################
############################# Red Channel model ##############################################
##############################################################################################
# Importing the Red dataset
red_data_features = pd.read_csv('Red_data.csv')
labels_file = pd.read_csv('Labels1.csv')
XR = red_data_features.iloc[:, 0:9].values
yR = labels_file.iloc[:, 0].values

# print(len(X))
# print(len(y))
# Training and Testing Data (divide the data into two part)
from sklearn.model_selection import train_test_split
X_trainR, X_testR, y_trainR, y_testR =train_test_split(XR,yR,test_size=0.3)

# Implementing Bernoulli Naive Bayes model
clfR = BernoulliNB()

# train the model
clfR.fit(X_trainR,y_trainR)

# predicting the response for test dataset
y_predR_testing=clfR.predict(X_testR)
# print("Testing Red predictions")
# print(len(y_predR_testing))
# print(y_predR_testing)

y_predR_training=clfR.predict(X_trainR)
# print("Training Red predictions")
# print(len(y_predR_training))
# print(y_predR_training)

#Model evaluation
from sklearn import metrics
print("NB Red Training Accuracy: ", metrics.accuracy_score(y_trainR, y_predR_training))
# print("Training Precision: ", metrics.precision_score(y_trainR,y_predR_training))
# print("Training Recall: ",metrics.recall_score(y_trainR,y_predR_training))

#Model evaluation
print("NB Red Testing Accuracy: ", metrics.accuracy_score(y_testR, y_predR_testing))
# print("Testing Precision: ", metrics.precision_score(y_testR,y_predR_testing))
# print("Testing Recall: ",metrics.recall_score(y_testR,y_predR_testing))


##############################################################################################
############################ Green Channel model ############################################
##############################################################################################
# Importing the Green dataset
green_data_features = pd.read_csv('Green_data.csv')
labels_file = pd.read_csv('Labels1.csv')
XG = green_data_features.iloc[:, 0:9].values
yG = labels_file.iloc[:, 0].values

# print(len(X))
# print(len(y))
# Training and Testing Data (divide the data into two part)
from sklearn.model_selection import train_test_split
X_trainG, X_testG, y_trainG, y_testG =train_test_split(XG,yG,test_size=0.3, random_state=109)

clfG = BernoulliNB()

# train the model
clfG.fit(X_trainG,y_trainG)

# predicting the reponse for test dataset
y_predG_testing=clfG.predict(X_testG)
# print("Testing Green Prediction")
# print(len(y_predG_testing))
# print(y_predG_testing)

y_predG_training=clfG.predict(X_trainG)
# print("Traning Green predictions")
# print(len(y_predG_training))
# print(y_predG_training)

#Model evaluation
from sklearn import metrics
print("NB Green Training Accuracy: ", metrics.accuracy_score(y_trainG, y_predG_training))
# print("Training Precision: ", metrics.precision_score(y_trainR,y_predR_training))
# print("Training Recall: ",metrics.recall_score(y_trainR,y_predR_training))

#Model evaluation
print("NB Green Testing Accuracy: ", metrics.accuracy_score(y_testG, y_predG_testing))
# print("Testing Precision: ", metrics.precision_score(y_testR,y_predR_testing))
# print("Testing Recall: ",metrics.recall_score(y_testR,y_predR_testing))



##############################################################################################
############################ Blue Channel model ############################################
##############################################################################################
# Importing the Blue dataset
blue_data_features = pd.read_csv('Blue_data.csv')
labels_file = pd.read_csv('Labels1.csv')
XB = blue_data_features.iloc[:, 0:9].values
yB = labels_file.iloc[:, 0].values

# print(len(X))
# print(len(y))
# Training and Testing Data (divide the data into two part)
from sklearn.model_selection import train_test_split
X_trainB, X_testB, y_trainB, y_testB =train_test_split(XB,yB,test_size=0.3, random_state=109)

clfB = BernoulliNB()

# train the model
clfB.fit(X_trainB,y_trainB)

# predicting the reponse for test dataset
y_predB_testing=clfB.predict(X_testB)

y_predB_training=clfB.predict(X_trainB)

#Model evaluation
from sklearn import metrics
print("NB Blue Training Accuracy: ", metrics.accuracy_score(y_trainB, y_predB_training))
# print("Training Precision: ", metrics.precision_score(y_trainR,y_predR_training))
# print("Training Recall: ",metrics.recall_score(y_trainR,y_predR_training))

#Model evaluation`
print("NB Blue Testing Accuracy: ", metrics.accuracy_score(y_testB, y_predB_testing))
# print("Testing Precision: ", metrics.precision_score(y_testR,y_predR_testing))
# print("Testing Recall: ",metrics.recall_score(y_testR,y_predR_testing))

################################################################
# Majority voting
import numpy as np

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

# # Making the Confusion Matrix
# import seaborn as sns
# cmtest = confusion_matrix(majority_votes_testing, y_testR)
# sns.heatmap(cmtest,annot=True)
# pyplot.savefig('NB_testing.png')
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
# import seaborn as sns
# cmtrain = confusion_matrix(majority_votes_training, y_trainR)
# sns.heatmap(cmtrain,annot=True)
# pyplot.savefig('NB_training.png')

accuracy_score(y_trainR,majority_votes_training)

print(classification_report(y_trainR,majority_votes_training))

#Model evaluation
# precision=metrics.precision_score(y_trainR,majority_votes_training)
# accuracy=metrics.accuracy_score(y_trainR, majority_votes_training)
# recall=metrics.recall_score(y_trainR,majority_votes_training)
print("Training Accuracy: ", metrics.accuracy_score(y_trainR, majority_votes_training))
print("Training Precision: ", metrics.precision_score(y_trainR,majority_votes_training))
print("Training Recall: ",metrics.recall_score(y_trainR,majority_votes_training))