import pickle

import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn import metrics
from keras.models import Sequential
from keras.layers import Dense, Dropout
from sklearn.metrics import confusion_matrix, accuracy_score
import seaborn as sns
from matplotlib import pyplot
##############################################################################################
############################# Red Channel model ##############################################
##############################################################################################
# Importing the Red dataset

red_data_features = pd.read_csv('../../Red_data.csv')
labels_file = pd.read_csv('../../Labels1.csv')
XR = red_data_features.iloc[:, 0:9].values
yR = labels_file.iloc[:, 0].values

X_trainR, X_testR, y_trainR, y_testR =train_test_split(XR,yR,test_size=0.3, random_state=109)


sc = StandardScaler()
X_trainR = sc.fit_transform(X_trainR)
X_testR = sc.transform(X_testR)

##############################################################################################
############################# Green Channel model ##############################################
##############################################################################################
# # Importing the Green dataset

green_data_features = pd.read_csv('../../Green_data.csv')
labels_file = pd.read_csv('../../Labels1.csv')
XG = green_data_features.iloc[:, 0:9].values
yG = labels_file.iloc[:, 0].values

X_trainG, X_testG, y_trainG, y_testG =train_test_split(XG,yG,test_size=0.3, random_state=109)


sc = StandardScaler()
X_trainG = sc.fit_transform(X_trainG)
X_testG = sc.transform(X_testG)

#########################################################################################################
##############################################################################################
############################# Blue Channel model ##############################################
##############################################################################################
# Importing the Blue dataset

blue_data_features = pd.read_csv('../../Blue_data.csv')
labels_file = pd.read_csv('../../Labels1.csv')
XB = blue_data_features.iloc[:, 0:9].values
yB = labels_file.iloc[:, 0].values

X_trainB, X_testB, y_trainB, y_testB =train_test_split(XB,yB,test_size=0.3, random_state=109)


sc = StandardScaler()
X_trainB = sc.fit_transform(X_trainB)
X_testB = sc.transform(X_testB)

########################################################################################
##################### Loading the saved model ##########################################
########################################################################################
# Red model
with open('model_dtRed.pkl', 'rb') as fileR:
    modelR= pickle.load(fileR)


# calculating the accuracy score and predict target values
scoreR_testing = modelR.score(X_testR,y_testR)
print("Test score Red Model: ", scoreR_testing)
scoreR_trianing = modelR.score(X_trainR,y_trainR)
print("Training score Red Model: ", scoreR_trianing)

# Predicting on training dataset
y_predR_training = modelR.predict(X_trainR)
# predicting on testing dataset
y_predR_testing = modelR.predict(X_testR)


# Green model
with open('model_dtGreen.pkl', 'rb') as fileG:
    modelG= pickle.load(fileG)

# calculating the accuracy score and predict target values
scoreG_testing = modelG.score(X_testG,y_testG)
print("Test score Green Model: ", scoreG_testing)
scoreG_trianing = modelG.score(X_trainG,y_trainG)
print("Traning score Green Model: ", scoreG_trianing)

# Predicting on training dataset
y_predG_training = modelG.predict(X_trainG)
# predicting on testing dataset
y_predG_testing = modelR.predict(X_testG)



# Blue model
with open('model_dtBlue.pkl', 'rb') as fileB:
    modelB= pickle.load(fileB)

# calculating the accuracy score and predict target values
scoreB_testing = modelB.score(X_testB,y_testB)
print("Test score Blue Model: ", scoreB_testing)
scoreB_trianing = modelB.score(X_trainB,y_trainB)
print("Training score Blue Model: ", scoreB_trianing)

# Predicting on training dataset
y_predB_training = modelB.predict(X_trainB)
# predicting on testing dataset
y_predB_testing = modelB.predict(X_testB)




# Majority Voting
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

# print(f"After majority voting the final predictions for testing: \n{majority_votes_testing}")

# Majority voting for training predictions
majority_votes_training=np.zeros(y_predR_training.shape,dtype=int)
print(len(majority_votes_training))

for i in range(len(y_predR_training)):
    if y_predR_training[i]==y_predG_training[i]:
        majority_votes_training[i]=y_predR_training[i]
    elif y_predR_training[i]==y_predB_training[i]:
        majority_votes_training[i]=y_predR_training[i]
    else:
        majority_votes_training[i]=y_predG_training[i]

# print(f"After majority voting the final predictions for training: \n{majority_votes_training}")


print("After majority voting \n Testing details: \n")
# Performance evaluation of the model for testing prediction
from sklearn.metrics import confusion_matrix
from sklearn.metrics import accuracy_score
accuracy_score(y_testR,majority_votes_testing)

# # Making the Confusion Matrix
# import seaborn as sns
# cmtest = confusion_matrix(majority_votes_testing, y_testR)
# sns.heatmap(cmtest,annot=True)
# pyplot.savefig('ANN_testing.png')
# accuracy_score(y_testR,majority_votes_testing)


from sklearn.metrics import classification_report
print(classification_report(y_testR,majority_votes_testing))
#
# #Model evaluation
from sklearn import metrics
# precision=metrics.precision_score(y_testR,majority_votes_testing)
# accuracy=metrics.accuracy_score(y_testR, majority_votes_testing)
# recall=metrics.recall_score(y_testR,majority_votes_testing)
print("Testing Accuracy: ", metrics.accuracy_score(y_testR, majority_votes_testing))
print("Testing Precision: ", metrics.precision_score(y_testR,majority_votes_testing))
print("Testing Recall: ",metrics.recall_score(y_testR,majority_votes_testing))

print("After majority voting \n Training details: \n")
# Performance evaluation of the model for training prediction
# # Making the Confusion Matrix
# import seaborn as sns
# cmtrain = confusion_matrix(majority_votes_training, y_trainR)
# sns.heatmap(cmtrain,annot=True)
# pyplot.savefig('ANN_training.png')

accuracy_score(y_trainR,majority_votes_training)
print(classification_report(y_trainR,majority_votes_training))

#Model evaluation
# precision=metrics.precision_score(y_trainR,majority_votes_training)
# accuracy=metrics.accuracy_score(y_trainR, majority_votes_training)
# recall=metrics.recall_score(y_trainR,majority_votes_training)
print("Training Accuracy: ", metrics.accuracy_score(y_trainR, majority_votes_training))
print("Training Precision: ", metrics.precision_score(y_trainR,majority_votes_training))
print("Training Recall: ",metrics.recall_score(y_trainR,majority_votes_training))