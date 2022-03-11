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

red_data_features = pd.read_csv('Red_data.csv')
labels_file = pd.read_csv('Labels1.csv')
XR = red_data_features.iloc[:, 0:9].values
yR = labels_file.iloc[:, 0].values

X_trainR, X_testR, y_trainR, y_testR =train_test_split(XR,yR,test_size=0.3, random_state=109)


sc = StandardScaler()
X_trainR = sc.fit_transform(X_trainR)
X_testR = sc.transform(X_testR)

classifierR = Sequential()
# intput layer
classifierR.add(Dense(units=16, activation='relu',input_dim=9))
#hidden layer
classifierR.add(Dense(units=64, activation='relu'))
classifierR.add(Dense(units=128, activation='relu'))
classifierR.add(Dropout(0.3))
classifierR.add(Dense(units=128, activation='relu'))

# output layer
classifierR.add(Dense(units=1,activation='sigmoid'))

#Compile the ANN
classifierR.compile(optimizer='adam',loss="binary_crossentropy",metrics=['accuracy'])

# Traning the ANN

classifierR.fit(X_trainR,y_trainR,batch_size=20,epochs=100)


# Predictions on the testing data
y_predR_testing=classifierR.predict(X_testR)
y_predR_testing=(y_predR_testing>0.5)
y_testR=y_testR>0.5

#confusion matrix
# cm=confusion_matrix(y_testR,y_predR_testing)
# print(cm)
print("ANN Testing Accuracy Red = ",accuracy_score(y_testR, y_predR_testing))
print("Testing Red  Accuracy: ", metrics.accuracy_score(y_testR, y_predR_testing))
print("Testing Red  Precision: ", metrics.precision_score(y_testR,y_predR_testing))
print("Testing Red Recall: ",metrics.recall_score(y_testR,y_predR_testing))

# Predictions on training Data
y_predR_training = classifierR.predict(X_trainR)
y_predR_training=(y_predR_training>0.5)
y_trainR=y_trainR>0.5

#confusion matrix
# cm=confusion_matrix(y_trainR,y_predR_training)
# print(cm)
print("ANN Training Accuracy Red = ",accuracy_score(y_trainR, y_predR_training))
print("ANN Training Red  Accuracy: ", metrics.accuracy_score(y_trainR, y_predR_training))
print("ANN Training Red  Precision: ", metrics.precision_score(y_trainR,y_predR_training))
print("ANN Training Red Recall: ",metrics.recall_score(y_trainR,y_predR_training))


##############################################################################################
############################# Green Channel model ##############################################
##############################################################################################
# # Importing the Red dataset

green_data_features = pd.read_csv('Green_data.csv')
labels_file = pd.read_csv('Labels1.csv')
XG = green_data_features.iloc[:, 0:9].values
yG = labels_file.iloc[:, 0].values

X_trainG, X_testG, y_trainG, y_testG =train_test_split(XG,yG,test_size=0.3, random_state=109)


sc = StandardScaler()
X_trainG = sc.fit_transform(X_trainG)
X_testG = sc.transform(X_testG)

classifierG = Sequential()
# intput layer
classifierG.add(Dense(units=16, activation='relu',input_dim=9))
#hidden layer
classifierG.add(Dense(units=64, activation='relu'))
classifierG.add(Dense(units=128, activation='relu'))
classifierG.add(Dropout(0.3))
classifierG.add(Dense(units=128, activation='relu'))

# output layer
classifierG.add(Dense(units=1,activation='sigmoid'))

#Compile the ANN
classifierG.compile(optimizer='adam',loss="binary_crossentropy",metrics=['accuracy'])

# Traning the ANN

classifierG.fit(X_trainG,y_trainG,batch_size=10,epochs=50)


# Predictions on the testing data
y_predG_testing=classifierG.predict(X_testG)
y_predG_testing=(y_predG_testing>0.5)
y_testG=y_testG>0.5
#confusion matrix
# cm=confusion_matrix(y_testG,y_predG_testing)
# print(cm)
# print("ANN Testing Accuracy Green = ",accuracy_score(y_testG, y_predG_testing))
# print("Testing Green  Accuracy: ", metrics.accuracy_score(y_testG, y_predG_testing))
# print("Testing Green  Precision: ", metrics.precision_score(y_testG,y_predG_testing))
# print("Testing Green Recall: ",metrics.recall_score(y_testG,y_predG_testing))

# Predictions on training Data
y_predG_training = classifierG.predict(X_trainG)
y_predG_training=(y_predG_training>0.5)
y_trainG=y_trainG>0.5

#confusion matrix
# cm=confusion_matrix(y_trainG,y_predG_training)
# print(cm)
# print("ANN Training Accuracy Green = ",accuracy_score(y_trainG, y_predG_training))
# print("Training Green  Accuracy: ", metrics.accuracy_score(y_trainG, y_predG_training))
# print("Training Green  Precision: ", metrics.precision_score(y_trainG,y_predG_training))
# print("Training Green Recall: ",metrics.recall_score(y_trainG,y_predG_training))



##############################################################################################
############################# Blue Channel model ##############################################
##############################################################################################
# Importing the Red dataset

blue_data_features = pd.read_csv('Blue_data.csv')
labels_file = pd.read_csv('Labels1.csv')
XB = blue_data_features.iloc[:, 0:9].values
yB = labels_file.iloc[:, 0].values

X_trainB, X_testB, y_trainB, y_testB =train_test_split(XB,yB,test_size=0.3, random_state=109)


sc = StandardScaler()
X_trainB = sc.fit_transform(X_trainB)
X_testB = sc.transform(X_testB)

classifierB = Sequential()
# intput layer
classifierB.add(Dense(units=32, activation='relu',input_dim=9))
#hidden layer
classifierB.add(Dense(units=64, activation='relu'))
classifierB.add(Dense(units=128, activation='relu'))
classifierB.add(Dropout(0.3))
classifierB.add(Dense(units=128, activation='relu'))

# output layer
classifierB.add(Dense(units=1,activation='sigmoid'))

#Compile the ANN
classifierB.compile(optimizer='adam',loss="binary_crossentropy",metrics=['accuracy'])

# Traning the ANN
classifierB.fit(X_trainB,y_trainB,batch_size=10,epochs=50)


# Predictions on the testing data
y_predB_testing=classifierB.predict(X_testB)
y_predB_testing=(y_predB_testing>0.5)
y_testB=y_testB>0.5
#confusion matrix
# cm=confusion_matrix(y_testB,y_predB_testing)
# print(cm)
# print("ANN Testing Accuracy Blue = ",accuracy_score(y_testB, y_predB_testing))
# print("Testing Blue Accuracy: ", metrics.accuracy_score(y_testB, y_predB_testing))
# print("Testing Blue  Precision: ", metrics.precision_score(y_testB,y_predB_testing))
# print("Testing Blue Recall: ",metrics.recall_score(y_testB,y_predB_testing))

# Predictions on training Data
y_predB_training = classifierB.predict(X_trainB)
y_predB_training=(y_predB_training>0.5)
y_trainB=y_trainB>0.5

#confusion matrix
# cm=confusion_matrix(y_trainB,y_predB_training)
# print(cm)
print("ANN Testing Accuracy Red = ",accuracy_score(y_testR, y_predR_testing))
print("ANN Training Accuracy Red = ",accuracy_score(y_trainR, y_predR_training))
print("ANN Testing Accuracy Green = ",accuracy_score(y_testG, y_predG_testing))
print("ANN Training Accuracy Green = ",accuracy_score(y_trainG, y_predG_training))
print("ANN Training Accuracy= ",accuracy_score(y_trainB, y_predB_training))
print("ANN Testing Accuracy Blue = ",accuracy_score(y_testB, y_predB_testing))

# print("Training Blue  Accuracy: ", metrics.accuracy_score(y_trainB, y_predB_training))
# print("Training Blue Precision: ", metrics.precision_score(y_trainB,y_predB_training))
# print("Training Blue Recall: ",metrics.recall_score(y_trainB,y_predB_training))

# sns.heatmap(cm,annot=True)
# pyplot.savefig('ANN_training.png')

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

# print(f"After majority voting the final predictions for training: \n{majority_votes_training}")

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

import pickle

pkl_file_nameR="model_annRed.pkl"
with open(pkl_file_nameR,'wb') as fileR:
    pickle.dump(classifierR, fileR)

pkl_file_nameG="model_annGreen.pkl"
with open(pkl_file_nameG,'wb') as fileG:
    pickle.dump(classifierG, fileG)

pkl_file_nameB="model_annBlue.pkl"
with open(pkl_file_nameB,'wb') as fileB:
    pickle.dump(classifierB, fileB)
