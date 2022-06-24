# -*- coding: utf-8 -*-
"""
Created on Mon Jun 20 15:32:20 2022

@author: apmel
"""


'''
This is a coding question, please read all the requirements listed below before starting. use any of the following IDE environments:

Spyder
Visual code
pycharm
sublime
(Notebooks generated through Colab and Jupyter are not accepted and will earn zero)

Once your python script and screenshots are ready please attach the python script all the screenshots to this question by clicking the "Add file" button and then follow the notes to upload your script & screenshots. 

Requirements:

1) Load the data and carry out initial exploration        (10 marks) 

Download the  "voice_reduced.csv" from here
Load the file into a data frame named df_firstname
Carry out initial analysis per column and show the types, number of missing values, range per feature(min,max), mean and median  information.
Provide a total count for each of the unique values within the label column.
Summarize your findings in the below box, make sure to use your own words and give a full picture of the dataset. 
Using seaborn generate a box_plot  of all the attributes in the dataframe. take a screenshot add it to your submission. 
2) Prepare the data for machine learning                  (15 marks)

Replace any infinite values in the data with nan, you can do that by applying the following transformation:
            df_firstname.replace([np.inf, -np.inf], np.nan, inplace=True)
Separate the features from the target class. Name the features dataframe df_firstname_features and name the target df_firstname_target.
Print the counts of df_firstname_features  and df_firstname_target.  (i.e. number of records)
Split the data into 75% for training and 25% for testing, set the random seed to the last two digits of your student id. Save the two training dataframes  to X_train_firstname and y_train_firstname. Save the testing data to X_test_firstname and y_train_firstname.
Print the training and testing data shapes. (All four dataframes)
Define an object to hold the LabelEncoder name it LR_firstname
Fit/transform the y_train_firstname using the LR_firstname.
Prepare a pipeline to handle the numeric data columns of the feature space. Name the pipeline num_pipeline_firstname. The pipeline should handle the following steps:
Fill in all the missing values using the median value of each column. 
Apply the standard scalar.
Take two screenshots showing the steps you carried out in points 5,6 &7  above, add them to your submissions.
3) Build, train and initial test of the model                            (15 marks) 

 Build a SVC classifier name it clf_svm_firstname, set gamma to equal "auto" and set the  random_state=last two digits of your student id.
Prepare another pipeline for the training process call it "pipe". This pipeline should contain two steps, the first step is the num_pipeline_firstname and the second is the SVC classifier you defined in step#1.
Fit the training data X_train_firstname to the pipeline "pipe" you defined in step#2.
Carryout 3 cross validation on the training and print out the mean of the scores. (Hint: use "pipe" instead of clf_svm_firstname)
Save the accuracy score of the training process to a variable named training_accuracy.
Take a screens shot for both of the above points and add it to your submission.
Use the model to predict the test data,  i.e. pass the test features to the model and save the results into a variable named initial_predictions. (Remember to transform the test data i.e. pass through the label encoder and the num_pipeline)
Calculate the accuracy score and save it to a variable named initial_accuracy and print it out.
Add one conclusion in relation to the training and testing results.
4) Fine tune the model                                      (10  marks)

Use grid search to fine tune the parameters of the SVM estimator, setup the following parameters:
kernel: ['linear','rbf']
C: [0.01, 0.1, 1]
gamma: [0.01,0.06, 0.1]
Apply GridSearchCV on the training data. set the estimator as clf_svm_firstname, and use accuracy as a scoring method, also set refit to 'True' and verbose  to 3.
Printout the best parameters 
Take a screen shot to show the results and add it to your submission.
 
5) Test & save the model                                        (10 marks)

Save the best model the grid search generated to a variable and name it  best_model_firstname. (Hint: check grid_search.best_estimator_ )
Predict the test data using the best model,  i.e. pass the test features to the model and save the results into a variable named final_predictions.
Calculate the accuracy score and save it to a variable named final_accuracy and print it out.
Take a screenshot to show the results and add it to your submission.
Record all three  accuracy scores in the below box and compare them and draw a conclusion.
Save the best model to your hard disk name the object best_model_firstname.pkl and attach it to your submission.
Save both the label encoder LR_firstname and the transformation pipeline "num_pipeline_firstname" to the disk, and attach the objects to your submission.(Hint: Use joblib dump)
Name your python script firstname_voice_midterm.py (Notebooks are not accepted and will earn zero)
Attach your python script i.e  to the submission.
'''







import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.svm import SVC
from sklearn.pipeline import Pipeline
from sklearn.metrics import accuracy_score, precision_score, recall_score, confusion_matrix
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import GridSearchCV
from sklearn.preprocessing import LabelEncoder
import joblib

#1) Load the data and carry out initial exploration

#Load the file into a data frame named df_firstname
df_adriano = pd.read_csv(r'C:\Users\apmel\OneDrive\IDEs\Anaconda-Files-Python\Semester4\SupervisedLearning\Assignments\MidTermTest\Exercise02\voice_reduced.csv')

pd.set_option('display.max_columns', None)

df_adriano.head()

#Carry out initial analysis per column and show the types, number of missing values, 
#range per feature(min,max), mean and median  information.

df_adriano.shape
df_adriano.dtypes
df_adriano.isna().sum()
df_adriano.info()
df_adriano.columns.values
df_adriano.describe()
df_adriano.median()
df_adriano.mean()

#Provide a total count for each of the unique values within the label column.
df_adriano['label'].value_counts()

#Using seaborn generate a box_plot  of all the attributes in the dataframe. 
figure = plt.figure(figsize=[10,7])
ax = sns.boxplot(data=df_adriano, palette="Set2")
plt.tight_layout()

#2) Prepare the data for machine learning

#Replace any infinite values in the data with nan, you can do that by applying the following 
#transformation:

df_adriano.replace([np.inf, -np.inf], np.nan, inplace=True)

#Separate the features from the target class. 
df_adriano_features = df_adriano.drop('label', axis = 1)
df_adriano_target = df_adriano['label']

#Print the counts of df_firstname_features and df_firstname_target.
df_adriano_features.count()
df_adriano_target.count()

#Split the data into 75% for training and 25% for testing, set the random seed to the last 
#two digits of your student id. Save the two training dataframes to X_train_firstname and 
#y_train_firstname. Save the testing data to X_test_firstname and y_train_firstname.
X_train_adriano, X_test_adriano, y_train_adriano, y_test_adriano = train_test_split(df_adriano_features, df_adriano_target, test_size=0.25, random_state=57)

#Print the training and testing data shapes. (All four dataframes)
print(X_train_adriano)
print(X_test_adriano)
print(y_train_adriano)
print(y_test_adriano)

print(X_train_adriano.shape)
print(X_test_adriano.shape)
print(y_train_adriano.shape)
print(y_test_adriano.shape)


#Define an object to hold the LabelEncoder name it LR_firstname

LR_adriano = LabelEncoder()

LR_adriano.fit_transform(y_train_adriano)




num_pipeline_adriano = Pipeline([
('imputer', SimpleImputer(strategy="median")),
('std_scaler', StandardScaler()),
])



#Build, train and initial test of the model    


clf_svm_adriano = SVC(gamma='auto', random_state=57)

pipe = Pipeline([
        ('transformer_pipeline', num_pipeline_adriano),
        ('svc', clf_svm_adriano)
        ])

pipe.fit(X_train_adriano, y_train_adriano)


# Fine tune the model 

#Creating the grid parameters
param_grid = [
    {'svc__kernel': ['linear', 'rbf'],
     'svc__C': [0.01, 0.1, 1],
     'svc__gamma': [0.01, 0.06, 0.1]}]
    
print(param_grid)

#Creating a grid search object
grid_search_adriano = GridSearchCV(estimator=pipe, 
                                   param_grid=param_grid, 
                                   scoring='accuracy', 
                                   refit=True, 
                                   verbose=3)
grid_search_adriano

#Fitting the data into the grid search object
grid_search_adriano.fit(X_train_adriano, y_train_adriano)

#Retreiving the best parameters and best estimator
grid_search_adriano.best_params_
grid_search_adriano.best_estimator_

#Creating the best model (with the best fine-tuned model) to predict the test data 
best_model_adriano = grid_search_adriano.best_estimator_
best_model_adriano



############################

from sklearn import cross_val_score
mean_score = cross_val_score(clf_svm_adriano, X_train_adriano, y_train_adriano, cv=3).mean()



#Saving the model using joblib
joblib.dump(best_model_adriano, 'SVC_model_adriano.pkl')

#Saving the full pipeline using joblib
joblib.dump(pipe_svm_adriano, 'full_pipeline_adriano.pkl')



