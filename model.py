import os
import joblib
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import confusion_matrix, accuracy_score, recall_score, f1_score
from sklearn.model_selection import KFold
from sklearn.model_selection import StratifiedKFold
from collections import Counter
import os, sys
import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import recall_score, classification_report, f1_score, accuracy_score
from sklearn.metrics import confusion_matrix
from sklearn.metrics import precision_score
from imblearn.under_sampling import RandomUnderSampler
from imblearn.over_sampling import SMOTE
from imblearn.pipeline import Pipeline
from imblearn.over_sampling import RandomOverSampler


trained_users = []
output_file = open('outputNew3.txt', mode = 'a',encoding='utf-8')

data=pd.read_csv("data/merge4.csv")
train_data=data[data["week"]<26]
removed_cols = ['actid','pcid','user','userID','time_stamp','mal','insider','mal_act']
x_cols = [i for i in data.columns if i not in removed_cols]

xTrain = train_data[x_cols].values
yTrain = train_data['mal'].values
yTrainBin = yTrain > 0

rf = RandomForestClassifier(n_jobs=-1)
rf.fit(xTrain, yTrainBin)

joblib.dump(rf, 'random_forest_model3.pkl')

rf = joblib.load('random_forest_model3.pkl')

data=[]
columns = ['user', 'cm00', 'cm01', 'cm10', 'cm11', 'mal_user']
df2 = pd.DataFrame(columns=columns)
df0 = pd.DataFrame(columns=columns)
df1 = pd.DataFrame(columns=columns)
# Training by week
data_folder = "data/weekData_ID"
for week in range(26, 74): 
    week_files = os.path.join(data_folder, str(week))
    files = [file for file in os.listdir(week_files) if file.endswith(".csv")]
    for file in files: 
        file_path = os.path.join(week_files, file) 
        print("-------------------------------------------------------------------------",file=output_file)
        print("At",week,"week,user",file,"situation",file=output_file)
        test_data = pd.read_csv(file_path)
        removed_cols = ['actid','pcid','user','userID','time_stamp','mal','insider','mal_act']
        x_cols = [i for i in test_data.columns if i not in removed_cols]
        xTest = test_data[x_cols].values
        yTest = test_data['mal'].values
        yTestBin = yTest > 0

        cm=confusion_matrix(yTestBin,rf.predict(xTest))
        # print(cm, file=output_file)


        if((len(cm)>1 and cm[0][1]>0) or (len(cm)>1 and cm[1][1]>0) or (len(cm)>1 and cm[1][0]>0)) :
            print(cm[0][1],"--------------",cm[1][1],file=output_file)
            print("!!!!!!!!!!!!!!!!!!!!",file=output_file)
            print("at",week,"week, user",file,"abnormal",file=output_file)

            if(file not in trained_users):
                print("at",week,"week, user",file,"abnormal, and this user haven't been used",file=output_file)
                trained_users.append(file)

                fold="data/userID"
                data=pd.read_csv(os.path.join(fold, file))

                data_shuffled = data.sample(frac=1, random_state=42).reset_index(drop=True)
                train_ratio = 0.8
                train_size = int(train_ratio * len(data_shuffled))
                trainData = data_shuffled.iloc[:train_size]
                testData = data_shuffled.iloc[train_size:int(len(data_shuffled))]
                # print(len(trainData))
                # print(len(testData))

               
                removed=['actid','pcid','userID','time_stamp','mal_act','insider']
                removedX=['actid','pcid','userID','time_stamp','mal','mal_act','insider']
                cols=[i for i in data.columns if i not in removed]
                colsX=[i for i in data.columns if i not in removedX]
                df=trainData.loc[:][cols]
                X = df.loc[:][colsX]
                y = df['mal'].values
                if(np.all(y == 0) or np.all(y == 1)):
                    print('without oversampling',file=output_file)
                    print(y,file=output_file)
                    overData=df
                else:
                    if(len(df[df['mal']==1])<5):
                        over = RandomOverSampler(random_state=0, sampling_strategy=0.1) 
                    else:
                        over = SMOTE(sampling_strategy=0.1, k_neighbors=4)
                    X_sos, y_sos = over.fit_resample(X, y)
                    X_sos['mal']=y_sos
                    print('oversampling:{}'.format(Counter(y_sos)),file=output_file)
                    overData=X_sos

                run = 1
                np.random.seed(run)
                x_Train = overData[x_cols].values
                y_Train = overData['mal'].values
                y_TrainBin = y_Train > 0

                rf = RandomForestClassifier(n_jobs=-1)
                rf.fit(x_Train, y_TrainBin)

                x_Test = testData[x_cols].values
                y_Test = testData['mal'].values
                y_TestBin = y_Test > 0
                    
                # results
                confusion = confusion_matrix(y_TestBin, rf.predict(x_Test))
                accuracy = accuracy_score(y_TestBin, rf.predict(x_Test))
                recall = recall_score(y_TestBin, rf.predict(x_Test))
                f1 = f1_score(y_TestBin, rf.predict(x_Test))
                if len(confusion)==1:
                    mal_user=0
                    # print(confusion)
                elif confusion[1][0]==0 and confusion[1][1] == 0:
                    mal_user=0
                    # print(confusion)
                else:
                    mal_user=1
                    # print(confusion)
                if len(confusion)==1:
                    df2.loc[len(df)]={
                        'user': file,
                        'cm00': confusion[0][0],
                        'cm01': 0,
                        'cm10': 0,
                        'cm11': 0,
                        'mal_user': mal_user
                    }
                    df0.loc[len(df)]={
                        'user': file,
                        'cm00': confusion[0][0],
                        'cm01': 0,
                        'cm10': 0,
                        'cm11': 0,
                        'mal_user': mal_user
                    }
                else:
                    if(mal_user==0):
                        df0.loc[len(df)] = {
                        'user': file,
                        'cm00': confusion[0][0],
                        'cm01': confusion[0][1],
                        'cm10': confusion[1][0],
                        'cm11': confusion[1][1],
                        'mal_user': mal_user
                        }
                    else:
                        df1.loc[len(df)] = {
                        'user': file,
                        'cm00': confusion[0][0],
                        'cm01': confusion[0][1],
                        'cm10': confusion[1][0],
                        'cm11': confusion[1][1],
                        'mal_user': mal_user
                        }
                    df2.loc[len(df)] = {
                        'user': file,
                        'cm00': confusion[0][0],
                        'cm01': confusion[0][1],
                        'cm10': confusion[1][0],
                        'cm11': confusion[1][1],
                        'mal_user': mal_user
                    }
                # print(df2)
                output_file.write(f"Confusion Matrix:\n{confusion}\n")
                output_file.write(f"Accuracy: {accuracy}\n")
                output_file.write(f"Recall: {recall}\n")
                output_file.write(f"F1 Score: {f1}\n")
                output_file.write(f"over!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!")

df2.to_csv('partRedo/data_0.8.csv',index=False)
df0.to_csv('partRedo/data_0.8_0.csv',index=False)
df1.to_csv('partRedo/data_0.8_1.csv',index=False)

output_file.close()
