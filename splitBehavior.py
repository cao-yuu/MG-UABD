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
output_file = open('file.txt', mode = 'a',encoding='utf-8')
data=pd.read_csv("data/merge4.csv")
train_data=data[data["week"]<26]
removed_cols = ['actid','pcid','user','userID','time_stamp','mal','insider','mal_act']
x_cols = [i for i in data.columns if i not in removed_cols]

# train_data_logon=train_data[train_data['act']==1]
# train_data_logoff=train_data[train_data['act']==2]
# train_data_connect=train_data[train_data['act']==3]
# train_data_disconnect=train_data[train_data['act']==4]
# train_data_http=train_data[train_data['act']==5]
# train_data_email=train_data[train_data['act']==6]
train_data_file=train_data[train_data['act']==7]


# xTrain_logon = train_data_logon[x_cols].values#这里去掉了mal_act
# yTrain_logon = train_data_logon['mal'].values
# yTrainBin_logon = yTrain_logon > 0
# #
# rf_logon = RandomForestClassifier(n_jobs=-1)
# rf_logon.fit(xTrain_logon, yTrainBin_logon)
# #
# joblib.dump(rf_logon, 'splitBehavior/logon.pkl')

#
# rf_logon = joblib.load('splitBehavior/logon.pkl')

# xTrain_logoff = train_data_logoff[x_cols].values
# yTrain_logoff = train_data_logoff['mal'].values
# yTrainBin_logoff = yTrain_logoff > 0
# #
# rf_logoff = RandomForestClassifier(n_jobs=-1)
# rf_logoff.fit(xTrain_logoff, yTrainBin_logoff)
# #
# joblib.dump(rf_logoff, 'splitBehavior/logoff.pkl')

# #
# rf_logoff = joblib.load('splitBehavior/logoff.pkl')

# xTrain_connect = train_data_connect[x_cols].values
# yTrain_connect = train_data_connect['mal'].values
# yTrainBin_connect = yTrain_connect > 0
# #
# rf_connect = RandomForestClassifier(n_jobs=-1)
# rf_connect.fit(xTrain_connect, yTrainBin_connect)
# #
# joblib.dump(rf_connect, 'splitBehavior/connect.pkl')

# #
# rf_connect = joblib.load('splitBehavior/connect.pkl')

# xTrain_disconnect = train_data_disconnect[x_cols].values
# yTrain_disconnect = train_data_disconnect['mal'].values
# yTrainBin_disconnect = yTrain_disconnect > 0
# #
# rf_disconnect = RandomForestClassifier(n_jobs=-1)
# rf_disconnect.fit(xTrain_disconnect, yTrainBin_disconnect)
# #
# joblib.dump(rf_disconnect, 'splitBehavior/disconnect.pkl')

# #
# rf_disconnect = joblib.load('splitBehavior/disconnect.pkl')

# xTrain_http = train_data_http[x_cols].values
# yTrain_http = train_data_http['mal'].values
# yTrainBin_http = yTrain_http > 0
# #
# rf_http = RandomForestClassifier(n_jobs=-1)
# rf_http.fit(xTrain_http, yTrainBin_http)
# #
# joblib.dump(rf_http, 'splitBehavior/http.pkl')

# #
# rf_http = joblib.load('splitBehavior/http.pkl')

# xTrain_email = train_data_email[x_cols].values
# yTrain_email = train_data_email['mal'].values
# yTrainBin_email = yTrain_email > 0
# #
# rf_email = RandomForestClassifier(n_jobs=-1)
# rf_email.fit(xTrain_email, yTrainBin_email)
# #
# joblib.dump(rf_email, 'splitBehavior/email.pkl')

# #
# rf_email = joblib.load('splitBehavior/email.pkl')

# xTrain_file = train_data_file[x_cols].values
# yTrain_file = train_data_file['mal'].values
# yTrainBin_file = yTrain_file > 0
# #
# rf_file = RandomForestClassifier(n_jobs=-1)
# rf_file.fit(xTrain_file, yTrainBin_file)
# #
# joblib.dump(rf_file, 'splitBehavior/file.pkl')

# # 
rf_file = joblib.load('splitBehavior/file.pkl')
##########################################

data=[]
columns = ['user', 'cm00', 'cm01', 'cm10', 'cm11', 'mal_user']
df2 = pd.DataFrame(columns=columns)
df0 = pd.DataFrame(columns=columns)
df1 = pd.DataFrame(columns=columns)

data_folder = "data/weekData_ID"
for week in range(26, 74): 
    print(week,"week",file=output_file)
    week_files = os.path.join(data_folder, str(week))
    files = [file for file in os.listdir(week_files) if file.endswith(".csv")]
    for file in files: 
        file_path = os.path.join(week_files, file) 
        test_data = pd.read_csv(file_path)
        test_data=test_data[test_data['act'] == 7]
        if(len(test_data)==0):
            continue
        removed_cols = ['actid','pcid','user','userID','time_stamp','mal','insider','mal_act']
        x_cols = [i for i in test_data.columns if i not in removed_cols]
        xTest = test_data[x_cols].values
        yTest = test_data['mal'].values
        yTestBin = yTest > 0

        cm=confusion_matrix(yTestBin,rf_file.predict(xTest))


        if((len(cm)>1 and cm[0][1]>0) or (len(cm)>1 and cm[1][1]>0) or (len(cm)>1 and cm[1][0]>0)) :
            print(cm[0][1],"--------------",cm[1][1],file=output_file)
            print("!!!!!!!!!!!!!!!!!!!!",file=output_file)
            print(week,"week, user",file,"abnormal user",file=output_file)

            if(file not in trained_users):
                print(week,"week, user",file,"abnormal user,and this user haven't been used",file=output_file)
                trained_users.append(file)
                fold="data/userID"
                data=pd.read_csv(os.path.join(fold, file))

                print(len(data),"data")
                data_shuffled = data.sample(frac=1, random_state=42).reset_index(drop=True)
                data_shuffled=data_shuffled[data_shuffled['act']==7]
                trainData = data_shuffled.iloc[0:int(0.7*len(data_shuffled))]
                testData = data_shuffled[~data_shuffled.index.isin(trainData.index)]
                print(len(data_shuffled),"data_shuffled")
                print(file,len(trainData))
                print(file,len(testData))
               
                removed=['actid','pcid','userID','time_stamp','mal_act','insider']
                removedX=['actid','pcid','userID','time_stamp','mal','mal_act','insider']
                cols=[i for i in data.columns if i not in removed]
                colsX=[i for i in data.columns if i not in removedX]
                df=trainData.loc[:][cols]
                X = df.loc[:][colsX]
                y = df['mal'].values
                print("df",len(df))
                print("X",len(X))
                if(np.all(y == 0) or np.all(y == 1)):
                    print('without oversampling',file=output_file)
                    overData=df
                elif(float(len(df[df['mal']==1]))/float(len(df[df['mal']==0])) < 0.2 and float(len(df[df['mal']==1])+1)/float(len(df[df['mal']==0])) < 0.4):
                    if(len(df[df['mal']==1])<5):
                        over = RandomOverSampler(random_state=0, sampling_strategy=0.4) #随机复制
                    else:
                        over = SMOTE(sampling_strategy=0.4, k_neighbors=4)
                    print(file,"\n",len(df[df['mal']==0]),len(df[df['mal']==1]),len(df[df['mal']==1])/len(df[df['mal']==0]),"\n",file=output_file)
                    print(file,"\n",len(df[df['mal']==0]),len(df[df['mal']==1]),len(df[df['mal']==1])/len(df[df['mal']==0]),"\n")
                    X_sos, y_sos = over.fit_resample(X, y)
                    X_sos['mal']=y_sos
                    print('oversampling：{}'.format(Counter(y_sos)),file=output_file)
                    overData=X_sos
                  
                    run = 1
                    np.random.seed(run)
                    trainData=overData
                x_Train = trainData[x_cols].values
                y_Train = trainData['mal'].values
                y_TrainBin = y_Train > 0
                if(len(x_Train)==0):
                    print(file,len(x_Train))
                    continue
                print("",file,"\n",len(trainData[trainData['mal']==0]),len(trainData[trainData['mal']==1]),"\n",file=output_file)
  
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

                elif confusion[1][0]==0 and confusion[1][1] == 0:
                    mal_user=0

                else:
                    mal_user=1

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

                output_file.write(f"Confusion Matrix:\n{confusion}\n")
                output_file.write(f"Accuracy: {accuracy}\n")
                output_file.write(f"Recall: {recall}\n")
                output_file.write(f"F1 Score: {f1}\n")
                output_file.write(f"over!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!")

df2.to_csv('partRedo/file.csv',index=False)
df0.to_csv('partRedo/file_0.csv',index=False)
df1.to_csv('partRedo/file_1.csv',index=False)

output_file.close()
