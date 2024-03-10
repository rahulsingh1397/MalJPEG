import pandas as pd
import numpy as np
from sklearn import utils 
from glob import glob
import lightgbm as lgb

train = pd.read_csv(r'E:\Python\MalJPEG\MalJPEG\Dataset\train.csv',header=None)
x_train = np.array(train.iloc[:, 0:9])        
y_train = np.array(train.iloc[:,10])

test = pd.read_csv(r'E:\Python\MalJPEG\MalJPEG\Dataset\test.csv',header=None)
x_test = np.array(test.iloc[:, 0:9])        
y_test = np.array(test.iloc[:,10])


lgb_train = lgb.Dataset(x_train, y_train)
lgb_test = lgb.Dataset(x_test, y_test)
lgb_eval = lgb.Dataset(x_test, y_test, reference=lgb_train)


params = {
    'boosting_type': 'gbdt',
    'objective': 'binary',
    'metric': {'l2', 'l1'},
    'num_leaves': 21,
    'learning_rate': 0.05,
    'feature_fraction': 0.9,
    'bagging_fraction': 0.8,
    'bagging_freq': 5,
    'verbose': 0   
}


#GridSearchRESULT
# params = {
#     'boosting_type': 'gbdt',
#     'objective': 'binary',
#     'metric': {'l2', 'l1'},
#     'bagging_fraction': 0.7,
#     'bagging_freq': 6,
#     'feature_fraction': 0.9,
#     'learning_rate': 0.1,
#     'num_leaves': 20,
#     'verbose': 0,
#     'device':'gpu',
#     }
# params = {
#   'boosting_type': 'gbdt', 
#     'objective': 'binary',
#     'metric': {'l2', 'l1'},
#     'num_leaves': 31,
#     'learning_rate': 0.05,
#     'feature_fraction': 0.9,
#     'bagging_fraction': 0.8,
#     'bagging_freq': 5,
#     'verbose': 0
# }
gbm = lgb.train(params,
                lgb_train,
                num_boost_round=20,
                valid_sets=lgb_eval,
                callbacks=[
                    lgb.early_stopping(stopping_rounds=5)]
                )

gbm.save_model('JPEG_LGBM.txt')


from sklearn.metrics import accuracy_score                
from sklearn.metrics import f1_score
from sklearn.metrics import recall_score
from sklearn.model_selection import GridSearchCV
from sklearn.model_selection import KFold

model_output = gbm.predict(x_test, num_iteration=gbm.best_iteration)

print(model_output)
print(y_test)


# Convert probabilities to binary class labels (0 or 1)
predicted_labels = model_output.round()

print("Accuracy = %.4f "%(accuracy_score(y_test.astype(int), predicted_labels.astype(int), normalize=True)*100))
print("F1 Score = %.4f "%f1_score(y_test.astype(int), predicted_labels.astype(int), average='weighted'))
print("Recall Score = %.4f "%recall_score(y_test.astype(int), predicted_labels.astype(int), average='weighted', zero_division =1))

A = (accuracy_score(y_test.astype(int), predicted_labels.astype(int), normalize=True)*100)
F = f1_score(y_test.astype(int), predicted_labels.astype(int), average='weighted')
R = recall_score(y_test.astype(int), predicted_labels.astype(int), average='weighted', zero_division =1)

gbm.save_model('MalJPEG.txt')

# import subprocess 

# try:
#     file4_path = (r"C:\Users\rahul\OneDrive\Desktop\Image\Code\Graph.py")          
#     subprocess.run(['python', file4_path])
# except:
#     print("Error Running the file")
