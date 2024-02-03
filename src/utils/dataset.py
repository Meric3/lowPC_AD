import pandas as pd
import sklearn.preprocessing 
from sklearn.decomposition import PCA
import torch
import numpy as np

def swat_preprocessing(train, test, validation = False):
    train = train.drop(['P2_P202', 'P4_P401', 'P4_P404', 'P5_P502', 'P6_P601', 'P6_P603'], axis=1)
    test = test.drop(['P2_P202', 'P4_P401', 'P4_P404', 'P5_P502', 'P6_P601', 'P6_P603'], axis=1)
    
    x_colname = list(train.columns)[:-1]
    y_colname = list(train.columns)[-1]

    train_x = train[x_colname]
    train_y = train[y_colname]
    test_x = test[x_colname]
    test_y= test[y_colname]

    test_y[test_y == 'Attack'] = 1
    test_y[test_y == 'Normal'] = 0
    return train_x, test_x, test_y

def pca_return(x_train, x_test, variance = True):
    if variance == False:
        scaler = sklearn.preprocessing.StandardScaler(with_std = False)
    else:
        scaler = sklearn.preprocessing.StandardScaler()
    scaler.fit(x_train)
    
    # train zero center about train data. For fairness test dataset scaled from x_train information
    x_train = scaler.transform(x_train)
    x_test = scaler.transform(x_test)

    pca = PCA(n_components = np.shape(x_train)[1])
    pca.fit(x_train)
    
    x_train = pca.transform(x_train)
    x_test = pca.transform(x_test)
    
    return x_train, x_test


def lstm_dataset(train_path, test_path):
    x_train, y_train, x_test, y_test = base_dataset(train_path, test_path)

    # SInce sklearn osvm property, set atk = -1, normal = 1
    y_train[y_train == 'Attack'] = 1
    y_train[y_train == 'Normal'] = 0
    y_test[y_test == 'Attack'] = 1
    y_test[y_test == 'Normal'] = 0
    
    y_train = torch.FloatTensor(y_train).cuda()
    y_test = torch.FloatTensor(y_test).cuda()    
    
    x_train, x_test = pca_return(x_train, x_test)
    
    x_train = torch.FloatTensor(x_train).cuda()
    x_test = torch.FloatTensor(x_test).cuda()
    
    return  x_train, x_test,  y_test    
    


def base_dataset(train_path, test_path, isprint = False):
    #load data
    normal_data = pd.read_csv(train_path, index_col = 0)
    attack_data = pd.read_csv(test_path, index_col = 0)

    x_col = list(normal_data.columns)[:-1]
    y_col = list(normal_data.columns)[-1]

    X_normal = normal_data[x_col]
    Y_normal = normal_data[y_col]

    X_attack = attack_data[x_col]
    Y_attack = attack_data[y_col]
    
    if (isprint == True):
        unique, counts = np.unique(Y_normal, return_counts=True)
        print('In train :', dict(zip(unique, counts)))
        unique, counts = np.unique(Y_attack, return_counts=True)
        print('In test :', dict(zip(unique, counts)))     
        print("data length is {}".format(len(x_col)))
    
    return X_normal, Y_normal, X_attack, Y_attack
 

def check(data) :
    if -1 in data :
        return -1
    else : 
        return 1
    
def svm_dataset(train_path, test_path):
    x_train, y_train, x_test, y_test = base_dataset(train_path, test_path)

    # SInce sklearn osvm property, set atk = -1, normal = 1
    y_train[y_train == 'Attack'] = -1
    y_train[y_train == 'Normal'] = +1
    y_test[y_test == 'Attack'] = -1
    y_test[y_test == 'Normal'] = +1      

    x_train, x_test = pca_return(x_train, x_test, True)

    x_train = pd.DataFrame(data = x_train
             , columns = ['PC%d'%(i) for i in range(np.shape(x_train)[1])])

    x_test = pd.DataFrame(data = x_test
             , columns = ['PC%d'%(i) for i in range(np.shape(x_train)[1])])

    # Moving average Window
    x_train = x_train.rolling(10, min_periods = 1).mean()
    x_test = x_test.rolling(10, min_periods = 1).mean()

    y_train = y_train.rolling(10, min_periods=1).apply(check)
    y_test = y_test.rolling(10, min_periods=1).apply(check)


    return  x_train.values, x_test.values,  y_test.values


# SWaT DATA LOAD Func
def dataset(train_path, test_path):
    
    #load data
    normal_data = pd.read_csv(train_path, index_col = 0)
    attack_data = pd.read_csv(test_path, index_col = 0)
        
#     normal_data = normal_data.drop(['P2_P202', 'P4_P401', 'P4_P404', 'P5_P502', 'P6_P601', 'P6_P603'], axis=1)
#     attack_data = attack_data.drop(['P2_P202', 'P4_P401', 'P4_P404', 'P5_P502', 'P6_P601', 'P6_P603'], axis=1)
    
    # columns setting 
    x_col = list(normal_data.columns)[:-1]
    y_col = list(normal_data.columns)[-1]
    
    # X, Y setting
    X_normal = normal_data[x_col]
    Y_normal = normal_data[y_col]
    
    X_attack = attack_data[x_col]
    Y_attack = attack_data[y_col]
    
    # label setting ( attack = 1, normal = 0)
    Y_train = Y_normal.copy()
    Y_test = Y_attack.copy()
    
    Y_train[Y_train == 'Attack'] = 1
    Y_train[Y_train == 'Normal'] = 0

    Y_test[Y_test == 'Attack'] = 1
    Y_test[Y_test == 'Normal'] = 0  
    
    Y_train = torch.FloatTensor(Y_train).cuda()
    Y_test = torch.FloatTensor(Y_test).cuda()
    
    unique, counts = np.unique(Y_normal, return_counts=True)
    print('In train :', dict(zip(unique, counts)))
    unique, counts = np.unique(Y_attack, return_counts=True)
    print('In test :', dict(zip(unique, counts)))     
    print("data length is {}".format(len(x_col)))
    
    # label index
    train_index = normal_data.index
    test_index = attack_data.index
    
    scaler = sklearn.preprocessing.StandardScaler()
    scaler.fit(X_normal)
    # train zero center about train data
    X_train = scaler.transform(X_normal)

    # test zero center about train data
    X_test = scaler.transform(X_attack)

    pca = PCA(n_components = len(x_col))

    pca.fit(X_train)
    X_train = pca.transform(X_train)
    X_test = pca.transform(X_test)
    
    X_train = torch.FloatTensor(X_train).cuda()
    X_test = torch.FloatTensor(X_test).cuda()
    
    train_row = X_train.size(0)
    train_col = X_train.size(1)
    test_row = X_test.size(0) 
    test_col = X_test.size(1)
        
    
    return  X_train, X_test,  Y_test