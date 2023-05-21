from imblearn.over_sampling import RandomOverSampler
from imblearn.under_sampling import RandomUnderSampler
from imblearn.over_sampling import SMOTE
from imblearn.over_sampling import ADASYN


def displayValues(y_train):
    print('Non-Frauds:', y_train.value_counts()[0], '/', round(y_train.value_counts()[0]/len(y_train) * 100,2), '% of the dataset')
    print('Frauds:', y_train.value_counts()[1], '/',round(y_train.value_counts()[1]/len(y_train) * 100,2), '% of the dataset')

def balancedWithRandomOverSampler(X_train, y_train):
    ros = RandomOverSampler(random_state=50)
    X_train_ros, y_train_ros = ros.fit_resample(X_train, y_train)
    
    displayValues(y_train_ros)
    
    return X_train_ros, y_train_ros

def balancedWithRandomUnderSampler(X_train, y_train):
    rus = RandomUnderSampler(random_state=50)
    X_train_rus, y_train_rus = rus.fit_resample(X_train, y_train)
    
    displayValues(y_train_rus)
    
    return X_train_rus, y_train_rus

def balanceWithSMOTE(X_train, y_train):
    smote = SMOTE(random_state=42)
    X_train_smote, y_train_smote = smote.fit_resample(X_train, y_train)

    displayValues(y_train_smote)
    
    return X_train_smote, y_train_smote

def balanceWithADASYN(X_train, y_train):
	adasyn = ADASYN(random_state=42)
	X_train_adasyn, y_train_adasyn = adasyn.fit_resample(X_train, y_train)

	displayValues(y_train_adasyn)
	return X_train_adasyn, y_train_adasyn