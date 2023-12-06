import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import RandomForestRegressor
from sklearn.svm import SVR
from sklearn.metrics import mean_squared_error
import numpy as np

def sort_dataset(dataset_df):
    #TODO: Implement this function
    return dataset_df.sort_values('year', ascending=True)

def split_dataset(dataset_df):
	#TODO: Implement this function
    dataset_df['salary'] = dataset_df['salary'] * 0.001
    train_df = dataset_df.iloc[:1718]
    test_df = dataset_df.iloc[1718:]
    
    return train_df.drop('salary', axis=1), test_df.drop('salary', axis=1), train_df['salary'], test_df['salary']

def extract_numerical_cols(dataset_df):
	#TODO: Implement this function
    num_cols = ['age', 'G', 'PA', 'AB', 'R', 'H', '2B', '3B', 'HR', 'RBI', 'SB', 'CS', 'BB', 'HBP', 'SO', 'GDP', 'fly', 'war']
    return dataset_df[num_cols]

def train_predict_decision_tree(X_train, Y_train, X_test):
    #TODO: Implement this function
    model = DecisionTreeRegressor(random_state=42)
    model.fit(X_train, Y_train)
    return model.predict(X_test)

def train_predict_random_forest(X_train, Y_train, X_test):
	#TODO: Implement this function
    model = RandomForestRegressor(random_state=42)
    model.fit(X_train, Y_train)
    return model.predict(X_test)

def train_predict_svm(X_train, Y_train, X_test):
	#TODO: Implement this function
    model = SVR()
    model.fit(X_train, Y_train)
    return model.predict(X_test)

def calculate_RMSE(labels, predictions):
    #TODO: Implement this function
    mse = mean_squared_error(labels, predictions)
    return np.sqrt(mse)

if __name__=='__main__':
    data_df = pd.read_csv('2019_kbo_for_kaggle_v2.csv')

    sorted_df = sort_dataset(data_df)  
    X_train, X_test, Y_train, Y_test = split_dataset(sorted_df)

    X_train = extract_numerical_cols(X_train)
    X_test = extract_numerical_cols(X_test)

    dt_predictions = train_predict_decision_tree(X_train, Y_train, X_test)
    rf_predictions = train_predict_random_forest(X_train, Y_train, X_test)
    svm_predictions = train_predict_svm(X_train, Y_train, X_test)

    print ("Decision Tree Test RMSE: ", calculate_RMSE(Y_test, dt_predictions))    
    print ("Random Forest Test RMSE: ", calculate_RMSE(Y_test, rf_predictions))    
    print ("SVM Test RMSE: ", calculate_RMSE(Y_test, svm_predictions))