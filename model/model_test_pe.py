import os
import pandas as pd
import xgboost as xgb
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.metrics import mean_squared_error, confusion_matrix
from sklearn.externals import joblib 


input_path = '../data/csv/.csv'
model_path = '../data/pkl/.pkl'


def main():
    data = pd.read_csv(input_path)
    x_data = data.drop(['filename', 'class'], axis = 1)
    
    model = joblib.load(model_path)
    print(model)
    software_df = pd.DataFrame({'filename': list(data['filename'].values), 'class': list(model.predict(x_data))})
    software_df.set_index('filename').to_csv('software.csv')
    print(software_df)
    

main()
