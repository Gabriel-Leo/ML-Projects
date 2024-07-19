import numpy as np
import pandas as pd
import matplotlib.ticker as ticker
from matplotlib import pyplot as plt
import seaborn as sn
from scipy import stats
from sklearn.svm import LinearSVC
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import GridSearchCV
from sklearn.preprocessing import MinMaxScaler as MMS
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score


'----------------------------------------------DATA VISUALIZATIONS--------------------------------------------------------------------'
'-------------------------------------------------------------------------------------------------------------------------------------'

# Plot distribution of values
def show_feature_bar(df):
    fig = plt.bar(np.sort(df.Pregnancies.unique()), df.Pregnancies.value_counts().sort_index())
    plt.gca().xaxis.set_major_locator(ticker.MaxNLocator(integer=True))
    plt.show()

def show_features_histograms(df):
    features = ['Glucose', 'BloodPressure', 'SkinThickness', 'Insulin', 'BMI', 'DiabetesPedigreeFunction']
    
    fig, axs = plt.subplots(2, 3, figsize=(15, 10))
    axs = axs.flatten()

    for i, feature in enumerate(features):
        axs[i].hist(df[feature], bins=20, facecolor='blue')
        axs[i].set_title(feature)
    
    plt.tight_layout()
    plt.show()

# Visualize correlations
def show_heatmap(df):
    sn.heatmap(data = df.corr(), annot = True, fmt='.1f')
    plt.show()
'''
BloodPressure, SkinThickness and Insulin show a very low correlation with the target. Thus they may be a source of noise and 
should be discarded.
'''

# Visualization of the relationship between pairs of variables
def show_pairplot(df):
    sn.pairplot(df, hue ='Outcome', vars = ['Pregnancies', 'Glucose', 'BMI', 'DiabetesPedigreeFunction'])
    plt.show()


'---------------------------------------------DATA CLEANING AND MODEL TRAINING--------------------------------------------------------'
'-------------------------------------------------------------------------------------------------------------------------------------'

def clean_data(df):
    # Drop potential sources of noise
    df.drop('BloodPressure', axis=1, inplace=True)
    df.drop('SkinThickness', axis=1, inplace=True)
    df.drop('Insulin', axis=1, inplace=True)

    # Find missing values
    for feature in df.columns:
        print(f'Total missing values for {feature}: {df[feature].isna().sum()}')
    print()
    'No missing data was found'

    # Find duplicated data
    print(f'Total duplicated observations: {df.duplicated().sum()}\n')
    'No duplicated data was found'

    # Remove invalid data (real life BMI and glucose can't be 0)
    conditions = (df['Glucose'] == 0) | (df['BMI'] == 0)
    df = df[~conditions]

    # Delete outliers with IQR method
    for feature in ['Glucose', 'BMI', 'DiabetesPedigreeFunction']:
        
        # Calculate Q1 (25th percentile) and Q3 (75th percentile)
        Q1 = df[feature].quantile(0.25)
        Q3 = df[feature].quantile(0.75)

        # Calculate IQR
        IQR = Q3 - Q1

        # Define the bounds for outliers
        lower_bound = Q1 - 1.5 * IQR
        upper_bound = Q3 + 1.5 * IQR

        # Filter out outliers
        df = df[(df[feature] >= lower_bound) & (df[feature] <= upper_bound)]

    return df

def transform_data(df, X_train, X_test):
    
    # Analize if the data follows a normal distribution
    features = ['Pregnancies','Glucose', 'BMI', 'DiabetesPedigreeFunction', 'Age']
    for feature in features:
        print(f'{feature} kurtosis: {stats.kurtosis(df[feature])}')
        print(f'{feature} skew: {stats.skew(df[feature])}')
    print()
    '''
    Pregnancies and DiabetesPedigreeFunction are too skew to be considered normal distribution, for the sake of simplicity standardization
    will be applied instead of normalization, to avoid non linear transformations
    '''

    scaler = MMS().fit(X_train)
    return scaler.transform(X_train), scaler.transform(X_test)

def train_lr(X80, X20, y80, y20):
    clf = LogisticRegression() 
 
    # Find best parameters for the model
    grid_values = {'C': [0.1, 0.5, 1.0, 10], 'penalty': ['l2']}
    grid_clf = GridSearchCV(clf, param_grid = grid_values, scoring='accuracy')
    grid_clf.fit(X80, y80)
    best_model = grid_clf.best_estimator_

    # Test accuracy for train and test (to look for best model and assure no overfitting is happening)
    y20_predicted = best_model.predict(X20)
    y80_predicted = best_model.predict(X80)

    return (f'train accuracy = {accuracy_score(y80, y80_predicted)}, test accuracy = {accuracy_score(y20, y20_predicted)}')

def train_svm(X80, X20, y80, y20):
    clf = LinearSVC()
    
    # Find best parameters for the model
    grid_values = {'C': [0.1, 0.5, 1.0, 10]}
    grid_clf = GridSearchCV(clf, param_grid = grid_values, scoring='accuracy')
    grid_clf.fit(X80, y80)
    best_model = grid_clf.best_estimator_

    # Test accuracy for train and test (to look for best model and assure no overfitting is happening)
    y20_predicted = best_model.predict(X20)
    y80_predicted = best_model.predict(X80)

    return (f'train accuracy = {accuracy_score(y80, y80_predicted)}, test accuracy = {accuracy_score(y20, y20_predicted)}')

def train_rf(X80, X20, y80, y20):
    clf = RandomForestClassifier()

    # Find best parameters for the model
    grid_values = {'n_estimators': [25, 50, 75, 100], 'max_features': [1, 2, 3], 'max_depth' : [2, 3]}
    grid_clf = GridSearchCV(clf, param_grid = grid_values, scoring='accuracy')
    grid_clf.fit(X80, y80)
    best_model = grid_clf.best_estimator_

    # Test accuracy for train and test (to look for best model and assure no overfitting is happening)
    y20_predicted = best_model.predict(X20)
    y80_predicted = best_model.predict(X80)

    return (f'train accuracy = {accuracy_score(y80, y80_predicted)}, test accuracy = {accuracy_score(y20, y20_predicted)}')

def train_models(X80, X20, y80, y20):
    results = dict()
    results['Logistic_Regression'] = train_lr(X80, X20, y80, y20)
    results['Support Vector Machine'] = train_svm(X80, X20, y80, y20)
    results['Random Forest'] = train_rf(X80, X20, y80, y20)
    return results

def main():
    diabetes_df = pd.read_csv('diabetes.csv')
    # https://www.kaggle.com/datasets/lara311/diabetes-dataset-using-many-medical-metrics/data

    # Analize data imbalance
    print(diabetes_df.Outcome.value_counts())
    '''
    The data presents a moderate imbalance of the target classes: 34,89% to 65,11%. 
    Since the imbalance is not severe and solving this imbalance would require oversampling or undersampling, 
    which can either overfit the data or produce an information loss, I'll fit a model only with the original data.
    '''

    # Clean the data
    clean_df = clean_data(diabetes_df)

    # Find number of observations and columns after data cleaning
    print(clean_df.shape, '\n')
    'Cleaning did not drastically reduce the number of observations. Still, 80% of 717 may be few observations when training a ML algorithm.' 

    # Split the data
    X = clean_df.loc[:, clean_df.columns != 'Outcome']
    y = clean_df['Outcome']    
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    
    # Transform the features used for prediction
    X_train_scaled, X_test_scaled = transform_data(clean_df, X_train, X_test)

    # Train different models and return their accuracy to compare their performance
    model_scores = train_models(X_train_scaled, X_test_scaled, y_train, y_test)
    print(model_scores)

    # Example Results
    '''
    {'Logistic_Regression': 'train accuracy = 0.7661431064572426, test accuracy = 0.8472222222222222', 
    'Support Vector Machine': 'train accuracy = 0.7678883071553229, test accuracy = 0.8472222222222222', 
    'Random Forest': 'train accuracy = 0.7958115183246073, test accuracy = 0.8402777777777778'}
    '''

main()