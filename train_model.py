# train_model.py
import argparse
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
import joblib


def load_data(path):
    df = pd.mobile.csv("C:\Users\rejir\Downloads\mobile-price-prediction\Data")
    return df


def preprocess(df):
    # Basic preprocessing: no missing values expected in this dataset
    X = df.drop('price_range', axis=1, errors='ignore')
    if 'Price_range' in df.columns:
        y = df['Price_range']
    elif 'price_range' in df.columns:
        y = df['price_range']
    else:
        raise ValueError('Target column not found. Expected Price_range or price_range')
    return X, y


def main(data_path, out_path):
    df = load_data(data_path)
    X, y = preprocess(df)

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)

    pipe = Pipeline([
        ('scaler', StandardScaler()),
        ('clf', RandomForestClassifier(random_state=42, n_jobs=-1))
    ])

    param_grid = {
        'clf__n_estimators': [100, 200],
        'clf__max_depth': [None, 10, 20]
    }

    gs = GridSearchCV(pipe, param_grid, cv=3, n_jobs=-1, verbose=1)
    gs.fit(X_train, y_train)

    print('Best params:', gs.best_params_)
    preds = gs.predict(X_test)
    print(classification_report(y_test, preds))
    print('Confusion matrix:\n', confusion_matrix(y_test, preds))

    # save model and the columns used
    joblib.dump({'model': gs.best_estimator_, 'columns': X.columns.tolist()}, out_path)
    print('Saved model to', out_path)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--data_path', required=True)
    parser.add_argument('--out', required=True)
    args = parser.parse_args()
    main(args.data_path, args.out)