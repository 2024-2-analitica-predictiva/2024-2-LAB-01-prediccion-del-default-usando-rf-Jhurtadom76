import pandas as pd
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.preprocessing import OneHotEncoder
from sklearn.ensemble import RandomForestClassifier
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.metrics import precision_score, balanced_accuracy_score, recall_score, f1_score, confusion_matrix
import joblib
import gzip
import os
import json

df = pd.read_csv("files/input/default_of_credit_card_clients.csv", sep=';')

df.rename(columns={'default payment next month': 'default'}, inplace=True)

df.drop(columns=['ID'], inplace=True)

df.dropna(inplace=True)

df['EDUCATION'] = df['EDUCATION'].apply(lambda x: 'others' if x > 4 else x)

categorical_features = df.select_dtypes(include=['object']).columns.tolist()
df[categorical_features] = df[categorical_features].astype(str)

X = df.drop(columns=['default'])  
y = df['default']  

x_train, x_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

categorical_features = X.select_dtypes(include=['object']).columns.tolist()

preprocessor = ColumnTransformer(
    transformers=[
        ('cat', OneHotEncoder(), categorical_features)
    ])

rf_model = RandomForestClassifier(random_state=42)

pipeline = Pipeline(steps=[
    ('preprocessor', preprocessor),  # Primero, aplicamos la transformación de variables categóricas
    ('classifier', rf_model)  # Después, ajustamos el modelo de Random Forest
])

param_grid = {
    'classifier__n_estimators': [50, 100, 200],  # Número de árboles en el bosque
    'classifier__max_depth': [None, 10, 20, 30],  # Profundidad máxima del árbol
    'classifier__min_samples_split': [2, 5, 10],  # Número mínimo de muestras necesarias para dividir un nodo
    'classifier__min_samples_leaf': [1, 2, 4],  # Número mínimo de muestras en una hoja
    'classifier__bootstrap': [True, False]  # Si usar bootstrap para muestrear los datos
}

grid_search = GridSearchCV(pipeline, param_grid, cv=10, scoring='balanced_accuracy', n_jobs=-1, verbose=2)

grid_search.fit(x_train, y_train)

y_train_pred = grid_search.predict(x_train)
y_test_pred = grid_search.predict(x_test)

metrics = [
    {
        'dataset': 'train',
        'precision': precision_score(y_train, y_train_pred),
        'balanced_accuracy': balanced_accuracy_score(y_train, y_train_pred),
        'recall': recall_score(y_train, y_train_pred),
        'f1_score': f1_score(y_train, y_train_pred),
        'confusion_matrix': confusion_matrix(y_train, y_train_pred).tolist()
    },
    {
        'dataset': 'test',
        'precision': precision_score(y_test, y_test_pred),
        'balanced_accuracy': balanced_accuracy_score(y_test, y_test_pred),
        'recall': recall_score(y_test, y_test_pred),
        'f1_score': f1_score(y_test, y_test_pred),
        'confusion_matrix': confusion_matrix(y_test, y_test_pred).tolist()
    }
]

os.makedirs('files/output', exist_ok=True)

metrics_filename = 'files/output/metrics.json'
with open(metrics_filename, 'w') as f:
    json.dump(metrics, f, indent=4)