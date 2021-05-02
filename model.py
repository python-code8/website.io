from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import train_test_split
from sklearn import metrics
import numpy as np
import pandas as pd
import pickle

df = pd.read_csv(r'C:\Users\Gopal\Desktop\HTML\diabetes.csv')
d = df[['Pregnancies', 'Glucose', 'BloodPressure', 'SkinThickness', 'Insulin', 'BMI', 'DiabetesPedigreeFunction', 'Age']] = df[[
    'Pregnancies', 'Glucose', 'BloodPressure', 'SkinThickness', 'Insulin', 'BMI', 'DiabetesPedigreeFunction', 'Age']].replace(0, np.nan)

df['Pregnancies'].fillna(df['Pregnancies'].mean(), inplace=True)
df['Glucose'].fillna(df['Glucose'].mean(), inplace=True)
df['BloodPressure'].fillna(df['BloodPressure'].mean(), inplace=True)
df['SkinThickness'].fillna(df['SkinThickness'].mean(), inplace=True)
df['Insulin'].fillna(df['Insulin'].mean(), inplace=True)
df['BMI'].fillna(df['BMI'].mean(), inplace=True)
df['DiabetesPedigreeFunction'].fillna(
    df['DiabetesPedigreeFunction'].mean(), inplace=True)
df['Age'].fillna(df['Age'].mean(), inplace=True)
df.head(30)

q_hi = []
q_low = []

for i in range(2, 7):
    hi = df[df.columns[i]].quantile(.15)
    low = df[df.columns[i]].quantile(.85)
    q_hi.append(hi)
    q_low.append(low)

for j in range(5):
    clean_data = df[(df[df.columns[j+2]] < q_hi[j]) &
                    (df[df.columns[j+2]] < q_low[j])]


x = clean_data.drop(['Outcome'], axis=1)
y = clean_data['Outcome']

clf = DecisionTreeClassifier(criterion="gini")
classifier = clf.fit(x, y)

pickle.dump(classifier, open('model.pkl', 'wb'))

model = pickle.load(open('model.pkl', 'rb'))


print(model.predict([[1.0, 90, 70, 17.0, 150, 20.2, 0.130, 42]]))
