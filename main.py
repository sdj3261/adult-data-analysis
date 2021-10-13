import pandas as pd
import numpy as np
import seaborn as sns
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
import sklearn
from sklearn import preprocessing
from sklearn.feature_selection import SelectKBest
from sklearn.feature_selection import f_classif
import matplotlib.pyplot as plt
import math
from sklearn import preprocessing
from sklearn import linear_model

# data_url = "C:/Users/jinkyung/Desktop/dataSet.csv"
data_url = "C:\\Users\\tlseh\\OneDrive\\바탕 화면\\과제\\3-1\\데이터과학\\termproject.csv"
# data_url = 'D:/VSCODE/python/TP/dataSet.csv'
dataSet = pd.read_csv(data_url)

# data preprocessing
data = pd.DataFrame(dataSet)

# print(data.head())
# print(data.info())
# print(data.describe())

print(data.isna().sum())

work_col = data['hours-per-week']
data = data.drop('hours-per-week', axis=1)
# Find Missing Value
missing_value = [" ?"]

# Handling Missing Value
for i in range(len(missing_value)):
    data.replace({missing_value[i]: np.NaN}, inplace=True)

print(data.isna().sum())
data = data.fillna(axis=0, method="bfill")
# Income data labeling
data.Income = data.Income.replace(' <=50K', 0)
data.Income = data.Income.replace(' >50K', 1)

# print(data)
# print(data.columns)
# print(data.isna().sum())

# importance hours-per-week
temp_data = data.copy()
temp_data = pd.concat([temp_data, work_col], axis=1)

# print(temp_data)
temp_data = temp_data.dropna()
x = pd.get_dummies(temp_data.drop(['hours-per-week'], axis=1))
y = temp_data['hours-per-week']

bestfeature = SelectKBest(f_classif, k='all')
fit = bestfeature.fit(x, y)

dfcolumns = pd.DataFrame(x.columns)
dfscores = pd.DataFrame(fit.scores_)

featureScores = pd.concat([dfcolumns, dfscores], axis=1)
featureScores.columns = ['feature', 'Score']

print(featureScores.nlargest(60, 'Score'))

# correlation hour-per-week with other feature
corrmat = temp_data.corr()  # corr() computes pairwise  correlations of features in a Data Frame
top_corr_features = corrmat.index
plt.figure(figsize=(20, 20)).show()
# plot the heat map
sns.heatmap(temp_data[top_corr_features].corr(), annot=True, cmap="RdYlGn")
plt.show()

# find outlier
fig, ax = plt.subplots(1, 2, figsize=(16, 4))
ax[0].boxplot(data['Age'])
ax[0].set_title("Age")
ax[1].boxplot(y)
ax[1].set_title("hours per week")
plt.show()
# Data preprocessing with regression
# hours-per-work regression by age
age_col = data['education-num']
age_col = pd.DataFrame(data['education-num'])
temp_age_col = age_col

# standard scaler visual
df = data.copy()
temp_work_col = work_col
df['hours-per-week'] = work_col
print(df)
score = np.array(df[['education-num', 'hours-per-week']])
scaler = preprocessing.StandardScaler()
scaled_df = scaler.fit_transform(score)
scaled_df = pd.DataFrame(scaled_df, columns=['education-num', 'hours-per-week'])
print(scaled_df)
print(scaled_df.tail(30))

edu_col = scaled_df['education-num']
edu_col = np.array(edu_col)
work_col = scaled_df['hours-per-week']
work_col = np.array(work_col)

# regression start
x_train = []
y_train = []
x_test = []
index = []
for i in range(len(edu_col)):
    if (np.isnan(work_col[i])):
        x_test.append(edu_col[i])
    else:
        x_train.append(edu_col[i])
        y_train.append(work_col[i])

# regression model
E = linear_model.LinearRegression()
E.fit(np.array(x_train)[:, np.newaxis], np.array(y_train))
y_test = E.predict(np.array(x_test)[:, np.newaxis])

# regression 시각화
x_train = np.array(x_train)
px_edu = np.array([x_train.min() - 1, x_train.max() + 1])
py_work = E.predict(px_edu[:, np.newaxis])

print("hour-per-week regreesion start")
# plt.scatter(x_train, y_train)
plt.scatter(x_test, y_test, color='red')
plt.plot(px_edu, py_work, color='black')
plt.xlabel("education-num")
plt.ylabel('Hours per week')
plt.title('Linear Regression - education-num / Hours-per-week')
plt.show()
# print predict hours-per-week
# print(y_test)

work_col = temp_work_col
age_col = temp_age_col

for i in range(len(work_col)):
    if (np.isnan(work_col[i])):
        work_col[i] = round(y_test[i])
print(work_col)
# dataSet merge
data = pd.concat([data, work_col], axis=1)
print(data)

# workclass 'private drop'
index = data[data['workclass'] == ' Private'].index
data = data.drop(index)
print(data)

# Category labeling
a = data.loc[data['native-country'] == ' United-States']
a.loc[a['education'] == ' Preschool', 'education'] = "NoSC"
a.loc[a['education'] == ' 1st-4th', 'education'] = "ElementarySC"
a.loc[a['education'] == ' 5th-6th', 'education'] = "ElementarySC"
a.loc[a['education'] == ' 7th-8th', 'education'] = "MiddleSC"
a.loc[a['education'] == ' 9th', 'education'] = "HighSC"
a.loc[a['education'] == ' 10th', 'education'] = "HighSC"
a.loc[a['education'] == ' 11th', 'education'] = "HighSC"
a.loc[a['education'] == ' 12th', 'education'] = "HighSC"
a.loc[a['education'] == ' Assoc-voc', 'education'] = "Assoc"
a.loc[a['education'] == ' Prof-school', 'education'] = "Prof-school"
a.loc[a['education'] == ' Some-college', 'education'] = "Techinical-Collage"
a.loc[a['education'] == ' Bachelors', 'education'] = "4-Collage"
a.loc[a['education'] == ' Doctorate', 'education'] = "Doctorate"
a.loc[a['education'] == ' Masters', 'education'] = "Masters"

data = a
# print(data)

# plot the Rich person
temp_data = data.copy()
temp_data['hours-per-week'] = work_col

# independenct feature and target feature setting
X = pd.get_dummies(data.drop(['Income'], axis=1))
y = data['Income']
# print(X)
# print(y)

# select important feature for target feature
bestfeature = SelectKBest(f_classif, k='all')
fit = bestfeature.fit(X, y)

dfcolumns = pd.DataFrame(X.columns)
dfscores = pd.DataFrame(fit.scores_)

featureScores = pd.concat([dfcolumns, dfscores], axis=1)
featureScores.columns = ['feature', 'Score']

print(featureScores.nlargest(60, 'Score'))

temp_data = data.drop(['native-country'], axis=1)
temp_data = temp_data.drop(['race'], axis=1)

temp_data.workclass = temp_data.workclass.replace(' Self-emp-not-inc', 0)
temp_data.workclass = temp_data.workclass.replace(' Federal-gov', 1)
temp_data.workclass = temp_data.workclass.replace(' Local-gov', 2)
temp_data.workclass = temp_data.workclass.replace(' Never-worked', 3)
temp_data.workclass = temp_data.workclass.replace(' Self-emp-inc', 4)
temp_data.workclass = temp_data.workclass.replace(' State-gov', 5)
temp_data.workclass = temp_data.workclass.replace(' Without-pay', 6)

temp_data.education = temp_data.education.replace(' Assoc-acdm', 0)
temp_data.education = temp_data.education.replace(' HS-grad', 1)
temp_data.education = temp_data.education.replace('4-Collage', 2)
temp_data.education = temp_data.education.replace('Assoc', 3)
temp_data.education = temp_data.education.replace('Doctorate', 4)
temp_data.education = temp_data.education.replace('ElementarySC', 5)
temp_data.education = temp_data.education.replace('HighSC', 6)
temp_data.education = temp_data.education.replace('Masters', 7)
temp_data.education = temp_data.education.replace('MiddleSC', 8)
temp_data.education = temp_data.education.replace('NoSC', 9)
temp_data.education = temp_data.education.replace('Prof-school', 10)
temp_data.education = temp_data.education.replace('Techinical-Collage', 11)

temp_data.marital = temp_data.marital.replace(' Divorced', 0)
temp_data.marital = temp_data.marital.replace(' Married-AF-spouse', 1)
temp_data.marital = temp_data.marital.replace(' Married-civ-spouse', 2)
temp_data.marital = temp_data.marital.replace(' Married-spouse-absent', 3)
temp_data.marital = temp_data.marital.replace(' Never-married', 4)
temp_data.marital = temp_data.marital.replace(' Separated', 5)
temp_data.marital = temp_data.marital.replace(' Widowed', 6)

temp_data.occupation = temp_data.occupation.replace(' Adm-clerical', 0)
temp_data.occupation = temp_data.occupation.replace(' Armed-Forces', 1)
temp_data.occupation = temp_data.occupation.replace(' Craft-repair', 2)
temp_data.occupation = temp_data.occupation.replace(' Exec-managerial', 3)
temp_data.occupation = temp_data.occupation.replace(' Farming-fishing', 4)
temp_data.occupation = temp_data.occupation.replace(' Handlers-cleaners', 5)
temp_data.occupation = temp_data.occupation.replace(' Machine-op-inspct', 6)
temp_data.occupation = temp_data.occupation.replace(' Other-service', 7)
temp_data.occupation = temp_data.occupation.replace(' Prof-specialty', 8)
temp_data.occupation = temp_data.occupation.replace(' Protective-serv', 9)
temp_data.occupation = temp_data.occupation.replace(' Sales', 10)
temp_data.occupation = temp_data.occupation.replace(' Tech-support', 11)
temp_data.occupation = temp_data.occupation.replace(' Transport-moving', 12)

temp_data.relationship = temp_data.relationship.replace(' Husband', 0)
temp_data.relationship = temp_data.relationship.replace(' Not-in-family', 1)
temp_data.relationship = temp_data.relationship.replace(' Other-relative', 2)
temp_data.relationship = temp_data.relationship.replace(' Own-child', 3)
temp_data.relationship = temp_data.relationship.replace(' Unmarried', 4)
temp_data.relationship = temp_data.relationship.replace(' Wife', 5)

corrmat = temp_data.corr()  # corr() computes pairwise  correlations of features in a Data Frame
print(corrmat)
top_corr_features = corrmat.index
print(top_corr_features)
plt.figure(figsize=(20, 20)).show()

# plot the heat map
sns.heatmap(temp_data[top_corr_features].corr(), annot=True, cmap="RdYlGn")
plt.show()

# Table Decomposition not important feature
a = a.drop("fnlwgt", axis=1)
a = a.drop("capital-gain", axis=1)
a = a.drop("capital-loss", axis=1)
a = a.drop('native-country', axis=1)
a = a.drop('education-num', axis=1)
a = a.drop('race', axis=1)

data = a

# data analysis start
print('Data analysis start')
from sklearn.model_selection import KFold
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import classification_report
from sklearn.model_selection import train_test_split
from sklearn.model_selection import cross_val_score
from sklearn.metrics import accuracy_score
from sklearn.metrics import classification_report
from sklearn.metrics import recall_score, precision_score, roc_auc_score, roc_curve
from sklearn.metrics import confusion_matrix
from sklearn.ensemble import BaggingClassifier
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn import metrics

data.to_csv("C:\\Users\\tlseh\\OneDrive\\바탕 화면\\과제\\3-1\\데이터과학\\ana_data.csv", index=False, encoding='cp949')

data_url = "C:\\Users\\tlseh\\OneDrive\\바탕 화면\\과제\\3-1\\데이터과학\\ana_data.csv"
# data_url = 'D:/VSCODE/python/ana_data.csv'
ana_data = pd.read_csv(data_url)
ana_data = pd.DataFrame(ana_data)
# print(ana_data)

# plot rich person
rich = ana_data.loc[ana_data['Income'] == 1]

sns.countplot(x='sex', data=rich)
plt.title("Rich person histogram with sex")
plt.show()

sns.countplot(x="workclass", data=rich)
plt.title("Rich person histogram with workclass")
plt.show()

sns.countplot(x="education", data=rich)
plt.title("Rich person histogram with education")
plt.show()

sns.countplot(x="relationship", data=rich)
plt.title("Rich person histogram with relationship")
plt.show()

sns.countplot(x="occupation", data=rich)
plt.title("Rich person histogram with occupation")
plt.show()

sns.countplot(x="marital", data=rich)
plt.title("Rich person histogram with marital-status")
plt.show()

plt.title("Age of Rich person")
plt.hist(rich['Age'], bins=10)
plt.show()

plt.title("hours per week of Rich person")
plt.hist(rich['hours-per-week'], bins=10)
plt.show()

kf = KFold(n_splits=2)
k2, k3, k4 = [], [], []
for train_index, test_index in kf.split(ana_data):
    data_train = ana_data.loc[[i for i in train_index], :]
    data_test = ana_data.loc[[i for i in test_index], :]

    x_train = pd.get_dummies(data_train.drop(['Income'], axis=1))
    y_train = data_train['Income']
    x_test = pd.get_dummies(data_train.drop(['Income'], axis=1))
    y_test = data_train['Income']
    train, test = [], []
    for i in range(2, 5):
        knn = KNeighborsClassifier(n_neighbors=i)
        knn.fit(x_train, y_train)
        predict = knn.predict(x_test)
        train.append(knn.score(x_train, y_train))
        test.append(knn.score(x_test, y_test))
        print("K = ", i, ", score : ", knn.score(x_test, y_test))
        if i == 2:
            k2.append(knn.score(x_test, y_test))
        elif i == 3:
            k3.append(knn.score(x_test, y_test))
        elif i == 4:
            k4.append(knn.score(x_test, y_test))

    print("classification_report using knn algorithm")
    print(classification_report(y_test, predict))
sum = 0.0
for i in range(len(k2)):
    sum = sum + k2[i]
print("k-fold-cross validation score(n_neighbors = 2) : ", sum / len(k2))
sum = 0.0
for i in range(len(k2)):
    sum = sum + k3[i]
print("k-fold-cross validation score(n_neighbors = 3) : ", sum / len(k2))
sum = 0.0
for i in range(len(k2)):
    sum = sum + k4[i]
print("k-fold-cross validation score(n_neighbors = 4): ", sum / len(k2))

# print(confusion_matrix(y_test, predict))

cm = sklearn.metrics.confusion_matrix(y_test, predict)
print(cm)
plt.figure(figsize=(2, 2))
sns.heatmap(data=cm, annot=True, fmt='.2f', linewidths=.1, cmap='Blues')
plt.show()

# adapting Knn to ensemble
kf = KFold(n_splits=2)
k2, k3, k4 = [], [], []
for train_index, test_index in kf.split(ana_data):
    data_train = ana_data.loc[[i for i in train_index], :]
    data_test = ana_data.loc[[i for i in test_index], :]

    x_train = pd.get_dummies(data_train.drop(['Income'], axis=1))
    y_train = data_train['Income']
    x_test = pd.get_dummies(data_train.drop(['Income'], axis=1))
    y_test = data_train['Income']
    train, test = [], []

    model = BaggingClassifier(base_estimator=KNeighborsClassifier(n_neighbors=2), random_state=0, n_estimators=5)
    model.fit(x_train, y_train)
    prediction = model.predict(x_test)
    print('When K =5, the accuracy for bagged KNN is : ', model.score(x_test, y_test))

    model = BaggingClassifier(base_estimator=KNeighborsClassifier(n_neighbors=2), random_state=0, n_estimators=10)
    model.fit(x_train, y_train)
    prediction = model.predict(x_test)
    print('When K =10, the accuracy for bagged KNN is : ', model.score(x_test, y_test))

    model = BaggingClassifier(base_estimator=KNeighborsClassifier(n_neighbors=2), random_state=0, n_estimators=20)
    model.fit(x_train, y_train)
    prediction = model.predict(x_test)
    print('When K =20, the accuracy for bagged KNN is : ', model.score(x_test, y_test))

    model = BaggingClassifier(base_estimator=KNeighborsClassifier(n_neighbors=2), random_state=0, n_estimators=50)
    model.fit(x_train, y_train)
    prediction = model.predict(x_test)
    print('When K =50, the accuracy for bagged KNN is : ', model.score(x_test, y_test))

    model = BaggingClassifier(base_estimator=KNeighborsClassifier(n_neighbors=2), random_state=0, n_estimators=100)
    model.fit(x_train, y_train)
    prediction = model.predict(x_test)
    print('When K =100, the accuracy for bagged KNN is : ', model.score(x_test, y_test))

    # Ensemble RandomForest
    forest = RandomForestClassifier(n_estimators=5)
    forest.fit(x_train, y_train)
    # Predict
    y_pred = forest.predict(x_test)
    # Predict use RandomForest
    print('K = 5 Random Forest Predict score :', metrics.accuracy_score(y_test, y_pred))
    # Ensemble RandomForest
    forest = RandomForestClassifier(n_estimators=10)
    forest.fit(x_train, y_train)
    # Predict
    y_pred = forest.predict(x_test)
    # Predict use RandomForest
    print('K = 10 Random Forest Predict score :', metrics.accuracy_score(y_test, y_pred))
    # Ensemble RandomForest
    forest = RandomForestClassifier(n_estimators=15)
    forest.fit(x_train, y_train)
    # Predict
    y_pred = forest.predict(x_test)
    # Predict use RandomForest
    print('K = 15 Random Forest Predict score :', metrics.accuracy_score(y_test, y_pred))
    # Ensemble RandomForest
    forest = RandomForestClassifier(n_estimators=20)
    forest.fit(x_train, y_train)
    # Predict
    y_pred = forest.predict(x_test)
    # Predict use RandomForest
    print('K = 20 Random Forest Predict score :', metrics.accuracy_score(y_test, y_pred))
    # Ensemble RandomForest
    forest = RandomForestClassifier(n_estimators=50)
    forest.fit(x_train, y_train)
    # Predict
    y_pred = forest.predict(x_test)
    # Predict use RandomForest
    print('K = 50 Random Forest Predict score :', metrics.accuracy_score(y_test, y_pred))
    # Ensemble RandomForest
    forest = RandomForestClassifier(n_estimators=100)
    forest.fit(x_train, y_train)
    # Predict
    y_pred = forest.predict(x_test)
    # Predict use RandomForest
    print('K = 100 Random Forest Predict score :', metrics.accuracy_score(y_test, y_pred))
