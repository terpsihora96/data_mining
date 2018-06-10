import pandas as pd
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import train_test_split
import sklearn.metrics
import matplotlib.pyplot as plt
import sys
import numpy as np

def isFloat(value):
  try:
    float(value)
    return True
  except ValueError:
    return False

def main():
    df = pd.read_csv('./food-choices/food_coded.csv')
    # print('\n{}'.format(df.head()))
    
    
    # print("\nStatistike skupa:\n{}".format(df.describe()))

    target_attribute = 'weight'
    attribute_1 = 'comfort_food_reasons_coded'
    attribute_2 = 'cook'
    attribute_3 = 'eating_out'

    df = df[[attribute_1, attribute_2, attribute_3, target_attribute]]
    
    """
    print(len(df))
    c = 0
    for i in range(len(df)):
        if str(df[attribute_1][i]) == 'nan':
            c += 1
    for i in range(len(df)):
        if str(df[attribute_2][i]) == 'nan':
            c += 1
    for i in range(len(df)):
        if str(df[attribute_3][i]) == 'nan':
            c += 1

    for i in range(len(df)):
        if str(df[target_attribute][i]) == 'nan':
            c += 1
    print(c) """

    df = df.replace('nan', np.nan)
    df = df.dropna()
  

    df = df[df[target_attribute].apply(lambda x: str(x).isdigit())]
    
    df.reset_index(drop=True, inplace=True)
    
    df[attribute_1] = df.comfort_food_reasons_coded.astype(int)
    df[attribute_2] = df.cook.astype(int)
    df[attribute_3] = df.eating_out.astype(int)
    df[target_attribute] = df.weight.astype(int)

    changes = {}
    weight = df[target_attribute].unique()
    for w in weight:
        if int(w) <150:
            changes[w] = 0
        elif int(w) < 190:
            changes[w] = 1
        else:
            changes[w] = 2

    df[target_attribute] = df[target_attribute].replace(changes)

    
    print("Ispitanici imaju sledece tezine: {}".format(weight))   
    
    X = df[[attribute_1, attribute_2, attribute_3]]
    y = df[[target_attribute]]

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3)
    print("\nVelicina skupa za obucavanje: {}".format(X_train.size))
    print("Velicina skupa za testiranje: {}".format(X_test.size))

    
    # clf = DecisionTreeClassifier(criterion='entropy')
    # clf = DecisionTreeClassifier(criterion='gini')

    #clf = RandomForestClassifier(n_estimators=31)
    clf = KNeighborsClassifier(5, 'distance')

    # Treniramo model
    clf.fit(X_train, y_train.values.ravel())

    # Vrsimo predikciju
    y_test_predicted = clf.predict(X_test)
    y_train_predicted = clf.predict(X_train)

    # Izracunavamo preciznost
    train_acc = clf.score(X_train, y_train)
    test_acc = clf.score(X_test, y_test)
    print('train preciznost: {}'.format(train_acc))
    print('test preciznost: {}'.format(test_acc))
    
    # Prikazujemo matricu konfuzije
    test_rep = sklearn.metrics.classification_report(y_test, y_test_predicted)
    train_rep = sklearn.metrics.classification_report(y_train, y_train_predicted)
    print("\nTest izvestaj:\n{}".format(test_rep))
    print("Train izvestaj:\n{}".format(train_rep))

    train_conf = sklearn.metrics.confusion_matrix(y_train, y_train_predicted)
    test_conf = sklearn.metrics.confusion_matrix(y_test, y_test_predicted)
    print("Matrica konfuzije za skup za obucavanje:\n{}".format(train_conf))
    print("\nMatrica konfuzije za skup za testiranje:\n{}".format(test_conf))

if __name__ == "__main__":
    main()
