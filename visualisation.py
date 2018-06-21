from sklearn.decomposition import PCA
import pandas as pd
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import train_test_split
import sklearn.metrics
import matplotlib.pyplot as plt
import sys
import numpy as np
from mpl_toolkits.mplot3d import Axes3D
from sklearn.preprocessing import StandardScaler

def isFloat(value):
  try:
    float(value)
    return True
  except ValueError:
    return False

def main():
    df = pd.read_csv('./food-choices/food_coded.csv')
    
    target_attribute = 'weight'
    attribute_1 = 'comfort_food_reasons_coded'
    attribute_2 = 'cook'
    attribute_3 = 'eating_out'


    df = df[[attribute_1, attribute_2, attribute_3, target_attribute]]

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
    weight = df[target_attribute].unique()

    features = [attribute_1, attribute_2, attribute_3]
    # Separating out the features
    x = df.loc[:, features].values
    # Separating out the target
    y = df.loc[:,[target_attribute]].values
    # Standardizing the features
    x = StandardScaler().fit_transform(x)
    pca = PCA(n_components=2)
    principalComponents = pca.fit_transform(x)
    principalDf = pd.DataFrame(data = principalComponents
                 , columns = ['principal component 1', 'principal component 2'])

    finalDf = pd.concat([principalDf, df[[target_attribute]]], axis = 1)

    fig = plt.figure(figsize = (8,8))
    ax = fig.add_subplot(1,1,1) 
    ax.set_xlabel('Principal Component 1', fontsize = 15)
    ax.set_ylabel('Principal Component 2', fontsize = 15)
    ax.set_title('2 component PCA', fontsize = 20)
    targets = [0,1,2]
    colors = ['r', 'g', 'b']
    for target, color in zip(targets,colors):
        indicesToKeep = finalDf[target_attribute] == target
        ax.scatter(finalDf.loc[indicesToKeep, 'principal component 1']
                   , finalDf.loc[indicesToKeep, 'principal component 2']
                   , c = color
                   , s = 50)
    ax.legend(targets)
    ax.grid()
    # fig = plt.figure()
    # ax = fig.add_subplot(111, projection='3d')

    
    # colors = ['green', 'blue','red']
    # for (v, color) in zip(weight, colors):
    #     subsamples = df.loc[df[target_attribute] == v]
    #     ax.scatter(subsamples[attribute_1], subsamples[attribute_2], subsamples[attribute_3],color=color, s=70, alpha=0.3)

    # ax.set_xlabel('comfort_food_reasons_coded')
    # ax.set_ylabel('cook')
    # ax.set_zlabel('eating_out')
    plt.show()

if __name__ == "__main__":
    main()
