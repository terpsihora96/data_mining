from sklearn.decomposition import PCA
import pandas as pd
# from sklearn.tree import DecisionTreeClassifier
# from sklearn.ensemble import RandomForestClassifier
# from sklearn.neighbors import KNeighborsClassifier
# from sklearn.model_selection import train_test_split
# import sklearn.metrics
import matplotlib.pyplot as plt
# import sys
import numpy as np
from mpl_toolkits.mplot3d import Axes3D
from sklearn.preprocessing import StandardScaler

def isFloat(value):
  try:
    float(value)
    return True
  except ValueError:
    return False

df = pd.read_csv('./food-choices/food_coded.csv')

target_attribute = 'weight'
# attributes = [ 'exercise', 'Gender' , 'eating_out', 'GPA', 'employment', 'breakfast', 'calories_chicken',
#         'calories_day','coffee','diet_current_coded', 'drink','cook']

attributes = [ 'exercise', 'Gender' , 'eating_out']

df = df[[attributes[0], attributes[1], attributes[2], target_attribute]]

# df = df[[attributes[0], attributes[1], attributes[2], attributes[3],attributes[4],attributes[5],attributes[6], attributes[7],
#     attributes[8],attributes[9], attributes[10], attributes[11],target_attribute]]

df = df.replace('nan', np.nan)
df = df.dropna()


df = df[df[target_attribute].apply(lambda x: str(x).isdigit())]
# df = df[df['GPA'].apply(lambda x: isFloat(str(x)))]

df.reset_index(drop=True, inplace=True)

df[attributes[0]] = df.exercise.astype(int)
df[attributes[1]] = df.Gender.astype(int)
df[attributes[2]] = df.eating_out.astype(int)
# df[attributes[3]] = df.GPA.astype(float)
# df[attributes[4]] = df.employment.astype(int)
# df[attributes[5]] = df.breakfast.astype(int)
# df[attributes[6]] = df.calories_chicken.astype(int)
# df[attributes[7]] = df.calories_day.astype(int)
# df[attributes[8]] = df.coffee.astype(int)
# df[attributes[9]] = df.diet_current_coded.astype(int)
# df[attributes[10]] = df.drink.astype(int)
# df[attributes[11]] = df.cook.astype(int)
df[target_attribute] = df.weight.astype(int)

changes = {}
weight = df[target_attribute].unique()

for w in weight:
    if int(w) <=128:
        changes[w] = 0
    elif int(w) <= 155:
        changes[w] = 1
    elif int(w) <= 180:
        changes[w] = 2
    else:
        changes[w] = 3

df[target_attribute] = df[target_attribute].replace(changes)
weight = df[target_attribute].unique()


# features =attributes 

# x = df.loc[:, features].values
# y = df.loc[:,[target_attribute]].values
# x = StandardScaler().fit_transform(x)
# pca = PCA(n_components=2)
# principalComponents = pca.fit_transform(x)
# principalDf = pd.DataFrame(data = principalComponents
#              , columns = ['principal component 1', 'principal component 2'])

# finalDf = pd.concat([principalDf, df[[target_attribute]]], axis = 1)

# fig = plt.figure(figsize = (8,8))
# ax = fig.add_subplot(1,1,1) 
# ax.set_xlabel('Principal Component 1', fontsize = 15)
# ax.set_ylabel('Principal Component 2', fontsize = 15)
# ax.set_title('2 component PCA', fontsize = 20)
# targets = [0,1,2,3]
# colors = ['green', 'blue','orange','red']
# for target, color in zip(targets,colors):
#     indicesToKeep = finalDf[target_attribute] == target
#     ax.scatter(finalDf.loc[indicesToKeep, 'principal component 1']
#                , finalDf.loc[indicesToKeep, 'principal component 2']
#                , c = color
#                , s = 50)

# ax.legend(['0-128', '129-155','156-180','180+'])
# ax.grid()

# df = df[[attributes[0], attributes[1],attributes[2], target_attribute]]
# print(df)


fig = plt.figure()
ax = fig.add_subplot(111,projection='3d')
colors = ['g', 'b', 'orange', 'r']
for (v, color) in zip(weight, colors):
    subsamples = df.loc[df[target_attribute] == v]
    ax.scatter(subsamples[attributes[0]], subsamples[attributes[1]], subsamples[attributes[2]],color=color, s=70, alpha=0.3)

ax.set_xlabel('exercise')
ax.set_ylabel('Gender')
ax.set_zlabel('eating_out')
plt.show()
print(len(df))

