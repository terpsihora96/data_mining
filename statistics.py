import pandas as pd 
import numpy as np
import matplotlib.pyplot as plt
from pandas.plotting import andrews_curves
from pandas.plotting import scatter_matrix
import seaborn

def main():
	data = pd.read_csv("./food_coded.csv", na_values="none")
	
	print(data.shape)
	print(data.columns)

	data.Gender = data.Gender.astype(int)
	
	print(data[data['Gender'] == 1]) # Koliko ima zena
	print(data[data['Gender'] == 2]) # Koliko ima muskaraca
	
	data = data[data['Gender'].apply(lambda x: str(x).isdigit())]
	data.reset_index(drop=True, inplace=True)
	groupby_gender = data.groupby('Gender')
	with pd.option_context('display.max_rows', None, 'display.max_columns', 50):
		print(groupby_gender.mean())

	data.hist()
	plt.show()
if __name__ == "__main__":
	main()