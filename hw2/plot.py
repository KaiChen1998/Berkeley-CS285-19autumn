import os
import numpy as np
import matplotlib.pyplot as plt

def find_npy(path):
	for each in os.listdir(path):
		if 'npy' in each:
			return each

def plot(path, prefix, title, save_name = 'result.jpg'):
	plt.figure(figsize=(10, 6))
	file_name = [os.path.join(path, each) for each in os.listdir(path) if prefix in each]
	npy_name = [os.path.join(each, find_npy(each)) for each in file_name]
	for each in npy_name:
		name = each.split('\\')[-1][0:-4]
		curve = np.load(each)
		plt.plot(range(len(curve)), curve, label = name)
	plt.legend()
	plt.title(title)
	plt.savefig(save_name)
	plt.show()

if __name__ == '__main__':
	plot('cs285/data', 'pg_Walker2d-v2', 'Walker2d-v2', 'fig/Walker2d-v2.jpg')
	