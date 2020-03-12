import os
import numpy as np
import matplotlib.pyplot as plt
from tensorboard.backend.event_processing import event_accumulator

def find_event(path):
	for each in os.listdir(path):
		if 'event' in each:
			return each

def get_curve(event_file):
	ea = event_accumulator.EventAccumulator(event_file)
	ea.Reload()
	curve = ea.scalars.Items('Eval_AverageReturn')
	step = np.array([each.step for each in curve])
	value = np.array([each.value for each in curve])
	print(len(step))
	return step, value

def plot(path, prefix, title, save_name = 'result.jpg'):
	plt.figure(figsize=(10, 6))
	file_name = [os.path.join(path, each) for each in os.listdir(path) if prefix in each]
	event_name = [os.path.join(each, find_event(each)) for each in file_name]
	print(event_name)
	
	for each in event_name:
		name = '_'.join(each.split('\\')[-2].split('_')[1:-3])
		step, value = get_curve(each)
		plt.plot(step, value, label = name)
	# plt.axhline(y = -250, ls = ":", label = 'Goal', c="green")
	
	plt.legend()
	plt.title(title)
	plt.savefig(save_name)
	plt.show()

if __name__ == '__main__':
	plot('cs285/data', 'cheetah_cheetah', 'Cheetah', 'fig/cheetah.jpg')
	