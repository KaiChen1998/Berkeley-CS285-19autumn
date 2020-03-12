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
	curve = ea.scalars.Items('Train_AverageReturn')
	step = np.array([each.step for each in curve])
	value = np.array([each.value for each in curve])
	print(len(step))
	return step, value

def plot(path, prefix, title, save_name = 'result.jpg'):
	plt.figure(figsize=(10, 6))
	file_name = [os.path.join(path, each) for each in os.listdir(path) if prefix in each]
	event_name = [os.path.join(each, find_event(each)) for each in file_name]
	print(event_name)
	
	# for each in event_name:
	# 	name = '_'.join(each.split('\\')[-2].split('_')[1:-3])
	# 	step, value = get_curve(each)
	# 	plt.plot(step, value, label = name)
	# plt.axhline(y = 20, ls = ":", label = 'Goal', c="green")
	
	plt.axhline(y = 150, ls = ":", label = 'Goal', c="green")
	event_name_1 = [each for each in event_name if 'double' not in each]	
	step_1, value_1 = get_curve(event_name_1[0])
	step_2, value_2 = get_curve(event_name_1[1])
	step_3, value_3 = get_curve(event_name_1[2])
	step = (step_1 + step_2 + step_3) / 3
	value = (value_1 + value_2 + value_3) / 3
	plt.plot(step, value, label = 'DQN')

	event_name_2 = [each for each in event_name if 'doubledqn' in each]
	step_1, value_1 = get_curve(event_name_2[0])
	step_2, value_2 = get_curve(event_name_2[1])
	step_3, value_3 = get_curve(event_name_2[2])
	step = (step_1 + step_2 + step_3) / 3
	value = (value_1 + value_2 + value_3) / 3
	plt.plot(step, value, label = 'DDQN')

	event_name_3 = [each for each in event_name if 'test' in each]
	step_1, value_1 = get_curve(event_name_3[0])
	step_2, value_2 = get_curve(event_name_3[1])
	step_3, value_3 = get_curve(event_name_3[2])
	step = (step_1 + step_2 + step_3) / 3
	value = (value_1 + value_2 + value_3) / 3
	plt.plot(step, value, label = 'POLYAK, lambda = 0.999')

	event_name_4 = [each for each in event_name if 'polyak' in each and 'test' not in each]
	step_1, value_1 = get_curve(event_name_4[0])
	step_2, value_2 = get_curve(event_name_4[1])
	step_3, value_3 = get_curve(event_name_4[2])
	step = (step_1 + step_2 + step_3) / 3
	value = (value_1 + value_2 + value_3) / 3
	plt.plot(step, value, label = 'POLYAK, lambda = 0.9999')


	plt.legend()
	plt.title(title)
	plt.savefig(save_name)
	plt.show()

if __name__ == '__main__':
	plot('cs285/data', 'LunarLander-v2', 'LunarLander-v2', 'fig/polyak.jpg')
	