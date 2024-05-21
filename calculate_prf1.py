import pandas as pd

column_names = ['text','true','pred', 'rule_llm_annotation', 'evaluation reasoning', 'extraction reasoning']
path_to_dir = '' # replace with your path to file.

files = [] # list of file names to calculate p,r and f1

for i in range(len(files)):

	list1 = list(df['rule_llm_annotation'])

	# Calculating Cohen's Kappa
	refined_list1 = [str(i).strip() for i in list1]

	def convert(l): #convert Similar/Dissimilar to Correct and Incorrect.
		for i in range(len(l)):
			if l[i] == 'Similar':
				l[i] = 'Correct'
			if l[i] == 'Dissimilar':
				l[i] = 'Incorrect'
		return l


	refined_list1 = convert(refined_list1)
	

	correct = sum([1 for item in refined_list1 if item == 'Correct'])
	incorrect = sum([1 for item in refined_list1 if item == 'Incorrect'])
	spurious = sum([1 for item in refined_list1 if item == 'Spurious'])
	missing = sum([1 for item in refined_list1 if item == 'Missing'])

	
	possible = correct + incorrect  + missing  
	actual = correct + incorrect  + spurious 


	precision = correct / actual
	recall = correct / possible
	f1 = (2 * (precision * recall)) / (precision + recall)

	print('precision', (precision,3))
	print('recall', (recall,3))
	print('f1', (f1,3))
	

