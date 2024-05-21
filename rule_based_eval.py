import pandas as pd
import re

files = [] # list of files to run rule based evaluation on

path_to_dir = '' # add path to directory where files with extracted entities are stored


for f in files:
	df = pd.read_csv(path_to_dir + f)
	df.columns = ['text','true','pred', 'extraction reasoning']
	true = list(df['true'])
	pred = list(df['pred'])


	n = len(true)
	rule_annotation = ['' for i in range(n)]

	for i in range(n):
		if true[i] == ' ' or str(true[i]) == 'nan':
			true_compare = '' # replace empty or nan with ''
		else:
			true_compare = str(true[i])

		if pred[i] == ' ' or str(pred[i]) == 'nan':
			pred_compare = '' 
		else:
			pred_compare = str(pred[i])

		pattern = r"\b((n|N)o\s|(n|N)ot\s|(n|N)othing\b|N/A|n/a|(w|W)ithout\s|(d|D)enies\s|(n|N)on[\w-]*)\b"

		no_in_true = bool(re.search(pattern, true_compare))
		no_in_pred =  bool(re.search(pattern, pred_compare))

		if no_in_true:
			true_compare = '' # if string contains any one of the words in the pattern replace with '' (equivalent to an empty string)
		if no_in_pred:
			pred_compare = ''

		# for more info: https://github.com/MantisAI/nervaluate
		if true_compare == '' and pred_compare != '':
			rule_annotation[i] = 'Spurious'

		elif pred_compare == '' and true_compare != '':
			rule_annotation[i]  = 'Missing'

		elif true_compare == '' and pred_compare == '':
			rule_annotation[i] = 'Similar'
		
		elif true_compare ==  pred_compare:
			
			rule_annotation[i] = 'Similar'

		else:
			rule_annotation[i] = 'to evaluate' # need to be evaluated by llm.


	df['rule_annotation'] = rule_annotation
	ordered_columns = ['text','true','pred','rule_annotation','extraction reasoning']

	df.to_csv(path_to_dir + 'rule_annotation_' + f, columns=ordered_columns)

	 





