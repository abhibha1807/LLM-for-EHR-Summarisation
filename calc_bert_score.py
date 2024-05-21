#/ihome/hdaqing/abg96/llm/cross_val_onset_5/exp8/8_llama2_13B_chat_2024-02-26 20:04:29_Onset_3_shot_val_step_Onset_reasoning_prompt_llm_and_rule_eval_complex_def_and_few_shot_5_times_run1.csv
from evaluate import load
import csv
import pandas as pd
from bert_score import score
import numpy as np
# Load bertscore
# bertscore = load('bertscore')
import math
import copy
from sklearn.metrics import cohen_kappa_score

# Initialize an empty list to store rows
csv_list = []

column_names = ['text','true','pred', 'rule_llm_annotation', 'evaluation reasoning', 'extraction reasoning']

path_to_dir = ''

files = ['']


for p in range(len(files)):

    df = pd.read_csv(path_to_dir + files[p], names = column_names)

    deberta_score = []
    roberta_score = []
   
    rule_llm_annotation = []
    reasoning_list = []

    results_deberta = []
    results_roberta = []

    preds = []
    trues = []



    for i in range(len(list(df['pred']))):
        if isinstance(list(df['pred'])[i], float) or isinstance(list(df['true'])[i], float): # if the string is 'nan' replace with ''
            preds.append('')
            trues.append('')
        else:
            preds.append(list(df['pred'])[i])
            trues.append(list(df['true'])[i])


    P, R, results_deberta = score(preds, trues, lang="en", verbose=True, model_type = 'khalidalt/DeBERTa-v3-large-mnli')
    P, R, results_roberta = score(preds, trues, lang="en", verbose=True, model_type = 'roberta-large')
    
    threshold_078_deberta = []
   
    threshold_093_roberta = []
   


    for i in range(len(preds)):
        if not isinstance(list(df['evaluation reasoning'])[i], float):
            
            if float(results_deberta[i]) < 0.78:
                threshold_078_deberta.append('Dissimilar')
              
            else:
                threshold_078_deberta.append('Similar')
        
            if float(results_roberta[i]) < 0.93:
                threshold_093_roberta.append('Dissimilar')
            else:
                threshold_093_roberta.append('Similar')
             
            
        else:
            threshold_078_deberta.append(list(df['llm annotation'])[i])
            threshold_093_roberta.append(list(df['llm annotation'])[i])


    df = pd.DataFrame({'text': list(df['text']) , 'true': trues, 'pred': preds, 'rule_llm_annotation' : list(df['rule_llm_annotation']), 'deberta_score': results_deberta.tolist(), 'roberta_score':results_roberta.tolist() ,  'threshold_078_deberta':threshold_078_deberta ,  'threshold_093_roberta':threshold_093_roberta, 'evaluation reasoning':list(df['evaluation reasoning']), 'extraction reasoning':list(df['extraction reasoning'])})

    path = '' # add your own path
    filename = path + f'new_roberta_deberta_score_{files[p]}.csv' 

    df.to_csv(filename, index=False)

    print(f"CSV file '{filename}' has been created successfully.")

