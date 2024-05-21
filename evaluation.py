
import torch
import transformers
import re
import json
import textwrap
import langchain
import torch
import csv
from tqdm import tqdm
import time
from transformers import AutoTokenizer, AutoModelForCausalLM
from transformers import LlamaForCausalLM, LlamaTokenizer
from accelerate import infer_auto_device_map, init_empty_weights
from accelerate import load_checkpoint_and_dispatch
from transformers import LlamaModel, LlamaConfig
from langchain import HuggingFacePipeline
from langchain import PromptTemplate, LLMChain
from langchain.prompts.few_shot import FewShotPromptTemplate
from langchain.prompts.prompt import PromptTemplate
from transformers import StoppingCriteria, StoppingCriteriaList
from transformers import pipeline
from datetime import datetime
import json
access_token = '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'
import pandas as pd
from datetime import datetime


config = LlamaConfig.from_pretrained("meta-llama/Llama-2-13b-chat-hf", cache_dir = '/zfs1/hdaqing/abg96/cache', torch_dtype=torch.float16, token=access_token)
with init_empty_weights():
    model = AutoModelForCausalLM.from_config(config)
   
model.tie_weights()
device_map = infer_auto_device_map(model, no_split_module_classes=["LlamaDecoderLayer"])

model = transformers.LlamaForCausalLM.from_pretrained("meta-llama/Llama-2-13b-chat-hf", torch_dtype=torch.float16, cache_dir = '/zfs1/hdaqing/abg96/cache', device_map = device_map, offload_folder="offload", offload_state_dict = True, token=access_token)

device_map = infer_auto_device_map(model)

tokenizer = LlamaTokenizer.from_pretrained("meta-llama/Llama-2-13b-chat-hf",unk_token="<unk>",bos_token="<s>",eos_token="</s>", cache_dir = '/zfs1/hdaqing/abg96/cache', token=access_token)




class StoppingCriteriaSub(StoppingCriteria):

    def __init__(self, stops = [], encounters=1):
        super().__init__()
        self.stops = [stop.to("cuda") for stop in stops]

    def __call__(self, input_ids: torch.LongTensor, scores: torch.FloatTensor):
        for stop in self.stops:
            if torch.all((stop == input_ids[0][-len(stop):])).item():
                return True
        return False


stop_words = ["</s>"]
stop_words_ids = [tokenizer(stop_word, return_tensors='pt')['input_ids'].squeeze() for stop_word in stop_words]

stopping_criteria = StoppingCriteriaList([StoppingCriteriaSub(stops=[torch.Tensor([669])])])

pipe = pipeline("text-generation",
                model=model,
                tokenizer= tokenizer,
                torch_dtype=torch.bfloat16,
                device_map="auto",
                max_new_tokens = 2000,
                do_sample=True,
                top_k=30,
                num_return_sequences=1,
                eos_token_id=tokenizer.eos_token_id,
                pad_token_id=tokenizer.eos_token_id,
                stopping_criteria=stopping_criteria
                )

entity = 'Onset'


sys_prom = f'''


You are a language model that is used for the evaluation of model 
predictions against the ground truth. I will be providing you with two phrases and your task is to tell me
if they are semantically similar or not. Please consider the meaning of words, the context in which they appear and any implied
or explicit relationships between the phrases.

Semantically similar phrases are phrases that describe the same event or action eventhough they dont use the exact same words.

Specifically,
a) The two phrases mean the same
b) The common subset of information means the same

Semantically dissimilar phrases are phrases that describe different events, actions, or concepts or if they differ
significantly in meaning, context or intent. In other words, phrases that express unrelated or contradictory ideas, or 
that have different connotations are considered semantically dissimilar.  

Specifically,
a) The two phrases do not mean the same:
b) The common subset of information does not mean the same: 



'''

few_shot = f'''

Now I will provide you with examples and the reasoning steps for semantically similar phrases.

Example1: Ground Truth: '24 hours ago' and Prediction: '1 day ago' 

1) Identify common topic of information: Both phrases are talking about time.
2) Determine the respective piece of information supporting the common topic:
    Phrase1: 24 hours ago 
    Phrase2:  1 day ago 
3) Evaluate the similarity of the two phrases: They mean the same as 24 hours ago and 1 day are the same time duration.
4) Final answer: @Similar@


Example2: Ground Truth: 'pain in left arm and shoulder' and Prediction: 'pain in left arm'

1) Identify common topic of information: Both phrases are talking about pain in the arm
2) Determine the respective piece of information supporting the common topic:
    Phrase1: pain in left arm
    Phrase2:  pain in left arm
3) Evaluate the similarity of the two phrases: They mean the same as both phrases mention that pain is in left arm.
4) Final answer: @Similar@

Example3: Ground Truth: 'pain in the left arm and shoulder' and Prediction:  'pain in the left arm and left side of chest'


1) Identify common topic of information: Both phrases are talking about pain in the arm.
2) Determine the respective piece of information supporting the common topic:
    Phrase1: pain in left arm
    Phrase2:  pain in left arm
3) Evaluate the similarity of the two phrases: They mean the same as both phrases mention that pain is in left arm.
4) Final answer: @Similar@



Now I will provide you with examples and the reasoning steps for semantically dissimilar phrases.

Example1: Ground Truth:  '24 hours ago' and   Prediction:'2 hours ago' 

1) Identify common topic of information: Both phrases are talking about time.
2) Determine the respective piece of information supporting the common topic:
    Phrase1: 24 hours ago 
    Phrase2: 2 hours ago
3) Evaluate the similarity of the two phrases: They do not mean the same as both phrases mention different time periods.
4) Final answer: @Dissimilar@


Example2: Ground Truth:  'pain in left arm and shoulder' and   Prediction: 'pain in right arm'

1) Identify common topic of information: Both phrases are talking about pain in the arm.
2) Determine the respective piece of information supporting the common topic:
    Phrase1: pain in the left arm
    Phrase2: pain in the right arm.
3) Evaluate the similarity of the two phrases: They do not mean the same as both phrases mention different locations of arm pain.
4) Final answer: @Dissimilar@


Example3: Ground Truth:  'pain in the left arm and shoulder' and  Prediction: 'pain in the right arm and right side of chest'

1) Identify common topic of information: Both phrases are talking about pain in the arm.
2) Determine the respective piece of information supporting the common topic:
    Phrase1: pain in the left arm
    Phrase2: pain in the right arm.
3) Evaluate the similarity of the two phrases: They do not mean the same as both phrases mention different locations of arm pain.
4) Final answer: @Dissimilar@



'''

instruct = '''

I will provide you with ground truth and predictions and your task is to tell me if they are 
'Similar', 'Dissimilar', 'Spurious' or 'Missing'. Answer '@Similar@', '@Dissimilar@', '@Spurious@', '@Missing@'.


Text: {text} 

'''

B_SEN, E_SEN = '<s>', '</s>'
B_INST, E_INST = "[INST]", "[/INST]"
B_SYS, E_SYS = "<<SYS>>\n", "\n<</SYS>>\n"


def get_prompt():

    SYSTEM_PROMPT = B_SYS + sys_prom + E_SYS + few_shot + '\n'
    prompt_template =  B_SEN + B_INST + SYSTEM_PROMPT + instruct + E_INST
    return prompt_template

template = get_prompt(instruct, sys_prom)


def remove_phrase(text, phrase):
    return text.replace(phrase, '').strip()



current_time = datetime.now()

formatted_time = current_time.strftime('%Y-%m-%d %H:%M:%S')




df = pd.read_csv('rule_annotation_{entity}')

df.columns = ['text','true','pred', 'rule_annotation', 'evaluation reasoning', 'extraction reasoning']

true = list(df['true'])
pred = list(df['pred'])
text = list(df['text'])
rule_annotation = list(df['rule_annotation'])
extraction_reasoning = list(df['extraction_reasoning'])


column_names = ['text','true','pred', 'rule_llm_annotation', 'evaluation reasoning', 'extraction reasoning']

temp_value = 1.0
top_p_value = 0.95

def evaluate_llm(true, pred):
    question = 'Ground Truth: ' + str(true) + ' \n ' + 'Prediction: ' + str(pred)

    prompt = PromptTemplate(template=template, input_variables=["text"])

    formatted_prompt = (prompt.format(text=question))

    tokenized_prompt = tokenizer.tokenize(formatted_prompt[0])
    prompt_length = len(tokenized_prompt)

    llm = HuggingFacePipeline(pipeline = pipe, model_kwargs = {'temperature':temp_value, 'top_p':top_p_value})

    llm_chain = LLMChain(prompt=prompt, llm=llm)

    output = llm_chain.run(text=question)

    pattern = r'@([^@]*)@' # extract answer between '@@'

    match = re.search(pattern, output)

    return (match, output)

with open(f'', 'w', newline='') as file: # add a suitable name to the file.
    writer = csv.writer(file)
    writer.writerow(experiment_specifications)
    writer.writerow(column_names)
    for i in range(len(true)):

        if rule_annotation[i] == 'to evaluate':
            match, output = evaluate_llm(true[i], pred[i])

            if match:
                writer.writerow([text[i], true[i], pred[i], str(match.group(1)), output, extraction_reasoning[i]])
               
            else:
                count_evaluate = 0 # evaluate 5 times max if we dont get suitable output once.
                while(match or count_evaluate > 5):
                    match, output = evaluate_llm(true[i], pred[i])
                    count_evaluate = count_evaluate + 1

                if match:
                    writer.writerow([text[i], true[i], pred[i], str(match.group(1)) , output, extraction_reasoning[i]])
                    break

                if count_evaluate > 5:
                    writer.writerow([text[i], true[i], pred[i], '', output, extraction_reasoning[i]])
                    break

               

        else:
            writer.writerow([text[i], true[i], pred[i], rule_annotation[i] , '', extraction_reasoning[i]])

                   

