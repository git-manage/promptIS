import os
import pickle
import pdb
import json
from transformers import BertModel,BertTokenizer
import torch
import tqdm

def list_all_files(root_folder):
    """
    递归遍历 root_folder 下的所有文件，并返回它们的完整路径列表。
    """
    processed_data = {}
    
    for foldername, subfolders, filenames in os.walk(root_folder):
        # pdb.set_trace()
        for filename in filenames:
            problem = []
            dataset_name = os.path.basename(foldername)
            if dataset_name.startswith('EvoEval_concise') or dataset_name.startswith('EvoEval_verbose'):
                dataset_index = 'HumanEval'
            elif dataset_name.startswith('EvoEval'):
                dataset_index = 'EvoEval'
            elif dataset_name.startswith('HumanEval'):
                dataset_index = 'HumanEval' #单纯用来组成索引
            else:
                dataset_index = 'BigCodeBench'
            if filename == 'tasks.pkl':
                if dataset_name not in processed_data:
                    processed_data[f'{dataset_name}'] = {}
                with open(f'{foldername}/{filename}', 'rb') as file:
                   f = pickle.load(file)
                   problem = f['question']
                processed_data[f'{dataset_name}']['problem'] = problem
            if filename == 'correctness.pkl':
                metric = 'correctness'
                if dataset_name not in processed_data:
                    processed_data[f'{dataset_name}']= {}
                if metric not in processed_data[f'{dataset_name}']:
                    processed_data[f'{dataset_name}'][metric] = {}
                with open(f'{foldername}/{filename}', 'rb') as file:
                    data = pickle.load(file)
                    for key,value in data.items():
                        if 'llama' in key and '7b' in key:
                            process(data[key],'codellama-7b',processed_data[f'{dataset_name}'][metric],dataset_index)
                        elif 'llama' in key and '13b' in key:
                            # pdb.set_trace()
                            process(data[key],'codellama-13b',processed_data[f'{dataset_name}'][metric],dataset_index)
                        elif 'llama' in key and '34b' in key:
                            process(data[key],'codellama-34b',processed_data[f'{dataset_name}'][metric],dataset_index)
                        elif 'llama' in key and '70b' in key:
                            process(data[key],'codellama-70b',processed_data[f'{dataset_name}'][metric],dataset_index)
                        else:
                            continue
            
            if filename == 'security.pkl':
                metric = 'security'
                if dataset_name not in processed_data:
                    processed_data[f'{dataset_name}']= {}
                if metric not in processed_data[f'{dataset_name}']:
                    processed_data[f'{dataset_name}'][metric] = {}
                with open(f'{foldername}/{filename}', 'rb') as file:
                    data = pickle.load(file)
                    for key,value in data.items():
                        if 'llama' in key and '7b' in key:
                            process(data[key],'codellama-7b',processed_data[f'{dataset_name}'][metric],dataset_index)
                        elif 'llama' in key and '13b' in key:
                            # pdb.set_trace()
                            process(data[key],'codellama-13b',processed_data[f'{dataset_name}'][metric],dataset_index)
                        elif 'llama' in key and '34b' in key:
                            process(data[key],'codellama-34b',processed_data[f'{dataset_name}'][metric],dataset_index)
                        elif 'llama' in key and '70b' in key:
                            process(data[key],'codellama-70b',processed_data[f'{dataset_name}'][metric],dataset_index)
                        else:
                            continue
            if filename == 'maintainability.pkl':
                metric = 'maintainability'
                if dataset_name not in processed_data:
                    processed_data[f'{dataset_name}']= {}
                if metric not in processed_data[f'{dataset_name}']:
                    processed_data[f'{dataset_name}'][metric] = {}
                with open(f'{foldername}/{filename}', 'rb') as file:
                    data = pickle.load(file)
                    for key,value in data.items():
                        if 'llama' in key and '7b' in key:
                            process(data[key],'codellama-7b',processed_data[f'{dataset_name}'][metric],dataset_index)
                        elif 'llama' in key and '13b' in key:
                            # pdb.set_trace()
                            process(data[key],'codellama-13b',processed_data[f'{dataset_name}'][metric],dataset_index)
                        elif 'llama' in key and '34b' in key:
                            process(data[key],'codellama-34b',processed_data[f'{dataset_name}'][metric],dataset_index)
                        elif 'llama' in key and '70b' in key:
                            process(data[key],'codellama-70b',processed_data[f'{dataset_name}'][metric],dataset_index)
                        else:
                            continue
            if filename == 'cc.pkl':
                metric = 'circle_complexity'
                if dataset_name not in processed_data:
                    processed_data[f'{dataset_name}']= {}
                if metric not in processed_data[f'{dataset_name}']:
                    processed_data[f'{dataset_name}'][metric] = {}
                with open(f'{foldername}/{filename}', 'rb') as file:
                    data = pickle.load(file)
                    for key,value in data.items():
                        if 'llama' in key and '7b' in key:
                            process(data[key],'codellama-7b',processed_data[f'{dataset_name}'][metric],dataset_index)
                        elif 'llama' in key and '13b' in key:
                            # pdb.set_trace()
                            process(data[key],'codellama-13b',processed_data[f'{dataset_name}'][metric],dataset_index)
                        elif 'llama' in key and '34b' in key:
                            process(data[key],'codellama-34b',processed_data[f'{dataset_name}'][metric],dataset_index)
                        elif 'llama' in key and '70b' in key:
                            process(data[key],'codellama-70b',processed_data[f'{dataset_name}'][metric],dataset_index)
                        else:
                            continue
            if filename == 'Halstead.pkl':
                metric = 'Halstead_complexity'
                if dataset_name not in processed_data:
                    processed_data[f'{dataset_name}']= {}
                if metric not in processed_data[f'{dataset_name}']:
                    processed_data[f'{dataset_name}'][metric] = {}
                with open(f'{foldername}/{filename}', 'rb') as file:
                    data = pickle.load(file)
                    for key,value in data.items():
                        if 'llama' in key and '7b' in key:
                            process(data[key],'codellama-7b',processed_data[f'{dataset_name}'][metric],dataset_index)
                        elif 'llama' in key and '13b' in key:
                            # pdb.set_trace()
                            process(data[key],'codellama-13b',processed_data[f'{dataset_name}'][metric],dataset_index)
                        elif 'llama' in key and '34b' in key:
                            process(data[key],'codellama-34b',processed_data[f'{dataset_name}'][metric],dataset_index)
                        elif 'llama' in key and '70b' in key:
                            process(data[key],'codellama-70b',processed_data[f'{dataset_name}'][metric],dataset_index)
                        else:
                            continue
                        
            if filename == 'codebleu.pkl':
                metric = 'codebleu'
                if dataset_name not in processed_data:
                    processed_data[f'{dataset_name}']= {}
                if metric not in processed_data[f'{dataset_name}']:
                    processed_data[f'{dataset_name}'][metric] = {}
                with open(f'{foldername}/{filename}', 'rb') as file:
                    data = pickle.load(file)
                    for key,value in data.items():
                        if 'llama' in key and '7b' in key:
                            process(data[key],'codellama-7b',processed_data[f'{dataset_name}'][metric],dataset_index)
                        elif 'llama' in key and '13b' in key:
                            # pdb.set_trace()
                            process(data[key],'codellama-13b',processed_data[f'{dataset_name}'][metric],dataset_index)
                        elif 'llama' in key and '34b' in key:
                            process(data[key],'codellama-34b',processed_data[f'{dataset_name}'][metric],dataset_index)
                        elif 'llama' in key and '70b' in key:
                            process(data[key],'codellama-70b',processed_data[f'{dataset_name}'][metric],dataset_index)
                        else:
                            continue
        
        #[key for key,value in processed_data['EvoEval_combine'].items()]
    return processed_data

def process(data, model, data_dict,dataset):
    # pdb.set_trace()
    if model not in data_dict:
        data_dict[model] = []
        for i in range(len(data)):
            key = f'{dataset}/{i}'
            if key in data:
                data_dict[model].append(data[key])

                
    # pdb.set_trace()
    
def text2embedding(data,full=False):
    tokenizer = BertTokenizer.from_pretrained("/root/autodl-tmp/tokenizer")  # 英文示例
    model = BertModel.from_pretrained('/root/autodl-tmp/Bert')
    Embedding = {}
    for key ,value in data.items():
        if key not in Embedding:
            Embedding[key] = []
        for _,problem in tqdm.tqdm(data[key]['problem'].items()):
            inputs = tokenizer(problem, return_tensors='pt', padding=True, truncation=True, max_length=512)
            with torch.no_grad():
                outputs = model(**inputs)
            # 获取最后一层的隐藏状态
            last_hidden_states = outputs.last_hidden_state
            # 使用[CLS]标记的嵌入作为句子的嵌入（对于分类任务特别有用）
            if full:
                sentence_embedding = last_hidden_states.numpy()
            else: 
                sentence_embedding = last_hidden_states[:, 0, :].numpy()
            Embedding[key].append(sentence_embedding)
    return Embedding
# 示例用法
def preprocess(folder_path,full=False,metric='correctness'):
    if not os.path.exists(f'{folder_path}/processed.pickle'):
        score = list_all_files(folder_path)
        with open(f'{folder_path}/processed.pickle', 'wb') as f:
            pickle.dump(score, f)
    else:
        with open(f'{folder_path}/processed.pickle', 'rb') as f:
            score = pickle.load(f)
    
    if not os.path.exists(f'{folder_path}/QuestionEmbedding.pickle'):
        with open(f'{folder_path}/processed.pickle', 'rb') as f:
            files = pickle.load(f)
        Embedding = text2embedding(files,full)
        with open(f'{folder_path}/QuestionEmbedding.pickle', 'wb') as f:
            pickle.dump(Embedding, f)
    else:
        with open(f'{folder_path}/QuestionEmbedding.pickle', 'rb') as f:
            Embedding = pickle.load(f)
    pdb.set_trace()
    score['Evoeval'] = merge_dicts(score['EvoEval_combine'],score['EvoEval_concise'],score['EvoEval_creative'],score['EvoEval_difficult'],
                                       score['HumanEvalPlus'],score['EvoEval_verbose'],score['EvoEval_tool_use'],score['EvoEval_subtle'])
    Embedding['Evoeval'] = merge_list(Embedding['EvoEval_combine'],Embedding['EvoEval_concise'],Embedding['EvoEval_creative'],Embedding['EvoEval_difficult'],
                                       Embedding['HumanEvalPlus'],Embedding['EvoEval_verbose'],Embedding['EvoEval_tool_use'],Embedding['EvoEval_subtle'])
    return Embedding, score

def merge_list(*lists):
    result = []
    for data in lists:
        result += data
    return result
    
    
    
def merge_dicts(*dicts):
    """
    递归合并多个结构相同的字典：
    - 遇到字典：继续递归合并
    - 遇到列表：直接相加（拼接）
    - 遇到其他类型：后面的覆盖前面的
    """
    result = {}

    def merge_values(key, values):
        # 过滤掉 None 值
        values = [v for v in values if v is not None]

        if not values:
            return None

        first_val = values[0]

        # 如果所有值都是列表，就拼接它们
        if all(isinstance(v, list) for v in values):
            merged_list = []
            for v in values:
                merged_list += v
            return merged_list

        # 如果所有值都是字典，就递归合并
        elif all(isinstance(v, dict) for v in values):
            return merge_dicts(*values)

        # 否则取最后一个非空值（可改为取第一个）
        return values[-1]

    # 收集团队中每个字典的所有键
    all_keys = set()
    for d in dicts:
        if isinstance(d, dict):
            all_keys.update(d.keys())

    for key in all_keys:
        values = [d.get(key) if isinstance(d, dict) else None for d in dicts]
        result[key] = merge_values(key, values)

    return result
    