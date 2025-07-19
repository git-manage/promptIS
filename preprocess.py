from util import ANSWER_PATH, RESULT_PATH, CODE_RESULT_PATH,DATASET_LIST,QUSETION_PATH

from correctness_analysis import AST_assemble_score, calculate_cfg_similarity, ERROR_TYPE
from security_analysis import scan_code
from c_compleyity_analysis.analysis import calculate_cyclomatic_complexity
from H_complexity_analysis.analysis import calculate_halstead_metrics
from Codebleu_analysis.analysis import calculate_codebleu_score
import pickle
import os
import pdb
import json
import builtins
import tqdm
from typing import List
import re
import ast

def fix_indentation(code_str):
    # 移除每一行开头的共同空格数
    lines = code_str.split('\n')
    # 找出所有非空行的最小缩进
    min_indent = None
    for line in lines:
        stripped_line = line.lstrip()
        if stripped_line:  # 跳过纯空白行
            current_indent = len(line) - len(stripped_line)
            if min_indent is None or current_indent < min_indent:
                min_indent = current_indent
    
    if min_indent is not None and min_indent > 0:
        fixed_lines = [line[min_indent:] for line in lines]
    else:
        fixed_lines = lines
    return '\n'.join(fixed_lines)

def get_file(path): #读取文件
    if os.path.exists(path):
        # 检查文件是否为.pkl文件
        if path.lower().endswith('.pkl'):
            # 打开并读取.pkl文件
            with open(path, 'rb') as file:
                data = pickle.load(file)
                return data
        elif path.lower().endswith('.jsonl'):
            with open(path, 'r', encoding='utf-8') as file:
                json_list = []
                for line in file:
                    context = line.strip()
                    json_list.append(context)
                return json_list
        elif path.lower().endswith('.json'):
            with open(path, 'r', encoding='utf-8') as file:
                context = json.load(file)
                return context
        elif os.path.isdir(path):
            file_dict = {}
            for dirpath, dirnames, filenames in os.walk(path):
                for filename in filenames:
                    # pdb.set_trace()
                    full_path = os.path.join(dirpath, filename)
                    # 使用仅文件名作为字典的键
                    file_dict[filename] = get_file(full_path)
            return file_dict
            
            
def preprocess_file(dataset):# 预处理的主要功能是对数据集的结果进行预处理，生成一个包含任务 ID、模型输出、解决方案以及错误类型的字典结构 (task_dict)
    if dataset == 'bigcodebench': # 标准格式为{'answer':{task_id},model:{task_id:{"status",'solution','error_type'}}}
        answer = get_file(ANSWER_PATH[dataset]) # 标准答案
        question = get_file(QUSETION_PATH[dataset]) # 问题
        result = get_file(RESULT_PATH[dataset]) # 结果
        code_result = get_file(CODE_RESULT_PATH[dataset]) # 模型输出
        task_dict = {}
        for line in question:
            dict_obj = json.loads(line)
            task_id = dict_obj['task_id']
            if 'question' not in task_dict:
                task_dict['question'] = {}
            task_dict['question'][task_id] = dict_obj["instruct_prompt"] # ['question'][task_id]:每一个问题的正确答案
        for line in answer:
            dict_obj = json.loads(line)
            task_id = dict_obj['task_id']
            if 'answer' not in task_dict:
                task_dict['answer'] = {}
            task_dict['answer'][task_id] = dict_obj["canonical_solution"] # ['answer'][task_id]:每一个问题的正确答案
        for model, code_result in code_result.items(): 
            for line in code_result:  
                dict_obj = json.loads(line)
                task_id = dict_obj['task_id']
                if model not in task_dict:
                    task_dict[model] = {}
                task_dict[model][task_id] = {'solution': dict_obj['solution']} # [model][task_id]['solution']:每一个问题的模型输出
                model_key = model.replace('.jsonl','_eval_results.json')
                task_dict[model][task_id]['status'] = result[model_key]['eval'][task_id][0]['status'] # [model][task_id]['status'] 每一个问题的模型输出正确与否
                error_set = set()
                if not task_dict[model][task_id]['status']:  
                    for test_case, error_report in result[model_key]['eval'][task_id][0]['details'].items():
                        for error in ERROR_TYPE:
                            if error in error_report:
                                error_set.add(error)
                task_dict[model][task_id]['error_type'] = error_set # [model][task_id]['error_type']:每一个问题的错误类型
    elif dataset.startswith('EvoEval') or dataset == 'HumanEvalPlus':
        task_dict = {}
        answer_path = f"{ANSWER_PATH['EvoEval']}/{dataset}.jsonl"
        answer = get_file(answer_path)
        for line in answer:
            dict_obj = json.loads(line)
            task_id = dict_obj['task_id']
            if 'answer' not in task_dict:
                task_dict['answer'] = {}
            task_dict['answer'][task_id] = dict_obj["canonical_solution"]
        for line in answer:
            dict_obj = json.loads(line)
            task_id = dict_obj['task_id']
            if 'question' not in task_dict:
                task_dict['question'] = {}
            task_dict['question'][task_id] = dict_obj["prompt"]
        for model in os.listdir(RESULT_PATH['EvoEval']):
            if model not in task_dict:
                task_dict[model] = {}
            result_path = f"{RESULT_PATH['EvoEval']}/{model}/{dataset}/eval_results.json"
            if dataset == 'HumanEvalPlus':
                result_path = f"{RESULT_PATH['EvoEval']}/{model}/humaneval/eval_results.json"
            result = get_file(result_path)
            for task_id,value in result['eval'].items():
                task_dict[model][task_id] ={'solution': result['eval'][task_id][0]['solution']}
                if result['eval'][task_id][0]['base_status'] == 'fail' or result['eval'][task_id][0]['plus_status'] == 'fail':
                    task_dict[model][task_id]['status'] = 'fail'
                else:
                    task_dict[model][task_id]['status'] = 'pass' 
    elif dataset == 'MBPP':
        answer_path = f"{ANSWER_PATH['EvoEval']}/{dataset}.jsonl"    
    return task_dict

def get_correct_score(tasks): 
    model_correctness_score = {}
    for model, code_result in tasks.items():
        total_score = {}
        if model != 'answer' and model != 'question':
            correctness_list = {} # 正确性分数
            AST_list = {}
            CFG_dict = {}
            for task_id, solution in tqdm.tqdm(code_result.items()):
                result = solution['solution'] 
                answer = tasks['answer'][task_id] # 正确性分数
                if solution['status'] == 'pass':
                    correctness_list[task_id] = 1
                else:
                    correctness_list[task_id] = 0
                AST_list[task_id] = AST_assemble_score(answer,result) # 相似性分数
                try:
                    fixed_answer = fix_indentation(answer)
                    CFG_dict[task_id] = calculate_cfg_similarity(fixed_answer,result)
                except Exception as e:
                    CFG_dict[task_id] = -1
            for key in correctness_list.keys():
                try:
                    if key in AST_list:
                        ast_score = AST_list[key]
                    if key in CFG_dict:
                        score = (CFG_dict[key] + ast_score)/2
                    else:
                        score = ast_score
                    total_score[key] = min(1,correctness_list[key] + score)
                except Exception as e:
                    total_score[key] = -1
            model_correctness_score[model] = total_score
    return model_correctness_score
     
def get_cyclomatic_complexity_score(tasks):
    model_complexity_score = {}
    for model, code_result in tasks.items():
        if model != 'answer' and model != 'question':
            complexity_score_list = {}  # 圈复杂度分数
            for task_id, solution in code_result.items():
                try:
                    result = solution['solution']
                    score = calculate_cyclomatic_complexity(result)
                    if score is not None:
                        complexity_score_list[task_id] = score
                    else:
                        complexity_score_list[task_id] = 0  # 表示出错或无法解析
                except Exception as e:
                    print(f"Error processing {model}/{task_id}: {e}")
                    complexity_score_list[task_id] = -1
            model_complexity_score[model] = complexity_score_list
    return model_complexity_score

def get_codebleu_score(tasks):
    model_codebleu_score = {}
    for model, code_result in tasks.items():
        if model != 'answer' and model != 'question':
            codebleu_score_list = {}  # 圈复杂度分数
            for task_id, solution in code_result.items():
                try:
                    answer = tasks['answer'][task_id]
                    result = solution['solution']
                    score = calculate_codebleu_score(answer,result)
                    if score is not None:
                        codebleu_score_list[task_id] = score
                    else:
                        codebleu_score_list[task_id] = -1  # 表示出错或无法解析
                except Exception as e:
                    print(f"Error processing {model}/{task_id}: {e}")
                    codebleu_score_list[task_id] = -1
            model_codebleu_score[model] = codebleu_score_list
    return model_codebleu_score
            
def get_security_score(tasks):
    model_security_score = {}
    for model, code_result in tasks.items():
        if model != 'answer' and model != 'question':
            security_index_list = {} # 安全性分数
            for task_id, solution in code_result.items():
                try:
                    code = solution['solution'] 
                    security_index_list[task_id] = scan_code(code)
                except Exception as e:
                    print(f"Error: {e}")
            model_security_score[model] = security_index_list
    return model_security_score

def get_halstead_complexity_score(tasks):
    model_halstead_score = {}
    for model, code_result in tasks.items():
        if model != 'answer' and model != 'question':
            halstead_score_list = {}  # 存储每个任务的 Halstead 指标
            for task_id, solution in code_result.items():
                try:
                    result = solution['solution']
                    metrics = calculate_halstead_metrics(result)
                    if metrics:
                        halstead_score_list[task_id] = metrics
                    else:
                        halstead_score_list[task_id] = {'error': 'Parsing failed'}
                except Exception as e:
                    print(f"Error processing {model}/{task_id}: {e}")
                    halstead_score_list[task_id] = {'error': str(e)}
            model_halstead_score[model] = halstead_score_list
    return model_halstead_score

def run(dataset):
    task_data_path = f'processed_data/{dataset}/tasks.pkl'
    correctness_data_path = f'processed_data/{dataset}/correctness.pkl'
    security_path = f'processed_data/{dataset}/security.pkl'
    codebleu_path = f'processed_data/{dataset}/codebleu.pkl'
    cc_path = f'processed_data/{dataset}/cc.pkl'
    h_path = f'processed_data/{dataset}/Halstead.pkl'
    if not os.path.exists(f'processed_data/{dataset}'):
        os.mkdir(f'processed_data/{dataset}')
    if not os.path.exists(task_data_path): # 数据结构预处理
        tasks = preprocess_file(dataset)
        with open(task_data_path, 'wb') as file:  # 注意模式为 'wb'（二进制写入）
            pickle.dump(tasks, file)
    else:
        with open(task_data_path,'rb') as file:
            tasks = pickle.load(file)
    
    if not os.path.exists(codebleu_path): # codebleu分析
        model_security_score = get_codebleu_score(tasks)
        with open(codebleu_path, 'wb') as file:
            pickle.dump(model_security_score, file)
    else:
        with open(codebleu_path,'rb') as file:
            model_security_score = pickle.load(file)
    
    if not os.path.exists(security_path): # 安全性分析
        model_security_score = get_security_score(tasks)
        with open(security_path, 'wb') as file:
            pickle.dump(model_security_score, file)
    else:
        with open(security_path,'rb') as file:
            model_security_score = pickle.load(file)
            
    if not os.path.exists(cc_path): # cc分析
        cc_score = get_cyclomatic_complexity_score(tasks)
        with open(cc_path, 'wb') as file:
            pickle.dump(cc_score, file)
    else:
        with open(cc_path,'rb') as file:
            cc_score = pickle.load(file)
    
    if not os.path.exists(h_path): # halstead分析
        h_score = get_halstead_complexity_score(tasks)
        with open(h_path, 'wb') as file:
            pickle.dump(h_score, file)
    else:
        with open(h_path,'rb') as file:
            h_score = pickle.load(file)
    if not os.path.exists(correctness_data_path): # 正确性分析
        model_correctness_score = get_correct_score(tasks)
        with open(correctness_data_path, 'wb') as file:
            pickle.dump(model_correctness_score, file)
    else:
        with open(correctness_data_path,'rb') as file:
            model_correctness_score = pickle.load(file)
    

    
if __name__ == '__main__':
    for dataset in tqdm.tqdm(DATASET_LIST):
        run(dataset)
    