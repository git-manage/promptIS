
DATASET_LIST = ['bigcodebench','EvoEval_combine','EvoEval_concise','EvoEval_creative','EvoEval_difficult',
                'EvoEval_subtle','EvoEval_tool_use','EvoEval_verbose','HumanEvalPlus'] # 数据集列表

ANSWER_PATH = { # 测试的答案路径
    'bigcodebench':'../dataset/bigcodebench/bigcode/full/BigCodeBench.jsonl',
    'EvoEval':'../dataset/evoeval/override',
    'MBPP':'../dataset/mbpp',
    
    }

QUSETION_PATH = { # 问题路径
    'bigcodebench':'../dataset/bigcodebench/bigcode/full/BigCodeBench.jsonl',
    'EvoEval':'../dataset/evoeval/dataset'
    }

CODE_RESULT_PATH = { # 模型输出路径
    'bigcodebench':'../dataset/bigcodebench/dataset/code_result',
    'EvoEval':'../dataset/evoeval/dataset'}

RESULT_PATH = { # 模型测试结果的存储路径
    'bigcodebench':'../dataset/bigcodebench/dataset/result',
    'EvoEval':'../dataset/evoeval/dataset'
    }