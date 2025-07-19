import correctness_analysis.TSED
import correctness_analysis.cfg
import correctness_analysis.AST_DFG_CFG
from correctness_analysis.staticfg.staticfg import CFGBuilder
from apted import APTED, PerEditOperationConfig
from apted.helpers import Tree
import ast
ERROR_TYPE = ['ArithmeticError', 'AssertionError', 'AttributeError', 'BaseException', 
              'BlockingIOError', 'BrokenPipeError', 'BufferError', 'BytesWarning', 
              'ChildProcessError', 'ConnectionAbortedError', 'ConnectionError', 
              'ConnectionRefusedError', 'ConnectionResetError', 'DeprecationWarning', 
              'EOFError', 'Ellipsis', 'EnvironmentError', 'Exception', 'FileExistsError', 
              'FileNotFoundError', 'FloatingPointError', 'FutureWarning', 'GeneratorExit', 
              'IOError', 'ImportError', 'ImportWarning', 'IndentationError', 'IndexError', 
              'InterruptedError', 'IsADirectoryError', 'KeyError', 'KeyboardInterrupt', 
              'LookupError', 'MemoryError', 'ModuleNotFoundError', 'NameError', 
              'NotADirectoryError', 'NotImplementedError', 'OSError', 'OverflowError', 
              'PendingDeprecationWarning', 'PermissionError', 'ProcessLookupError', 
              'RecursionError', 'ReferenceError', 'ResourceWarning', 'RuntimeError', 
              'RuntimeWarning', 'StopAsyncIteration', 'StopIteration', 'SyntaxError', 
              'SyntaxWarning', 'SystemError', 'SystemExit', 'TabError', 'TimeoutError', 
              'TypeError', 'UnboundLocalError', 'UnicodeDecodeError', 'UnicodeEncodeError', 
              'UnicodeError', 'UnicodeTranslateError', 'UnicodeWarning', 'UserWarning', 
              'ValueError', 'Warning', 'ZeroDivisionError']

class CFGNodeTree:
    def __init__(self, block_id, label):
        self.id = block_id
        self.label = label
        self.children = []
        self.name = None

    def add_child(self, node):
        self.children.append(node)

def convert_cfg_to_tree(cfg_block, visited=None):
    """
    Convert a CFG block and its children into a tree structure suitable for comparison.
    """
    if visited is None:
        visited = set()
    
    if cfg_block in visited:
        return None
    
    visited.add(cfg_block)
    root = CFGNodeTree(cfg_block.id, str(cfg_block.get_source()))
    
    for exit_ in cfg_block.exits:
        child_node = convert_cfg_to_tree(exit_.target, visited)
        if child_node:
            root.add_child(child_node)
    
    return root

def calculate_cfg_similarity(code1, code2):
    """
    Calculate the similarity between two CFGs.
    """
    Builder = CFGBuilder()
    cfg1 = Builder.build_from_src(name='solution1',src=code1)
    cfg2 = Builder.build_from_src(name='solution2',src=code2)
    # Convert CFGs to trees
    tree1 = convert_cfg_to_tree(cfg1.entryblock)
    tree2 = convert_cfg_to_tree(cfg2.entryblock)
    
    # Compute edit distance
    apted = APTED(tree1, tree2, PerEditOperationConfig(1, 1, 1))
    res = apted.compute_edit_distance()
    
    # Calculate maximum possible length (total nodes in the larger tree)
    max_len = max(calculate_node_count(tree1), calculate_node_count(tree2))
    
    if max_len > 0:
        similarity_score = (max_len - res) / max_len
    else:
        similarity_score = 1.0
    
    return similarity_score

def calculate_node_count(node):
    count = 1
    for child in node.children:
        count += calculate_node_count(child)
    return count

def AST_assemble_score(code1, code2):
    AST_score = correctness_analysis.TSED.Calaulte("python",code1, code2, 1.0, 0.8, 1.0)
    return AST_score

