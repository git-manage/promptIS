import ast
import networkx as nx
from correctness_analysis.cfg import PyParser, CFGVisitor
from zss import simple_distance, Node
import asttokens
import math

def CFG_generate(code,task):
    parser = PyParser(code)
    parser.removeCommentsAndDocstrings()
    parser.formatCode()
    cfg = CFGVisitor().build(task, ast.parse(parser.script))
    return cfg

def ast_to_tree(node, parent=None):
    """将AST节点转换为自定义的树节点"""
    name = type(node).__name__
    tree_node = Node(name)
    for child in ast.iter_child_nodes(node):
        tree_node.addkid(ast_to_tree(child, tree_node))
    return tree_node

def get_tree_size(tree):
    """计算树的节点数量"""
    return len(list(tree.leaves())) + len(list(tree.children()))

def ast_similarity(ast1, ast2):
    tree1 = ast_to_tree(ast1)
    tree2 = ast_to_tree(ast2)
    distance = simple_distance(tree1, tree2)
    size1 = get_tree_size(tree1)
    size2 = get_tree_size(tree2)
    normalized_distance = distance / math.sqrt(size1, size2)
    return normalized_distance