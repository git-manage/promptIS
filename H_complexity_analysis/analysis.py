from radon.metrics import h_visit
import json
import pdb
def calculate_halstead_metrics(code):
    """计算单段代码的 Halstead 复杂度"""
    try:
        halstead = h_visit(code)
        return {
            'length': halstead.total.length,
            'volume': halstead.total.volume,
            'difficulty': halstead.total.difficulty,
            'effort': halstead.total.effort,
            'time': halstead.total.time,
            'bugs': halstead.total.bugs
        }
    except Exception as e:
        print(f"Halstead calculation error: {e}")
        return None