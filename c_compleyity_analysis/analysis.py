from radon.complexity import cc_visit
import ast


def calculate_cyclomatic_complexity(code):
    """计算单段代码的圈复杂度"""
    try:
        complexity_scores = cc_visit(code)
        # 返回所有函数/类的平均复杂度，或者只取整体复杂度
        if complexity_scores:
            return sum(c.complexity for c in complexity_scores) / len(complexity_scores) \
                if complexity_scores else 0
        return 0
    except (SyntaxError, TypeError, AttributeError, Exception) as e:
        print(f"Complexity calculation error: {e}")
        return None