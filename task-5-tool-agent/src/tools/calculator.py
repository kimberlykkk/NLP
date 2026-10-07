"""任务五 M1：计算器工具（AST 白名单求值，禁止任意代码执行）。

自检：run({"expression": "2 + 3 * 4"}) 应返回包含 "14" 的字符串。
"""
import ast
import math

TOOL_SCHEMA = {
    "name": "calculator",
    "description": "计算数学表达式。支持四则运算、乘方、取模、括号，"
                   "以及数学函数 sqrt/sin/cos/tan/log/log10/exp/pow/factorial/floor/ceil/round "
                   "和常量 pi/e。",
    "parameters": {
        "type": "object",
        "properties": {"expression": {"type": "string", "description": "数学表达式，如 (123 + 456) * 789"}},
        "required": ["expression"],
    },
}

_FUNCS = {
    "sqrt": math.sqrt, "sin": math.sin, "cos": math.cos, "tan": math.tan,
    "asin": math.asin, "acos": math.acos, "atan": math.atan,
    "log": math.log, "log10": math.log10, "log2": math.log2,
    "exp": math.exp, "pow": pow, "factorial": math.factorial,
    "floor": math.floor, "ceil": math.ceil, "round": round, "abs": abs,
}
_CONSTS = {"pi": math.pi, "e": math.e}
_BINOPS = {
    ast.Add: lambda a, b: a + b, ast.Sub: lambda a, b: a - b,
    ast.Mult: lambda a, b: a * b, ast.Div: lambda a, b: a / b,
    ast.FloorDiv: lambda a, b: a // b, ast.Mod: lambda a, b: a % b,
    ast.Pow: lambda a, b: a ** b,
}


def _eval(node):
    if isinstance(node, ast.Constant):
        if isinstance(node.value, (int, float)):
            return node.value
        raise ValueError(f"不支持的常量: {node.value!r}")
    if isinstance(node, ast.BinOp):
        fn = _BINOPS.get(type(node.op))
        if fn is None:
            raise ValueError(f"不支持的运算符: {type(node.op).__name__}")
        return fn(_eval(node.left), _eval(node.right))
    if isinstance(node, ast.UnaryOp):
        if isinstance(node.op, ast.UAdd):
            return +_eval(node.operand)
        if isinstance(node.op, ast.USub):
            return -_eval(node.operand)
        raise ValueError(f"不支持的运算符: {type(node.op).__name__}")
    if isinstance(node, ast.Call):
        if not isinstance(node.func, ast.Name) or node.func.id not in _FUNCS:
            raise ValueError("只允许调用白名单数学函数")
        args = [_eval(a) for a in node.args]
        return _FUNCS[node.func.id](*args)
    if isinstance(node, ast.Name):
        if node.id in _CONSTS:
            return _CONSTS[node.id]
        raise ValueError(f"未知名称: {node.id}")
    raise ValueError(f"不支持语法: {type(node).__name__}")


def run(args: dict) -> str:
    expression = str(args.get("expression", "")).strip()
    if not expression:
        raise ValueError("缺少 expression 参数")
    tree = ast.parse(expression, mode="eval")
    value = _eval(tree.body)
    if isinstance(value, float):
        return f"{value:.12g}"
    return str(value)
