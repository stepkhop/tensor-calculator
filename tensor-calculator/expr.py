"""
Разбор и вычисление выражений калькулятора

Вместо eval строка разбирается модулем ast, и вычисляются только
разрешенные узлы: числа, имена A и B, операции + - * / ^ и вызовы функций
из таблицы FUNCS. Все остальное (атрибуты, импорты, лямбды, индексы)
отклоняется с сообщением ExprError

Операторы:
    A + B, A - B, -A        сложение, вычитание (типы должны совпадать)
    A * B                   тензорное произведение; если один операнд
                            число - умножение на скаляр
    A / 2                   деление на число
    A ^ B                   внешнее произведение
Функции (номера осей - с единицы, отдельно в группе):
    dot(A, B)               однократная свертка: A^i_j B^j_k, A^i_j v^j
    ddot(A, B)              двойная свертка: C^{ij}_{kl} e^{kl}
    contract(T, i, j)       свертка i-го нижнего индекса с j-м верхним
    contract2(A, B, g, i, j) свертка i-й оси группы g ('p'|'q') тензора A
                            с j-й осью противоположной группы тензора B
    contract2(A, B, [('p', i, j), ('q', k, l)])  то же сразу по нескольким парам
    wedge(A, B)             внешнее произведение (то же, что A ^ B, но без
                            ловушки с приоритетом: в Python ^ слабее + и *)
    tr(T), det(T)           след (1,1) и определитель (ранг 2) - числа
    sym(T), alt(T)          симметрирование / альтернирование верхних осей
    symq(T), altq(T)        то же для нижних осей
    perm(T, [2,1,3])        перестановка верхних осей, permq - нижних
    dev(T), sph(T)          девиатор и шаровая часть тензора ранга 2
    delta(n), eps(n)        символ Кронекера и символ Леви-Чивиты
    low(T, i), up(T, i)     опускание i-го верхнего / поднятие i-го нижнего
                            индекса с текущей фундаментальной матрицей
                            (по умолчанию g_ij = delta_ij, ОНБ)
    low(T, i, G), up(T, i, G)  то же с метрикой G (тензор ранга 2)
    scalar(x)               число как тензор ранга 0
"""

import ast
from fractions import Fraction

import tensor_ext as tx
import tensor_eigen as te

__all__ = ["evaluate", "ExprError", "FUNCS", "as_tensor"]


class ExprError(ValueError):
    pass


def _tensor(x, what):
    if isinstance(x, Fraction):
        raise ExprError(f"{what}: ожидался тензор, получено число {x}")
    return x


def _num(x, what):
    if not isinstance(x, Fraction):
        raise ExprError(f"{what}: ожидалось число")
    return x


def _int(x, what):
    x = _num(x, what)
    if x.denominator != 1:
        raise ExprError(f"{what}: ожидалось целое число")
    return int(x)


def _perm_list(x, what):
    if not isinstance(x, list):
        raise ExprError(f"{what}: перестановку задают списком, например [2, 1, 3]")
    return [_int(v, what) for v in x]


def _contract(t, i, j):
    return tx.contract(_tensor(t, "contract"), _int(i, "contract") - 1,
                       _int(j, "contract") - 1)


def _contract2(a, b, *spec):
    """contract2(A, B, 'p', i, j) или contract2(A, B, [('p', i, j), ...])"""
    if len(spec) == 3:
        pairs = [tuple(spec)]
    elif len(spec) == 1 and isinstance(spec[0], list):
        pairs = spec[0]
    else:
        raise ExprError("contract2: укажите группу и два номера или список пар")
    norm = []
    for pair in pairs:
        if not isinstance(pair, (list, tuple)) or len(pair) != 3 or pair[0] not in ("p", "q"):
            raise ExprError("contract2: пара задается как ('p'|'q', i, j)")
        g, i, j = pair
        norm.append((g, _int(i, "contract2") - 1, _int(j, "contract2") - 1))
    a, b = _tensor(a, "contract2"), _tensor(b, "contract2")
    _guard_rank(a.p + a.q + b.p + b.q - 2 * len(norm), a.n, "contract2")
    return tx.contract_ab(a, b, norm)


_METRIC = None


def _low(t, i, g=None):
    t = _tensor(t, "low")
    return tx.lower_index(t, _int(i, "low") - 1,
                          _METRIC if g is None else _tensor(g, "low"))


def _up(t, i, g=None):
    t = _tensor(t, "up")
    return tx.raise_index(t, _int(i, "up") - 1,
                          _METRIC if g is None else _tensor(g, "up"))


def _wedge(a, b):
    """Внешнее произведение по верхним индексам, для ковекторов - по нижним"""
    a, b = _tensor(a, "wedge"), _tensor(b, "wedge")
    _guard_rank(a.p + a.q + b.p + b.q, a.n, "wedge")
    group = "q" if a.p == 0 and b.p == 0 else "p"
    k = (a.p + b.p) if group == "p" else (a.q + b.q)
    _guard_perm(k, "wedge")
    return tx.wedge(a, b, group)


MAX_COMPONENTS = 6561
MAX_PERM_AXES = 7


def _guard_rank(rank, n, what):
    if n ** rank > MAX_COMPONENTS:
        raise ExprError(f"{what}: результат содержал бы {n}^{rank} = {n ** rank} "
                        f"компонент, предел {MAX_COMPONENTS}")


def _guard_perm(k, what):
    if k > MAX_PERM_AXES:
        raise ExprError(f"{what}: {k}! перестановок - слишком долго, предел "
                        f"{MAX_PERM_AXES} осей")


def _sym_checked(t, group, signed, what):
    t = _tensor(t, what)
    _guard_perm(t.p if group == "p" else t.q, what)
    return tx.antisymmetrize(t, group) if signed else tx.symmetrize(t, group)


def _sized(fn, rank_of, what):
    def call(n):
        n = _int(n, what)
        if n < 1:
            raise ExprError(f"{what}: нужна размерность n >= 1")
        _guard_rank(rank_of(n), n, what)
        return fn(n)
    return call


def _scalar_tensor(x, n=1):
    return tx.from_list(0, 0, n, _num(x, "scalar"))


FUNCS = {
    "dot": lambda a, b: tx.dot(_tensor(a, "dot"), _tensor(b, "dot")),
    "ddot": lambda a, b: tx.ddot(_tensor(a, "ddot"), _tensor(b, "ddot")),
    "contract": _contract,
    "contract2": _contract2,
    "tr": lambda t: tx.trace(_tensor(t, "tr")),
    "det": lambda t: tx.det(_tensor(t, "det")),
    "sym": lambda t: _sym_checked(t, "p", False, "sym"),
    "alt": lambda t: _sym_checked(t, "p", True, "alt"),
    "symq": lambda t: _sym_checked(t, "q", False, "symq"),
    "altq": lambda t: _sym_checked(t, "q", True, "altq"),
    "wedge": _wedge,
    "perm": lambda t, p: tx.permute(_tensor(t, "perm"), _perm_list(p, "perm"), "p"),
    "permq": lambda t, p: tx.permute(_tensor(t, "permq"), _perm_list(p, "permq"), "q"),
    "dev": lambda t: te.deviator(_tensor(t, "dev"), _METRIC),
    "sph": lambda t: te.spherical_part(_tensor(t, "sph"), _METRIC),
    "delta": _sized(tx.delta, lambda n: 2, "delta"),
    "eps": _sized(tx.levi_civita, lambda n: n, "eps"),
    "low": _low,
    "up": _up,
    "scalar": _scalar_tensor,
}


def _binop(op, a, b):
    an, bn = isinstance(a, Fraction), isinstance(b, Fraction)
    if isinstance(op, ast.Add):
        if an and bn:
            return a + b
        if an or bn:
            raise ExprError("нельзя складывать число и тензор")
        return tx.add(a, b)
    if isinstance(op, ast.Sub):
        if an and bn:
            return a - b
        if an or bn:
            raise ExprError("нельзя вычитать число и тензор")
        return tx.sub(a, b)
    if isinstance(op, ast.Mult):
        if an and bn:
            return a * b
        if an:
            return tx.scalar_mul(b, a)
        if bn:
            return tx.scalar_mul(a, b)
        _guard_rank(a.p + a.q + b.p + b.q, a.n, "тензорное произведение")
        return tx.tensor_product(a, b)
    if isinstance(op, ast.Div):
        if not bn:
            raise ExprError("делить можно только на число")
        if b == 0:
            raise ExprError("деление на ноль")
        return a / b if an else tx.scalar_mul(a, 1 / b)
    if isinstance(op, ast.BitXor):
        if an or bn:
            raise ExprError("внешнее произведение определено для тензоров")
        return _wedge(a, b)
    raise ExprError(f"операция {type(op).__name__} не поддерживается")


def _eval(node, env):
    if isinstance(node, ast.Expression):
        return _eval(node.body, env)
    if isinstance(node, ast.Constant):
        if isinstance(node.value, bool):
            raise ExprError("логические значения не поддерживаются")
        if isinstance(node.value, (int, float)):
            return Fraction(str(node.value))
        if isinstance(node.value, str) and node.value in ("p", "q"):
            return node.value
        raise ExprError(f"недопустимая константа {node.value!r}")
    if isinstance(node, ast.Name):
        if node.id in env:
            return env[node.id]
        raise ExprError(f"неизвестное имя {node.id}")
    if isinstance(node, ast.List):
        return [_eval(x, env) for x in node.elts]
    if isinstance(node, ast.Tuple):
        return tuple(_eval(x, env) for x in node.elts)
    if isinstance(node, ast.UnaryOp):
        val = _eval(node.operand, env)
        if isinstance(node.op, ast.USub):
            return -val if isinstance(val, Fraction) else tx.neg(val)
        if isinstance(node.op, ast.UAdd):
            return val
        raise ExprError("недопустимая унарная операция")
    if isinstance(node, ast.BinOp):
        return _binop(node.op, _eval(node.left, env), _eval(node.right, env))
    if isinstance(node, ast.Call):
        if not isinstance(node.func, ast.Name) or node.func.id not in FUNCS:
            name = getattr(node.func, "id", "?")
            raise ExprError(f"неизвестная функция {name}")
        if node.keywords:
            raise ExprError("именованные аргументы не поддерживаются")
        args = [_eval(a, env) for a in node.args]
        try:
            return FUNCS[node.func.id](*args)
        except TypeError:
            raise ExprError(f"{node.func.id}: неверное число аргументов") from None
    raise ExprError(f"конструкция {type(node).__name__} не поддерживается")


def evaluate(text, env, metric=None):
    """
    Вычислить выражение. env - словарь имен, обычно {"A": ..., "B": ..., "G": ...}.
    metric - фундаментальная матрица g_ij (None - ортонормированный базис).
    Результат - тензор или Fraction
    """
    global _METRIC
    _METRIC = metric
    try:
        return _evaluate(text, env)
    finally:
        _METRIC = None


def _evaluate(text, env):
    text = (text or "").strip()
    if not text:
        raise ExprError("пустое выражение")
    try:
        tree = ast.parse(text, mode="eval")
    except SyntaxError as e:
        raise ExprError(f"синтаксическая ошибка: {e.msg}") from None
    try:
        return _eval(tree, env)
    except ExprError:
        raise
    except ValueError as e:
        raise ExprError(str(e)) from None


def as_tensor(value, n):
    """Число превращается в тензор ранга 0, чтобы его можно было показать"""
    if isinstance(value, Fraction):
        return tx.from_list(0, 0, n, value)
    return value
