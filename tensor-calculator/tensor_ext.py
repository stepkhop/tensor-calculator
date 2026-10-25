"""
ДОГОВОРЕННОСТЬ ОБ ИНДЕКСАХ:

Все модули работают с компонентами тензоров в фиксированном базисе: сам
тензор от базиса не зависит, а RecurTensor(p, q, n) хранит набор его
компонент как дерево глубины r = p + q. Тип (p, q) хранится вместе с
компонентами и определяет, какие операции допустимы

RecurTensor(p, q, n) хранит данные как дерево глубины r = p + q

Уровни вложенности:

    уровни 0 .. q-1        - группа 'q'
    уровни q .. q+p-1      - группа 'p'

Внутри самой библиотеки списки координат (coo/read/write) идут в обратном
порядке относительно уровней вложенности, поэтому permutation в
RecurTensor.transpose нумерует оси задом наперед, а все функции модуля
нумеруют оси в естественном порядке вложенности, 0-based, отдельно внутри
каждой группы

Пример: t - тензор (p=1, q=1, n=2), тогда to_list(t)[a][b] - это элемент
с индексом a из группы q и индексом b из группы p

Все числовые значения приводятся к Fraction, чтобы арифметика была более точной
"""

from fractions import Fraction
from itertools import permutations, product

from recur_tensor import RecurTensor

__all__ = [
    "to_list", "from_list", "zeros", "equal", "pretty", "shape",
    "add", "sub", "neg", "scalar_mul",
    "tensor_product", "permute", "swap_axes",
    "contract", "trace", "full_contraction",
    "symmetrize", "antisymmetrize", "wedge",
    "is_symmetric", "is_antisymmetric",
    "delta", "levi_civita", "det", "contract_with_matrix",
    "contract_ab", "dot", "ddot",
    "to_conventional", "from_conventional",
    "inverse_matrix", "basis_matrix", "gram_matrix", "is_cartesian",
    "metric_matrix", "is_orthonormal",
    "lower_index", "raise_index",
]


def shape(t):
    return t.p, t.q, t.n


def to_list(t):
    if t.p + t.q == 0:
        return Fraction(t.data)
    return [to_list(child) for child in t.data]


def to_conventional(t):
    """
    Вложенный список в общепринятом порядке индексов: сначала верхние,
    затем нижние, C[i1]...[ip][j1]...[jq] = T^{i1...ip}_{j1...jq}.
    Внутреннее хранение идет в обратном порядке групп (сначала нижние)
    """
    r = t.p + t.q
    if r == 0:
        return Fraction(t.data)
    src = to_list(t)
    out = _const_nested(r, t.n, Fraction(0))
    for idx in product(range(t.n), repeat=r):
        _set(out, idx, _get(src, idx[t.p:] + idx[:t.p]))
    return out


def from_conventional(p, q, n, data):
    """Обратное к to_conventional: data[i1]...[ip][j1]...[jq]"""
    r = p + q
    if r == 0:
        return from_list(p, q, n, data)
    _check_nested(data, r, n)
    return _build(p, q, n, lambda idx: _get(data, idx[q:] + idx[:q]))


def _check_nested(data, rank, n):
    if rank == 0:
        if isinstance(data, (list, tuple)):
            raise ValueError("ожидалось число, получен список")
        return
    if not isinstance(data, (list, tuple)) or len(data) != n:
        raise ValueError(f"на уровне ранга {rank} ожидался список длины {n}")
    for item in data:
        _check_nested(item, rank - 1, n)


def from_list(p, q, n, data):
    if p < 0 or q < 0 or n < 0:
        raise ValueError("p, q, n должны быть неотрицательными")
    _check_nested(data, p + q, n)
    t = RecurTensor(p, q, n)

    def fill(node, src):
        if node.p + node.q == 0:
            node.data = Fraction(src)
        else:
            for i in range(n):
                fill(node.data[i], src[i])

    fill(t, data)
    return t


def zeros(p, q, n):
    return from_list(p, q, n, _const_nested(p + q, n, Fraction(0)))


def _const_nested(rank, n, value):
    if rank == 0:
        return value
    return [_const_nested(rank - 1, n, value) for _ in range(n)]


def equal(a, b):
    """Совпадают ли тип (p, q, n) и все компоненты"""
    return shape(a) == shape(b) and to_list(a) == to_list(b)


def pretty(t):
    """Тип тензора + компоненты"""
    def fmt(x):
        if isinstance(x, list):
            return "[" + ", ".join(fmt(i) for i in x) + "]"
        return str(x) if x.denominator != 1 else str(x.numerator)

    return f"T(p={t.p}, q={t.q}, n={t.n}) " + fmt(to_list(t))


def _axes(t):
    return product(range(t.n), repeat=t.p + t.q)


def _get(data, idx):
    for i in idx:
        data = data[i]
    return data


def _set(data, idx, value):
    for i in idx[:-1]:
        data = data[i]
    data[idx[-1]] = value


def _build(p, q, n, func):
    """Собрать тензор по функции func(мультииндекс) -> число"""
    r = p + q
    if r == 0:
        return from_list(p, q, n, Fraction(func(())))
    data = _const_nested(r, n, Fraction(0))
    for idx in product(range(n), repeat=r):
        _set(data, idx, Fraction(func(idx)))
    return from_list(p, q, n, data)


def _group_slice(t, group):
    """Границы группы осей в порядке вложенности: q - первые, p - последние"""
    if group == "q":
        return 0, t.q
    if group == "p":
        return t.q, t.q + t.p
    raise ValueError("group должна быть 'p' или 'q'")


def _same_type(a, b, op):
    if shape(a) != shape(b):
        raise ValueError(
            f"{op}: несовместимые тензоры {shape(a)} и {shape(b)}; "
            "требуется совпадение p, q и n"
        )


def add(a, b):
    """Поэлементная сумма"""
    _same_type(a, b, "add")
    return _build(a.p, a.q, a.n,
                  lambda i: _get(to_list(a), i) + _get(to_list(b), i))


def sub(a, b):
    """Поэлементная разность"""
    _same_type(a, b, "sub")
    return _build(a.p, a.q, a.n,
                  lambda i: _get(to_list(a), i) - _get(to_list(b), i))


def neg(a):
    return scalar_mul(a, -1)


def scalar_mul(t, c):
    """Умножение на скаляр"""
    c = Fraction(c)
    return _build(t.p, t.q, t.n, lambda i: c * _get(to_list(t), i))


def tensor_product(a, b):
    """
    Тензорное произведение с математически ожидаемым порядком осей:

        (A (x) B)[qa, qb, pa, pb] = A[qa, pa] * B[qb, pb]

    """
    if a.n != b.n:
        raise ValueError(f"разные размерности пространства: {a.n} и {b.n}")
    la, lb = to_list(a), to_list(b)

    def value(idx):
        qa = idx[:a.q]
        qb = idx[a.q:a.q + b.q]
        pa = idx[a.q + b.q:a.q + b.q + a.p]
        pb = idx[a.q + b.q + a.p:]
        return _get(la, qa + pa) * _get(lb, qb + pb)

    return _build(a.p + b.p, a.q + b.q, a.n, value)


def permute(t, perm, group="p"):
    """
    Транспонирование по осям одной группы:

    perm - перестановка чисел 1..k (k - число осей группы) в естественном
    порядке вложенности: perm[x] = номер оси исходного тензора, которая
    встанет на место x. Перестановка проверяется на корректность
    (в исходном transpose [1,1,2] и [0,1,2] проходят молча)
    """
    start, stop = _group_slice(t, group)
    k = stop - start
    perm = list(perm)
    if sorted(perm) != list(range(1, k + 1)):
        raise ValueError(
            f"perm={perm} не является перестановкой 1..{k} "
            f"для группы '{group}' тензора {shape(t)}"
        )
    zero = [x - 1 for x in perm]
    src = to_list(t)

    def value(idx):
        head = list(idx[:start])
        mid = list(idx[start:stop])
        tail = list(idx[stop:])
        moved = [mid[zero[x]] for x in range(k)]
        return _get(src, tuple(head + moved + tail))

    return _build(t.p, t.q, t.n, value)


def swap_axes(t, i, j, group="p"):
    """Поменять местами две оси группы"""
    start, stop = _group_slice(t, group)
    k = stop - start
    if not (0 <= i < k and 0 <= j < k):
        raise ValueError(f"оси {i}, {j} вне группы '{group}' размера {k}")
    perm = list(range(1, k + 1))
    perm[i], perm[j] = perm[j], perm[i]
    return permute(t, perm, group)


def contract(t, i_q, i_p):
    """
    Свертка i_q-й оси группы q с i_p-й осью группы p (обе 0-based)
    Результат - тензор (p-1, q-1, n)
    """
    if t.p == 0 or t.q == 0:
        raise ValueError("для свертки нужен хотя бы один верхний и один нижний индекс")
    if not (0 <= i_q < t.q and 0 <= i_p < t.p):
        raise ValueError(f"оси ({i_q}, {i_p}) вне диапазона для {shape(t)}")
    src = to_list(t)
    pos_q, pos_p = i_q, t.q + i_p

    def value(idx):
        rest_q = list(idx[:t.q - 1])
        rest_p = list(idx[t.q - 1:])
        total = Fraction(0)
        for s in range(t.n):
            full_q = rest_q[:i_q] + [s] + rest_q[i_q:]
            full_p = rest_p[:i_p] + [s] + rest_p[i_p:]
            total += _get(src, tuple(full_q + full_p))
        return total

    assert pos_q < pos_p
    return _build(t.p - 1, t.q - 1, t.n, value)


def trace(t):
    """След тензора типа (1, 1) - скаляр Fraction"""
    if (t.p, t.q) != (1, 1):
        raise ValueError("trace определен для тензора типа (1, 1)")
    return to_list(contract(t, 0, 0))


def full_contraction(a, b):
    """Полная свертка по всем осям (скалярное произведение), Fraction"""
    _same_type(a, b, "full_contraction")
    la, lb = to_list(a), to_list(b)
    return sum((_get(la, i) * _get(lb, i) for i in _axes(a)), Fraction(0))


def _inversions(perm):
    return sum(1 for i in range(len(perm)) for j in range(i + 1, len(perm))
               if perm[j] < perm[i])


def _sym(t, group, signed):
    start, stop = _group_slice(t, group)
    k = stop - start
    if k == 0:
        return from_list(t.p, t.q, t.n, to_list(t))
    acc = zeros(t.p, t.q, t.n)
    for perm in permutations(range(1, k + 1)):
        term = permute(t, perm, group)
        if signed and _inversions(perm) % 2:
            acc = sub(acc, term)
        else:
            acc = add(acc, term)
    factor = Fraction(1, 1)
    for i in range(2, k + 1):
        factor /= i
    return scalar_mul(acc, factor)


def symmetrize(t, group="p"):
    """Симметризация по осям группы: Sym(T) = (1/k!) * сумма по перестановкам"""
    return _sym(t, group, signed=False)


def antisymmetrize(t, group="p"):
    """Альтернирование: Alt(T) = (1/k!) * знакопеременная сумма"""
    return _sym(t, group, signed=True)


def wedge(a, b, group="p"):
    """
    Внешнее произведение с правильным знаком:

        A ^ B = C(k+m, k) * Alt(A (x) B)

    Для двух векторов дает (u ^ v)[i][j] = u[i]*v[j] - u[j]*v[i]

    """
    prod = tensor_product(a, b)
    start, stop = _group_slice(prod, group)
    k = stop - start
    ka = a.p if group == "p" else a.q
    coef = Fraction(1)
    for i in range(1, k + 1):
        coef *= i
    for i in range(1, ka + 1):
        coef /= i
    for i in range(1, k - ka + 1):
        coef /= i
    return scalar_mul(antisymmetrize(prod, group), coef)


def is_symmetric(t, group="p"):
    return equal(t, symmetrize(t, group))


def is_antisymmetric(t, group="p"):
    return equal(t, antisymmetrize(t, group))


def delta(n):
    """Символ Кронекера как тензор типа (1, 1)"""
    return _build(1, 1, n, lambda i: 1 if i[0] == i[1] else 0)


def levi_civita(n):
    """Полностью антисимметричный символ Леви-Чивиты, тип (n, 0)"""
    def value(idx):
        if len(set(idx)) != len(idx):
            return 0
        return -1 if _inversions(idx) % 2 else 1

    return _build(n, 0, n, value)


def det(t):
    """Определитель тензора ранга 2 (любой группы) по формуле Лейбница"""
    if t.p + t.q != 2:
        raise ValueError("det определен для тензора ранга 2")
    src = to_list(t)
    n = t.n
    total = Fraction(0)
    for perm in permutations(range(n)):
        term = Fraction(1)
        for i in range(n):
            term *= src[i][perm[i]]
        total += -term if _inversions(perm) % 2 else term
    return total


def contract_with_matrix(t, group, axis, matrix):
    """
    Свернуть одну ось тензора с матрицей n x n:

        R[..., i, ...] = sum_s M[i][s] * T[..., s, ...]

    Годится для смены базиса, поднятия/опускания индекса метрикой и т.п
    """
    start, stop = _group_slice(t, group)
    k = stop - start
    if not (0 <= axis < k):
        raise ValueError(f"ось {axis} вне группы '{group}' размера {k}")
    m = [[Fraction(x) for x in row] for row in matrix]
    if len(m) != t.n or any(len(row) != t.n for row in m):
        raise ValueError(f"матрица должна быть {t.n}x{t.n}")
    src = to_list(t)
    pos = start + axis

    def value(idx):
        total = Fraction(0)
        for s in range(t.n):
            shifted = list(idx)
            shifted[pos] = s
            total += m[idx[pos]][s] * _get(src, tuple(shifted))
        return total

    return _build(t.p, t.q, t.n, value)


def contract_ab(a, b, pairs):
    """
    Свертка двух тензоров по нескольким парам индексов без построения
    полного тензорного произведения:

        pairs = [(group, i_a, i_b), ...]

    group = 'p': i_a-я верхняя ось A сворачивается с i_b-й нижней осью B,
    group = 'q': i_a-я нижняя ось A сворачивается с i_b-й верхней осью B.
    Номера осей 0-based, внутри своей группы.

    Порядок осей результата такой же, как у tensor_product:
    нижние - оставшиеся A, затем оставшиеся B; верхние - так же.
    Свертка с одной парой совпадает с contract(tensor_product(a, b), ...)
    """
    if a.n != b.n:
        raise ValueError(f"разные размерности пространства: {a.n} и {b.n}")
    used = {("a", "p"): set(), ("a", "q"): set(),
            ("b", "p"): set(), ("b", "q"): set()}
    norm = []
    for pair in pairs:
        if len(pair) != 3 or pair[0] not in ("p", "q"):
            raise ValueError(f"пара {pair} должна иметь вид ('p'|'q', i_a, i_b)")
        g, ia, ib = pair
        gb = "q" if g == "p" else "p"
        size_a = a.p if g == "p" else a.q
        size_b = b.p if gb == "p" else b.q
        if not (0 <= ia < size_a and 0 <= ib < size_b):
            raise ValueError(
                f"пара {pair} вне диапазона: у A {size_a} осей группы '{g}', "
                f"у B {size_b} осей группы '{gb}'")
        if ia in used[("a", g)] or ib in used[("b", gb)]:
            raise ValueError(f"ось из пары {pair} уже участвует в свертке")
        used[("a", g)].add(ia)
        used[("b", gb)].add(ib)
        norm.append((g, ia, ib))

    free_qa = [i for i in range(a.q) if i not in used[("a", "q")]]
    free_pa = [i for i in range(a.p) if i not in used[("a", "p")]]
    free_qb = [i for i in range(b.q) if i not in used[("b", "q")]]
    free_pb = [i for i in range(b.p) if i not in used[("b", "p")]]
    rq = len(free_qa) + len(free_qb)
    rp = len(free_pa) + len(free_pb)
    la, lb = to_list(a), to_list(b)
    k = len(norm)

    def value(idx):
        lower, upper = idx[:rq], idx[rq:]
        qa_val = dict(zip(free_qa, lower[:len(free_qa)]))
        qb_val = dict(zip(free_qb, lower[len(free_qa):]))
        pa_val = dict(zip(free_pa, upper[:len(free_pa)]))
        pb_val = dict(zip(free_pb, upper[len(free_pa):]))
        total = Fraction(0)
        for s in product(range(a.n), repeat=k):
            qa, pa, qb, pb = dict(qa_val), dict(pa_val), dict(qb_val), dict(pb_val)
            for (g, ia, ib), v in zip(norm, s):
                if g == "p":
                    pa[ia], qb[ib] = v, v
                else:
                    qa[ia], pb[ib] = v, v
            ia_idx = tuple(qa[i] for i in range(a.q)) + tuple(pa[i] for i in range(a.p))
            ib_idx = tuple(qb[i] for i in range(b.q)) + tuple(pb[i] for i in range(b.p))
            total += _get(la, ia_idx) * _get(lb, ib_idx)
        return total

    return _build(rp, rq, a.n, value)


def _dot_pairs(a, b, count):
    """
    Пары для count-кратного скалярного произведения A . B:
    последние count нижних индексов A с первыми count верхними индексами B
    (как в A^i_j B^j_k); если так нельзя - последние верхние A с первыми
    нижними B (как в v^i f_i)
    """
    if a.q >= count and b.p >= count:
        return [("q", a.q - count + s, s) for s in range(count)]
    if a.p >= count and b.q >= count:
        return [("p", a.p - count + s, s) for s in range(count)]
    raise ValueError(
        f"для {count}-кратной свертки нужно {count} нижних индекса у A и "
        f"{count} верхних у B (или наоборот); типы {shape(a)} и {shape(b)}")


def dot(a, b):
    """Однократная свертка A . B, например A^i_j v^j или A^i_j B^j_k"""
    return contract_ab(a, b, _dot_pairs(a, b, 1))


def ddot(a, b):
    """Двойная свертка A : B, например C^{ij}_{kl} e^{kl}"""
    return contract_ab(a, b, _dot_pairs(a, b, 2))


def inverse_matrix(m):
    """Обратная матрица методом Гаусса-Жордана в Fraction (точно)"""
    n = len(m)
    a = [[Fraction(x) for x in row] + [Fraction(int(i == j)) for j in range(n)]
         for i, row in enumerate(m)]
    for col in range(n):
        piv = next((r for r in range(col, n) if a[r][col] != 0), None)
        if piv is None:
            raise ValueError("матрица метрики вырождена")
        a[col], a[piv] = a[piv], a[col]
        lead = a[col][col]
        a[col] = [x / lead for x in a[col]]
        for r in range(n):
            if r != col and a[r][col] != 0:
                f = a[r][col]
                a[r] = [x - f * y for x, y in zip(a[r], a[col])]
    return [row[n:] for row in a]


def basis_matrix(basis, n):
    """
    Матрица базиса: строка i - декартовы координаты вектора e_i.
    None - ортонормированный базис (единичная матрица). Базис должен быть
    линейно независимым (det != 0)
    """
    if basis is None:
        return [[Fraction(int(i == j)) for j in range(n)] for i in range(n)]
    if hasattr(basis, "p"):
        if basis.p + basis.q != 2 or basis.n != n:
            raise ValueError("базис должен задаваться тензором ранга 2 той же размерности")
        basis = to_list(basis)
    b = [[Fraction(x) for x in row] for row in basis]
    if len(b) != n or any(len(row) != n for row in b):
        raise ValueError(f"базис должен задаваться матрицей {n}x{n}")
    if _det_exact(b) == 0:
        raise ValueError("векторы базиса линейно зависимы")
    return b


def gram_matrix(basis, n):
    """
    Фундаментальная матрица базиса: g_ij = (e_i, e_j) = (B B^T)_ij,
    где строка i матрицы B - декартовы координаты вектора e_i
    """
    b = basis_matrix(basis, n)
    return [[sum((b[i][k] * b[j][k] for k in range(n)), Fraction(0))
             for j in range(n)] for i in range(n)]


def is_cartesian(basis, n):
    """Совпадает ли базис с декартовым ортонормированным"""
    return basis is None or basis_matrix(basis, n) == basis_matrix(None, n)


def metric_matrix(metric, n):
    """
    Фундаментальная (метрическая) матрица g_ij = (e_i, e_j).
    None - ортонормированный базис, g_ij = delta_ij. Допускается вложенный
    список или тензор ранга 2. Матрица должна быть симметричной и
    положительно определенной (критерий Сильвестра, точно в Fraction)
    """
    if metric is None:
        return [[Fraction(int(i == j)) for j in range(n)] for i in range(n)]
    if hasattr(metric, "p"):
        if metric.p + metric.q != 2 or metric.n != n:
            raise ValueError("метрика должна быть тензором ранга 2 той же размерности")
        metric = to_list(metric)
    g = [[Fraction(x) for x in row] for row in metric]
    if len(g) != n or any(len(row) != n for row in g):
        raise ValueError(f"метрика должна быть матрицей {n}x{n}")
    if any(g[i][j] != g[j][i] for i in range(n) for j in range(n)):
        raise ValueError("метрика должна быть симметричной")
    for k in range(1, n + 1):
        if _det_exact([row[:k] for row in g[:k]]) <= 0:
            raise ValueError("метрика должна быть положительно определенной "
                             f"(главный минор порядка {k} не положителен)")
    return g


def is_orthonormal(metric, n):
    """Совпадает ли метрика с delta_ij (или не задана)"""
    return metric is None or metric_matrix(metric, n) == metric_matrix(None, n)


def _det_exact(m):
    """Определитель методом Гаусса в Fraction"""
    a = [list(row) for row in m]
    n = len(a)
    det = Fraction(1)
    for c in range(n):
        piv = next((r for r in range(c, n) if a[r][c] != 0), None)
        if piv is None:
            return Fraction(0)
        if piv != c:
            a[c], a[piv] = a[piv], a[c]
            det = -det
        det *= a[c][c]
        for r in range(c + 1, n):
            f = a[r][c] / a[c][c]
            a[r] = [x - f * y for x, y in zip(a[r], a[c])]
    return det


def lower_index(t, axis, metric=None):
    """
    Опускание axis-го верхнего индекса (0-based):

        T_{... k}^{...} = g_{ks} T^{... s ...}

    Работа идет с компонентами в текущем базисе: новый нижний индекс
    становится последним в группе q. Без метрики берется g_ij = delta_ij
    (ортонормированный базис), компоненты при этом не меняются, меняется
    только тип тензора
    """
    if not (0 <= axis < t.p):
        raise ValueError(
            f"low опускает верхний индекс, а у тензора типа ({t.p}, {t.q}) "
            f"верхних индексов {t.p}: номера {axis + 1} нет"
            + ("; поднять нижний индекс можно функцией up" if t.p == 0 else ""))
    g = metric_matrix(metric, t.n)
    src = to_list(t)

    def value(idx):
        lower, upper = list(idx[:t.q + 1]), list(idx[t.q + 1:])
        k = lower.pop()
        total = Fraction(0)
        for s in range(t.n):
            if g[k][s]:
                total += g[k][s] * _get(src, tuple(lower + upper[:axis] + [s] + upper[axis:]))
        return total

    return _build(t.p - 1, t.q + 1, t.n, value)


def raise_index(t, axis, metric=None):
    """
    Поднятие axis-го нижнего индекса (0-based) обратной метрикой:

        T^{... k} = g^{ks} T_{... s ...},  g^{ks} g_{sm} = delta^k_m

    Новый верхний индекс становится последним в группе p
    """
    if not (0 <= axis < t.q):
        raise ValueError(
            f"up поднимает нижний индекс, а у тензора типа ({t.p}, {t.q}) "
            f"нижних индексов {t.q}: номера {axis + 1} нет"
            + ("; опустить верхний индекс можно функцией low" if t.q == 0 else ""))
    ginv = inverse_matrix(metric_matrix(metric, t.n))
    src = to_list(t)

    def value(idx):
        lower, upper = list(idx[:t.q - 1]), list(idx[t.q - 1:])
        k = upper.pop()
        total = Fraction(0)
        for s in range(t.n):
            if ginv[k][s]:
                total += ginv[k][s] * _get(src, tuple(lower[:axis] + [s] + lower[axis:] + upper))
        return total

    return _build(t.p + 1, t.q - 1, t.n, value)
