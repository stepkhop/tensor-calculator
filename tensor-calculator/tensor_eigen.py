"""
Задача на собственные значения для тензора второго ранга

Тензор типа (1, 1) задает линейный оператор v^i -> T^i_j v^j, и задача
T v = lambda v для него поставлена корректно в любом базисе. Для тензоров
типа (2, 0) и (0, 2) задача ставится с фундаментальной матрицей g_ij:
det(T_ij - lambda g_ij) = 0, то есть для смешанного тензора g^ik T_kj
(mixed_matrix). Без матрицы базис считается ортонормированным, g_ij = delta_ij

Что считается точно (Fraction):
    - коэффициенты характеристического многочлена (главные инварианты),
    - разложение многочлена на свободные от квадратов множители,
      а значит и кратности корней,
    - рациональные корни и базисы характеристических пространств для них
Что считается приближенно (float):
    - иррациональные корни и соответствующие им собственные векторы

Характеристическое пространство (ХП) корня lambda - ядро оператора
T - lambda*E. Его размерность равна числу независимых собственных векторов:
1 - прямая (рисуется вектором), 2 - плоскость, 3 - все пространство
(шаровой тензор, рисуется сферой)
"""

import cmath
import math
from fractions import Fraction

from tensor_ext import to_list, from_list, sub, metric_matrix, inverse_matrix

__all__ = [
    "as_matrix", "from_matrix", "mixed_matrix", "char_poly", "invariants",
    "eigen", "EigenSpace", "is_symmetric_matrix", "classify",
    "spherical_part", "deviator", "spectral_check", "report",
]

TOL = 1e-9


def as_matrix(t):
    """
    Матрица компонент тензора ранга 2: M[i][j]

    (1, 1): M[i][j] = T^i_j, то есть строка - верхний индекс
    (2, 0), (0, 2): M[i][j] = T^{ij} или T_{ij}
    """
    if t.p + t.q != 2:
        raise ValueError("нужен тензор ранга 2")
    data = to_list(t)
    n = t.n
    if (t.p, t.q) == (1, 1):
        return [[data[j][i] for j in range(n)] for i in range(n)]
    return [[data[i][j] for j in range(n)] for i in range(n)]


def from_matrix(m, p=1, q=1):
    """Обратное к as_matrix"""
    n = len(m)
    if (p, q) == (1, 1):
        return from_list(1, 1, n, [[m[i][j] for i in range(n)] for j in range(n)])
    return from_list(p, q, n, m)


def mixed_matrix(t, metric=None):
    """
    Матрица смешанного тензора T^i_j, для которой ставится задача на
    собственные значения при фундаментальной матрице g_ij:

        (1, 1): T^i_j как есть;
        (0, 2): T^i_j = g^{ik} T_kj, то есть det(T_ij - lambda g_ij) = 0;
        (2, 0): T^i_j = T^{ik} g_kj.

    В ортонормированном базисе (metric=None) это просто as_matrix
    """
    m = as_matrix(t)
    if metric is None or (t.p, t.q) == (1, 1):
        return m
    g = metric_matrix(metric, t.n)
    if (t.p, t.q) == (0, 2):
        return _matmul(inverse_matrix(g), m)
    return _matmul(m, g)


def _matrix_of(t_or_m, metric):
    if hasattr(t_or_m, "p"):
        return mixed_matrix(t_or_m, metric)
    return [[Fraction(x) for x in row] for row in t_or_m]


def is_symmetric_matrix(m):
    n = len(m)
    return all(m[i][j] == m[j][i] for i in range(n) for j in range(n))


def _matmul(a, b):
    n = len(a)
    return [[sum((a[i][k] * b[k][j] for k in range(n)), Fraction(0))
             for j in range(n)] for i in range(n)]


def char_poly(m):
    """
    Коэффициенты det(lambda*E - M) по убыванию степеней:
    [1, -I1, I2, -I3, ...]

    Метод Фаддеева-Леверье: n умножений матриц, только сложения,
    умножения и деление на целое k, поэтому в Fraction результат точный
    """
    n = len(m)
    m = [[Fraction(x) for x in row] for row in m]
    coeffs = [Fraction(1)]
    mk = [[Fraction(0)] * n for _ in range(n)]
    for k in range(1, n + 1):
        prev = [[mk[i][j] + (coeffs[-1] if i == j else 0) for j in range(n)]
                for i in range(n)]
        mk = _matmul(m, prev)
        ck = -sum((mk[i][i] for i in range(n)), Fraction(0)) / k
        coeffs.append(ck)
    return coeffs


def invariants(m):
    """Главные инварианты I1, ..., In (I1 - след, In - определитель)"""
    c = char_poly(m)
    return [c[k] * (-1) ** k for k in range(1, len(c))]


def _trim(a):
    a = list(a)
    while len(a) > 1 and a[0] == 0:
        a.pop(0)
    return a


def _divmod(a, b):
    a, b = _trim(a), _trim(b)
    if len(a) < len(b):
        return [Fraction(0)], a
    q = [Fraction(0)] * (len(a) - len(b) + 1)
    r = list(a)
    for i in range(len(q)):
        coef = r[i] / b[0]
        q[i] = coef
        for j in range(len(b)):
            r[i + j] -= coef * b[j]
    r = _trim(r[len(q):]) if len(r) > len(q) else [Fraction(0)]
    if not r:
        r = [Fraction(0)]
    return q, r


def _monic(a):
    a = _trim(a)
    return [x / a[0] for x in a]


def _gcd(a, b):
    a, b = _trim(a), _trim(b)
    while not (len(b) == 1 and b[0] == 0):
        _, r = _divmod(a, b)
        a, b = b, r
    return _monic(a)


def _deriv(a):
    deg = len(a) - 1
    if deg == 0:
        return [Fraction(0)]
    return [a[i] * (deg - i) for i in range(deg)]


def _squarefree(f):
    """
    Алгоритм Юна: f = prod g_k^k, где g_k свободны от квадратов и
    попарно взаимно просты. Возвращает [(g_k, k), ...] с deg g_k >= 1
    """
    f = _monic(f)
    result = []
    a = _gcd(f, _deriv(f))
    b, _ = _divmod(f, a)
    c, _ = _divmod(_deriv(f), a)
    d = [x - y for x, y in zip(*_pad(c, _deriv(b)))]
    k = 1
    while len(_trim(b)) > 1:
        g = _gcd(b, d)
        if len(g) > 1:
            result.append((g, k))
        b, _ = _divmod(b, g)
        c, _ = _divmod(d, g)
        d = [x - y for x, y in zip(*_pad(c, _deriv(b)))]
        k += 1
    return result


def _pad(a, b):
    n = max(len(a), len(b))
    return [Fraction(0)] * (n - len(a)) + list(a), [Fraction(0)] * (n - len(b)) + list(b)


def _peval(a, x):
    res = 0
    for c in a:
        res = res * x + c
    return res


def _roots_numeric(a):
    """Все корни многочлена (комплексные числа)"""
    a = [complex(x) for x in _monic(a)]
    deg = len(a) - 1
    if deg == 1:
        return [-a[1]]
    if deg == 2:
        disc = cmath.sqrt(a[1] ** 2 - 4 * a[2])
        return [(-a[1] + disc) / 2, (-a[1] - disc) / 2]
    if deg == 3 and all(abs(x.imag) < 1e-15 for x in a):
        return _cubic_real_coeffs(a[1].real, a[2].real, a[3].real)
    roots = [(0.4 + 0.9j) ** k for k in range(deg)]
    for _ in range(500):
        new = []
        for i, r in enumerate(roots):
            den = 1
            for j, s in enumerate(roots):
                if i != j:
                    den *= (r - s)
            new.append(r - _peval(a, r) / den)
        if max(abs(x - y) for x, y in zip(new, roots)) < 1e-14:
            roots = new
            break
        roots = new
    return roots


def _cubic_real_coeffs(a, b, c):
    """
    Корни x^3 + a x^2 + b x + c. Подстановка x = y - a/3 дает
    y^3 + p y + q = 0. При трех вещественных корнях - тригонометрическая
    формула Виета, иначе - формула Кардано
    """
    p = b - a * a / 3
    q = 2 * a ** 3 / 27 - a * b / 3 + c
    shift = -a / 3
    disc = -(4 * p ** 3 + 27 * q ** 2)
    if disc >= 0 and p < 0:
        r = 2 * math.sqrt(-p / 3)
        arg = 3 * q / (p * r)
        arg = max(-1.0, min(1.0, arg))
        phi = math.acos(arg) / 3
        return [complex(r * math.cos(phi - 2 * math.pi * k / 3) + shift)
                for k in range(3)]
    sq = cmath.sqrt((q / 2) ** 2 + (p / 3) ** 3)
    u = _cbrt(-q / 2 + sq)
    v = -p / (3 * u) if abs(u) > 1e-300 else _cbrt(-q / 2 - sq)
    w = complex(-0.5, math.sqrt(3) / 2)
    return [u + v + shift, u * w + v * w.conjugate() + shift,
            u * w.conjugate() + v * w + shift]


def _cbrt(z):
    z = complex(z)
    if abs(z.imag) < 1e-15:
        x = z.real
        return complex(math.copysign(abs(x) ** (1 / 3), x))
    return z ** (1 / 3)


def _exact_or_float(g, z):
    """
    Попытаться угадать рациональный корень по приближенному и проверить
    его подстановкой в многочлен точно. Иначе - вещественное или
    комплексное приближение
    """
    if abs(z.imag) > 1e-9 * max(1.0, abs(z)):
        return z, False
    x = z.real
    for den in (10 ** 6, 10 ** 9):
        guess = Fraction(x).limit_denominator(den)
        if _peval(g, guess) == 0:
            return guess, True
    return x, False


def _sqrt_parts(x):
    """sqrt(x) = c * sqrt(m) для рационального x >= 0: c рационально, m целое без квадратов"""
    num = x.numerator * x.denominator
    c, m, f = 1, 1, 2
    while f * f <= num:
        while num % (f * f) == 0:
            num //= f * f
            c *= f
        f += 1
    m = num
    return Fraction(c, x.denominator), m


def _fmt_coef_sqrt(c, m, imag):
    """Запись c*sqrt(m) (с множителем i, если imag)"""
    unit = ("i" if imag else "") + (f"√{m}" if m != 1 else "")
    if not unit:
        return fmt_num(c)
    if c == 1:
        return unit
    if c.denominator == 1:
        return f"{c.numerator}{unit}"
    return (f"{c.numerator}{unit}/{c.denominator}" if c.numerator != 1
            else f"{unit}/{c.denominator}")


def _quadratic_text(a):
    """
    Корни квадратного множителя x^2 + b x + c с рациональными
    коэффициентами в радикалах: -b/2 ± sqrt(b^2/4 - c).
    Возвращает [(приближенное значение, запись), ...]
    """
    _, b, c = _monic(a)
    re = -b / 2
    d = b * b / 4 - c
    cf, m = _sqrt_parts(abs(d))
    part = _fmt_coef_sqrt(cf, m, d < 0)
    head = "" if re == 0 else fmt_num(re) + " "
    out = []
    for sign in (1, -1):
        mark = ("+ " if head else "") if sign > 0 else ("- " if head else "-")
        text = f"{head}{mark}{part}"
        size = float(cf) * math.sqrt(m) * sign
        z = complex(float(re), size) if d < 0 else complex(float(re) + size, 0)
        out.append((z, text))
    return out


def _null_space(m, exact):
    """
    Базис ядра матрицы методом Гаусса. В Fraction - точно, во float -
    с частичным выбором ведущего элемента и порогом TOL
    """
    n = len(m)
    a = [list(row) for row in m]
    scale = max([abs(x) for row in a for x in row] + [1])
    pivots = []
    row = 0
    for col in range(n):
        if exact:
            piv = next((r for r in range(row, n) if a[r][col] != 0), None)
        else:
            piv = max(range(row, n), key=lambda r: abs(a[r][col]), default=None)
            if piv is not None and abs(a[piv][col]) < TOL * scale:
                piv = None
        if piv is None:
            continue
        a[row], a[piv] = a[piv], a[row]
        lead = a[row][col]
        a[row] = [x / lead for x in a[row]]
        for r in range(n):
            if r != row and a[r][col] != 0:
                f = a[r][col]
                a[r] = [x - f * y for x, y in zip(a[r], a[row])]
        pivots.append(col)
        row += 1
        if row == n:
            break
    free = [c for c in range(n) if c not in pivots]
    basis = []
    for fcol in free:
        v = [Fraction(0) if exact else 0.0 for _ in range(n)]
        v[fcol] = Fraction(1) if exact else 1.0
        for r, pcol in enumerate(pivots):
            v[pcol] = -a[r][fcol]
        basis.append(v)
    return basis


class EigenSpace:
    """
    Один корень характеристического уравнения и его характеристическое
    пространство

    value     - Fraction (точный корень), float или complex
    exact     - найден ли корень точно
    mult      - алгебраическая кратность (точно, по разложению Юна)
    basis     - базис ХП (пустой для комплексного корня)
    """

    def __init__(self, value, exact, mult, basis, text=None):
        self.value = value
        self.exact = exact
        self.mult = mult
        self.basis = basis
        self.text = text

    @property
    def label(self):
        """Запись корня: точная в радикалах, если она найдена, иначе число"""
        return self.text or fmt_num(self.value)

    @property
    def real(self):
        return not isinstance(self.value, complex)

    @property
    def dim(self):
        return len(self.basis)

    @property
    def kind(self):
        n = len(self.basis[0]) if self.basis else 0
        if not self.real:
            return "комплексный корень"
        if self.dim == 1:
            return "прямая (вектор)"
        if self.dim == n:
            return "все пространство (шар)"
        if self.dim == 2:
            return "плоскость"
        return f"подпространство размерности {self.dim}"

    def __repr__(self):
        return f"EigenSpace({fmt_num(self.value)}, кратность {self.mult}, dim {self.dim})"


def eigen(t_or_m, metric=None):
    """
    Собственные значения и характеристические пространства тензора
    ранга 2 (или квадратной матрицы). metric - фундаментальная матрица
    g_ij (None - ортонормированный базис). Базисы ХП - контравариантные
    компоненты v^i. Корни упорядочены по убыванию вещественной части
    """
    m = _matrix_of(t_or_m, metric)
    n = len(m)
    poly = char_poly(m)
    spaces = []
    for g, k in _squarefree(poly):
        found = [(z,) + _exact_or_float(g, z) for z in _roots_numeric(g)]
        rest = g
        for _, val, exact in found:
            if exact:
                rest, _ = _divmod(rest, [Fraction(1), -val])
        radicals = _quadratic_text(rest) if len(_trim(rest)) == 3 else []
        for z, val, exact in found:
            text = None
            if not exact and radicals:
                text = min(radicals, key=lambda r: abs(r[0] - z))[1]
            if isinstance(val, complex):
                spaces.append(EigenSpace(val, False, k, [], text))
                continue
            if exact:
                shifted = [[m[i][j] - (val if i == j else 0) for j in range(n)]
                           for i in range(n)]
            else:
                shifted = [[float(m[i][j]) - (val if i == j else 0.0)
                            for j in range(n)] for i in range(n)]
            basis = _null_space(shifted, exact)
            spaces.append(EigenSpace(val, exact, k, basis, text))
    spaces.sort(key=lambda s: (-complex(s.value).real, -complex(s.value).imag))
    return spaces


def classify(t_or_m, metric=None):
    """
    Словесная характеристика спектра. Симметрия понимается относительно
    метрики: тензор симметричен, если g_ik T^k_j симметрична
    """
    m = _matrix_of(t_or_m, metric)
    n = len(m)
    sp = eigen(m)
    gm = m if metric is None else _matmul(metric_matrix(metric, n), m)
    if any(not s.real for s in sp):
        return "есть комплексно-сопряженные корни"
    if len(sp) == 1 and sp[0].dim == n:
        return "шаровой тензор"
    if n == 3 and len(sp) == 2 and is_symmetric_matrix(gm):
        return "осесимметричный тензор"
    if all(s.mult == 1 for s in sp):
        return "все корни простые"
    if any(s.dim < s.mult for s in sp):
        return "кратный корень с неполным ХП (тензор недиагонализуем)"
    return "есть кратные корни"


def spherical_part(t, metric=None):
    """
    Шаровая часть (I1 / n) E того же типа, что и t; I1 - след смешанного
    тензора. Единичный тензор E имеет компоненты delta^i_j, g_ij или g^ij
    """
    m = mixed_matrix(t, metric)
    n = t.n
    i1 = sum((m[i][i] for i in range(n)), Fraction(0))
    if (t.p, t.q) == (1, 1):
        e = metric_matrix(None, n)
    elif (t.p, t.q) == (0, 2):
        e = metric_matrix(metric, n)
    else:
        e = inverse_matrix(metric_matrix(metric, n))
    return from_matrix([[i1 / n * x for x in row] for row in e], t.p, t.q)


def deviator(t, metric=None):
    """Девиатор: T минус шаровая часть, след смешанного девиатора равен нулю"""
    return sub(t, spherical_part(t, metric))


def spectral_check(t_or_m):
    """
    Для симметричного тензора: max |M - sum lambda_k P_k|, где P_k -
    ортогональные проекторы на ХП. Ноль означает, что найденные корни
    и ХП полностью восстанавливают тензор
    """
    m = as_matrix(t_or_m) if hasattr(t_or_m, "p") else t_or_m
    n = len(m)
    acc = [[0.0] * n for _ in range(n)]
    for s in eigen(m):
        if not s.real:
            raise ValueError("комплексный спектр")
        ortho = _gram_schmidt([[float(x) for x in v] for v in s.basis])
        for e in ortho:
            for i in range(n):
                for j in range(n):
                    acc[i][j] += float(s.value) * e[i] * e[j]
    return max(abs(float(m[i][j]) - acc[i][j]) for i in range(n) for j in range(n))


def _gram_schmidt(vectors):
    out = []
    for v in vectors:
        w = list(v)
        for e in out:
            d = sum(x * y for x, y in zip(w, e))
            w = [x - d * y for x, y in zip(w, e)]
        norm = math.sqrt(sum(x * x for x in w))
        if norm > TOL:
            out.append([x / norm for x in w])
    return out


def orthonormal_basis(space):
    """Ортонормированный базис ХП (float) - для рисования"""
    return _gram_schmidt([[float(x) for x in v] for v in space.basis])


def fmt_num(x):
    if isinstance(x, Fraction):
        return str(x.numerator) if x.denominator == 1 else f"{x.numerator}/{x.denominator}"
    if isinstance(x, complex):
        return f"{x.real + 0.0:.6g} {'+' if x.imag >= 0 else '-'} {abs(x.imag):.6g}i"
    return f"{x:.6g}"


def report(t, metric=None):
    """Многострочный отчет: инварианты, корни, кратности, базисы ХП"""
    m = mixed_matrix(t, metric)
    inv = invariants(m)
    lines = ["Инварианты: " + ", ".join(f"I{k + 1} = {fmt_num(v)}"
                                        for k, v in enumerate(inv))]
    if (t.p, t.q) != (1, 1):
        if metric is None:
            lines.append(f"Базис ортонормированный, для типа ({t.p}, {t.q}) "
                         "берется матрица компонент")
        else:
            lines.append(f"Тип ({t.p}, {t.q}): задача решается для смешанного тензора с метрикой G")
    lines.append("Характер спектра: " + classify(t, metric))
    if metric is not None:
        lines.append("Векторы ХП - компоненты v^i в базисе с метрикой G")
    for s in eigen(m):
        mark = "точно" if s.exact else "приближенно"
        value = fmt_num(s.value)
        if s.text:
            mark = "точно, в радикалах"
            value = f"{s.text} ≈ {value.replace('-0 ', '0 ')}"
        head = f"lambda = {value} ({mark}), кратность {s.mult}"
        if not s.real:
            lines.append(head + " - ХП в вещественном пространстве нет")
            continue
        extra = f", dim ХП = {s.dim} < {s.mult}" if s.dim < s.mult else ""
        lines.append(head + f", ХП: {s.kind}{extra}")
        for v in s.basis:
            lines.append("    (" + ", ".join(fmt_num(x) for x in v) + ")")
    return "\n".join(lines)
