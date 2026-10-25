
import math
from fractions import Fraction

import matplotlib
matplotlib.use("Agg")

import pytest
from matplotlib.backends.backend_agg import FigureCanvasAgg

import tensor_ext as tx
import tensor_eigen as te
from expr import evaluate, ExprError
from tensor_viz import make_figure, describe_mode, start_rotation
from conftest import F, frac_list


def mat11(m):
    """Тензор типа (1, 1) по матрице M[i][j] = T^i_j"""
    return te.from_matrix([[Fraction(x) for x in row] for row in m])


def matvec(m, v):
    return [sum(float(m[i][j]) * float(v[j]) for j in range(len(v)))
            for i in range(len(m))]


def test_contract_ab_matches_contract_of_product():
    a = tx.from_list(2, 1, 2, [[[1, 2], [3, 4]], [[5, 6], [7, 8]]])
    b = tx.from_list(1, 1, 2, [[1, -1], [2, 3]])
    prod = tx.tensor_product(a, b)
    assert tx.equal(tx.contract_ab(a, b, [("p", 1, 0)]),
                    tx.contract(prod, a.q + 0, 1))
    assert tx.equal(tx.contract_ab(a, b, [("q", 0, 0)]),
                    tx.contract(prod, 0, a.p + 0))


def test_dot_is_matrix_product():
    a = mat11([[1, 2], [3, 4]])
    b = mat11([[5, 6], [7, 8]])
    assert te.as_matrix(tx.dot(a, b)) == frac_list([[19, 22], [43, 50]])


def test_dot_matrix_vector_equals_contract_with_matrix():
    m = [[1, 2, 0], [0, 1, 3], [4, 0, 1]]
    a = mat11(m)
    v = tx.from_list(1, 0, 3, [1, 2, 3])
    expected = tx.contract_with_matrix(v, "p", 0, m)
    assert tx.equal(tx.dot(a, v), expected)
    assert tx.to_list(tx.dot(a, v)) == frac_list([5, 11, 7])


def test_ddot_equals_full_contraction():
    a = tx.from_list(0, 2, 2, [[1, 2], [3, 4]])
    b = tx.from_list(2, 0, 2, [[5, 6], [7, 8]])
    assert tx.to_list(tx.ddot(a, b)) == F(70)
    assert tx.to_list(tx.ddot(a, b)) == tx.full_contraction(
        tx.from_list(2, 0, 2, tx.to_list(a)), b)


def test_ddot_isotropic_elasticity():
    """sigma = C : eps для изотропного тела равно lambda*tr(eps)*E + 2*mu*eps"""
    n, lam, mu = 3, F(2), F(3)

    def c(idx):
        k, l_, i, j = idx
        d = lambda x, y: 1 if x == y else 0  # noqa: E731
        return lam * d(i, j) * d(k, l_) + mu * (d(i, k) * d(j, l_) + d(i, l_) * d(j, k))

    C = tx._build(2, 2, n, c)
    eps_m = [[1, 2, 0], [2, -1, 3], [0, 3, 4]]
    eps = tx.from_list(2, 0, n, eps_m)
    sigma = tx.ddot(C, eps)
    tr = sum(eps_m[i][i] for i in range(n))
    expected = [[lam * tr * (i == j) + 2 * mu * eps_m[i][j] for j in range(n)]
                for i in range(n)]
    assert (sigma.p, sigma.q) == (2, 0)
    assert tx.to_list(sigma) == frac_list(expected)


def test_contract_ab_validates_pairs():
    a = tx.from_list(1, 0, 2, [1, 2])
    b = tx.from_list(1, 0, 2, [3, 4])
    with pytest.raises(ValueError):
        tx.dot(a, b)
    c = tx.from_list(0, 1, 2, [1, 1])
    with pytest.raises(ValueError):
        tx.contract_ab(a, c, [("p", 0, 0), ("p", 0, 0)])
    with pytest.raises(ValueError):
        tx.contract_ab(a, c, [("p", 1, 0)])


def test_invariants_of_3x3():
    m = [[1, 2, 3], [0, 4, 5], [1, 0, 6]]
    i1, i2, i3 = te.invariants(m)
    assert i1 == 11
    assert i2 == 31
    assert i3 == tx.det(tx.from_list(2, 0, 3, m))


def test_last_invariant_is_determinant_for_n4():
    m = [[2, 1, 0, 3], [1, 0, 4, 1], [0, 5, 1, 2], [3, 1, 2, 0]]
    inv = te.invariants(m)
    assert inv[0] == 3
    assert inv[-1] == tx.det(tx.from_list(2, 0, 4, m))


def test_axisymmetric_tensor_has_plane():
    sp = te.eigen(mat11([[2, 1, 0], [1, 2, 0], [0, 0, 3]]))
    assert [(s.value, s.mult, s.dim) for s in sp] == [(3, 2, 2), (1, 1, 1)]
    assert all(s.exact for s in sp)
    assert sp[0].kind == "плоскость"
    assert te.classify(mat11([[2, 1, 0], [1, 2, 0], [0, 0, 3]])) == "осесимметричный тензор"


def test_spherical_tensor_whole_space():
    sp = te.eigen(mat11([[F(7, 2), 0, 0], [0, F(7, 2), 0], [0, 0, F(7, 2)]]))
    assert len(sp) == 1
    assert (sp[0].value, sp[0].mult, sp[0].dim) == (F(7, 2), 3, 3)
    assert sp[0].kind == "все пространство (шар)"


def test_distinct_roots_satisfy_definition():
    m = [[1, 2, 3], [2, 4, 5], [3, 5, 6]]
    sp = te.eigen(mat11(m))
    i1, _, i3 = te.invariants(m)
    assert len(sp) == 3 and all(s.mult == 1 and s.dim == 1 for s in sp)
    assert math.isclose(sum(s.value for s in sp), i1, rel_tol=1e-12)
    assert math.isclose(math.prod(s.value for s in sp), i3, rel_tol=1e-9)
    for s in sp:
        v = s.basis[0]
        tv = matvec(m, v)
        assert all(math.isclose(tv[i], s.value * v[i], abs_tol=1e-9) for i in range(3))


def test_rational_roots_are_exact():
    m = [[3, 1, 1], [1, 3, 1], [1, 1, 3]]
    sp = te.eigen(mat11(m))
    assert [(s.value, s.mult, s.exact) for s in sp] == [(5, 1, True), (2, 2, True)]
    for s in sp:
        for v in s.basis:
            tv = [sum(Fraction(m[i][j]) * v[j] for j in range(3)) for i in range(3)]
            assert tv == [s.value * x for x in v]


def test_jordan_block_has_incomplete_space():
    sp = te.eigen(mat11([[2, 1, 0], [0, 2, 0], [0, 0, 1]]))
    two = sp[0]
    assert (two.value, two.mult, two.dim) == (2, 2, 1)
    assert "недиагонализуем" in te.classify(mat11([[2, 1, 0], [0, 2, 0], [0, 0, 1]]))


def test_teacher_nilpotent_tensor():
    """
    Тест преподавателя: ((0,1,2),(0,0,3),(0,0,0)). Тензор нильпотентный
    (T^3 = 0), все инварианты нулевые, корень 0 трехкратный, а ХП -
    только прямая вдоль первого базисного вектора
    """
    t = tx.from_conventional(1, 1, 3, [[0, 1, 2], [0, 0, 3], [0, 0, 0]])
    assert te.invariants(te.as_matrix(t)) == [0, 0, 0]
    (s,) = te.eigen(t)
    assert s.value == 0 and s.exact and s.mult == 3
    assert s.basis == [[1, 0, 0]]
    assert "неполным" in te.classify(t)


def test_teacher_spherical_plus_antisymmetric():
    """
    Тест преподавателя: ((2,1,2),(-1,2,3),(-2,-3,2)) = 2E + W, W = -W^T.
    Вещественный корень 2 с ХП вдоль оси вращения (3, -2, 1), то есть
    вектора w антисимметричной части; пара корней 2 ± i*sqrt(14),
    |w|^2 = 1 + 4 + 9 = 14
    """
    m = [[2, 1, 2], [-1, 2, 3], [-2, -3, 2]]
    t = tx.from_conventional(1, 1, 3, m)
    sp = te.eigen(t)
    real = [s for s in sp if s.real]
    cplx = [s for s in sp if not s.real]
    assert len(real) == 1 and real[0].value == 2 and real[0].exact
    assert real[0].basis == [[3, -2, 1]]
    assert sorted(s.text for s in cplx) == ["2 + i√14", "2 - i√14"]
    for s in cplx:
        assert abs(s.value.real - 2) < 1e-12
        assert abs(abs(s.value.imag) - math.sqrt(14)) < 1e-12
    sph = te.as_matrix(te.spherical_part(t))
    assert sph == [[2, 0, 0], [0, 2, 0], [0, 0, 2]]


def test_input_order_is_conventional():
    """Первая строка поля ввода - первая строка матрицы T^i_j"""
    m = [[0, 1, 2], [0, 0, 3], [0, 0, 0]]
    t = tx.from_conventional(1, 1, 3, m)
    assert te.as_matrix(t) == m
    assert tx.to_conventional(t) == m
    c = tx.from_conventional(2, 1, 2, [[[1, 2], [3, 4]], [[5, 6], [7, 8]]])
    assert tx.to_list(c)[0][0][1] == 3
    assert tx.to_conventional(c) == [[[1, 2], [3, 4]], [[5, 6], [7, 8]]]


def test_metric_lower_and_raise_index():
    """
    В ОНБ (g = E) опускание индекса меняет только тип, компоненты те же.
    С метрикой G: up(low(T)) = T, а задача на собственные значения для
    T_ij в косоугольном базисе решается для смешанного G^-1 T
    """
    a = tx.from_conventional(2, 0, 3, [[1, 2, 3], [4, 5, 6], [7, 8, 9]])
    low = tx.lower_index(a, 1)
    assert (low.p, low.q) == (1, 1)
    assert te.as_matrix(low) == te.as_matrix(a)
    g = [[2, 1, 0], [1, 1, 0], [0, 0, 1]]
    b = tx.from_conventional(0, 2, 3, [[1, 0, 0], [0, 2, 0], [0, 0, 3]])
    mixed = tx.raise_index(b, 0, g)
    assert te.as_matrix(mixed) == [[1, -2, 0], [-1, 4, 0], [0, 0, 3]]
    assert tx.equal(tx.lower_index(mixed, 0, g), b)
    gt = tx.from_conventional(0, 2, 3, g)
    r = evaluate("up(A, 1, B)", {"A": b, "B": gt})
    assert tx.equal(r, mixed)
    with pytest.raises(ValueError):
        tx.raise_index(b, 0, [[1, 1, 0], [1, 1, 0], [0, 0, 1]])


SKEW_A = [[1, 1, 0], [0, 1, 0], [0, 0, 2]]
AXIS_T = [[3, -1, 1], [-1, 3, -1], [1, -1, 3]]


def _mm(a, b):
    return [[sum(Fraction(a[i][k]) * b[k][j] for k in range(len(b)))
             for j in range(len(b[0]))] for i in range(len(a))]


def _tr(a):
    return [list(r) for r in zip(*a)]


def test_metric_eigen_covariant_tensor_in_skew_basis():
    """
    T_ij из ОНБ переводится в косоугольный базис: T' = A^T T A, g' = A^T A.
    С фундаментальной матрицей корни те же (5 и двукратный 2), без нее - другие
    """
    tp = _mm(_mm(_tr(SKEW_A), AXIS_T), SKEW_A)
    g = _mm(_tr(SKEW_A), SKEW_A)
    t = tx.from_conventional(0, 2, 3, tp)
    sp = te.eigen(t, g)
    assert [(s.value, s.mult, s.dim) for s in sp] == [(5, 1, 1), (2, 2, 2)]
    assert all(s.exact for s in sp)
    x = [sum(SKEW_A[i][j] * sp[0].basis[0][j] for j in range(3)) for i in range(3)]
    assert x[0] == -x[1] == x[2] != 0
    assert te.classify(t, g) == "осесимметричный тензор"
    wrong = te.eigen(t)
    assert sorted(float(s.value) for s in wrong) != pytest.approx([2, 2, 5])


def test_metric_eigen_contravariant_tensor_in_skew_basis():
    """T^ij: T' = B T B^T, B = A^-1; задача для T^ik g_kj дает те же корни"""
    b = tx.inverse_matrix(SKEW_A)
    tp = _mm(_mm(b, AXIS_T), _tr(b))
    g = _mm(_tr(SKEW_A), SKEW_A)
    sp = te.eigen(tx.from_conventional(2, 0, 3, tp), g)
    assert [(s.value, s.mult) for s in sp] == [(5, 1), (2, 2)]


def test_metric_spherical_part_and_deviator():
    g = _mm(_tr(SKEW_A), SKEW_A)
    t = tx.from_conventional(0, 2, 3, _mm(_mm(_tr(SKEW_A), AXIS_T), SKEW_A))
    sph = te.spherical_part(t, g)
    dev = te.deviator(t, g)
    assert (sph.p, sph.q) == (0, 2)
    assert te.as_matrix(sph) == [[3 * x for x in row] for row in g]
    assert tx.equal(tx.add(sph, dev), t)
    m = te.mixed_matrix(dev, g)
    assert sum(m[i][i] for i in range(3)) == 0


def test_metric_must_be_positive_definite():
    with pytest.raises(ValueError):
        tx.metric_matrix([[1, 2], [2, 1]], 2)
    with pytest.raises(ValueError):
        tx.metric_matrix([[1, 2], [3, 4]], 2)
    assert tx.is_orthonormal(None, 3)
    assert tx.is_orthonormal([[1, 0], [0, 1]], 2)
    assert not tx.is_orthonormal([[1, 1], [1, 2]], 2)


def test_expr_uses_current_metric():
    g = [[2, 1, 0], [1, 1, 0], [0, 0, 1]]
    a = tx.from_conventional(1, 0, 3, [1, 2, 3])
    r = evaluate("low(A, 1)", {"A": a}, metric=g)
    assert tx.to_conventional(r) == [4, 3, 3]
    assert tx.to_conventional(evaluate("low(A, 1)", {"A": a})) == [1, 2, 3]


def test_viz_uses_given_basis():
    """
    Рисунок строится по явно заданному базису: строки матрицы - векторы
    e_i. Фундаментальная матрица получается из него как g_ij = (e_i, e_j)
    """
    from tensor_viz import _frame, _to_cartesian
    basis = _tr(SKEW_A)
    g = tx.gram_matrix(basis, 3)
    assert g == _mm(_tr(SKEW_A), SKEW_A)
    frame = _frame(basis, 3)
    assert frame == [[float(x) for x in row] for row in basis]
    assert _frame(None, 3) is None and _frame([[1, 0], [0, 1]], 2) is None
    assert _to_cartesian([4, -2, 1], frame) == [2.0, -2.0, 2.0]
    t = tx.from_conventional(0, 2, 3, _mm(_mm(_tr(SKEW_A), AXIS_T), SKEW_A))
    v = tx.from_conventional(0, 1, 3, [1, 0, 0])
    fig = make_figure([("T", t), ("f", v)], basis=basis)
    FigureCanvasAgg(fig).draw()
    assert any(ax.name == "3d" for ax in fig.get_axes())


def test_basis_must_be_independent():
    with pytest.raises(ValueError):
        tx.basis_matrix([[1, 1], [2, 2]], 2)
    assert tx.is_cartesian(None, 3)
    assert not tx.is_cartesian([[1, 1], [0, 1]], 2)


def test_rotation_has_complex_roots():
    sp = te.eigen(mat11([[0, -1, 0], [1, 0, 0], [0, 0, 1]]))
    real = [s for s in sp if s.real]
    cplx = [s for s in sp if not s.real]
    assert len(real) == 1 and real[0].value == 1 and real[0].basis == [frac_list([0, 0, 1])]
    assert len(cplx) == 2
    assert all(math.isclose(abs(s.value), 1.0) for s in cplx)


def test_spectral_decomposition_restores_symmetric_tensor():
    for m in ([[1, 2, 3], [2, 4, 5], [3, 5, 6]], [[2, 1, 0], [1, 2, 0], [0, 0, 3]],
              [[4, 0, 0], [0, 4, 0], [0, 0, 4]]):
        assert te.spectral_check(m) < 1e-12


def test_eigenvalues_do_not_depend_on_basis():
    """Корни тензора (1, 1) - инварианты: T' = B T A при B = A^-1"""
    t = mat11([[2, 1, 0], [1, 2, 0], [0, 0, 3]])
    a = [[1, 1, 0], [0, 1, 2], [1, 0, 1]]
    b = [[F(1, 3), F(-1, 3), F(2, 3)], [F(2, 3), F(1, 3), F(-2, 3)],
         [F(-1, 3), F(1, 3), F(1, 3)]]
    a_t = [[a[j][i] for j in range(3)] for i in range(3)]
    t2 = tx.contract_with_matrix(t, "p", 0, b)
    t2 = tx.contract_with_matrix(t2, "q", 0, a_t)
    assert te.as_matrix(t2) != te.as_matrix(t)
    assert te.invariants(te.as_matrix(t2)) == te.invariants(te.as_matrix(t))
    assert [(s.value, s.mult) for s in te.eigen(t2)] == [(3, 2), (1, 1)]


def test_deviator_and_spherical_part():
    t = mat11([[1, 2, 3], [4, 5, 6], [7, 8, 10]])
    dev = te.deviator(t)
    assert tx.trace(dev) == 0
    assert tx.equal(tx.add(dev, te.spherical_part(t)), t)


@pytest.fixture
def env():
    return {"A": mat11([[1, 2], [3, 4]]), "B": tx.from_list(1, 0, 2, [1, 1])}


def test_expr_operations(env):
    a, b = env["A"], env["B"]
    assert tx.equal(evaluate("A * B", env), tx.tensor_product(a, b))
    assert tx.equal(evaluate("dot(A, B)", env), tx.dot(a, b))
    assert tx.equal(evaluate("A + 2*A - A/2", env), tx.scalar_mul(a, F(5, 2)))
    assert evaluate("tr(A) + det(A)", env) == F(5) + F(-2)


def test_expr_functions(env):
    assert tx.equal(evaluate("dev(A) + sph(A)", env), env["A"])
    assert tx.is_symmetric(evaluate("sym(A*A)", env))
    assert tx.equal(evaluate("contract(A*B, 1, 2)", env), tx.dot(env["A"], env["B"]))
    assert tx.equal(evaluate("delta(2)", env), tx.delta(2))
    f = tx.from_list(0, 1, 2, [1, 2])
    env2 = dict(env, C=tx.tensor_product(env["A"], f))
    assert tx.equal(evaluate("contract2(C, B, [('q', 1, 1)])", env2),
                    tx.contract_ab(env2["C"], env["B"], [("q", 0, 0)]))
    g = tx.from_list(0, 1, 2, [3, 5])
    w = evaluate("wedge(F, G)", {"F": f, "G": g})
    assert tx.to_list(w) == frac_list([[0, -1], [1, 0]])
    assert tx.equal(evaluate("F ^ G", {"F": f, "G": g}), w)


def test_expr_rejects_unsafe_code(env):
    for bad in ("__import__('os')", "A.p", "(lambda: 1)()", "open('x')", "A[0]"):
        with pytest.raises(ExprError):
            evaluate(bad, env)


def test_expr_reports_errors(env):
    with pytest.raises(ExprError, match="пустое"):
        evaluate("   ", env)
    with pytest.raises(ExprError, match="синтаксическая"):
        evaluate("A +", env)
    with pytest.raises(ExprError, match="несовместимые"):
        evaluate("A + B", env)
    with pytest.raises(ExprError):
        evaluate("A / 0", env)
    with pytest.raises(ExprError, match="предел"):
        evaluate("eps(9)", env)
    with pytest.raises(ExprError, match="предел"):
        evaluate("delta(3)*delta(3)*delta(3)*delta(3)*delta(3)", env)


def test_describe_mode():
    assert describe_mode(tx.from_list(0, 0, 3, 5)) == "скаляр"
    assert describe_mode(tx.from_list(1, 0, 3, [1, 2, 3])) == "вектор"
    assert describe_mode(mat11([[1, 0, 0], [0, 1, 0], [0, 0, 1]])) == \
        "характеристические пространства"
    assert describe_mode(tx.zeros(3, 0, 3)) == "срезы"
    assert describe_mode(tx.zeros(2, 0, 4)) == "срезы"


def test_make_figure_draws_every_kind():
    items = [
        ("осесимметричный", mat11([[2, 1, 0], [1, 2, 0], [0, 0, 3]])),
        ("шаровой", mat11([[2, 0, 0], [0, 2, 0], [0, 0, 2]])),
        ("поворот", mat11([[0, -1, 0], [1, 0, 0], [0, 0, 1]])),
        ("2D", mat11([[1, 1], [1, 1]])),
        ("вектор", tx.from_list(1, 0, 3, [1, 2, 3])),
        ("ранг 3", tx.zeros(3, 0, 3)),
        ("скаляр", tx.from_list(0, 0, 3, 7)),
    ]
    fig = make_figure(items, size=(20, 4))
    fig.canvas.draw()
    assert sum(1 for ax in fig.get_axes() if ax.name == "3d") == 4


def test_rotation_turns_3d_axes():
    fig = make_figure([("T", mat11([[2, 1, 0], [1, 2, 0], [0, 0, 3]]))])
    ax = [a for a in fig.get_axes() if a.name == "3d"][0]
    before = ax.azim
    FigureCanvasAgg(fig)
    anim = start_rotation(fig, step=10)
    fig.canvas.draw()
    before = ax.azim
    anim._func(0)
    anim.event_source.stop()
    assert math.isclose((ax.azim - before) % 360, 10)


def test_show_basis_toggle():
    """
    Флажок «Показывать базис» убирает с рисунка только векторы e_i:
    положение ХП и подписи корней не меняются
    """
    from tensor_viz import _frame
    basis = _tr(SKEW_A)
    t = tx.from_conventional(0, 2, 3, _mm(_mm(_tr(SKEW_A), AXIS_T), SKEW_A))
    texts = {}
    for show in (True, False):
        fig = make_figure([("T", t)], basis=basis, show_basis=show)
        FigureCanvasAgg(fig).draw()
        ax = next(a for a in fig.get_axes() if a.name == "3d")
        texts[show] = sorted(x.get_text() for x in ax.texts)
    assert "e1" in texts[True] and "e1" not in texts[False]
    assert [x for x in texts[True] if not x.startswith("e")] == texts[False]
    assert _frame(basis, 3) == [[float(x) for x in row] for row in basis]


def test_low_up_reject_missing_index():
    """
    low опускает верхний индекс, поэтому для типа (0, 2) она не определена:
    программа сообщает об ошибке, а не делает что-то молча. То же для up
    и типа (2, 0)
    """
    cov = tx.from_conventional(0, 2, 3, [[1, 0, 0], [0, 2, 0], [0, 0, 3]])
    con = tx.from_conventional(2, 0, 3, [[1, 0, 0], [0, 2, 0], [0, 0, 3]])
    for expr_text, t in (("low(A, 1)", cov), ("low(A, 2)", cov), ("up(A, 1)", con)):
        with pytest.raises((ExprError, ValueError)) as err:
            evaluate(expr_text, {"A": t})
        assert "индекс" in str(err.value)
    assert (tx.raise_index(cov, 0).p, tx.raise_index(cov, 0).q) == (1, 1)
    assert (tx.lower_index(con, 0).p, tx.lower_index(con, 0).q) == (1, 1)
