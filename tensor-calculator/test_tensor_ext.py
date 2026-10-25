
from fractions import Fraction

import pytest

import tensor_ext as tx
from conftest import build, dump, F, frac_list


def test_from_list_to_list_roundtrip():
    data = [[[1, 2], [3, 4]], [[5, 6], [7, 8]]]
    t = tx.from_list(2, 1, 2, data)
    assert tx.shape(t) == (2, 1, 2)
    assert tx.to_list(t) == frac_list(data)


def test_from_list_validates_shape():
    with pytest.raises(ValueError):
        tx.from_list(2, 0, 2, [[1, 2, 3], [4, 5, 6]])
    with pytest.raises(ValueError):
        tx.from_list(2, 0, 2, [1, 2])


def test_zeros_and_equal():
    z = tx.zeros(2, 0, 2)
    assert tx.to_list(z) == frac_list([[0, 0], [0, 0]])
    assert tx.equal(z, tx.from_list(2, 0, 2, [[0, 0], [0, 0]]))
    assert not tx.equal(z, tx.zeros(0, 2, 2))


def test_pretty_prints_components():
    t = tx.from_list(1, 0, 2, [F(1, 2), 3])
    s = tx.pretty(t)
    assert "p=1" in s and "1/2" in s and "3" in s


def test_add_and_sub_match_original_for_valid_input():
    a = build(2, 0, 2, [[1, 2], [3, 4]])
    b = build(2, 0, 2, [[5, 6], [7, 8]])
    assert tx.to_list(tx.add(a, b)) == dump(a + b)
    assert tx.to_list(tx.sub(a, b)) == dump(a - b)
    c = tx.from_list(1, 1, 3, [[1, 2, 3], [4, 5, 6], [7, 8, 9]])
    d = tx.from_list(1, 1, 3, [[F(1, 2), 0, -1], [2, F(1, 3), 0], [5, 1, 1]])
    assert tx.equal(tx.add(c, d), tx.add(d, c))


def test_add_rejects_mismatched_types():
    v = tx.from_list(1, 0, 2, [1, 2])
    w = tx.from_list(0, 1, 2, [3, 4])
    with pytest.raises(ValueError):
        tx.add(v, w)
    with pytest.raises(ValueError):
        tx.add(v, tx.from_list(1, 0, 3, [1, 2, 3]))


def test_scalar_mul():
    v = tx.from_list(1, 0, 2, [F(1, 3), 2])
    assert tx.to_list(tx.scalar_mul(v, 3)) == [F(1), F(6)]
    assert tx.to_list(tx.scalar_mul(v, F(1, 2))) == [F(1, 6), F(1)]


def test_neg_is_scalar_mul_by_minus_one():
    v = tx.from_list(1, 0, 2, [1, -2])
    assert tx.to_list(tx.neg(v)) == [F(-1), F(2)]


def test_tensor_product_index_order():
    a = tx.from_list(1, 0, 2, [1, 2])
    b = tx.from_list(1, 0, 2, [10, 30])
    assert tx.to_list(tx.tensor_product(a, b)) == frac_list([[10, 30], [20, 60]])


def test_tensor_product_fixes_original_order():
    a = build(1, 0, 2, [1, 2])
    b = build(1, 0, 2, [10, 30])
    assert dump(a * b) == tx.to_list(tx.tensor_product(b, a))


def test_tensor_product_mixed_types():
    a = tx.from_list(1, 1, 2, [[1, 2], [3, 4]])
    b = tx.from_list(1, 0, 2, [10, 20])
    prod = tx.tensor_product(a, b)
    assert tx.shape(prod) == (2, 1, 2)
    assert tx.to_list(prod) == frac_list([[[10, 20], [20, 40]],
                                          [[30, 60], [40, 80]]])


def test_tensor_product_is_associative():
    a = tx.from_list(1, 0, 2, [1, 2])
    b = tx.from_list(1, 0, 2, [3, 5])
    c = tx.from_list(1, 0, 2, [7, 11])
    left = tx.tensor_product(tx.tensor_product(a, b), c)
    right = tx.tensor_product(a, tx.tensor_product(b, c))
    assert tx.equal(left, right)
    u = tx.from_list(1, 0, 3, [1, 2, 3])
    v = tx.from_list(1, 0, 3, [4, 5, 6])
    f = tx.from_list(0, 1, 3, [7, 8, 10])
    left = tx.tensor_product(tx.tensor_product(u, v), f)
    right = tx.tensor_product(u, tx.tensor_product(v, f))
    assert tx.equal(left, right) and len(list(tx._axes(left))) == 27


def test_tensor_product_bilinear():
    a = tx.from_list(1, 0, 2, [1, 2])
    b = tx.from_list(1, 0, 2, [3, 5])
    c = tx.from_list(1, 0, 2, [7, 11])
    assert tx.equal(tx.tensor_product(a, tx.add(b, c)),
                    tx.add(tx.tensor_product(a, b), tx.tensor_product(a, c)))
    assert tx.equal(tx.tensor_product(tx.add(a, b), c),
                    tx.add(tx.tensor_product(a, c), tx.tensor_product(b, c)))
    assert tx.equal(tx.tensor_product(tx.scalar_mul(a, F(3, 2)), b),
                    tx.scalar_mul(tx.tensor_product(a, b), F(3, 2)))


def test_tensor_product_rejects_different_dimension():
    with pytest.raises(ValueError):
        tx.tensor_product(tx.from_list(1, 0, 2, [1, 2]),
                          tx.from_list(1, 0, 3, [1, 2, 3]))


def test_permute_matrix_transpose():
    m = tx.from_list(2, 0, 2, [[1, 2], [3, 4]])
    assert tx.to_list(tx.permute(m, [2, 1])) == frac_list([[1, 3], [2, 4]])
    m3 = tx.from_list(2, 0, 3, [[1, 2, 3], [4, 5, 6], [7, 8, 10]])
    assert tx.equal(tx.permute(tx.permute(m3, [2, 1]), [2, 1]), m3)


def test_permute_uses_natural_axis_order():
    t = tx.from_list(3, 0, 2, [[[100 * i + 10 * j + k for k in range(2)]
                               for j in range(2)] for i in range(2)])
    expected = [[[100 * j + 10 * i + k for k in range(2)]
                 for j in range(2)] for i in range(2)]
    assert tx.to_list(tx.permute(t, [2, 1, 3])) == frac_list(expected)


def test_permute_cycle():
    t = tx.from_list(3, 0, 2, [[[100 * i + 10 * j + k for k in range(2)]
                               for j in range(2)] for i in range(2)])
    expected = [[[100 * j + 10 * k + i for k in range(2)]
                 for j in range(2)] for i in range(2)]
    assert tx.to_list(tx.permute(t, [2, 3, 1])) == frac_list(expected)
    once = tx.permute(t, [2, 3, 1])
    assert tx.equal(tx.permute(tx.permute(once, [2, 3, 1]), [2, 3, 1]), t)


def test_permute_lower_group():
    t = tx.from_list(1, 2, 2, [[[1, 2], [3, 4]], [[5, 6], [7, 8]]])
    expected = [[[1, 2], [5, 6]], [[3, 4], [7, 8]]]
    assert tx.to_list(tx.permute(t, [2, 1], group="q")) == frac_list(expected)


def test_permute_validates_permutation():
    t = tx.from_list(3, 0, 2, [[[0, 0], [0, 0]], [[0, 0], [0, 0]]])
    for bad in ([1, 1, 2], [0, 1, 2], [1, 2], [1, 2, 3, 4], [4, 1, 2]):
        with pytest.raises(ValueError):
            tx.permute(t, bad)
    with pytest.raises(ValueError):
        tx.permute(t, [2, 1], group="q")
    with pytest.raises(ValueError):
        tx.permute(t, [1, 2, 3], group="x")


def test_swap_axes():
    m = tx.from_list(2, 0, 2, [[1, 2], [3, 4]])
    assert tx.to_list(tx.swap_axes(m, 0, 1)) == frac_list([[1, 3], [2, 4]])
    with pytest.raises(ValueError):
        tx.swap_axes(m, 0, 5)


def test_trace_of_matrix():
    t = tx.from_list(1, 1, 2, [[1, 2], [3, 4]])
    assert tx.trace(t) == F(5)


def test_trace_of_delta_is_dimension():
    for n in range(1, 6):
        assert tx.trace(tx.delta(n)) == F(n)


def test_contract_reduces_rank():
    t = tx.from_list(2, 1, 2, [[[1, 2], [3, 4]], [[5, 6], [7, 8]]])
    r = tx.contract(t, 0, 0)
    assert tx.shape(r) == (1, 0, 2)
    assert tx.to_list(r) == [F(8), F(10)]


def test_contract_second_axis():
    t = tx.from_list(2, 1, 2, [[[1, 2], [3, 4]], [[5, 6], [7, 8]]])
    assert tx.to_list(tx.contract(t, 0, 1)) == [F(7), F(11)]


def test_contract_of_product_is_scalar_product():
    v = tx.from_list(1, 0, 3, [1, 2, 3])
    w = tx.from_list(0, 1, 3, [4, 5, 6])
    scalar = tx.contract(tx.tensor_product(v, w), 0, 0)
    assert tx.shape(scalar) == (0, 0, 3)
    assert tx.to_list(scalar) == F(32)


def test_contract_validates_arguments():
    m = tx.from_list(2, 0, 2, [[1, 2], [3, 4]])
    with pytest.raises(ValueError):
        tx.contract(m, 0, 0)
    t = tx.from_list(1, 1, 2, [[1, 2], [3, 4]])
    with pytest.raises(ValueError):
        tx.contract(t, 0, 3)


def test_full_contraction():
    a = tx.from_list(2, 0, 2, [[1, 2], [3, 4]])
    b = tx.from_list(2, 0, 2, [[5, 6], [7, 8]])
    assert tx.full_contraction(a, b) == F(70)
    with pytest.raises(ValueError):
        tx.full_contraction(a, tx.from_list(1, 0, 2, [1, 2]))


def test_symmetrize_matrix():
    m = tx.from_list(2, 0, 2, [[1, 2], [3, 4]])
    assert tx.to_list(tx.symmetrize(m)) == [[F(1), F(5, 2)], [F(5, 2), F(4)]]


def test_antisymmetrize_matrix():
    m = tx.from_list(2, 0, 2, [[1, 2], [3, 4]])
    assert tx.to_list(tx.antisymmetrize(m)) == [[F(0), F(-1, 2)], [F(1, 2), F(0)]]


def test_sym_plus_alt_restores_rank_two_tensor():
    m = tx.from_list(2, 0, 2, [[1, 2], [3, 4]])
    assert tx.equal(tx.add(tx.symmetrize(m), tx.antisymmetrize(m)), m)
    m3 = tx.from_list(2, 0, 3, [[1, 2, 3], [4, 5, 6], [7, 8, 10]])
    assert tx.equal(tx.add(tx.symmetrize(m3), tx.antisymmetrize(m3)), m3)
    t3 = tx.from_list(3, 0, 3, [[[9 * i + 3 * j + k + 1 for k in range(3)]
                                for j in range(3)] for i in range(3)])
    assert not tx.equal(tx.add(tx.symmetrize(t3), tx.antisymmetrize(t3)), t3)


def test_projectors_are_idempotent():
    t = tx.from_list(3, 0, 2, [[[100 * i + 10 * j + k for k in range(2)]
                               for j in range(2)] for i in range(2)])
    assert tx.equal(tx.symmetrize(tx.symmetrize(t)), tx.symmetrize(t))
    assert tx.equal(tx.antisymmetrize(tx.antisymmetrize(t)), tx.antisymmetrize(t))


def test_is_symmetric_and_is_antisymmetric():
    s = tx.from_list(2, 0, 2, [[1, 7], [7, 4]])
    a = tx.from_list(2, 0, 2, [[0, 5], [-5, 0]])
    assert tx.is_symmetric(s) and not tx.is_antisymmetric(s)
    assert tx.is_antisymmetric(a) and not tx.is_symmetric(a)
    assert tx.is_symmetric(tx.symmetrize(tx.from_list(2, 0, 2, [[1, 2], [3, 4]])))


def test_wedge_sign_matches_definition():
    u = tx.from_list(1, 0, 3, [1, 2, 3])
    v = tx.from_list(1, 0, 3, [4, 5, 6])
    assert tx.to_list(tx.wedge(u, v)) == frac_list([[0, -3, -6],
                                                    [3, 0, -3],
                                                    [6, 3, 0]])


def test_wedge_anticommutes_and_is_nilpotent():
    u = tx.from_list(1, 0, 3, [1, 2, 3])
    v = tx.from_list(1, 0, 3, [4, 5, 6])
    assert tx.equal(tx.wedge(u, v), tx.neg(tx.wedge(v, u)))
    assert tx.equal(tx.wedge(u, u), tx.zeros(2, 0, 3))


def test_wedge_of_three_basis_vectors_is_levi_civita():
    e1 = tx.from_list(1, 0, 3, [1, 0, 0])
    e2 = tx.from_list(1, 0, 3, [0, 1, 0])
    e3 = tx.from_list(1, 0, 3, [0, 0, 1])
    assert tx.equal(tx.wedge(tx.wedge(e1, e2), e3), tx.levi_civita(3))


def test_wedge_is_totally_antisymmetric():
    u = tx.from_list(1, 0, 3, [1, 2, 3])
    v = tx.from_list(1, 0, 3, [4, 5, 6])
    assert tx.is_antisymmetric(tx.wedge(u, v))


def test_delta():
    assert tx.to_list(tx.delta(3)) == frac_list([[1, 0, 0], [0, 1, 0], [0, 0, 1]])


def test_levi_civita_values():
    e = tx.to_list(tx.levi_civita(3))
    assert e[0][1][2] == F(1)
    assert e[0][2][1] == F(-1)
    assert e[1][2][0] == F(1)
    assert e[1][1][2] == F(0)


def test_det_of_matrix():
    assert tx.det(tx.from_list(2, 0, 2, [[1, 2], [3, 4]])) == F(-2)
    assert tx.det(tx.from_list(2, 0, 3, [[2, 0, 1], [1, 3, 2], [1, 1, 1]])) == F(0)


def test_det_of_identity_and_fractions():
    assert tx.det(tx.delta(4)) == F(1)
    assert tx.det(tx.from_list(2, 0, 2, [[F(1, 2), 0], [0, F(1, 3)]])) == F(1, 6)


def test_det_matches_levi_civita_formula():
    data = [[1, 2, 3], [4, 5, 7], [2, 0, 1]]
    m = tx.from_list(2, 0, 3, data)
    eps = tx.to_list(tx.levi_civita(3))
    manual = sum(eps[i][j][k] * data[0][i] * data[1][j] * data[2][k]
                 for i in range(3) for j in range(3) for k in range(3))
    assert tx.det(m) == Fraction(manual)


def _det_by_two_epsilons(data):
    """Формула (2.30): det T = 1/n! eps^{i..} eps^{j..} T_{i1 j1} ... T_{in jn}"""
    n = len(data)
    t = tx.from_list(0, 2, n, data)
    prod = t
    for _ in range(n - 1):
        prod = tx.tensor_product(prod, t)
    ee = tx.tensor_product(tx.levi_civita(n), tx.levi_civita(n))
    pairs = [("p", k, 2 * k) for k in range(n)] + [("p", n + k, 2 * k + 1) for k in range(n)]
    total = tx.to_list(tx.contract_ab(ee, prod, pairs))
    fact = 1
    for k in range(2, n + 1):
        fact *= k
    return total / fact


def test_det_two_epsilon_formula_n2_n3():
    for data in ([[1, 2], [3, 4]], [[1, 2, 3], [4, 5, 7], [2, 0, 1]]):
        n = len(data)
        assert _det_by_two_epsilons(data) == tx.det(tx.from_list(2, 0, n, data))


def test_det_rejects_wrong_rank():
    with pytest.raises(ValueError):
        tx.det(tx.from_list(1, 0, 2, [1, 2]))


def test_contract_with_matrix_is_matrix_multiplication():
    t = tx.from_list(2, 0, 2, [[1, 2], [3, 4]])
    m = [[0, 1], [1, 0]]
    assert tx.to_list(tx.contract_with_matrix(t, "p", 0, m)) == frac_list([[3, 4], [1, 2]])
    assert tx.to_list(tx.contract_with_matrix(t, "p", 1, m)) == frac_list([[2, 1], [4, 3]])


def test_contract_with_identity_changes_nothing():
    t = tx.from_list(2, 0, 2, [[1, 2], [3, 4]])
    eye = [[1, 0], [0, 1]]
    assert tx.equal(tx.contract_with_matrix(t, "p", 0, eye), t)


def test_contract_with_matrix_lowers_index_via_metric():
    v = tx.from_list(1, 0, 2, [1, 2])
    g = [[2, 0], [0, 3]]
    assert tx.to_list(tx.contract_with_matrix(v, "p", 0, g)) == [F(2), F(6)]


def test_contract_with_matrix_validates_input():
    t = tx.from_list(2, 0, 2, [[1, 2], [3, 4]])
    with pytest.raises(ValueError):
        tx.contract_with_matrix(t, "p", 5, [[1, 0], [0, 1]])
    with pytest.raises(ValueError):
        tx.contract_with_matrix(t, "p", 0, [[1, 0, 0], [0, 1, 0], [0, 0, 1]])
