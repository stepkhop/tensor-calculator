"""
Графическое отображение тензоров (matplotlib)

Единого геометрического образа для тензора произвольного ранга нет,
поэтому способ рисования выбирается по рангу и размерности:

    ранг 0                 - число
    ранг 1, n = 2, 3       - стрелка из начала координат
    ранг 2, n = 3          - характеристические пространства (ХП):
                             одномерное ХП - прямая (двусторонняя стрелка),
                             двумерное - плоскость,
                             все пространство (шаровой тензор) - сфера
    ранг 2, n = 2          - то же на плоскости: прямые или окружность
    остальные случаи       - срезы: тензор режется на матрицы n x n по двум
                             последним индексам (порядок i1..ip, j1..jq),
                             каждая матрица - тепловая карта

Модуль ничего не считает сам: корни и ХП берутся из tensor_eigen
"""

import math
import textwrap

import matplotlib
import matplotlib.patches
from matplotlib.animation import FuncAnimation
from matplotlib.figure import Figure

from tensor_ext import (to_list, to_conventional, basis_matrix, gram_matrix,
                        inverse_matrix, is_cartesian)
from tensor_eigen import (eigen, orthonormal_basis, fmt_num, classify,
                          _gram_schmidt)

__all__ = ["draw_tensor", "make_figure", "start_rotation", "describe_mode"]

POS = "#c0392b"
NEG = "#2c6fbb"
ZERO = "#2e7d32"
MAX_PANELS = 27


def _color(value):
    v = float(value)
    if abs(v) < 1e-12:
        return ZERO
    return POS if v > 0 else NEG


def describe_mode(t):
    """Как будет нарисован тензор - для подписи и для текста отчета"""
    if t is None:
        return "нет данных"
    r = t.p + t.q
    if r == 0:
        return "скаляр"
    if r == 1 and t.n in (2, 3):
        return "вектор"
    if r == 2 and t.n in (2, 3):
        return "характеристические пространства"
    return "срезы"


def _draw_scalar(fig, t, title):
    ax = fig.add_subplot(1, 1, 1)
    ax.axis("off")
    ax.text(0.5, 0.5, fmt_num(to_list(t)), ha="center", va="center", fontsize=28)
    ax.set_title(title, fontsize=10)


def _setup_3d(ax, lim):
    ax.set_xlim(-lim, lim)
    ax.set_ylim(-lim, lim)
    ax.set_zlim(-lim, lim)
    ax.set_box_aspect((1, 1, 1))
    ax.set_xlabel("x1")
    ax.set_ylabel("x2")
    ax.set_zlabel("x3")
    for k in range(3):
        a = [0, 0, 0]
        b = [0, 0, 0]
        a[k], b[k] = -lim, lim
        ax.plot(*zip(a, b), color="0.75", lw=0.6)


def _draw_vector(fig, t, title, basis=None):
    comp = to_list(t)
    if t.q == 1 and basis is not None:
        ginv = inverse_matrix(gram_matrix(basis, t.n))
        comp = [sum(ginv[i][j] * comp[j] for j in range(t.n)) for i in range(t.n)]
    v = _to_cartesian(comp, _frame(basis, t.n))
    norm = math.sqrt(sum(x * x for x in v)) or 1.0
    if t.n == 3:
        ax = fig.add_subplot(1, 1, 1, projection="3d")
        _setup_3d(ax, 1.2 * norm)
        ax.quiver(0, 0, 0, *v, color=POS, lw=2, arrow_length_ratio=0.12)
    else:
        ax = fig.add_subplot(1, 1, 1)
        ax.set_aspect("equal")
        ax.axhline(0, color="0.75", lw=0.6)
        ax.axvline(0, color="0.75", lw=0.6)
        ax.annotate("", xy=v, xytext=(0, 0),
                    arrowprops=dict(arrowstyle="->", color=POS, lw=2))
        ax.set_xlim(-1.2 * norm, 1.2 * norm)
        ax.set_ylim(-1.2 * norm, 1.2 * norm)
    kind = "вектор" if t.p == 1 else "ковектор"
    ax.set_title(f"{title}\n{kind} ({', '.join(fmt_num(x) for x in to_list(t))})",
                 fontsize=10)
    return ax


def _frame(basis, n):
    """
    Декартовы координаты базисных векторов: строка i - вектор e_i.
    None, если базис декартов ортонормированный. Базис задается явно:
    по одной фундаментальной матрице восстановить его нельзя
    """
    if basis is None or is_cartesian(basis, n):
        return None
    return [[float(x) for x in row] for row in basis_matrix(basis, n)]


def _to_cartesian(v, frame):
    """Вектор с контравариантными компонентами v^i -> декартовы координаты"""
    if frame is None:
        return [float(x) for x in v]
    n = len(frame)
    return [sum(float(v[i]) * frame[i][k] for i in range(n)) for k in range(n)]


def _space_basis(space, frame):
    """Ортонормированный (в декартовых координатах) базис ХП для рисования"""
    if frame is None:
        return orthonormal_basis(space)
    return _gram_schmidt([_to_cartesian(v, frame) for v in space.basis])


def _draw_frame_3d(ax, frame, show):
    """
    Базисные векторы e_i стрелками и параллелепипед, натянутый на них, в
    тех же декартовых осях, в которых задан базис
    """
    if frame is None or not show:
        return
    top = max(math.sqrt(sum(x * x for x in e)) for e in frame)
    vec = [[0.95 * x / top for x in e] for e in frame]
    for i, w in enumerate(vec):
        ax.quiver(0, 0, 0, *w, color="0.35", lw=1.2, arrow_length_ratio=0.1)
        ax.text(*[1.1 * x for x in w], f"e{i + 1}", color="0.3", fontsize=9)
    a, b, c = vec
    corners = [[i * a[k] + j * b[k] + m * c[k] for k in range(3)]
               for i in (0, 1) for j in (0, 1) for m in (0, 1)]
    for p in range(8):
        for q in range(p + 1, 8):
            if bin(p ^ q).count("1") == 1:
                ax.plot(*zip(corners[p], corners[q]), color="0.82", lw=0.6)


def _scaled_lengths(spaces):
    """Размер элемента пропорционален |lambda|, но не меньше 0.35"""
    vals = [abs(float(s.value)) for s in spaces]
    if not vals or max(vals) == 0:
        return [0.9] * len(vals)
    top = max(vals)
    return [0.35 + 0.65 * v / top for v in vals]


def _draw_eigen_3d(fig, t, title, basis=None, show_basis=True):
    ax = fig.add_subplot(1, 1, 1, projection="3d")
    frame = _frame(basis, 3)
    metric = None if frame is None else gram_matrix(basis, 3)
    _setup_3d(ax, 1.1)
    _draw_frame_3d(ax, frame, show_basis)
    spaces = eigen(t, metric)
    real = [s for s in spaces if s.real]
    sizes = _scaled_lengths(real)
    legend = []
    for s, size in zip(real, sizes):
        col = _color(s.value)
        basis = _space_basis(s, frame)
        label = f"λ = {s.label}" + (f" (×{s.mult})" if s.mult > 1 else "")
        if len(basis) == 1:
            e = basis[0]
            ax.quiver(0, 0, 0, *[size * x for x in e], color=col, lw=2.2,
                      arrow_length_ratio=0.15)
            ax.quiver(0, 0, 0, *[-size * x for x in e], color=col, lw=2.2,
                      arrow_length_ratio=0.15)
            tip = [1.08 * size * x for x in e]
            ax.text(*tip, label, color=col, fontsize=9)
        elif len(basis) == 2:
            e1, e2 = basis
            steps = 12
            xs, ys, zs = [], [], []
            for i in range(steps + 1):
                u = -size + 2 * size * i / steps
                rx, ry, rz = [], [], []
                for j in range(steps + 1):
                    w = -size + 2 * size * j / steps
                    rx.append(u * e1[0] + w * e2[0])
                    ry.append(u * e1[1] + w * e2[1])
                    rz.append(u * e1[2] + w * e2[2])
                xs.append(rx)
                ys.append(ry)
                zs.append(rz)
            ax.plot_surface(_arr(xs), _arr(ys), _arr(zs), color=col, alpha=0.35,
                            linewidth=0, shade=False)
            corner = [size * (e1[k] + e2[k]) for k in range(3)]
            ax.text(*corner, label, color=col, fontsize=9)
        else:
            steps = 24
            xs, ys, zs = [], [], []
            for i in range(steps + 1):
                th = math.pi * i / steps
                rx, ry, rz = [], [], []
                for j in range(steps + 1):
                    ph = 2 * math.pi * j / steps
                    rx.append(size * math.sin(th) * math.cos(ph))
                    ry.append(size * math.sin(th) * math.sin(ph))
                    rz.append(size * math.cos(th))
                xs.append(rx)
                ys.append(ry)
                zs.append(rz)
            ax.plot_surface(_arr(xs), _arr(ys), _arr(zs), color=col, alpha=0.3,
                            linewidth=0, shade=False)
            ax.text(0, 0, size * 1.1, label, color=col, fontsize=9)
        lack = f", неполное: dim {s.dim} < {s.mult}" if s.dim < s.mult else ""
        legend.append(f"{label}: {s.kind}{lack}")
    cplx = [s for s in spaces if not s.real]
    note = "\n".join(textwrap.wrap(classify(t, metric), 34))
    if cplx:
        note += "\nкомплексные корни: " + ", ".join(s.label for s in cplx)
    ax.set_title(f"{title}\n{note}", fontsize=10)
    fig.subplots_adjust(bottom=0.16)
    fig.text(0.5, 0.01, "\n".join(legend), ha="center", va="bottom", fontsize=8)
    return ax


def _draw_eigen_2d(fig, t, title, basis=None, show_basis=True):
    ax = fig.add_subplot(1, 1, 1)
    ax.set_aspect("equal")
    ax.set_xlim(-1.1, 1.1)
    ax.set_ylim(-1.1, 1.1)
    frame = _frame(basis, 2)
    metric = None if frame is None else gram_matrix(basis, 2)
    ax.axhline(0, color="0.8", lw=0.6)
    ax.axvline(0, color="0.8", lw=0.6)
    if frame is not None and show_basis:
        top = max(math.hypot(*e) for e in frame)
        for i, e in enumerate(frame):
            w = [0.95 * x / top for x in e]
            ax.annotate("", xy=w, xytext=(0, 0),
                        arrowprops=dict(arrowstyle="->", color="0.35", lw=1.2))
            ax.text(1.08 * w[0], 1.08 * w[1], f"e{i + 1}", color="0.3", fontsize=9)
    spaces = eigen(t, metric)
    real = [s for s in spaces if s.real]
    for s, size in zip(real, _scaled_lengths(real)):
        col = _color(s.value)
        basis = _space_basis(s, frame)
        label = f"λ = {s.label}" + (f" (×{s.mult})" if s.mult > 1 else "")
        if len(basis) == 1:
            e = basis[0]
            for sign in (1, -1):
                ax.annotate("", xy=(sign * size * e[0], sign * size * e[1]),
                            xytext=(0, 0),
                            arrowprops=dict(arrowstyle="->", color=col, lw=2))
            ax.text(1.05 * size * e[0], 1.05 * size * e[1], label, color=col)
        else:
            circ = matplotlib.patches.Circle((0, 0), size, color=col, alpha=0.3)
            ax.add_patch(circ)
            ax.text(0, size * 1.05, label + ", вся плоскость", color=col,
                    ha="center")
    cplx = [s for s in spaces if not s.real]
    note = "\n".join(textwrap.wrap(classify(t, metric), 34))
    if cplx:
        note += "\nкомплексные корни: " + ", ".join(s.label for s in cplx)
    ax.set_title(f"{title}\n{note}", fontsize=10)
    return ax


def _draw_slices(fig, t, title):
    """
    Срезы по двум последним индексам в общепринятом порядке
    (сначала верхние i1..ip, затем нижние j1..jq), как в полях ввода
    """
    data = to_conventional(t)
    r, n = t.p + t.q, t.n
    if r == 1:
        ax = fig.add_subplot(1, 1, 1)
        ax.bar(range(1, n + 1), [float(x) for x in data], color=POS)
        ax.set_title(f"{title}\nкомпоненты (n = {n})", fontsize=10)
        return
    lead = r - 2
    panels = n ** lead
    if panels > MAX_PANELS:
        ax = fig.add_subplot(1, 1, 1)
        ax.axis("off")
        ax.text(0.5, 0.5, f"{n ** r} компонент:\nслишком много срезов ({panels})",
                ha="center", va="center")
        ax.set_title(title, fontsize=10)
        return
    cols = n if lead >= 1 else 1
    rows = math.ceil(panels / cols)
    flat = []
    _collect(data, lead, (), flat)
    vmax = max([abs(float(x)) for _, mat in flat for row in mat for x in row] + [1e-12])
    names = [f"i{k + 1}" for k in range(t.p)] + [f"j{k + 1}" for k in range(t.q)]
    for k, (prefix, mat) in enumerate(flat):
        ax = fig.add_subplot(rows, cols, k + 1)
        vals = [[float(x) for x in row] for row in mat]
        ax.imshow(vals, cmap="RdBu_r", vmin=-vmax, vmax=vmax)
        ax.set_xticks([])
        ax.set_yticks([])
        if n <= 4:
            for i in range(n):
                for j in range(n):
                    dark = abs(float(mat[i][j])) > 0.6 * vmax
                    ax.text(j, i, fmt_num(mat[i][j]), ha="center", va="center",
                            fontsize=7, color="white" if dark else "black")
        if prefix:
            ax.set_title("[" + ",".join(str(x + 1) for x in prefix) + "]", fontsize=8)
    order = ", ".join(names)
    fig.suptitle(f"{title}\nтип ({t.p}, {t.q}), порядок осей: {order}", fontsize=10)


def _collect(data, lead, prefix, out):
    if lead == 0:
        out.append((prefix, data))
        return
    for i, sub in enumerate(data):
        _collect(sub, lead - 1, prefix + (i,), out)


def _arr(x):
    import numpy as np
    return np.array(x)


def draw_tensor(fig, t, title, basis=None, show_basis=True):
    """
    Нарисовать тензор t на (под)фигуре fig. basis - матрица базиса
    (строка i - вектор e_i в декартовых координатах); векторы и ХП
    переводятся в декартовы координаты, фундаментальная матрица
    вычисляется как g_ij = (e_i, e_j). show_basis - рисовать ли сами
    векторы e_i: на вычисления и на положение ХП это не влияет
    """
    if t is None:
        ax = fig.add_subplot(1, 1, 1)
        ax.axis("off")
        ax.set_title(title + "\nнет данных", fontsize=10)
        return
    mode = describe_mode(t)
    if mode == "скаляр":
        _draw_scalar(fig, t, title)
    elif mode == "вектор":
        _draw_vector(fig, t, title, basis)
    elif mode == "характеристические пространства":
        if t.n == 3:
            _draw_eigen_3d(fig, t, title, basis, show_basis)
        else:
            _draw_eigen_2d(fig, t, title, basis, show_basis)
    else:
        _draw_slices(fig, t, title)


def make_figure(items, fig=None, size=(12, 4.2), basis=None, show_basis=True):
    """
    items - список пар (подпись, тензор или None). Каждый тензор рисуется
    в своей подфигуре, рядом друг с другом: A, B, результат
    """
    if fig is None:
        fig = Figure(figsize=size)
    fig.clear()
    subs = fig.subfigures(1, len(items), squeeze=False)[0]
    for sub, (title, t) in zip(subs, items):
        b = basis if (t is not None and basis is not None
                      and len(basis) == t.n) else None
        draw_tensor(sub, t, title, b, show_basis)
    return fig


def start_rotation(fig, step=2, interval=60):
    """
    Вращение всех 3D-осей фигуры вокруг вертикали. Возвращает объект
    анимации - его нужно хранить, иначе сборщик мусора остановит вращение
    """
    def axes3d():
        out = []
        for ax in fig.get_axes():
            if ax.name == "3d":
                out.append(ax)
        return out

    def update(_):
        for ax in axes3d():
            ax.view_init(elev=ax.elev, azim=(ax.azim + step) % 360)
        return []

    return FuncAnimation(fig, update, interval=interval, cache_frame_data=False)
