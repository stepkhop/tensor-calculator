from fractions import Fraction
from itertools import product
from tkinter import (Tk, Toplevel, Frame, LabelFrame, Entry, Button, Label, Text, Canvas,
                     Checkbutton, BooleanVar, StringVar, OptionMenu, Scrollbar,
                     messagebox, END)

import matplotlib
matplotlib.use("TkAgg")
from matplotlib.backends.backend_tkagg import (FigureCanvasTkAgg,
                                               NavigationToolbar2Tk)
from matplotlib.figure import Figure

import tensor_ext as tx
import tensor_eigen as te
from expr import evaluate, as_tensor, ExprError
from tensor_viz import make_figure, start_rotation

COLOUR = "#F8ECE0"
MAX_COMPONENTS = 729
HELP = [
    ("A + B, A - B", "сумма и разность, типы должны совпадать"),
    ("2*A, A/2", "умножение и деление на число"),
    ("A*B", "тензорное произведение"),
    ("wedge(A, B)", "внешнее произведение"),
    ("dot(A, B)", "свертка по одной паре индексов, например T^i_j v^j"),
    ("ddot(A, B)", "свертка по двум парам, например C^ij_kl e^kl"),
    ("contract(T, i, j)", "свертка i-го нижнего индекса T с j-м верхним"),
    ("contract2(A, B, 'p', i, j)", "i-й верхний индекс A с j-м нижним B ('q' - наоборот)"),
    ("tr(T), det(T)", "след и определитель"),
    ("sym(T), alt(T)", "симметрирование и альтернирование верхних индексов"),
    ("symq(T), altq(T)", "то же для нижних индексов"),
    ("perm(T, [2, 1]), permq", "перестановка верхних (нижних) индексов"),
    ("sph(T), dev(T)", "шаровая часть и девиатор"),
    ("delta(n), eps(n)", "символ Кронекера и символ Леви-Чивиты"),
    ("low(T, i), up(T, i)", "опустить i-й верхний / поднять i-й нижний индекс метрикой G;\n"
                            "например, для типа (0, 2) low не определена, нужна up"),
    ("low(T, i, B), up(T, i, B)", "то же с метрикой B"),
    ("G", "фундаментальная матрица g_ij заданного базиса"),
    ("E", "обратная матрица g^ij"),
]
HELP_NOTE = ("Вводятся и выводятся компоненты тензоров в текущем базисе, "
             "сам тензор от базиса не зависит. "
             "Базис задается матрицей справа: строка i - вектор e_i в декартовых "
             "координатах. По нему считается фундаментальная матрица g_ij = (e_i, e_j), "
             "она учитывается в low, up, sph, dev и в задаче на собственные значения, "
             "а сам базис нужен для рисунка. Индексы нумеруются с 1. "
             "Компоненты вводятся так: сначала верхние индексы, потом нижние.")


class TensorInput(LabelFrame):
    """
    Ввод тензора: сначала p, q, n, затем сетка полей.

    Индексы берутся в общепринятом порядке: сначала верхние i1..ip,
    затем нижние j1..jq. Два последних индекса образуют матрицу n x n
    (строка - предпоследний индекс, столбец - последний), остальные
    индексы нумеруют блоки. Для тензора ранга 2 это обычная матрица:
    первая строка поля ввода - первая строка матрицы
    """
    WDTH = 5

    def __init__(self, root, name):
        super().__init__(root, text=f"Компоненты тензора {name} в базисе", bg=COLOUR)
        self.name = name
        self.p = self.q = self.n = 0
        self.cells = None
        head = Frame(self, bg=COLOUR)
        head.pack(anchor="w", padx=4, pady=2)
        self.p_in = self._param(head, "p =", 0, "1")
        self.q_in = self._param(head, "q =", 2, "1")
        self.n_in = self._param(head, "n =", 4, "3")
        Button(head, text="Создать поля", command=self.build).grid(row=0, column=6, padx=6)
        self.hint = Label(self, text="p - число верхних индексов, q - нижних, n - размерность",
                          bg=COLOUR, anchor="w", justify="left", font=("TkDefaultFont", 9))
        self.hint.pack(fill="x", padx=4)
        box = Frame(self, bg=COLOUR)
        box.pack(fill="both", expand=True, padx=4, pady=4)
        self.canvas = Canvas(box, bg=COLOUR, highlightthickness=0, width=520, height=230)
        ys = Scrollbar(box, orient="vertical", command=self.canvas.yview)
        xs = Scrollbar(box, orient="horizontal", command=self.canvas.xview)
        self.canvas.configure(yscrollcommand=ys.set, xscrollcommand=xs.set)
        ys.pack(side="right", fill="y")
        xs.pack(side="bottom", fill="x")
        self.canvas.pack(side="left", fill="both", expand=True)
        self.area = Frame(self.canvas, bg=COLOUR)
        self.canvas.create_window((0, 0), window=self.area, anchor="nw")
        self.area.bind("<Configure>", lambda _: self.canvas.configure(
            scrollregion=self.canvas.bbox("all")))

    def _param(self, head, text, col, default):
        Label(head, text=text, bg=COLOUR).grid(row=0, column=col)
        e = Entry(head, width=4)
        e.insert(0, default)
        e.grid(row=0, column=col + 1, padx=2)
        return e

    def set_shape(self, p, q, n):
        for e, v in ((self.p_in, p), (self.q_in, q), (self.n_in, n)):
            e.delete(0, END)
            e.insert(0, str(v))
        self.build()

    def index_names(self):
        return [f"i{k + 1}" for k in range(self.p)] + [f"j{k + 1}" for k in range(self.q)]

    def build(self):
        for w in self.area.winfo_children():
            w.destroy()
        self.cells = None
        try:
            p, q, n = (int(e.get()) for e in (self.p_in, self.q_in, self.n_in))
        except ValueError:
            messagebox.showerror("Ввод", f"{self.name}: p, q, n должны быть целыми")
            return
        if p < 0 or q < 0 or n < 1:
            messagebox.showerror("Ввод", f"{self.name}: нужно p, q >= 0 и n >= 1")
            return
        if n ** (p + q) > MAX_COMPONENTS:
            messagebox.showerror("Ввод", f"{self.name}: слишком много компонент "
                                         f"({n ** (p + q)} > {MAX_COMPONENTS})")
            return
        self.p, self.q, self.n = p, q, n
        r = p + q
        names = self.index_names()
        self.cells = {}
        if r == 0:
            self.hint["text"] = "скаляр"
            e = Entry(self.area, width=self.WDTH)
            e.grid(row=0, column=0, padx=1, pady=1)
            self.cells[()] = e
            return
        if r == 1:
            self.hint["text"] = f"компоненты по индексу {names[0]} = 1..{n}"
            for k in range(n):
                Label(self.area, text=str(k + 1), bg=COLOUR).grid(row=0, column=k)
                e = Entry(self.area, width=self.WDTH)
                e.grid(row=1, column=k, padx=1, pady=1)
                self.cells[(k,)] = e
            return
        lead = r - 2
        text = f"порядок индексов: {', '.join(names)}; строка - {names[-2]}, столбец - {names[-1]}"
        if lead:
            text += f"; блок [{', '.join(names[:lead])}]"
        self.hint["text"] = text
        for idx in product(range(n), repeat=r):
            blk, i, j = idx[:lead], idx[-2], idx[-1]
            brow = 0
            for x in blk[:-1]:
                brow = brow * n + x
            bcol = blk[-1] if blk else 0
            row0, col0 = brow * (n + 1), bcol * (n + 1)
            if i == 0 and j == 0 and blk:
                Label(self.area, text="[" + ", ".join(str(x + 1) for x in blk) + "]",
                      bg=COLOUR, font=("TkDefaultFont", 8)).grid(
                    row=row0, column=col0, columnspan=n, sticky="w")
            e = Entry(self.area, width=self.WDTH)
            e.grid(row=row0 + 1 + i, column=col0 + j, padx=1, pady=1)
            self.cells[idx] = e
        if lead:
            for b in range(n):
                self.area.grid_columnconfigure(b * (n + 1) + n, minsize=14)

    def fill(self, data):
        """Заполнить поля из вложенного списка в общепринятом порядке индексов"""
        for idx, e in self.cells.items():
            v = data
            for x in idx:
                v = v[x]
            e.delete(0, END)
            e.insert(0, str(v))

    def get_tensor(self):
        if self.cells is None:
            raise ExprError(f"тензор {self.name} не задан: нажмите «Создать поля»")
        values = {}
        for idx, e in self.cells.items():
            text = e.get().strip().replace(",", ".")
            if not text:
                values[idx] = Fraction(0)
                continue
            try:
                values[idx] = Fraction(text)
            except (ValueError, ZeroDivisionError):
                pos = ", ".join(f"{nm}={x + 1}" for nm, x in zip(self.index_names(), idx))
                raise ExprError(f"{self.name}[{pos}]: «{text}» не число") from None
        r = self.p + self.q

        def nest(prefix):
            if len(prefix) == r:
                return values[prefix]
            return [nest(prefix + (k,)) for k in range(self.n)]

        return tx.from_conventional(self.p, self.q, self.n, nest(()))


def _nested_text(x):
    if isinstance(x, list):
        return "[" + ", ".join(_nested_text(v) for v in x) + "]"
    return te.fmt_num(x)


def format_tensor(t):
    """
    Текстовый вид: тип и компоненты в общепринятом порядке индексов
    (сначала верхние, затем нижние); ранг 2 - матрицей, строка - первый индекс
    """
    if isinstance(t, Fraction):
        return te.fmt_num(t)
    head = f"тип ({t.p}, {t.q}), n = {t.n}"
    if t.p + t.q == 2:
        m = te.as_matrix(t)
        width = max(len(te.fmt_num(x)) for row in m for x in row)
        rows = ["  [" + "  ".join(te.fmt_num(x).rjust(width) for x in row) + "]"
                for row in m]
        return head + "\n" + "\n".join(rows)
    names = [f"i{k + 1}" for k in range(t.p)] + [f"j{k + 1}" for k in range(t.q)]
    order = f", порядок индексов: {', '.join(names)}" if names else ""
    return head + order + "\n  " + _nested_text(tx.to_conventional(t))


EXAMPLES = [
    ("нильпотентный ((0,1,2),(0,0,3),(0,0,0))", 1, 1, 3,
     [[0, 1, 2], [0, 0, 3], [0, 0, 0]], "A"),
    ("шаровой + антисимметричный ((2,1,2),(-1,2,3),(-2,-3,2))", 1, 1, 3,
     [[2, 1, 2], [-1, 2, 3], [-2, -3, 2]], "A"),
    ("осесимметричный ((3,-1,1),(-1,3,-1),(1,-1,3))", 1, 1, 3,
     [[3, -1, 1], [-1, 3, -1], [1, -1, 3]], "A"),
    ("шаровой 2E", 1, 1, 3, [[2, 0, 0], [0, 2, 0], [0, 0, 2]], "A"),
    ("три простых корня ((1,2,3),(2,4,5),(3,5,6))", 1, 1, 3,
     [[1, 2, 3], [2, 4, 5], [3, 5, 6]], "A"),
    ("косоугольный базис: T_ij и e_i", 0, 2, 3,
     [[3, 2, 2], [2, 4, 0], [2, 0, 12]], "A",
     [[1, 0, 0], [1, 1, 0], [0, 0, 2]]),
]


class BasisInput(LabelFrame):
    """
    Базис e_1, ..., e_n: строка i - декартовы координаты вектора e_i.
    По фундаментальной матрице g_ij = (e_i, e_j) базис восстановить нельзя
    (она не меняется при повороте), поэтому задается именно базис, а G
    вычисляется по нему и показывается рядом. По умолчанию базис декартов
    """
    def __init__(self, root, on_change=None):
        super().__init__(root, text="Базис (строки - векторы e_i)", bg=COLOUR)
        self.on_change = on_change
        head = Frame(self, bg=COLOUR)
        head.pack(anchor="w", padx=4, pady=2)
        Label(head, text="n =", bg=COLOUR).grid(row=0, column=0)
        self.n_in = Entry(head, width=4)
        self.n_in.insert(0, "3")
        self.n_in.grid(row=0, column=1, padx=2)
        Button(head, text="Декартов базис", command=self.reset).grid(row=0, column=2, padx=6)
        self.show = BooleanVar(value=True)
        Checkbutton(head, text="Рисовать базис", variable=self.show, bg=COLOUR,
                    command=self._changed).grid(row=1, column=0, columnspan=3, sticky="w")
        self.area = Frame(self, bg=COLOUR)
        self.area.pack(fill="x", padx=4, pady=(4, 2))
        self.gram = Label(self, text="", bg=COLOUR, anchor="w", justify="left",
                          font=("Courier", 9))
        self.gram.pack(fill="x", padx=4, pady=(0, 4))
        self.cells = []
        self.reset()

    def reset(self, values=None):
        """Поля n x n; без values - декартов базис"""
        for w in self.area.winfo_children():
            w.destroy()
        try:
            n = int(self.n_in.get())
        except ValueError:
            n = 3
        if values is not None:
            n = len(values)
            self.n_in.delete(0, END)
            self.n_in.insert(0, str(n))
        n = max(1, min(n, 9))
        self.cells = []
        for i in range(n):
            Label(self.area, text=f"e{i + 1}", bg=COLOUR,
                  font=("TkDefaultFont", 9)).grid(row=i, column=0, padx=(0, 4))
            row = []
            for j in range(n):
                e = Entry(self.area, width=5)
                e.grid(row=i, column=j + 1, padx=1, pady=1)
                e.insert(0, str(values[i][j] if values is not None else int(i == j)))
                e.bind("<KeyRelease>", lambda _: self.show_gram())
                row.append(e)
            self.cells.append(row)
        self.show_gram()

    def _changed(self):
        if self.on_change is not None:
            self.on_change()

    def _read(self):
        b = []
        for i, row in enumerate(self.cells):
            out = []
            for j, e in enumerate(row):
                text = e.get().strip().replace(",", ".") or "0"
                try:
                    out.append(Fraction(text))
                except (ValueError, ZeroDivisionError):
                    raise ExprError(f"e{i + 1}[{j + 1}]: «{text}» не число") from None
            b.append(out)
        return b

    def show_gram(self):
        """Показать фундаментальную матрицу, вычисленную по базису"""
        try:
            g = tx.gram_matrix(self._read(), len(self.cells))
        except (ExprError, ValueError) as e:
            self.gram["text"] = f"G: {e}"
            return
        rows = [" ".join(te.fmt_num(x).rjust(4) for x in row) for row in g]
        self.gram["text"] = "g_ij = (e_i, e_j):\n" + "\n".join(rows)

    def get_basis(self):
        """Матрица базиса (строки - e_i) или None для декартова базиса"""
        b = self._read()
        try:
            tx.basis_matrix(b, len(b))
        except ValueError as e:
            raise ExprError(f"базис: {e}") from None
        return None if tx.is_cartesian(b, len(b)) else b


class App(Frame):
    def __init__(self, root):
        super().__init__(root, bg=COLOUR)
        self.root = root
        self.anim = None
        self.result = None

        top = Frame(self, bg=COLOUR)
        top.pack(fill="x")
        self.A = TensorInput(top, "A")
        self.B = TensorInput(top, "B")
        self.G = BasisInput(top, self._redraw)
        self.A.pack(side="left", fill="both", expand=True, padx=4)
        self.B.pack(side="left", fill="both", expand=True, padx=4)
        self.G.pack(side="left", fill="y", padx=4)

        mid = Frame(self, bg=COLOUR)
        mid.pack(fill="x", pady=4)
        Label(mid, text="Выражение:", bg=COLOUR).pack(side="left", padx=4)
        self.expr = Entry(mid, width=60)
        self.expr.pack(side="left", padx=4)
        self.expr.bind("<Return>", lambda _: self.calculate())
        Button(mid, text="Вычислить", command=self.calculate).pack(side="left", padx=4)
        Button(mid, text="Собственные значения", command=self.show_eigen).pack(side="left", padx=4)
        Button(mid, text="Справка", command=self.show_help).pack(side="left", padx=4)
        self.example = StringVar(value="Пример...")
        om = OptionMenu(mid, self.example, *[e[0] for e in EXAMPLES],
                        command=self.load_example)
        om.config(width=16)
        om.pack(side="left", padx=4)
        self.rotate = BooleanVar(value=False)
        Checkbutton(mid, text="Вращать", variable=self.rotate, bg=COLOUR,
                    command=self.toggle_rotation).pack(side="left", padx=8)

        bottom = Frame(self, bg=COLOUR)
        bottom.pack(fill="both", expand=True)
        left = Frame(bottom, bg=COLOUR)
        left.pack(side="left", fill="y", padx=4)
        self.out = Text(left, width=48, height=24, font=("Courier", 10))
        scroll = Scrollbar(left, command=self.out.yview)
        self.out.configure(yscrollcommand=scroll.set)
        self.out.pack(side="left", fill="y")
        scroll.pack(side="left", fill="y")

        right = Frame(bottom, bg=COLOUR)
        right.pack(side="left", fill="both", expand=True)
        self.fig = Figure(figsize=(9, 3.6))
        self.canvas = FigureCanvasTkAgg(self.fig, master=right)
        toolbar = NavigationToolbar2Tk(self.canvas, right, pack_toolbar=False)
        toolbar.update()
        toolbar.pack(side="bottom", fill="x")
        self.canvas.get_tk_widget().pack(fill="both", expand=True)
        self.A.build()

    def _write(self, text):
        self.out.delete("1.0", END)
        self.out.insert(END, text)

    def _redraw(self):
        items = []
        for title, t in (("A", self._safe(self.A)), ("B", self._safe(self.B)),
                         ("Результат", self.result)):
            items.append((title, t))
        make_figure(items, fig=self.fig, basis=self._basis_or_none(),
                    show_basis=self.G.show.get())
        self.canvas.draw()
        if self.rotate.get():
            self.toggle_rotation()

    def _basis_or_none(self):
        try:
            return self.G.get_basis()
        except ExprError:
            return None

    def _metric_or_none(self):
        basis = self._basis_or_none()
        return None if basis is None else tx.gram_matrix(basis, len(basis))

    @staticmethod
    def _safe(inp):
        try:
            return inp.get_tensor()
        except ExprError:
            return None

    def calculate(self):
        try:
            env = {"A": self.A.get_tensor()}
            if self.B.cells is not None:
                env["B"] = self.B.get_tensor()
            basis = self.G.get_basis()
            n = env["A"].n
            g = tx.gram_matrix(basis, len(self.G.cells))
            metric = None if basis is None else g
            if len(g) == n:
                env["G"] = tx.from_conventional(0, 2, n, g)
                env["E"] = tx.from_conventional(2, 0, n, tx.inverse_matrix(g))
            elif basis is not None:
                raise ExprError(f"размерность базиса ({len(g)}) не совпадает с n = {n}")
            value = evaluate(self.expr.get(), env, metric)
        except (ExprError, ValueError) as e:
            messagebox.showerror("Ошибка", str(e))
            return
        n = env["A"].n
        self.result = as_tensor(value, n)
        self._write(f"{self.expr.get()} =\n{format_tensor(value)}\n")
        self._redraw()

    def show_eigen(self):
        lines = []
        for title, t in (("A", self._safe(self.A)), ("B", self._safe(self.B)),
                         ("Результат", self.result)):
            if t is None or t.p + t.q != 2:
                continue
            metric = self._metric_or_none()
            if metric is not None and len(metric) != t.n:
                metric = None
            lines.append(f"--- {title} ---\n{te.report(t, metric)}\n")
        if not lines:
            lines = ["Нет тензоров ранга 2: задача на собственные значения "
                     "ставится только для них"]
        self._write("\n".join(lines))
        self._redraw()

    def show_help(self):
        win = Toplevel(self.root)
        win.title("Справка")
        win["bg"] = COLOUR
        win.resizable(False, False)
        for row, (code, text) in enumerate(HELP):
            Label(win, text=code, bg=COLOUR, anchor="w", font=("Courier", 10)).grid(
                row=row, column=0, sticky="w", padx=(10, 16), pady=1)
            Label(win, text=text, bg=COLOUR, anchor="w").grid(
                row=row, column=1, sticky="w", padx=(0, 10), pady=1)
        Label(win, text=HELP_NOTE, bg=COLOUR, anchor="w", justify="left",
              wraplength=620).grid(row=len(HELP), column=0, columnspan=2,
                                   sticky="w", padx=10, pady=(8, 10))

    def load_example(self, label):
        """Заполнить A готовым примером, вычислить и показать ХП"""
        for title, p, q, n, data, text, *rest in EXAMPLES:
            if title == label:
                self.G.reset(rest[0] if rest else None)
                self.A.set_shape(p, q, n)
                self.A.fill(data)
                self.expr.delete(0, END)
                self.expr.insert(0, text)
                self.calculate()
                self.show_eigen()
                return

    def toggle_rotation(self):
        if self.anim is not None:
            self.anim.event_source.stop()
            self.anim = None
        if self.rotate.get():
            self.anim = start_rotation(self.fig)
            self.canvas.draw_idle()


def main():
    root = Tk()
    root["bg"] = COLOUR
    root.title("Тензорный калькулятор")
    root.geometry("1400x900")
    app = App(root)
    app.pack(fill="both", expand=True)
    root.mainloop()


if __name__ == "__main__":
    main()
