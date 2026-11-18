# -*- coding: utf-8 -*-
"""标记工具条 + 标记统计面板（对标 iCogtiveFlow 的标记栏与「标注管理」一栏）。

两块东西：

    MarkToolbar   —— 四个标记按钮（正常/过杀/漏杀/存疑）+ 清空 + 备注。
                     作用域 = 图片列表里**当前选中的行**（单选或多选批量）。

    MarkStatsPanel—— 四态计数与占比 + **缺陷类型 × 标记** 交叉表。
                     点某个格子 = 直接筛出"这个缺陷 + 这个标记"的那批图 ——
                     这是路线图 5.8 要的"点格子即筛选"，也是闭环的入口。

信号设计成"请求"而不是"直接改数据"：面板不知道工程目录、不知道列表怎么刷，
只负责喊一声，由 MainWindow 执行并回灌数据。这样面板可以脱离主窗口单测。
"""
from __future__ import annotations
from core.i18n import tr

from PyQt5.QtCore import Qt, pyqtSignal
from PyQt5.QtWidgets import (
    QWidget, QVBoxLayout, QHBoxLayout, QGridLayout, QLabel, QPushButton,
    QTableWidget, QTableWidgetItem, QHeaderView, QGroupBox, QSizePolicy, QFrame,
)

from core import marks as mk

_BTN_QSS = """
QPushButton {
    font-size: 11px; padding: 2px 8px; border: 1px solid %(bd)s;
    border-radius: 3px; color: %(fg)s; background: %(bg)s;
}
QPushButton:hover { border: 1px solid %(fg)s; }
QPushButton:pressed { background: %(fg)s; color: #FFFFFF; }
"""


def _btn_style(kind):
    c = mk.kind_color(kind)
    return _BTN_QSS % {"bd": c, "fg": c, "bg": "rgba(0,0,0,0)"}


#: 三个附加按钮（清空 / 备注 / 统计）的紧凑度量。
#: ⚠️ 只写度量、**不写颜色** —— 颜色交给 themes.py 的主题表（内联写死的白名单色会被
#:    sweep_inline_styles 剥掉，只剩度量，等于骗自己；底栏滑块"占位不画"就栽在这上面）。
#: 为什么需要它（2026-10-15 用户反馈「7 个按钮挤一行」后实测）：
#:    这三个按钮原来**没有自己的样式**，走全局默认按钮样式，sizeHint 高达 80px
#:    （四态按钮是 40px）—— 一个量级；再加上四个态按钮硬钉的 minWidth 42×4+38+48+38=294，
#:    而左栏可用宽度只有 ~300px ⇒ **Qt 必然牺牲某几个按钮的宽度来塞**，
#:    表现就是文字被裁。实测最小需求 421px vs 可用 300px。
_AUX_QSS = "QPushButton { font-size: 11px; padding: 2px 6px; }"

#: 四个态按钮的最小宽度：够放两个汉字（11px 字体约 22px + 左右内边距 16px + 边框），
#: 再小就裁字。**下限**，不是固定宽 —— 富余由 Expanding 均分。
_CHIP_MIN_W = 40


class MarkToolbar(QWidget):
    """四态标记按钮 + 清空 + 备注。"""

    mark_requested = pyqtSignal(str)      # kind；"" = 清空
    note_requested = pyqtSignal()
    stats_requested = pyqtSignal()

    def __init__(self, parent=None):
        super().__init__(parent)
        lay = QVBoxLayout(self)
        lay.setContentsMargins(0, 0, 0, 0)
        lay.setSpacing(2)

        # 一行放下：四态 + 分隔线 + 三个附带操作。
        # 原来拆成两行（四态一行、清空/备注/统计一行），在本来就窄的左栏里白占 26px 高度，
        # 而且第二行只有 3 个左对齐按钮、右边一大片空白 —— 视觉上像"没排完"。
        #
        # ★ 2026-10-15「7 个按钮挤一行」的修法（实测左栏可用 ~300px）：
        #   ① 四个态按钮 Expanding —— 宽裕时**均分富余**（面板拉宽不会右半截空着）；
        #      最低只保 _CHIP_MIN_W（够放两个汉字，不裁字）。
        #   ② 三个附加按钮改用 _AUX_QSS 紧凑度量并**去掉硬 minWidth**（原来 38/48/38
        #      是猜的，不是量的；真正需求由 sizeHint 说话）。
        #   ③ 「备注…」去省略号 —— 左栏空间金贵，"…" 表达的"点开有对话框"已由 tooltip 承担。
        #   ⇒ 结果：一行放得下（最小 ~272px vs 可用 300px），文字一个都不裁。
        row = QHBoxLayout(); row.setSpacing(2)
        self._btns = {}
        for k in mk.MARK_KINDS:
            b = QPushButton(tr(mk.MARK_LABELS[k]))
            b.setToolTip(tr(mk.MARK_HINTS[k]))
            b.setStyleSheet(_btn_style(k))
            b.setMinimumWidth(_CHIP_MIN_W)
            b.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Fixed)
            b.clicked.connect(lambda _c=False, kk=k: self.mark_requested.emit(kk))
            row.addWidget(b, 1)          # stretch=1：宽裕时四态均分，不给右半截留白
            self._btns[k] = b
        # 分隔线：左边是"给这张图定性"，右边是"附加操作"，语义不同不该糊成一片
        sep = QFrame(); sep.setFrameShape(QFrame.VLine); sep.setFrameShadow(QFrame.Sunken)
        row.addWidget(sep)
        self.btn_clear = QPushButton(tr("清空"))
        self.btn_clear.setToolTip(tr("清掉选中图片的标记"))
        self.btn_clear.setStyleSheet(_AUX_QSS)
        self.btn_clear.clicked.connect(lambda: self.mark_requested.emit(""))
        row.addWidget(self.btn_clear)
        self.btn_note = QPushButton(tr("备注"))
        self.btn_note.setToolTip(tr("给已标记的图写理由（现场经验最难传承，理由要留下来）"))
        self.btn_note.setStyleSheet(_AUX_QSS)
        self.btn_note.clicked.connect(self.note_requested.emit)
        row.addWidget(self.btn_note)
        self.btn_stats = QPushButton(tr("统计"))
        self.btn_stats.setToolTip(tr("四态计数与占比、缺陷类型 × 标记 交叉表"))
        self.btn_stats.setStyleSheet(_AUX_QSS)
        self.btn_stats.clicked.connect(self.stats_requested.emit)
        row.addWidget(self.btn_stats)
        row.addStretch(1)
        lay.addLayout(row)

    def set_scope_text(self, n):
        """把"当前作用于几张图"显示在按钮 tooltip 上（不占版面）。"""
        for b in list(self._btns.values()) + [self.btn_clear, self.btn_note]:
            base = b.toolTip().split(" ｜")[0].split(" |")[0]
            b.setToolTip(base + (tr("  ｜  当前选中 %d 张") % n if n
                                else tr("  ｜  未选中任何图")))


class MarkStatsPanel(QGroupBox):
    """四态计数与占比 + 缺陷类型 × 标记 交叉表。"""

    #: 点某个格子：(缺陷类型, 标记kind)；缺陷类型为 "" 表示不限
    cell_filter_requested = pyqtSignal(str, str)
    kind_filter_requested = pyqtSignal(str)
    note_filter_requested = pyqtSignal()   # 只看有备注的
    back_requested = pyqtSignal()          # 「← 返回列表」：不带筛选地回图片列表页

    def __init__(self, parent=None):
        super().__init__(tr("标记统计"), parent)
        self._records = []
        root = QVBoxLayout(self)
        root.setContentsMargins(6, 4, 4, 4)
        root.setSpacing(3)

        # ── 顶部一行：「← 返回列表」──
        # 2026-10-15：左栏页头那行（含「标记统计」入口按钮）因冗余整行删除后，
        # 从本页回图片列表**只剩"点格子带筛选跳回"一条路**。补一个不带筛选的返回。
        top_row = QHBoxLayout(); top_row.setContentsMargins(0, 0, 0, 0); top_row.setSpacing(4)
        self.btn_back = QPushButton(tr("← 返回列表"))
        self.btn_back.setToolTip(tr("回到图片列表（不改变当前筛选条件）"))
        self.btn_back.setFixedHeight(22)
        self.btn_back.clicked.connect(self.back_requested.emit)
        top_row.addWidget(self.btn_back)
        top_row.addStretch(1)
        root.addLayout(top_row)

        # ── 四态计数（可点的按钮，点了就按该态筛选） ──
        self._chips = {}
        chip_row = QGridLayout(); chip_row.setHorizontalSpacing(4); chip_row.setVerticalSpacing(2)
        for i, k in enumerate(mk.MARK_KINDS):
            b = QPushButton("%s 0" % tr(mk.MARK_LABELS[k]))
            b.setStyleSheet(_btn_style(k))
            b.setToolTip(tr(mk.MARK_HINTS[k]) + "  " + tr("｜") + "  " + tr("点击 = 只看这个标记"))
            b.clicked.connect(lambda _c=False, kk=k: self.kind_filter_requested.emit(kk))
            chip_row.addWidget(b, 0, i)
            self._chips[k] = b
        root.addLayout(chip_row)

        self.lbl_total = QLabel(tr("尚无标记"))
        self.lbl_total.setWordWrap(True)
        root.addWidget(self.lbl_total)

        self.lbl_ratio = QLabel("")
        self.lbl_ratio.setWordWrap(True)
        root.addWidget(self.lbl_ratio)

        self.lbl_warn = QLabel("")
        self.lbl_warn.setWordWrap(True)
        self.lbl_warn.setVisible(False)
        root.addWidget(self.lbl_warn)

        # ── 交叉表：缺陷类型 × 标记 ──
        self.table = QTableWidget(0, len(mk.MARK_KINDS) + 2)
        heads = [tr("缺陷类型")] + [tr(mk.MARK_LABELS[k]) for k in mk.MARK_KINDS] + [tr("无标记")]
        self.table.setHorizontalHeaderLabels(heads)
        self.table.verticalHeader().setVisible(False)
        self.table.setEditTriggers(QTableWidget.NoEditTriggers)
        self.table.setSelectionMode(QTableWidget.SingleSelection)
        self.table.setAlternatingRowColors(True)
        self.table.horizontalHeader().setSectionResizeMode(0, QHeaderView.Stretch)
        for c in range(1, self.table.columnCount()):
            self.table.horizontalHeader().setSectionResizeMode(c, QHeaderView.ResizeToContents)
        self.table.cellClicked.connect(self._on_cell)
        self.table.setToolTip(tr("点某个格子 = 只看「这个缺陷 + 这个标记」的图"))
        self.table.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Expanding)
        self.table.setMinimumHeight(120)
        root.addWidget(self.table, 1)

    # ────────── 对外 ──────────
    def set_records(self, records):
        """records: [{base, defect, mark, note, pred_cnt, shape_count}, ...]"""
        self._records = list(records or [])
        cnt = mk.count_marks(self._records)
        for k in mk.MARK_KINDS:
            self._chips[k].setText("%s %d" % (tr(mk.MARK_LABELS[k]), cnt.get(k, 0)))
        marked = sum(cnt.get(k, 0) for k in mk.MARK_KINDS)
        self.lbl_total.setText(tr("共 %d 张　已标记 %d　无标记 %d")
                               % (len(self._records), marked, cnt.get("", 0)))
        if marked:
            cells = []
            for k, (n, r) in mk.mark_ratios(cnt).items():
                if n:
                    cells.append(tr("%s %d（%.1f%%）") % (tr(mk.MARK_LABELS[k]), n, r * 100))
            self.lbl_ratio.setText(tr("占比（分母=已标记）：") + tr("　").join(cells))
        else:
            self.lbl_ratio.setText("")

        bad = []
        for r in self._records:
            ok, why = mk.is_consistent(r.get("mark"), r.get("pred_cnt"), r.get("shape_count"))
            if not ok:
                bad.append(why)
        if bad:
            from collections import Counter
            top = Counter(bad).most_common(3)
            self.lbl_warn.setText(tr("⚠ 自相矛盾 %d 条：%s")
                                  % (len(bad), tr("；").join("%s×%d" % (t, n) for t, n in top)))
            self.lbl_warn.setVisible(True)
        else:
            self.lbl_warn.setVisible(False)

        self._fill_table()

    def _fill_table(self):
        table = mk.cross_table(self._records, type_key="defect")
        self.table.setRowCount(len(table))
        for i, (name, row) in enumerate(table.items()):
            it = QTableWidgetItem(str(name))
            it.setData(Qt.UserRole, name)
            self.table.setItem(i, 0, it)
            for j, k in enumerate(mk.MARK_KINDS):
                n = row.get(k, 0)
                cell = QTableWidgetItem(str(n) if n else "")
                cell.setTextAlignment(Qt.AlignCenter)
                if n:
                    cell.setToolTip(tr("%s ｜ %s：%d 张（点此筛选）") % (name, tr(mk.MARK_LABELS[k]), n))
                    cell.setForeground(_qcolor(mk.kind_color(k)))
                self.table.setItem(i, 1 + j, cell)
            n0 = row.get("", 0)
            c0 = QTableWidgetItem(str(n0) if n0 else "")
            c0.setTextAlignment(Qt.AlignCenter)
            self.table.setItem(i, 1 + len(mk.MARK_KINDS), c0)

    def _on_cell(self, row, col):
        if row < 0 or col < 1:
            # 点第 0 列（缺陷类型名）= 只看这个缺陷类型
            if row >= 0:
                it = self.table.item(row, 0)
                self.cell_filter_requested.emit((it.text() if it else ""), "")
            return
        it = self.table.item(row, 0)
        defect = it.text() if it else ""
        idx = col - 1
        if idx < len(mk.MARK_KINDS):
            self.cell_filter_requested.emit(defect, mk.MARK_KINDS[idx])
        else:
            self.cell_filter_requested.emit(defect, "")


def _qcolor(hexstr):
    from PyQt5.QtGui import QColor
    return QColor(hexstr)
