# -*- coding: utf-8 -*-
"""滚轮守卫：数字输入框/下拉框**不响应滚轮**，滚轮改为滚动外层面板。

为什么需要：Qt 里 `QSpinBox` / `QDoubleSpinBox` / `QComboBox` 默认吃滚轮事件，
鼠标只要滑过就改数值。放进滚动面板（右侧参数管理、训练设置对话框）后就成了灾难 ——
用户想上下滑面板，结果 Epochs 从 200 变成 199/201、模型从 unet 跳到别的（用户实测反馈：
"所有填数字的格子，滚轮在里面不要生效，很影响整个面板上下滑动"）。

做法：装一个应用级事件过滤器 —— 遇到这三类控件上的滚轮事件时，
先把它**转交给最近的滚动区**（面板照常滚动），再吞掉事件（数值不变）。
应用级安装的好处：以后新建的对话框、动态生成的控件都自动受保护，不用逐个改。
"""
from __future__ import annotations


def install(app):
    """在 QApplication 上安装滚轮守卫，返回过滤器对象（挂在 app 上防 GC）。"""
    from PyQt5.QtCore import QObject, QEvent
    from PyQt5.QtWidgets import (
        QAbstractSpinBox, QComboBox, QAbstractScrollArea, QApplication,
    )

    guard_types = (QAbstractSpinBox, QComboBox)   # 数字框 + 下拉框

    class _WheelGuard(QObject):
        def eventFilter(self, obj, ev):
            if ev.type() == QEvent.Wheel and isinstance(obj, guard_types):
                # ① 转交给最近的滚动区，让面板正常滚动
                p = obj.parentWidget()
                while p is not None and not isinstance(p, QAbstractScrollArea):
                    p = p.parentWidget()
                if p is not None and p.viewport() is not None:
                    try:
                        QApplication.sendEvent(p.viewport(), ev)
                    except Exception:
                        pass
                # ② 吞掉：控件自己不处理，数值不会被滚轮改掉
                return True
            return super().eventFilter(obj, ev)

    g = _WheelGuard(app)
    app.installEventFilter(g)
    app._wheel_guard = g          # 防止被垃圾回收
    return g
