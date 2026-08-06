# -*- coding: utf-8 -*-
"""Windows 全局热键封装（ctypes，无第三方依赖）。

RegisterHotKey 注册的热键在窗口未聚焦、甚至窗口隐藏时也会触发，
消息会投递到注册时指定的窗口句柄（WM_HOTKEY）。
"""
import ctypes
from ctypes import wintypes

WM_HOTKEY = 0x0312

MOD_ALT = 0x0001
MOD_CONTROL = 0x0002
MOD_SHIFT = 0x0004
MOD_WIN = 0x0008
MOD_NOREPEAT = 0x4000  # 按住不连发（Win7+）

VK_X = 0x58


def register_global_hotkey(hwnd, hotkey_id,
                           modifiers=MOD_SHIFT | MOD_NOREPEAT, vk=VK_X):
    """注册全局热键到指定窗口句柄，返回是否成功。"""
    user32 = ctypes.windll.user32
    user32.RegisterHotKey.argtypes = [wintypes.HWND, ctypes.c_int,
                                      wintypes.UINT, wintypes.UINT]
    user32.RegisterHotKey.restype = wintypes.BOOL
    res = user32.RegisterHotKey(wintypes.HWND(int(hwnd)), int(hotkey_id),
                                wintypes.UINT(modifiers), wintypes.UINT(vk))
    return bool(res)


def unregister_global_hotkey(hwnd, hotkey_id):
    """注销全局热键。"""
    user32 = ctypes.windll.user32
    user32.UnregisterHotKey.argtypes = [wintypes.HWND, ctypes.c_int]
    user32.UnregisterHotKey.restype = wintypes.BOOL
    res = user32.UnregisterHotKey(wintypes.HWND(int(hwnd)), int(hotkey_id))
    return bool(res)