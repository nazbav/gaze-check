"""Камеры DirectShow по порядку — те же номера, что у cv2.VideoCapture(номер, cv2.CAP_DSHOW).

Имена берутся из системного перечислителя устройств (COM через ctypes, без лишних пакетов).
ИК-камеры Windows Hello здесь нет — она видна только через WinRT.
"""
import ctypes
from ctypes import POINTER, byref, c_ulong, c_ushort, c_void_p, c_wchar_p

CLSID_SYSTEM_DEVICE_ENUM = "{62BE5D10-60EB-11d0-BD3B-00A0C911CE86}"
IID_ICREATE_DEV_ENUM = "{29840822-5B84-11D0-BD3B-00A0C911CE86}"
CLSID_VIDEO_INPUT_CATEGORY = "{860BB310-5D01-11d0-BD3B-00A0C911CE86}"
IID_IPROPERTY_BAG = "{55272A00-42CB-11CE-8135-00AA004BB851}"
VT_BSTR = 8


class GUID(ctypes.Structure):
    _fields_ = [("d1", ctypes.c_uint32), ("d2", ctypes.c_uint16), ("d3", ctypes.c_uint16), ("d4", ctypes.c_ubyte * 8)]

    @classmethod
    def of(cls, text):
        guid = cls()
        ctypes.oledll.ole32.CLSIDFromString(c_wchar_p(text), byref(guid))
        return guid


class VARIANT(ctypes.Structure):  # 24 байта на x64: тип, 3 запасных слова, значение
    _fields_ = [("vt", c_ushort), ("r1", c_ushort), ("r2", c_ushort), ("r3", c_ushort),
                ("val", c_void_p), ("pad", c_void_p)]


def _method(obj, index, *argtypes):
    vtbl = ctypes.cast(obj, POINTER(POINTER(c_void_p)))[0]
    return ctypes.WINFUNCTYPE(ctypes.HRESULT, c_void_p, *argtypes)(vtbl[index])


def _release(obj):
    if obj:
        vtbl = ctypes.cast(obj, POINTER(POINTER(c_void_p)))[0]
        ctypes.WINFUNCTYPE(c_ulong, c_void_p)(vtbl[2])(obj)


def list_cameras():
    """→ [(номер, имя)]; пусто, если перечислить не вышло (не Windows, камер нет)."""
    try:
        ole32 = ctypes.windll.ole32
    except AttributeError:
        return []
    inited = ole32.CoInitializeEx(None, 0x2) in (0, 1)  # поток уже в другом режиме COM — тоже работает
    names, dev_enum, enum = [], c_void_p(), c_void_p()
    try:
        ctypes.oledll.ole32.CoCreateInstance(byref(GUID.of(CLSID_SYSTEM_DEVICE_ENUM)), None, 1,
                                             byref(GUID.of(IID_ICREATE_DEV_ENUM)), byref(dev_enum))
        create = _method(dev_enum, 3, POINTER(GUID), POINTER(c_void_p), c_ulong)  # CreateClassEnumerator
        if create(dev_enum, byref(GUID.of(CLSID_VIDEO_INPUT_CATEGORY)), byref(enum), 0) != 0 or not enum:
            return []  # S_FALSE — камер нет
        nxt = _method(enum, 3, c_ulong, POINTER(c_void_p), POINTER(c_ulong))  # IEnumMoniker::Next
        while True:
            mon, got = c_void_p(), c_ulong()
            if nxt(enum, 1, byref(mon), byref(got)) != 0 or not got.value:
                break
            bag, name = c_void_p(), None
            try:
                _method(mon, 9, c_void_p, c_void_p, POINTER(GUID), POINTER(c_void_p))(  # BindToStorage
                    mon, None, None, byref(GUID.of(IID_IPROPERTY_BAG)), byref(bag))
                var = VARIANT()
                _method(bag, 3, c_wchar_p, POINTER(VARIANT), c_void_p)(bag, "FriendlyName", byref(var), None)
                if var.vt == VT_BSTR and var.val:
                    name = ctypes.wstring_at(var.val)
                ctypes.windll.oleaut32.VariantClear(byref(var))
            except OSError:
                pass
            finally:
                _release(bag)
                _release(mon)
            names.append(name or "Камера %d" % len(names))
    except OSError:
        pass
    finally:
        _release(enum)
        _release(dev_enum)
        if inited:
            ole32.CoUninitialize()
    return list(enumerate(names))
