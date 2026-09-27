"""Камеры DirectShow по порядку — те же номера, что у cv2.VideoCapture(номер, cv2.CAP_DSHOW).

Имена берутся из системного перечислителя устройств (COM через ctypes, без лишних пакетов).
ИК-камеры Windows Hello здесь нет — она видна только через WinRT. Режимы камеры (разрешение и частота)
берутся у её пина захвата (IAMStreamConfig).
"""
import ctypes
from ctypes import POINTER, byref, c_ulong, c_ushort, c_void_p, c_wchar_p

CLSID_SYSTEM_DEVICE_ENUM = "{62BE5D10-60EB-11d0-BD3B-00A0C911CE86}"
IID_ICREATE_DEV_ENUM = "{29840822-5B84-11D0-BD3B-00A0C911CE86}"
CLSID_VIDEO_INPUT_CATEGORY = "{860BB310-5D01-11d0-BD3B-00A0C911CE86}"
IID_IPROPERTY_BAG = "{55272A00-42CB-11CE-8135-00AA004BB851}"
IID_IBASE_FILTER = "{56A86895-0AD4-11CE-B03A-0020AF0BA770}"
IID_IAM_STREAM_CONFIG = "{C6E13340-30AC-11D0-A18C-00A0C9118956}"
FORMAT_VIDEO_INFO = "{05589F80-C356-11CE-BF01-00AA0055595A}"
FORMAT_VIDEO_INFO2 = "{F72A76A0-EB0A-11D0-ACE4-0000C0CC16BA}"
VT_BSTR = 8
PINDIR_OUTPUT = 1
MIN_MODE_HEIGHT = 480  # ниже — глаза в кадре слишком мелкие для взгляда


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


class AM_MEDIA_TYPE(ctypes.Structure):
    _fields_ = [("majortype", GUID), ("subtype", GUID), ("fixed", ctypes.c_int), ("temporal", ctypes.c_int),
                ("sample_size", c_ulong), ("formattype", GUID), ("unk", c_void_p), ("cb_format", c_ulong),
                ("pb_format", c_void_p)]


def _method(obj, index, *argtypes):
    vtbl = ctypes.cast(obj, POINTER(POINTER(c_void_p)))[0]
    return ctypes.WINFUNCTYPE(ctypes.HRESULT, c_void_p, *argtypes)(vtbl[index])


def _release(obj):
    if obj:
        vtbl = ctypes.cast(obj, POINTER(POINTER(c_void_p)))[0]
        ctypes.WINFUNCTYPE(c_ulong, c_void_p)(vtbl[2])(obj)


def _monikers(visit):
    """visit(номер, моникер камеры) для каждой камеры DirectShow по порядку; моникер освобождается после."""
    try:
        ole32 = ctypes.windll.ole32
    except AttributeError:
        return
    inited = ole32.CoInitializeEx(None, 0x2) in (0, 1)  # поток уже в другом режиме COM — тоже работает
    dev_enum, enum = c_void_p(), c_void_p()
    try:
        ctypes.oledll.ole32.CoCreateInstance(byref(GUID.of(CLSID_SYSTEM_DEVICE_ENUM)), None, 1,
                                             byref(GUID.of(IID_ICREATE_DEV_ENUM)), byref(dev_enum))
        create = _method(dev_enum, 3, POINTER(GUID), POINTER(c_void_p), c_ulong)  # CreateClassEnumerator
        if create(dev_enum, byref(GUID.of(CLSID_VIDEO_INPUT_CATEGORY)), byref(enum), 0) != 0 or not enum:
            return  # S_FALSE — камер нет
        nxt = _method(enum, 3, c_ulong, POINTER(c_void_p), POINTER(c_ulong))  # IEnumMoniker::Next
        index = 0
        while True:
            mon, got = c_void_p(), c_ulong()
            if nxt(enum, 1, byref(mon), byref(got)) != 0 or not got.value:
                break
            try:
                visit(index, mon)
            finally:
                _release(mon)
            index += 1
    except OSError:
        pass
    finally:
        _release(enum)
        _release(dev_enum)
        if inited:
            ole32.CoUninitialize()


def list_cameras():
    """→ [(номер, имя)]; пусто, если перечислить не вышло (не Windows, камер нет)."""
    names = []

    def visit(index, mon):
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
        names.append(name or "Камера %d" % index)

    _monikers(visit)
    return list(enumerate(names))


def _stream_caps(cfg):
    """IAMStreamConfig → [(ширина, высота, наибольшая частота)] по всем форматам пина."""
    count, size = ctypes.c_int(), ctypes.c_int()
    _method(cfg, 5, POINTER(ctypes.c_int), POINTER(ctypes.c_int))(cfg, byref(count), byref(size))  # GetNumberOfCapabilities
    get = _method(cfg, 6, ctypes.c_int, POINTER(POINTER(AM_MEDIA_TYPE)), c_void_p)  # GetStreamCaps
    info, info2 = bytes(GUID.of(FORMAT_VIDEO_INFO)), bytes(GUID.of(FORMAT_VIDEO_INFO2))
    out = []
    for i in range(count.value):
        pmt, caps = POINTER(AM_MEDIA_TYPE)(), ctypes.create_string_buffer(max(128, size.value))
        if get(cfg, i, byref(pmt), caps) != 0 or not pmt:
            continue
        mt = pmt.contents
        try:
            kind = bytes(mt.formattype)
            if mt.pb_format and kind in (info, info2):
                head = 48 if kind == info else 72  # BITMAPINFOHEADER в VIDEOINFOHEADER / VIDEOINFOHEADER2
                if mt.cb_format >= head + 12:
                    w = ctypes.c_int32.from_address(mt.pb_format + head + 4).value
                    h = abs(ctypes.c_int32.from_address(mt.pb_format + head + 8).value)
                    interval = ctypes.c_int64.from_buffer_copy(caps.raw[104:112]).value  # MinFrameInterval
                    if interval <= 0:
                        interval = ctypes.c_int64.from_address(mt.pb_format + 40).value  # AvgTimePerFrame
                    if w > 0 and h > 0:
                        out.append((w, h, round(1e7 / interval) if interval > 0 else 0))
        finally:
            if mt.pb_format:
                ctypes.windll.ole32.CoTaskMemFree(c_void_p(mt.pb_format))
            _release(c_void_p(mt.unk))
            ctypes.windll.ole32.CoTaskMemFree(pmt)
    return out


def list_modes(camera):
    """Режимы камеры номер camera → [(ширина, высота, наибольшая частота)], крупные первыми; режимы
    ниже MIN_MODE_HEIGHT — только если других нет. Пусто — перечислить не вышло. Камеру не занимает:
    работает и когда она уже снимает."""
    found = []

    def visit(index, mon):
        if index != camera:
            return
        filt, pins = c_void_p(), c_void_p()
        try:
            _method(mon, 8, c_void_p, c_void_p, POINTER(GUID), POINTER(c_void_p))(  # BindToObject
                mon, None, None, byref(GUID.of(IID_IBASE_FILTER)), byref(filt))
            _method(filt, 10, POINTER(c_void_p))(filt, byref(pins))  # IBaseFilter::EnumPins
            nxt = _method(pins, 3, c_ulong, POINTER(c_void_p), POINTER(c_ulong))
            while not found:
                pin, got = c_void_p(), c_ulong()
                if nxt(pins, 1, byref(pin), byref(got)) != 0 or not got.value:
                    break
                cfg = c_void_p()
                try:
                    direction = ctypes.c_int()
                    _method(pin, 9, POINTER(ctypes.c_int))(pin, byref(direction))  # QueryDirection
                    if direction.value != PINDIR_OUTPUT:
                        continue
                    # первый выходной пин с форматами — «Capture» (у «Still» свои снимки)
                    if _method(pin, 0, POINTER(GUID), POINTER(c_void_p))(
                            pin, byref(GUID.of(IID_IAM_STREAM_CONFIG)), byref(cfg)) == 0 and cfg:
                        found.extend(_stream_caps(cfg))
                except OSError:
                    pass
                finally:
                    _release(cfg)
                    _release(pin)
        except OSError:
            pass
        finally:
            _release(pins)
            _release(filt)

    _monikers(visit)
    best = {}
    for w, h, fps in found:
        best[(w, h)] = max(best.get((w, h), 0), fps)
    modes = sorted(((w, h, fps) for (w, h), fps in best.items()), key=lambda m: (-m[0] * m[1], -m[2]))
    return [m for m in modes if m[1] >= MIN_MODE_HEIGHT] or modes

