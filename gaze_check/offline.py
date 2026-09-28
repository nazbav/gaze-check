"""Без сети. MediaPipe 1.0 сам отправляет статистику использования в Google (play.googleapis.com/log,
через WinINet), выключателя у неё нет. Программа никуда ничего не передаёт: сразу после загрузки
библиотеки MediaPipe её вызовы WinINet (таблица импорта) подменяются:
- HttpSendRequestA — единственный вызов, который что-то передаёт, — сразу «не удалось соединиться»;
- InternetReadFile — «нечего читать»;
- InternetOpenA — настоящая сессия, но через прокси 127.0.0.1:9, где никого нет (второй рубеж).
Сессию открыть надо: без неё MediaPipe падает (CHECK «Couldn't open a WININET session»).
Действует только внутри этого процесса и только на MediaPipe; прав администратора не нужно."""
import ctypes
import struct
import sys
import time

ERROR_INTERNET_CANNOT_CONNECT = 12029
INTERNET_OPEN_TYPE_PROXY = 3
DEAD_PROXY = b"127.0.0.1:9"  # порт discard: на нём никто не слушает — соединение отклоняется сразу
PAGE_READWRITE = 0x04
_stubs = {}  # имя функции → заглушка (ссылка держит её живой)
attempts = []  # какие отправки библиотека пыталась сделать


def _note(name):
    if not attempts:
        print(time.strftime("%H:%M:%S"), "MediaPipe пытался отправить данные (%s) — не пускаю" % name, flush=True)
    attempts.append(name)


def _refuse(name):
    # x64: аргументы в регистрах, стек чистит вызывающий — описывать их не нужно
    @ctypes.WINFUNCTYPE(ctypes.c_void_p)
    def stub():
        _note(name)
        ctypes.windll.kernel32.SetLastError(ERROR_INTERNET_CANNOT_CONNECT)
        return 0
    return stub


def _open_via_dead_proxy():
    real = ctypes.windll.wininet.InternetOpenA
    real.argtypes = [ctypes.c_char_p, ctypes.c_ulong, ctypes.c_char_p, ctypes.c_char_p, ctypes.c_ulong]
    real.restype = ctypes.c_void_p

    @ctypes.WINFUNCTYPE(ctypes.c_void_p, ctypes.c_char_p, ctypes.c_ulong, ctypes.c_char_p, ctypes.c_char_p,
                        ctypes.c_ulong)
    def stub(agent, _access, _proxy, _bypass, flags):
        return real(agent, INTERNET_OPEN_TYPE_PROXY, DEAD_PROXY, None, flags)
    return stub


REPLACE = {"HttpSendRequestA": lambda: _refuse("HttpSendRequestA"),
           "InternetReadFile": lambda: _refuse("InternetReadFile"),
           "InternetOpenA": _open_via_dead_proxy}


def block_network(handle, dll="WININET.dll"):
    """Подменить в загруженном модуле (handle) сетевые функции из dll (см. REPLACE).
    → [имена подменённых]. Не Windows или не 64 бита — ничего не делает."""
    if sys.platform != "win32" or struct.calcsize("P") != 8 or not handle:
        return []
    k32 = ctypes.windll.kernel32
    k32.VirtualProtect.argtypes = [ctypes.c_void_p, ctypes.c_size_t, ctypes.c_ulong, ctypes.POINTER(ctypes.c_ulong)]

    def u32(addr):
        return ctypes.c_uint32.from_address(addr).value

    base = handle
    pe = base + u32(base + 0x3C)
    imports = u32(pe + 24 + 120)  # PE32+: каталог импорта — второй в DataDirectory
    done, desc = [], base + imports
    while imports:
        names, _, _, name_rva, iat = struct.unpack("<IIIII", ctypes.string_at(desc, 20))
        if not name_rva:
            break
        if ctypes.string_at(base + name_rva).decode("ascii", "replace").lower() == dll.lower():
            i = 0
            while True:
                thunk = ctypes.c_uint64.from_address(base + (names or iat) + 8 * i).value
                if not thunk:
                    break
                name = None if thunk >> 63 else ctypes.string_at(base + (thunk & 0x7FFFFFFF) + 2).decode("ascii")
                if name in REPLACE:
                    stub = _stubs.setdefault(name, REPLACE[name]())
                    slot, old = base + iat + 8 * i, ctypes.c_ulong()
                    target = ctypes.cast(stub, ctypes.c_void_p).value
                    if ctypes.c_uint64.from_address(slot).value == target:  # уже подменено
                        done.append(name)
                    elif k32.VirtualProtect(slot, 8, PAGE_READWRITE, ctypes.byref(old)):
                        ctypes.c_uint64.from_address(slot).value = target
                        k32.VirtualProtect(slot, 8, old.value, ctypes.byref(old))
                        done.append(name)
                i += 1
        desc += 20
    return done
