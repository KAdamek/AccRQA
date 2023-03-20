# See the LICENSE file at the top-level directory of this distribution.

import ctypes

class Error:
    error_codes = {
        0: "No error",
        1: "Generic runtime error",
        2: "Invalid function argument",
        3: "Unsupported data type(s)",
        4: "Memory allocation failure",
        5: "Memory copy failure",
        6: "Memory location mismatch"
    }

    def __init__(self):
        self._error = ctypes.c_int(0)

    def check(self):
        if self._error.value != 0:
            raise RuntimeError(Error.error_codes[self._error.value])

    def handle(self):
        return ctypes.byref(self._error)

    @staticmethod
    def handle_type():
        return ctypes.POINTER(ctypes.c_int)
