# See the LICENSE file at the top-level directory of this distribution.

import ctypes

class accrqaError:
    error_codes = {
        0: "AccRQA: No error",
        1: "AccRQA: Generic runtime error",
        2: "AccRQA: Invalid function argument",
        3: "AccRQA: Unsupported data type(s)",
        4: "AccRQA: Memory allocation failure",
        5: "AccRQA: Memory copy failure",
        6: "AccRQA: Memory location mismatch"
    }

    def __init__(self):
        self._error = ctypes.c_int(0)

    def check(self):
        if self._error.value != 0:
            raise RuntimeError(accrqaError.error_codes[self._error.value])

    def handle(self):
        return ctypes.byref(self._error)

    @staticmethod
    def handle_type():
        return ctypes.POINTER(ctypes.c_int)
