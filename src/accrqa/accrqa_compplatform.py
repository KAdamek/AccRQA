# See the LICENSE file at the top-level directory of this distribution.

import ctypes

class accrqaCompPlatform:
    error_codes = {
        1: "cpu",
        1024: "nv_gpu"
    }

    def __init__(self):
        self._plt = ctypes.c_int(0)

    def handle(self):
        return ctypes.byref(self._plt)

    @staticmethod
    def handle_type():
        return ctypes.POINTER(ctypes.c_int)
