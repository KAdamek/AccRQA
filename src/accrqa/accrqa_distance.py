# See the LICENSE file at the top-level directory of this distribution.

import ctypes

class accrqaDistance:
    distance_types = {
        1: "euclidean",
        2: "maximal"
    }

    def __init__(self):
        self._distance = ctypes.c_int(0)

    def handle(self):
        return ctypes.byref(self._distance)

    @staticmethod
    def handle_type():
        return ctypes.POINTER(ctypes.c_int)
