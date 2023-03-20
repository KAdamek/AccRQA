# See the LICENSE file at the top-level directory of this distribution.

import ctypes
import numpy
try:
    import cupy
except ImportError:
    cupy = None

from accrqa_error import Error
from accrqa_lib import Lib

class Mem:
    class Handle(ctypes.Structure):
        pass

    class MemType:
        MEM_CHAR = 1
        MEM_INT = 2
        MEM_FLOAT = 4
        MEM_DOUBLE = 8
        MEM_COMPLEX_FLOAT = 36
        MEM_COMPLEX_DOUBLE = 40

    class MemLocation:
        MEM_CPU = 0
        MEM_GPU = 1

    def __init__(self, *args):
        self._handle = None
        obj = args[0] if len(args) == 1 else None
        mem_create_wrapper = Lib.handle().mem_create_wrapper
        mem_create_wrapper.restype = Mem.handle_type()
        mem_create_wrapper.argtypes = [
            ctypes.c_void_p,
            ctypes.c_int,
            ctypes.c_int,
            ctypes.c_int32,
            ctypes.POINTER(ctypes.c_int64),
            ctypes.POINTER(ctypes.c_int64),
            Error.handle_type()
        ]
        mem_set_read_only = Lib.handle().mem_set_read_only
        mem_set_read_only.argtypes = [Mem.handle_type(), ctypes.c_int32]
        error_status = Error()
        if type(obj) == numpy.ndarray:
            if obj.dtype == numpy.int8 or obj.dtype == numpy.byte:
                mem_type = self.MemType.MEM_CHAR
            elif obj.dtype == numpy.int32:
                mem_type = self.MemType.MEM_INT
            elif obj.dtype == numpy.float32:
                mem_type = self.MemType.MEM_FLOAT
            elif obj.dtype == numpy.float64:
                mem_type = self.MemType.MEM_DOUBLE
            elif obj.dtype == numpy.complex64:
                mem_type = self.MemType.MEM_COMPLEX_FLOAT
            elif obj.dtype == numpy.complex128:
                mem_type = self.MemType.MEM_COMPLEX_DOUBLE
            else:
                raise TypeError("Unsupported type of numpy array")
            shape = (ctypes.c_int64 * obj.ndim)(*obj.shape)
            strides = (ctypes.c_int64 * obj.ndim)(*obj.strides)
            self._handle = mem_create_wrapper(
                obj.ctypes.data_as(ctypes.POINTER(ctypes.c_void_p)),
                mem_type,
                self.MemLocation.MEM_CPU,
                obj.ndim,
                shape,
                strides,
                error_status.handle()
            )
            mem_set_read_only(self._handle, not obj.flags.writeable)
        elif cupy:
            if type(obj) == cupy.ndarray:
                if obj.dtype == cupy.int8 or obj.dtype == cupy.byte:
                    mem_type = self.MemType.MEM_CHAR
                elif obj.dtype == cupy.int32:
                    mem_type = self.MemType.MEM_INT
                elif obj.dtype == cupy.float32:
                    mem_type = self.MemType.MEM_FLOAT
                elif obj.dtype == cupy.float64:
                    mem_type = self.MemType.MEM_DOUBLE
                elif obj.dtype == cupy.complex64:
                    mem_type = self.MemType.MEM_COMPLEX_FLOAT
                elif obj.dtype == cupy.complex128:
                    mem_type = self.MemType.MEM_COMPLEX_DOUBLE
                else:
                    raise TypeError("Unsupported type of cupy array")
                shape = (ctypes.c_int64 * obj.ndim)(*obj.shape)
                strides = (ctypes.c_int64 * obj.ndim)(*obj.strides)
                self._handle = mem_create_wrapper(
                    ctypes.cast(obj.data.ptr, ctypes.POINTER(ctypes.c_void_p)),
                    mem_type,
                    self.MemLocation.MEM_GPU,
                    obj.ndim,
                    shape,
                    strides,
                    error_status.handle()
                )
                # cupy doesn't appear to have a "writeable" flag.
                mem_set_read_only(self._handle, 0)
        if not self._handle and obj is not None:
            raise TypeError("Unknown array type")
        error_status.check()

    def __del__(self):
        if self._handle:
            mem_free = Lib.handle().mem_free
            mem_free.argtypes = [Mem.handle_type()]
            mem_free(self._handle)

    def handle(self):
        return self._handle

    @staticmethod
    def handle_type():
        return ctypes.POINTER(Mem.Handle)
