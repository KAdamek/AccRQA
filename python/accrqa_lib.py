# See the LICENSE file at the top-level directory of this distribution.

import glob
import os
import threading

import numpy

# We don't want a module-level variable for the library handle,
# as that would mean the documentation doesn't build properly if the library
# can't be found when it gets imported - and unfortunately nested packages
# trigger a bug with mock imports. We use this static class instead
# to hold the library handle.

class Lib:
    name = "libAccRQA"
    env_name = "ACCRQA_LIB_DIR"
    search_dirs = [".", "/usr/local/lib"]
    lib = None
    mutex = threading.Lock()

    @staticmethod
    def handle():
        if not Lib.lib:
            lib_dir = Lib.find_dir(Lib.search_dirs)
            Lib.mutex.acquire()
            if not Lib.lib:
                try:
                    Lib.lib = numpy.ctypeslib.load_library(Lib.name, lib_dir)
                except OSError:
                    pass
            Lib.mutex.release()
            if not Lib.lib:
                raise RuntimeError(
                    f"Cannot find {Lib.name} in {Lib.search_dirs}. "
                    f"Try setting the environment variable {Lib.env_name}"
                )
        return Lib.lib

    @staticmethod
    def find_dir(lib_search_dirs):
        # Try to find the shared library in the listed directories.
        lib_dir = ""
        env_dir = os.environ.get(Lib.env_name)
        if env_dir and env_dir not in Lib.search_dirs:
            lib_search_dirs.insert(0, env_dir)
        for test_dir in lib_search_dirs:
            if glob.glob(os.path.join(test_dir, Lib.name) + "*"):
                lib_dir = test_dir
                break
        return lib_dir
