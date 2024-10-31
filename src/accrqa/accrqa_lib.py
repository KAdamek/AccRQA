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

class accrqaLib:
    name = "libaccrqa"
    env_name = "ACCRQA_LIB_DIR"
    this_dir = os.path.dirname(os.path.abspath(__file__))
    package_root = os.path.join(this_dir, "..")
    search_dirs = [".", package_root, "/usr/local/lib"]
    lib = None
    mutex = threading.Lock()

    @staticmethod
    def handle():
        if not accrqaLib.lib:
            lib_dir = accrqaLib.find_dir(accrqaLib.search_dirs)
            accrqaLib.mutex.acquire()
            if not accrqaLib.lib:
                try:
                    accrqaLib.lib = numpy.ctypeslib.load_library(accrqaLib.name, lib_dir)
                except OSError:
                    pass
            accrqaLib.mutex.release()
            if not accrqaLib.lib:
                raise RuntimeError(
                    f"Cannot find {accrqaLib.name} in {accrqaLib.search_dirs}. "
                    f"Try setting the environment variable {accrqaLib.env_name}"
                )
        return accrqaLib.lib

    @staticmethod
    def find_dir(lib_search_dirs):
        # Try to find the shared library in the listed directories.
        lib_dir = ""
        env_dir = os.environ.get(accrqaLib.env_name)
        if env_dir and env_dir not in accrqaLib.search_dirs:
            lib_search_dirs.insert(0, env_dir)
        for test_dir in lib_search_dirs:
            if glob.glob(os.path.join(test_dir, accrqaLib.name) + "*"):
                lib_dir = test_dir
                break
        return lib_dir
