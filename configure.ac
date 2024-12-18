AC_INIT([accrqa],[1.0.0.])
AC_CONFIG_SRCDIR([src/])

AC_MSG_CHECKING(environment variable CUDA_HOME)
if test -z "${CUDA_HOME}"; then
	CUDA_HOME=`find /usr/local/ -maxdepth 1 -type d -name "cuda*" | sort -V | tail -1`
	AC_MSG_RESULT(CUDA_HOME not set; using highest version found ${CUDA_HOME})
else
	AC_MSG_RESULT(using CUDA_HOME=${CUDA_HOME})
fi

AC_CHECK_FILE([${CUDA_HOME}/bin/nvcc], [HAS_NVCC="yes"])
if test "$HAS_NVCC" != "no"; then
    AC_MSG_RESULT([Adding to compile R bindings with cuda])
    ACCRQA_CXXFLAGS="-DACCRQA_R_FOUND"
else
    AC_MSG_RESULT([not found])
    ACCRQA_CXXFLAGS=""
fi

# Substitute into Makevars
AC_SUBST(ACCRQA_CXXFLAGS)

# Detect CUDA compute capability
AC_MSG_CHECKING([for nvcc compute capability])
# Use nvidia-smi or a default compute capability
NVCC_FLAGS="-arch=sm_50" # Default
if command -v nvidia-smi > /dev/null; then
    GPU_ARCH=$(nvidia-smi --query-gpu=compute_cap --format=csv,noheader | head -n 1)
    GPU_ARCH_NUM=$(echo "$GPU_ARCH" | tr -d ".")
    NVCC_FLAGS="-arch=sm_${GPU_ARCH_NUM}"
fi
AC_MSG_RESULT([$NVCC_FLAGS])
AC_SUBST(NVCC_FLAGS)


## look for Rscript, but use the one found via R_HOME to allow for multiple installations
# Detect R home
R_HOME=$(R RHOME)
R_INCL=$(R CMD config --cppflags)
R_LIB=$(R CMD config --ldflags)
AC_SUBST(R_HOME)

AC_SUBST(CUDA_HOME)
AC_SUBST(R_INCL)
AC_SUBST(R_LIB)
AC_CONFIG_FILES([src/Makevars])

# Generate .gitignore from .gitignore.in
AC_SUBST([CUSTOM_IGNORE_RULES], [
# Additional ignore rules
*.log
*.tmp
build/
])

AC_CONFIG_FILES([
.gitignore
])
AC_OUTPUT


