AC_INIT([accrqa],[0.0.0])
AC_CONFIG_SRCDIR([src/])

VERSION_FILE="version.txt"

if test -f "$VERSION_FILE"; then
	ACCRQA_VERSION_STRING=$(cat $VERSION_FILE)
else
	AC_MSG_ERROR([version.txt not found])
fi

# Split version into parts
ACCRQA_VERSION_MAJOR=$(echo "$ACCRQA_VERSION_STRING" | cut -d. -f1)
ACCRQA_VERSION_MINOR=$(echo "$ACCRQA_VERSION_STRING" | cut -d. -f2)
ACCRQA_VERSION_PATCH=$(echo "$ACCRQA_VERSION_STRING" | cut -d. -f3)

AC_DEFINE_UNQUOTED([ACCRQA_VER_MAJOR], [$ACCRQA_VERSION_MAJOR], [Major version])
AC_DEFINE_UNQUOTED([ACCRQA_VER_MINOR], [$ACCRQA_VERSION_MINOR], [Minor version])
AC_DEFINE_UNQUOTED([ACCRQA_VER_PATCH], [$ACCRQA_VERSION_PATCH], [Patch version])

AC_SUBST([ACCRQA_VERSION_MAJOR])
AC_SUBST([ACCRQA_VERSION_MINOR])
AC_SUBST([ACCRQA_VERSION_PATCH])
AC_SUBST([ACCRQA_VERSION_STRING])



# Use system R for reliable config output (avoids R_check_bin wrapper issues)
SYS_R=`which R`

if test -x "$SYS_R"; then
  R_HOME=`R RHOME`
  R_INCL=`R CMD config --cppflags`
  R_LIB=`R CMD config --ldflags`
  AC_MSG_NOTICE([System R found: $SYS_R])
  AC_MSG_NOTICE([R_INCL flags: $R_INCL])
  ACCRQA_CXXFLAGS="-DACCRQA_R_FOUND"
else
  AC_MSG_ERROR([System R not found.])
fi

AC_SUBST(R_HOME)
AC_SUBST(R_INCL)
AC_SUBST(R_LIB)

# Check for CUDA
AC_MSG_CHECKING(environment variable CUDA_HOME)
if test -z "${CUDA_HOME}"; then
    CUDA_HOME=`find /usr/local/ -maxdepth 1 -type d -name "cuda*" | sort -V | tail -1`
    AC_MSG_RESULT(CUDA_HOME not set; using highest version found ${CUDA_HOME})
else
    AC_MSG_RESULT(using CUDA_HOME=${CUDA_HOME})
fi
AC_SUBST(CUDA_HOME)

AC_CHECK_FILE([${CUDA_HOME}/bin/nvcc], [HAS_NVCC="yes"], [HAS_NVCC="no"])

if test "$HAS_NVCC" = "yes"; then
    ACCRQA_CXXFLAGS="$ACCRQA_CXXFLAGS -DCUDA_FOUND"
    AC_MSG_RESULT([CUDA found and will be used])
    # Detect CUDA compute capability
    AC_MSG_CHECKING([for nvcc compute capability])
    NVCC_FLAGS="-arch=native"  # default fallback

    if command -v nvidia-smi > /dev/null; then
      GPU_ARCH=$(nvidia-smi --query-gpu=compute_cap --format=csv,noheader | head -n 1)
      GPU_ARCH_NUM=$(echo "$GPU_ARCH" | tr -d ".")
      NVCC_FLAGS="-arch=sm_${GPU_ARCH_NUM}"
    fi
  AC_MSG_RESULT([$NVCC_FLAGS])
  AC_SUBST(NVCC_FLAGS)
else
    AC_MSG_RESULT([CUDA not found. CUDA support will be disabled])
fi

AC_SUBST(ACCRQA_CXXFLAGS)


AC_CONFIG_FILES([src/Makevars])
AC_OUTPUT
