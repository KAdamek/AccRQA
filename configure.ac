AC_PREREQ([2.69])
AC_INIT([AccRQA],[0.9.9])
AC_CONFIG_SRCDIR([src])

dnl ============================================================
dnl  Version: read from version.txt and split into parts
dnl ============================================================

VERSION_FILE="version.txt"

if test -f "$VERSION_FILE"; then
  ACCRQA_VERSION_STRING=`cat "$VERSION_FILE"`
else
  AC_MSG_ERROR([version.txt not found])
fi

dnl Strip whitespace
ACCRQA_VERSION_STRING=`echo "$ACCRQA_VERSION_STRING" | tr -d '[:space:]'`

dnl Basic sanity check (allow digits/letters/dot/dash)
case "$ACCRQA_VERSION_STRING" in
  *[!0-9A-Za-z.\-]*)
    AC_MSG_ERROR([Invalid characters in version string: $ACCRQA_VERSION_STRING])
    ;;
esac

AC_MSG_NOTICE([Using AccRQA version: $ACCRQA_VERSION_STRING])

dnl Split version into major.minor.patch
ACCRQA_VERSION_MAJOR=`echo "$ACCRQA_VERSION_STRING" | cut -d. -f1`
ACCRQA_VERSION_MINOR=`echo "$ACCRQA_VERSION_STRING" | cut -d. -f2`
ACCRQA_VERSION_PATCH=`echo "$ACCRQA_VERSION_STRING" | cut -d. -f3`

AC_DEFINE_UNQUOTED([ACCRQA_VER_MAJOR], [$ACCRQA_VERSION_MAJOR], [Major version])
AC_DEFINE_UNQUOTED([ACCRQA_VER_MINOR], [$ACCRQA_VERSION_MINOR], [Minor version])
AC_DEFINE_UNQUOTED([ACCRQA_VER_PATCH], [$ACCRQA_VERSION_PATCH], [Patch version])

AC_SUBST([ACCRQA_VERSION_MAJOR])
AC_SUBST([ACCRQA_VERSION_MINOR])
AC_SUBST([ACCRQA_VERSION_PATCH])
AC_SUBST([ACCRQA_VERSION_STRING])

dnl ============================================================
dnl  Find R (minimal; you can expand if you later need flags)
dnl ============================================================

SYS_R=`which R 2>/dev/null`

if test -x "$SYS_R"; then
  R_INCL=""
  R_LIB=""
  AC_MSG_NOTICE([System R found: $SYS_R])
else
  AC_MSG_ERROR([System R not found.])
fi

AC_SUBST([R_INCL])
AC_SUBST([R_LIB])

dnl ============================================================
dnl  OpenMP detection via R (CRAN-safe)
dnl ============================================================

AC_MSG_CHECKING([for OpenMP flags from R])
OPENMP_CXXFLAGS=`"$SYS_R" CMD config SHLIB_OPENMP_CXXFLAGS 2>/dev/null`

if test -n "$OPENMP_CXXFLAGS"; then
  ACCRQA_USE_OPENMP=yes
  AC_MSG_RESULT([$OPENMP_CXXFLAGS])
else
  ACCRQA_USE_OPENMP=no
  OPENMP_CXXFLAGS=""
  AC_MSG_RESULT([none])
fi

AC_MSG_NOTICE([Use OpenMP: $ACCRQA_USE_OPENMP])
AC_MSG_NOTICE([OpenMP CXX flags: $OPENMP_CXXFLAGS])

AC_SUBST([ACCRQA_USE_OPENMP])
AC_SUBST([OPENMP_CXXFLAGS])

dnl ============================================================
dnl  CUDA / nvcc detection (optional)
dnl ============================================================

AC_ARG_WITH([cuda],
  [AS_HELP_STRING([--with-cuda@<:@=PATH@:>@],
                  [build with CUDA GPU support if available (default=yes). Optionally provide CUDA prefix PATH])],
  [with_cuda="$withval"],
  [with_cuda="yes"])

AC_ARG_WITH([gpu-arch],
  [AS_HELP_STRING([--with-gpu-arch=XX],
                  [set CUDA GPU arch, e.g. 80 for sm_80])],
  [GPU_ARCH="$withval"],
  [GPU_ARCH=""])

AC_MSG_CHECKING([whether to use CUDA])
AC_MSG_RESULT([$with_cuda])

ACCRQA_USE_CUDA="no"
CUDA_HOME="${CUDA_HOME-""}"
NVCC_FLAGS=""
NVCC=""

dnl Extra flags that you may want to pass into C++ compilation
ACCRQA_CXXFLAGS="-DACCRQA_R_FOUND"

if test "x$with_cuda" = "xno"; then
  AC_MSG_NOTICE([CUDA explicitly disabled])
  CUDA_HOME=""
  NVCC=""
  ACCRQA_USE_CUDA="no"
else
  dnl If user passed a directory: --with-cuda=/path
  if test "x$with_cuda" != "xyes"; then
    CUDA_HOME="$with_cuda"
  fi

  dnl Locate nvcc (use CUDA_HOME as a hint if set)
  if test -n "$CUDA_HOME"; then
    AC_PATH_TOOL([NVCC],[nvcc],[no],[$CUDA_HOME/bin:$PATH])
  else
    AC_PATH_TOOL([NVCC],[nvcc],[no])
  fi

  if test "x$NVCC" = "xno"; then
    AC_MSG_WARN([nvcc not found; building AccRQA without GPU support])
    NVCC=""
    CUDA_HOME=""
    ACCRQA_USE_CUDA="no"
  else
    dnl If CUDA_HOME still unknown, search typical prefixes by header
    if test -z "$CUDA_HOME"; then
      for d in /usr/local/cuda /usr; do
        if test -f "$d/include/cuda_runtime_api.h"; then
          CUDA_HOME="$d"
          break
        fi
      done
    fi

    dnl Only enable CUDA if we actually see cuda_runtime_api.h
    if test -n "$CUDA_HOME" && test -f "$CUDA_HOME/include/cuda_runtime_api.h"; then
      ACCRQA_USE_CUDA="yes"
      AC_MSG_NOTICE([nvcc found: $NVCC])
      AC_MSG_NOTICE([CUDA home: $CUDA_HOME])

      dnl Enable macro for C++ code:
      ACCRQA_CXXFLAGS="$ACCRQA_CXXFLAGS -DCUDA_FOUND"

      dnl Optional: GPU arch flag
      if test -n "$GPU_ARCH"; then
        GPU_ARCH_NUM=`echo "$GPU_ARCH" | tr -d "."`
        NVCC_FLAGS="-arch=sm_${GPU_ARCH_NUM}"
      else
        NVCC_FLAGS=""
        AC_MSG_WARN([no --with-gpu-arch given; nvcc will use default arch])
      fi
    else
      AC_MSG_WARN([cuda_runtime_api.h not found; building AccRQA without GPU support])
      NVCC=""
      CUDA_HOME=""
      ACCRQA_USE_CUDA="no"
    fi
  fi
fi

AC_MSG_NOTICE([Use CUDA: $ACCRQA_USE_CUDA])
AC_MSG_NOTICE([NVCC flags: $NVCC_FLAGS])
AC_MSG_NOTICE([CXX flags (extra): $ACCRQA_CXXFLAGS])

AC_SUBST([CUDA_HOME])
AC_SUBST([NVCC])
AC_SUBST([NVCC_FLAGS])
AC_SUBST([ACCRQA_USE_CUDA])
AC_SUBST([ACCRQA_CXXFLAGS])

dnl ============================================================
dnl  Compute portable Makevars substitutions:
dnl   - CUDA_CXXFLAGS, CUDA_LIBS
dnl   - OBJECTS list (from src/*.cpp and src/*.cu when enabled)
dnl ============================================================

CUDA_CXXFLAGS=""
CUDA_LIBS=""

if test "x$ACCRQA_USE_CUDA" = "xyes"; then
  CUDA_CXXFLAGS="-I$CUDA_HOME/include -DCUDA_FOUND"
  CUDA_LIBS="-L$CUDA_HOME/lib64 -lcudart -lcuda -lcublas -lcurand"
fi

OBJECTS=""

dnl CPU objects from src/*.cpp
for f in "$srcdir"/src/*.cpp; do
  if test -f "$f"; then
    b=`basename "$f" .cpp`
    OBJECTS="$OBJECTS $b.o"
  fi
done

dnl CUDA objects from src/*.cu (only if enabled)
if test "x$ACCRQA_USE_CUDA" = "xyes"; then
  for f in "$srcdir"/src/*.cu; do
    if test -f "$f"; then
      b=`basename "$f" .cu`
      OBJECTS="$OBJECTS $b.o"
    fi
  done
fi

dnl Trim leading whitespace
OBJECTS=`echo "$OBJECTS" | sed 's/^ *//'`

AC_SUBST([CUDA_CXXFLAGS])
AC_SUBST([CUDA_LIBS])
AC_SUBST([OBJECTS])

dnl ============================================================
dnl  Generate src/Makevars from src/Makevars.in
dnl ============================================================

AC_CONFIG_FILES([src/Makevars])
AC_OUTPUT

