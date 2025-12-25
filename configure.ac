AC_PREREQ([2.69])
AC_INIT([AccRQA],[0.9.9])
AC_CONFIG_SRCDIR([src])
AC_LANG([C++])

dnl ============================================================
dnl  Version handling
dnl ============================================================

VERSION_FILE="version.txt"

if test -f "$VERSION_FILE"; then
  ACCRQA_VERSION_STRING=`cat "$VERSION_FILE" | tr -d '[:space:]'`
else
  AC_MSG_ERROR([version.txt not found])
fi

case "$ACCRQA_VERSION_STRING" in
  *[!0-9A-Za-z.\-]*)
    AC_MSG_ERROR([Invalid characters in version string: $ACCRQA_VERSION_STRING])
    ;;
esac

AC_MSG_NOTICE([Using AccRQA version: $ACCRQA_VERSION_STRING])

ACCRQA_VERSION_MAJOR=`echo "$ACCRQA_VERSION_STRING" | cut -d. -f1`
ACCRQA_VERSION_MINOR=`echo "$ACCRQA_VERSION_STRING" | cut -d. -f2`
ACCRQA_VERSION_PATCH=`echo "$ACCRQA_VERSION_STRING" | cut -d. -f3`

AC_SUBST([ACCRQA_VERSION_MAJOR])
AC_SUBST([ACCRQA_VERSION_MINOR])
AC_SUBST([ACCRQA_VERSION_PATCH])
AC_SUBST([ACCRQA_VERSION_STRING])

dnl ============================================================
dnl  Find R
dnl ============================================================

SYS_R=`which R 2>/dev/null`
if test -x "$SYS_R"; then
  AC_MSG_NOTICE([System R found: $SYS_R])
else
  AC_MSG_ERROR([System R not found])
fi

R_INCL=""
R_LIB=""
AC_SUBST([R_INCL])
AC_SUBST([R_LIB])

dnl ============================================================
dnl  Ensure C++ compiler is detected early (clean output)
dnl ============================================================

AC_PROG_CXX

dnl ============================================================
dnl  OpenMP detection (portable compile+link test)
dnl ============================================================

OPENMP_CXXFLAGS=""
OPENMP_LIBS=""
ACCRQA_USE_OPENMP=no

AC_LANG_PUSH([C++])
AC_MSG_CHECKING([for OpenMP support])

save_CXXFLAGS="$CXXFLAGS"
save_LIBS="$LIBS"

dnl 1) GCC / Clang / mingw-w64
CXXFLAGS="$save_CXXFLAGS -fopenmp"
LIBS="$save_LIBS"
AC_LINK_IFELSE(
  [AC_LANG_PROGRAM([[#include <omp.h>]],
                   [[int n = omp_get_max_threads(); return (n>0)?0:1;]])],
  [ACCRQA_USE_OPENMP=yes; OPENMP_CXXFLAGS="-fopenmp"]
)

dnl 2) Apple Clang + libomp
if test "x$ACCRQA_USE_OPENMP" = "xno"; then
  CXXFLAGS="$save_CXXFLAGS -Xpreprocessor -fopenmp"
  LIBS="$save_LIBS -lomp"
  AC_LINK_IFELSE(
    [AC_LANG_PROGRAM([[#include <omp.h>]],
                     [[int n = omp_get_max_threads(); return (n>0)?0:1;]])],
    [ACCRQA_USE_OPENMP=yes;
     OPENMP_CXXFLAGS="-Xpreprocessor -fopenmp";
     OPENMP_LIBS="-lomp"]
  )
fi

CXXFLAGS="$save_CXXFLAGS"
LIBS="$save_LIBS"

if test "x$ACCRQA_USE_OPENMP" = "xyes"; then
  AC_MSG_RESULT([yes])
  AC_MSG_NOTICE([OpenMP CXX flags: $OPENMP_CXXFLAGS])
  AC_MSG_NOTICE([OpenMP libs: ${OPENMP_LIBS:-none}])
else
  AC_MSG_RESULT([no])
  OPENMP_CXXFLAGS=""
  OPENMP_LIBS=""
  AC_MSG_NOTICE([OpenMP disabled])
fi

AC_SUBST([OPENMP_CXXFLAGS])
AC_SUBST([OPENMP_LIBS])
AC_SUBST([ACCRQA_USE_OPENMP])
AC_LANG_POP([C++])

dnl ============================================================
dnl  CUDA detection (optional)
dnl ============================================================

AC_ARG_WITH([cuda],
  [AS_HELP_STRING([--with-cuda],
                  [build with CUDA GPU support if available @<:@default=yes@:>@])],
  [with_cuda=$withval],
  [with_cuda=yes])

AC_ARG_WITH([gpu-arch],
  [AS_HELP_STRING([--with-gpu-arch=XX],
                  [set CUDA GPU arch, e.g. 80 for sm_80])],
  [GPU_ARCH="$withval"],
  [GPU_ARCH=""])

AC_MSG_CHECKING([whether to use CUDA])
AC_MSG_RESULT([$with_cuda])

ACCRQA_USE_CUDA=no
CUDA_HOME=${CUDA_HOME-""}
NVCC_FLAGS=""

if test "x$with_cuda" = "xno"; then
	AC_MSG_RESULT([CUDA explicitly disabled])
	NVCC=""
	CUDA_HOME=""
else
	dnl 1) If user passed a directory: --with-cuda=/path
	if test "x$with_cuda" != "xyes"; then
		CUDA_HOME="$with_cuda"
	fi

	dnl 2) Locate nvcc (use CUDA_HOME as a hint if set)
	if test -n "$CUDA_HOME"; then
		AC_PATH_TOOL([NVCC],[nvcc],[no],[$CUDA_HOME/bin:$PATH])
	else
		AC_PATH_TOOL([NVCC],[nvcc],[no])
	fi

	if test "x$NVCC" = "xno"; then
		AC_MSG_WARN([nvcc not found; building AccRQA without GPU support])
		NVCC=""
		CUDA_HOME=""
		ACCRQA_USE_CUDA=no
	else
		dnl 3) If CUDA_HOME still unknown, search typical prefixes **by header**
		if test -z "$CUDA_HOME"; then
			for d in /usr/local/cuda /usr; do
				if test -f "$d/include/cuda_runtime_api.h"; then
					CUDA_HOME="$d"
					break
				fi
			done
		fi

		dnl 4) Only enable CUDA if we actually see cuda_runtime_api.h
		if test -n "$CUDA_HOME" && test -f "$CUDA_HOME/include/cuda_runtime_api.h"; then
			ACCRQA_USE_CUDA=yes
			AC_MSG_RESULT([nvcc found: $NVCC])
			#AC_MSG_RESULT([CUDA home: $CUDA_HOME])

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
			ACCRQA_USE_CUDA=no
		fi
	fi
fi

AC_MSG_NOTICE([Use CUDA: $ACCRQA_USE_CUDA])
AC_MSG_NOTICE([NVCC flags: ${NVCC_FLAGS:-none}])

AC_SUBST([CUDA_HOME])
AC_SUBST([NVCC])
AC_SUBST([NVCC_FLAGS])
AC_SUBST([ACCRQA_USE_CUDA])

dnl ============================================================
dnl  CUDA flags
dnl ============================================================

CUDA_CXXFLAGS=""
CUDA_LIBS=""

if test "x$ACCRQA_USE_CUDA" = "xyes"; then
  CUDA_CXXFLAGS="-I$CUDA_HOME/include -DCUDA_FOUND"
  CUDA_LIBS="-L$CUDA_HOME/lib64 -lcudart -lcuda -lcublas -lcurand"
fi

AC_MSG_NOTICE([CUDA CXXFLAGS: ${CUDA_CXXFLAGS:-none}])
AC_MSG_NOTICE([CUDA LIBS: ${CUDA_LIBS:-none}])

AC_SUBST([CUDA_CXXFLAGS])
AC_SUBST([CUDA_LIBS])

dnl ============================================================
dnl  Object list (portable, no sed)
dnl ============================================================

OBJECTS=""

add_obj () {
  if test -z "$OBJECTS"; then
    OBJECTS="$1"
  else
    OBJECTS="$OBJECTS $1"
  fi
}

for f in "$srcdir"/src/*.cpp; do
  test -f "$f" && add_obj "`basename "$f" .cpp`.o"
done

if test "x$ACCRQA_USE_CUDA" = "xyes"; then
  for f in "$srcdir"/src/*.cu; do
    test -f "$f" && add_obj "`basename "$f" .cu`.o"
  done
fi

AC_SUBST([OBJECTS])

dnl ============================================================
dnl  Output
dnl ============================================================

AC_CONFIG_FILES([src/Makevars])
AC_OUTPUT

