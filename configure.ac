AC_PREREQ([2.69])
AC_INIT([accrqa],[0.0.0])
AC_CONFIG_SRCDIR([src/])

dnl ------------------------------------------------------------
dnl  Version: read from version.txt and split into parts
dnl ------------------------------------------------------------

VERSION_FILE="version.txt"

if test -f "$VERSION_FILE"; then
	ACCRQA_VERSION_STRING=$(cat $VERSION_FILE)
else
	AC_MSG_ERROR([version.txt not found])
fi
dnl Strip whitespace
ACCRQA_VERSION_STRING=`echo "$ACCRQA_VERSION_STRING" | tr -d '[:space:]'`

dnl Basic sanity check
case "$ACCRQA_VERSION_STRING" in
	*[!0-9A-Za-z.\-]*)
	AC_MSG_ERROR([Invalid characters in version string: $ACCRQA_VERSION_STRING])
	;;
esac

AC_MSG_RESULT([Using AccRQA version: $ACCRQA_VERSION_STRING])

dnl Split version into major.minor.patch (simple cut)
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

dnl ------------------------------------------------------------
dnl  Common extra C++ flags (we will APPEND here)
dnl ------------------------------------------------------------

ACCRQA_CXXFLAGS=""
dnl -> later: + -DR_FOUND, + -DCUDA_FOUND, etc.

dnl ------------------------------------------------------------
dnl  Find R and its flags
dnl ------------------------------------------------------------

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

dnl ------------------------------------------------------------
dnl  CUDA / nvcc detection (optional)
dnl ------------------------------------------------------------

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

AC_MSG_RESULT([CUDA home: $CUDA_HOME])
AC_MSG_RESULT([Use CUDA: $ACCRQA_USE_CUDA])
AC_MSG_RESULT([NVCC flags: $NVCC_FLAGS])
AC_MSG_RESULT([CXX flags (extra): $ACCRQA_CXXFLAGS])

AC_SUBST([CUDA_HOME])
AC_SUBST([NVCC])
AC_SUBST([NVCC_FLAGS])
AC_SUBST([ACCRQA_USE_CUDA])
AC_SUBST([ACCRQA_CXXFLAGS])

dnl ------------------------------------------------------------
dnl  Generate src/Makevars from src/Makevars.in
dnl ------------------------------------------------------------

AC_CONFIG_FILES([src/Makevars])
AC_OUTPUT
