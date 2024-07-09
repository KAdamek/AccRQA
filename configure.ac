AC_INIT([accrqa],[1.0.0.])

AC_LANG(C++)
AC_REQUIRE_CPP

echo "Checking for C++ Compiler"
AC_PROG_CXX

GPURCUDA_CPPFLAGS="-I. -I../inst/include"

AC_MSG_CHECKING(environment variable CUDA_HOME)
if test -z "${CUDA_HOME}"; then
	CUDA_HOME=`find /usr/local/ -maxdepth 1 -type d -name "cuda*" | sort -V | tail -1`
	AC_MSG_RESULT(CUDA_HOME not set; using highest version found ${CUDA_HOME})
else
	AC_MSG_RESULT(using CUDA_HOME=${CUDA_HOME})
fi

AC_SUBST(CUDA_HOME)

AC_CHECK_FILE([${CUDA_HOME}/bin/nvcc], [HAS_NVCC="yes"])

AC_MSG_CHECKING(for NVCC)
if test -z "${HAS_NVCC}"; then
	AC_MSG_ERROR(NVCC not found; check CUDA install)
else
	AC_MSG_RESULT(NVCC found)
	NVCC="${CUDA_HOME}/bin/nvcc"
	AC_SUBST(NVCC)
fi

## look for Rscript, but use the one found via R_HOME to allow for multiple installations
AC_DEFUN([AC_PROG_RSCRIPT], [AC_CHECK_PROG(RSCRIPT,Rscript,yes)])
AC_PROG_RSCRIPT

if test x"${RSCRIPT}" == x"yes" ; then
		
		R_HOME=`Rscript -e 'cat(R.home())'`

    rcpp_incl=`${R_HOME}/bin/Rscript -e 'Rcpp:::CxxFlags()'`
		thrust_incl=`${R_HOME}/bin/Rscript -e 'cat(paste("-I", system.file("include", package = "thrust"), sep = ""))'`
		AC_SUBST(RCPP_INCL,["$rcpp_incl"])
    AC_SUBST(THRUST_INCL,["$thrust_incl"])
else
    echo "
  Your installation does not appear to have Rscript installed.

  Please make sure that you have a working and complete R installation.
"
    exit 1
fi

R_INCL=`"${R_HOME}/bin/R" CMD config --cppflags`
R_LIBS=`"${R_HOME}/bin/R" CMD config --ldflags`
R_CPIC=`"${R_HOME}/bin/R" CMD config CPICFLAGS`

AC_SUBST(GPURCUDA_CPPFLAGS)
AC_SUBST(R_INCL)
AC_SUBST(R_CPIC)
AC_SUBST(R_LIBS)
AC_CONFIG_FILES([src/Makevars])
AC_OUTPUT
