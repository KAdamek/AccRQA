#ifdef ACCRQA_R_FOUND
	#include <R_ext/Print.h>
	#define ACCRQA_PRINT(...)  Rprintf(__VA_ARGS__)
	#define ACCRQA_EPRINT(...) REprintf(__VA_ARGS__)
#else
	#include <cstdio>
	#define ACCRQA_PRINT(...)  std::printf(__VA_ARGS__)
	#define ACCRQA_EPRINT(...) std::fprintf(stderr, __VA_ARGS__)
#endif


