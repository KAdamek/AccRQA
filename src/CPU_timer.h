#include <sys/time.h>
#ifndef CPU_TIMER_H__
#define CPU_TIMER_H__


class CPU_Timer {
private:
	struct timeval begin, end;
	
public:
	void Start() {
		gettimeofday(&begin, 0);
	}
	
	void Stop() {
		gettimeofday(&end, 0);
	}
	
	double Elapsed() {
		long seconds = end.tv_sec - begin.tv_sec;
		long microseconds = end.tv_usec - begin.tv_usec;
		double elapsed = ((double) (seconds + (double) microseconds*1e-6))*1.0e3;
		return(elapsed);
	}
};

#endif  /* CPU_TIMER_H__ */
