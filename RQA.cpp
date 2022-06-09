#include <stdio.h>
#include <string.h>
#include <math.h>
#include <time.h>
#include <stdlib.h>
#include <iostream>
#include <fstream>
#include <iomanip>
#include <vector>

using namespace std;

#include <dirent.h>

#include "AccRQA_library.hpp"

typedef double RQAdp;

bool UNIT_TESTS = false;

bool CPU_RQA = false;
bool CPU_RQA_RR = true;
bool CPU_RQA_DET = false;
bool CPU_RQA_LAM = false;

bool GPU_RQA = true;
bool GPU_RQA_RR = true;
bool GPU_RQA_DET = false;
bool GPU_RQA_LAM = false;
bool GPU_RQA_ALL = true;


long int get_file_size(ifstream &FILEIN){
	long int count = 0;
	FILEIN.seekg(0,ios::beg);
	for(std::string line; std::getline(FILEIN, line); ++count){}
	return(count);
}


int Load_data(std::vector<float> *data, char *filename){
	int error;
	error=0;

	ifstream FILEIN;
	FILEIN.open(filename,ios::in);
	if (!FILEIN.fail()){
		long int file_size = get_file_size(FILEIN);
		printf("nSamples:%ld;\n", file_size );
		
		// read data
		FILEIN.clear();
		FILEIN.seekg(0,ios::beg);
		for(long int f = 0; f < file_size; f++){
			double tp1;
			FILEIN >> tp1;
			data->push_back((float) tp1);
		}

		if(file_size==0){
			printf("\nFile is empty!\n");
			error++;
		}
	}
	else {
		cout << "File not found -> " << filename << " <-" << endl;
		error++;
	}
	FILEIN.close();
	return(error);
}


int Load_data(std::vector<double> *data, char *filename){
	int error;
	error=0;

	ifstream FILEIN;
	FILEIN.open(filename,ios::in);
	if (!FILEIN.fail()){
		long int file_size = get_file_size(FILEIN);
		printf("nSamples:%ld;\n", file_size );
		
		// read data
		FILEIN.clear();
		FILEIN.seekg(0,ios::beg);
		for(long int f = 0; f < file_size; f++){
			double tp1;
			FILEIN >> tp1;
			data->push_back(tp1);
		}

		if(file_size==0){
			printf("\nFile is empty!\n");
			error++;
		}
	}
	else {
		cout << "File not found -> " << filename << " <-" << endl;
		error++;
	}
	FILEIN.close();
	return(error);
}


int main(int argc, char* argv[]) {
	int device = 0;
	char input_threshold_file[1000];
	char input_data_file[1000];
	int tau = 1;
	int emb = 1;
	int lmin = 2;
	int vmin = 2;
	char * pEnd;
	
	printf("args=%d;", argc);
	if (argc==7) {
		if (strlen(argv[1])>1000) {printf("Filename of the data file is too long\n"); exit(2);}
		sprintf(input_data_file,"%s", argv[1]);
		if (strlen(argv[1])>1000) {printf("Filename of the threshold file is too long\n"); exit(2);}
		sprintf(input_threshold_file,"%s", argv[2]);
		tau  = strtol(argv[3],&pEnd,10);
		emb  = strtol(argv[4],&pEnd,10);
		lmin = strtol(argv[5],&pEnd,10);
		vmin = strtol(argv[6],&pEnd,10);
	}
	else {
		printf("Parameters error!\n");
		printf("Input parameters are as follows:\n");
		printf(" 1) filename of the file which contains a time-series to process \n");
		printf(" 2) filename containing desired thresholds\n");
		printf(" 3) value for the time step (tau)\n");
		printf(" 4) value for the embedding (emb)\n");
		printf(" 5) l_min\n");
		printf(" 6) v_min\n");
		printf(" Example: RQA.exe data.txt thresholds.txt 5 12 2 2\n");
		return(1);
	}
	
	printf("Program parameters:\n");
	printf("data file: %s;\n", input_data_file);
	printf("threshold file: %s;\n", input_threshold_file);
	printf("tau  = %d;\n", tau);
	printf("emb  = %d;\n", emb);
	printf("lmin = %d;\n", lmin);
	printf("vmin = %d;\n", vmin);
	
	
	int filein_length = strlen(input_data_file);
	
	// Conversion to string
	string infile_test(input_data_file, input_data_file + filein_length);
	
	size_t found = 0, found_old = 0;
	// removing dot and extension
	while(found!=std::string::npos){
		found_old = found;
		found = found + 1;
		found = infile_test.find(".", found, 1);
	}
	std::string temporary_filename = infile_test.substr (0,found_old);
	size_t temporary_filesize = temporary_filename.length();
	
	// extracting filename
	found = 0; found_old = 0;
	while(found!=std::string::npos){
		found_old = found;
		found = found + 1;
		found = temporary_filename.find("/", found, 1);
	}
	if(found_old > 0) {
		found_old = found_old + 1;
	}
	string basename = temporary_filename.substr(found_old, temporary_filesize);
	string directory = temporary_filename.substr(0, found_old);
	string output_filename = directory + basename + "_t" + to_string(tau) + "_e" + to_string(emb) + "_lmin" + to_string(lmin) + "_vmin" + to_string(vmin) + ".rqa";
	
	vector<RQAdp> input_data;
	vector<RQAdp> threshold_list;
	Load_data(&input_data, input_data_file);
	Load_data(&threshold_list, input_threshold_file);
	
	
	if(input_data.size()==0) {
		printf("Error: No data to process.\n");
		return(1);
	}
	if(threshold_list.size()==0) {
		printf("Error: No thresholds file is empty.\n");
		return(1);
	}
	if(tau<1){
		printf("Error: Time step must be tau>0.\n");
		return(1);
	}
	if(emb<1){
		printf("Error: Embedding must be emb>0.\n");
		return(1);
	}
	if(lmin<2){
		printf("Error: lmin must be lmin>1.\n");
		return(1);
	}
	if(vmin<2){
		printf("Error: vmin must be vmin>1.\n");
		return(1);
	}	
	
	
	char str[200];
	
	if(GPU_RQA==true){
		//---------------------> RR
		if(GPU_RQA_RR){
			printf("--> GPU recurrent rate\n");
			vector<double> result_l2_RR;
			for(size_t th_idx = 0; th_idx<threshold_list.size(); th_idx++){
				RQAdp threshold = threshold_list[th_idx];
				RQAdp *RR;
				RR = new RQAdp[emb];

				accrqaRecurrentRateERGPU(RR, threshold, input_data.data(), input_data.size(), tau, emb, RQA_METRIC_MAXIMAL, device);
				

				result_l2_RR.push_back(RR[emb-1]); // Shift the index left by one: since dimensions start at 1 but indices is from 0


				delete [] RR;
			}


			//writing results to disk
			std::ofstream FILEOUT;
			sprintf(str, "test_GPU_RR_t%d_e%d_l%d.dat", tau, emb, lmin);
			FILEOUT.open(str);
			for(int i = 0; i< (int) threshold_list.size(); i++){
				RQAdp RR;
				RR = result_l2_RR[i];
				FILEOUT << threshold_list[i] << " " << RR << std::endl;
			}
			FILEOUT.close();
			
		}
		//------------------------------<

		//---------------------> Determinism
		if(GPU_RQA_DET){
			printf("--> GPU determinism\n");
			vector<double> result_l2_DET;
			vector<double> result_l2_L;
			for(size_t th_idx = 0; th_idx<threshold_list.size(); th_idx++){
				RQAdp threshold = threshold_list[th_idx];
				RQAdp DET, L, Lmax;
				
				accrqaDeterminismGPU(&DET, &L, &Lmax, input_data.data(), input_data.size(),  threshold, tau, emb, lmin, RQA_METRIC_MAXIMAL, device);
				
				result_l2_DET.push_back(DET);
				result_l2_L.push_back(L);
			}
			
			std::ofstream FILEOUT;
			sprintf(str, "test_GPU_DET_L_t%d_e%d_l%d.dat", tau, emb, lmin);
			FILEOUT.open(str);
			size_t size = threshold_list.size();
			for(size_t i = 0; i<size; i++){
				RQAdp DET, L, th;
				DET = result_l2_DET[i];
				L = result_l2_L[i];
				th = threshold_list[i];
				FILEOUT << th << " " << DET << " " << L << endl;
			}
			FILEOUT.close();
		}
		//------------------------------<
		
		//---------------------> Laminarity
		if(GPU_RQA_LAM){
			printf("--> GPU laminarity\n");
			vector<double> result_l2_LAM;
			vector<double> result_l2_TT;
			for(size_t th_idx = 0; th_idx<threshold_list.size(); th_idx++){
				RQAdp threshold = threshold_list[th_idx];
				RQAdp LAM, TT, TTmax;
				
				accrqaLaminarityGPU(&LAM, &TT, &TTmax, input_data.data(), input_data.size(), threshold, tau, emb, vmin, RQA_METRIC_MAXIMAL, device);
				
				result_l2_LAM.push_back(LAM);
				result_l2_TT.push_back(TT);
			}
			
			
			std::ofstream FILEOUT;
			sprintf(str, "test_GPU_LAM_TT_t%d_e%d_l%d.dat", tau, emb, vmin);
			FILEOUT.open(str);
			size_t size = threshold_list.size();
			for(size_t i = 0; i<size; i++){
				RQAdp LAM, TT, th;
				LAM = result_l2_LAM[i];
				TT = result_l2_TT[i];
				th = threshold_list[i];
				FILEOUT << th << " " << LAM << " " << TT << endl;
			}
			FILEOUT.close();
		}
		//------------------------------<
		
		//---------------------> Everything
		if(GPU_RQA_ALL){
			if(DEBUG) printf("--> GPU recurrent rate\n");
			int nThresholds = (int) threshold_list.size();
			RQAdp *RR;
			RR = new RQAdp[nThresholds];
			
			accrqaRecurrentRateCPU(RR, threshold_list.data(), nThresholds, input_data.data(), input_data.size(), tau, emb, RQA_METRIC_MAXIMAL);
			
			if(DEBUG) printf("--> GPU DET and LAM\n");
			vector<double> result_DET;
			vector<double> result_L;
			vector<double> result_Lmax;
			vector<double> result_LAM;
			vector<double> result_TT;
			vector<double> result_TTmax;
			for(size_t th_idx = 0; th_idx<threshold_list.size(); th_idx++){
				RQAdp threshold = threshold_list[th_idx];
				RQAdp DET, L, Lmax, LAM, TT, TTmax;
				
				accrqaDeterminismGPU(&DET, &L, &Lmax, input_data.data(), input_data.size(),  threshold, tau, emb, lmin, RQA_METRIC_MAXIMAL, device);
				accrqaLaminarityGPU(&LAM, &TT, &TTmax, input_data.data(), input_data.size(), threshold, tau, emb, vmin, RQA_METRIC_MAXIMAL, device);
				
				result_DET.push_back(DET);
				result_L.push_back(L);
				result_Lmax.push_back(Lmax);
				result_LAM.push_back(LAM);
				result_TT.push_back(TT);
				result_TTmax.push_back(TTmax);
			}
			
			std::ofstream FILEOUT;
			FILEOUT.open(output_filename);
			size_t size = threshold_list.size();
			FILEOUT << "#threshold RR DET LAM Lmean Vmean Lmax Vmax" << endl;
			for(size_t i = 0; i<size; i++){
				RQAdp th, DET, LAM, L, Lmax, TT, TTmax;
				th = threshold_list[i];
				DET = result_DET[i];
				L = result_L[i];
				Lmax = result_Lmax[i];
				LAM = result_LAM[i];
				TT = result_TT[i];
				TTmax = result_TTmax[i];
				
				FILEOUT << th << " " << RR[i] << " " << DET << " " << LAM << " " << L << " " << TT << " " << Lmax << " " << TTmax << endl;
			}
			FILEOUT.close();
			
			delete [] RR;
		}
	}
	
	if(CPU_RQA==true){
		//================================================================
		//============================ CPU ===============================
		//================================================================
		
		//-------------- Recurrent rate
		if(CPU_RQA_RR){
			printf("--> CPU recurrent rate\n");
			int nThresholds = (int) threshold_list.size();
			RQAdp *RR;
			RR = new RQAdp[nThresholds];
			
			accrqaRecurrentRateCPU(RR, threshold_list.data(), nThresholds, input_data.data(), input_data.size(), tau, emb, RQA_METRIC_MAXIMAL);
			
			std::ofstream FILEOUT;
			sprintf(str, "test_CPU_RR_t%d_e%d_l%d.dat", tau, emb, lmin);
			FILEOUT.open(str);
			for(int i = 0; i<nThresholds; i++){
				FILEOUT << threshold_list[i] << " " << RR[i] << std::endl;
			}
			FILEOUT.close();
			
			delete [] RR;
		}
		//----------------------------------------------------------<
		
		//-------------- Laminarity
		if(CPU_RQA_LAM){
			printf("--> CPU laminarity\n");
			std::vector<double> result_l2_LAM;
			std::vector<double> result_l2_TT;
			for(size_t th_idx = 0; th_idx<threshold_list.size(); th_idx++){
				RQAdp threshold = threshold_list[th_idx];
				RQAdp LAM, TT, TTmax;
				
				accrqaLaminarityCPU(&LAM, &TT, &TTmax, input_data.data(), input_data.size(), threshold, tau, emb, vmin, RQA_METRIC_MAXIMAL);
				
				result_l2_LAM.push_back(LAM);
				result_l2_TT.push_back(TT);
			}
			
			std::ofstream FILEOUT;
			sprintf(str, "test_CPU_LAM_TT_t%d_e%d_l%d.dat", tau, emb, vmin);
			FILEOUT.open(str);
			size_t size = threshold_list.size();
			for(size_t i = 0; i<size; i++){
				RQAdp LAM, TT, th;
				LAM = result_l2_LAM[i];
				TT = result_l2_TT[i];
				th = threshold_list[i];
				FILEOUT << th << " " << LAM << " " << TT << std::endl;
			}
			FILEOUT.close();
		}
		//----------------------------------------------------------<
		
		//-------------- Determinism
		if(CPU_RQA_DET){
			printf("--> CPU determinism\n");
			std::vector<double> result_l2_DET;
			std::vector<double> result_l2_L;
			for(size_t th_idx = 0; th_idx<threshold_list.size(); th_idx++){
				RQAdp threshold = threshold_list[th_idx];
				RQAdp DET, L, Lmax;
				
				accrqaDeterminismCPU(&DET, &L, &Lmax, input_data.data(), input_data.size(), threshold, tau, emb, lmin, RQA_METRIC_MAXIMAL);
				
				result_l2_DET.push_back(DET);
				result_l2_L.push_back(L);
			}
			
			std::ofstream FILEOUT;
			sprintf(str, "test_CPU_DET_L_t%d_e%d_l%d.dat", tau, emb, lmin);
			FILEOUT.open(str);
			size_t size = threshold_list.size();
			for(size_t i = 0; i<size; i++){
				RQAdp DET, L, th;
				DET = result_l2_DET[i];
				L = result_l2_L[i];
				th = threshold_list[i];
				FILEOUT << th << " " << DET << " " << L << std::endl;
			}
			FILEOUT.close();
		}
		//----------------------------------------------------------<
	}
	

	return (0);
}

