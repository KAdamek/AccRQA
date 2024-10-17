#include <stdio.h>
#include <string.h>
#include <math.h>
#include <time.h>
#include <stdlib.h>
#include <iostream>
#include <fstream>
#include <iomanip>
#include <vector>

#define VERBOSE true

using namespace std;

#include <dirent.h>

#include "../include/AccRQA_library.hpp"

typedef double RQAdp;

bool UNIT_TESTS = false;

bool CPU_RQA = false;
bool CPU_RQA_RR = false;
bool CPU_RQA_DET = false;
bool CPU_RQA_LAM = false;

bool GPU_RQA = true;
bool GPU_RQA_RR = false;
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
	
	if(VERBOSE) {
		printf("Program parameters:\n");
		printf("  data file: %s;\n", input_data_file);
		printf("  threshold file: %s;\n", input_threshold_file);
		printf("  tau  = %d;\n", tau);
		printf("  emb  = %d;\n", emb);
		printf("  lmin = %d;\n", lmin);
		printf("  vmin = %d;\n", vmin);
	}
	
	
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
			int nThresholds = (int) threshold_list.size();
			RQAdp *RR;
			RR = new RQAdp[nThresholds];
			
			int tau_values = tau;
			int emb_values = emb;
			Accrqa_Error error;
			accrqa_RR(RR, input_data.data(), input_data.size(), &tau_values, 1, &emb_values, 1, threshold_list.data(), nThresholds, DST_MAXIMAL, PLT_NV_GPU, &error);
			
			//writing results to disk
			std::ofstream FILEOUT;
			sprintf(str, "test_GPU_RR_t%d_e%d_l%d.dat", tau, emb, lmin);
			FILEOUT.open(str);
			for(int i = 0; i<nThresholds; i++){
				FILEOUT << threshold_list[i] << " " << RR[i] << std::endl;
			}
			FILEOUT.close();
			
			delete [] RR;
		}
		//------------------------------<

		//---------------------> Determinism
		if(GPU_RQA_DET){
			printf("--> GPU determinism\n");
			vector<RQAdp> result_l2_DET;
			vector<RQAdp> result_l2_L;
			vector<RQAdp> result_l2_Lmax;
			vector<RQAdp> result_l2_ENTR;
			for(size_t th_idx = 0; th_idx<threshold_list.size(); th_idx++){
				RQAdp threshold_values = threshold_list[th_idx];
				//=============
				int tau_values = tau;
				int emb_values = emb;
				int lmin_values = lmin;
				Accrqa_Error error;
				int calc_ENTR = 1;
				RQAdp *output;
				output = new RQAdp[5];
				accrqa_DET(output, input_data.data(), input_data.size(), &tau_values, 1, &emb_values, 1, &lmin_values, 1, &threshold_values, 1, DST_MAXIMAL, calc_ENTR, PLT_NV_GPU, &error);
				result_l2_DET.push_back(output[0]);
				result_l2_L.push_back(output[1]);
				result_l2_Lmax.push_back(output[2]);
				result_l2_ENTR.push_back(output[3]);
				delete[] output;
				//=============
			}
			
			std::ofstream FILEOUT;
			sprintf(str, "test_GPU_DET_L_t%d_e%d_l%d.dat", tau, emb, lmin);
			FILEOUT.open(str);
			size_t size = threshold_list.size();
			for(size_t i = 0; i<size; i++){
				RQAdp DET, L, Lmax, ENTR, th;
				DET = result_l2_DET[i];
				L = result_l2_L[i];
				Lmax = result_l2_Lmax[i];
				ENTR = result_l2_L[i];
				th = threshold_list[i];
				FILEOUT << th << " " << DET << " " << L << " " << Lmax << " " << ENTR << std::endl;
			}
			FILEOUT.close();
		}
		//------------------------------<
		
		//---------------------> Laminarity
		if(GPU_RQA_LAM){
			printf("--> GPU laminarity\n");
			vector<RQAdp> result_l2_LAM;
			vector<RQAdp> result_l2_TT;
			vector<RQAdp> result_l2_TTmax;
			vector<RQAdp> result_l2_ENTR;
			for(size_t th_idx = 0; th_idx<threshold_list.size(); th_idx++){
				RQAdp threshold_values = threshold_list[th_idx];
				int tau_values = tau;
				int emb_values = emb;
				int vmin_values = lmin;
				Accrqa_Error error;
				int calc_ENTR = 1;
				RQAdp *output;
				output = new RQAdp[5];
				accrqa_LAM(output, input_data.data(), input_data.size(), &tau_values, 1, &emb_values, 1, &vmin_values, 1, &threshold_values, 1, DST_MAXIMAL, calc_ENTR, PLT_NV_GPU, &error);
				result_l2_LAM.push_back(output[0]);
				result_l2_TT.push_back(output[1]);
				result_l2_TTmax.push_back(output[2]);
				result_l2_ENTR.push_back(output[3]);
				delete[] output;
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
			if(VERBOSE) printf("--> GPU recurrent rate\n");
			int nThresholds = (int) threshold_list.size();
			RQAdp *RR;
			RR = new RQAdp[nThresholds];
			
			int tau_values = tau;
			int emb_values = emb;
			Accrqa_Error error;
			accrqa_RR(RR, input_data.data(), input_data.size(), &tau_values, 1, &emb_values, 1, threshold_list.data(), nThresholds, DST_MAXIMAL, PLT_NV_GPU, &error);
			if(VERBOSE) printf("----> Done\n");
			
			if(VERBOSE) printf("--> GPU DET and LAM\n");
			vector<RQAdp> result_DET;
			vector<RQAdp> result_L;
			vector<RQAdp> result_Lmax;
			vector<RQAdp> result_ENTR;
			vector<RQAdp> result_LAM;
			vector<RQAdp> result_TT;
			vector<RQAdp> result_TTmax;
			for(size_t th_idx = 0; th_idx<threshold_list.size(); th_idx++){
				RQAdp threshold_values = threshold_list[th_idx];
				int tau_values = tau;
				int emb_values = emb;
				int lmin_values = lmin;
				int vmin_values = vmin;
				Accrqa_Error error;
				int calc_ENTR = 1;
				RQAdp *output_DET, *output_LAM;
				output_DET = new RQAdp[5];
				output_LAM = new RQAdp[5];
				
				RQAdp LAM, TT, TTmax;
				
				accrqa_DET(output_DET, input_data.data(), input_data.size(), &tau_values, 1, &emb_values, 1, &lmin_values, 1, &threshold_values, 1, DST_MAXIMAL, calc_ENTR, PLT_NV_GPU, &error);
				if(VERBOSE) printf("----> Done\n");
				
				accrqa_LAM(output_LAM, input_data.data(), input_data.size(), &tau_values, 1, &emb_values, 1, &vmin_values, 1, &threshold_values, 1, DST_MAXIMAL, calc_ENTR, PLT_NV_GPU, &error);
				if(VERBOSE) printf("----> Done\n");
				
				result_DET.push_back(output_DET[0]);
				result_L.push_back(output_DET[1]);
				result_Lmax.push_back(output_DET[2]);
				result_ENTR.push_back(output_DET[3]);
				result_LAM.push_back(output_LAM[0]);
				result_TT.push_back(output_LAM[1]);
				result_TTmax.push_back(output_LAM[2]);
				
				delete[] output_DET;
				delete[] output_LAM;
			}
			
			std::ofstream FILEOUT;
			FILEOUT.open(output_filename);
			size_t size = threshold_list.size();
			FILEOUT << "#threshold RR DET LAM Lmean Vmean Lmax Vmax ENTR" << endl;
			for(size_t i = 0; i<size; i++){
				RQAdp th, DET, LAM, L, Lmax, TT, TTmax, ENTR;
				th = threshold_list[i];
				DET = result_DET[i];
				L = result_L[i];
				Lmax = result_Lmax[i];
				ENTR = result_ENTR[i];
				LAM = result_LAM[i];
				TT = result_TT[i];
				TTmax = result_TTmax[i];
				
				FILEOUT << th << " " << RR[i] << " " << DET << " " << LAM << " " << L << " " << TT << " " << Lmax << " " << TTmax << " " << ENTR << std::endl;
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
			
			int tau_values = tau;
			int emb_values = emb;
			Accrqa_Error error;
			accrqa_RR(RR, input_data.data(), input_data.size(), &tau_values, 1, &emb_values, 1, threshold_list.data(), nThresholds, DST_MAXIMAL, PLT_CPU, &error);
			if(VERBOSE) printf("----> Done\n");
			
			
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
			if(VERBOSE) printf("--> CPU laminarity\n");
			std::vector<RQAdp> result_l2_LAM;
			std::vector<RQAdp> result_l2_TT;
			std::vector<RQAdp> result_l2_TTmax;
			for(size_t th_idx = 0; th_idx<threshold_list.size(); th_idx++){
				RQAdp threshold_values = threshold_list[th_idx];
				
				//=============
				int tau_values = tau;
				int emb_values = emb;
				int vmin_values = vmin;
				Accrqa_Error error;
				int calc_ENTR = 0;
				RQAdp *output;
				output = new RQAdp[5];
				accrqa_LAM(output, input_data.data(), input_data.size(), &tau_values, 1, &emb_values, 1, &vmin_values, 1, &threshold_values, 1, DST_MAXIMAL, calc_ENTR, PLT_CPU, &error);
				result_l2_LAM.push_back(output[0]);
				result_l2_TT.push_back(output[1]);
				result_l2_TTmax.push_back(output[2]);
				delete[] output;
				//=============
			}
			if(VERBOSE) printf("----> Done\n");
			
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
			if(VERBOSE) printf("--> CPU determinism\n");
			std::vector<RQAdp> result_l2_DET;
			std::vector<RQAdp> result_l2_L;
			std::vector<RQAdp> result_l2_Lmax;
			std::vector<RQAdp> result_l2_ENTR;
			for(size_t th_idx = 0; th_idx<threshold_list.size(); th_idx++){
				RQAdp threshold_values = threshold_list[th_idx];
				
				//=============
				int tau_values = tau;
				int emb_values = emb;
				int lmin_values = lmin;
				Accrqa_Error error;
				int calc_ENTR = 1;
				RQAdp *output;
				output = new RQAdp[5];
				accrqa_DET(output, input_data.data(), input_data.size(), &tau_values, 1, &emb_values, 1, &lmin_values, 1, &threshold_values, 1, DST_MAXIMAL, calc_ENTR, PLT_CPU, &error);
				result_l2_DET.push_back(output[0]);
				result_l2_L.push_back(output[1]);
				result_l2_Lmax.push_back(output[2]);
				result_l2_ENTR.push_back(output[3]);
				delete[] output;
				//=============
			}
			if(VERBOSE) printf("----> Done\n");
			
			std::ofstream FILEOUT;
			sprintf(str, "test_CPU_DET_L_t%d_e%d_l%d.dat", tau, emb, lmin);
			FILEOUT.open(str);
			size_t size = threshold_list.size();
			for(size_t i = 0; i<size; i++){
				RQAdp DET, L, Lmax, ENTR, th;
				DET = result_l2_DET[i];
				L = result_l2_L[i];
				Lmax = result_l2_Lmax[i];
				ENTR = result_l2_ENTR[i];
				th = threshold_list[i];
				FILEOUT << th << " " << DET << " " << L << " " << Lmax << " " << ENTR << std::endl;
			}
			FILEOUT.close();
		}
		//----------------------------------------------------------<
	}
	

	return (0);
}

