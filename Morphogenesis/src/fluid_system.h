
//-------------------------------------------------------------------
// FLUIDS v.3 - SPH Fluid Simulator for CPU and GPU
// Copyright (C) 2012-2013. Rama Hoetzlein, http://fluids3.com
//
// Attribute-ZLib license (* See additional part 4)
// This software is provided 'as-is', without any express or implied
// warranty. In no event will the authors be held liable for any damages
// arising from the use of this software.
//
// Permission is granted to anyone to use this software for any purpose,
// including commercial applications, and to alter it and redistribute it
// freely, subject to the following restrictions:
//
// 1. The origin of this software must not be misrepresented; you must not
//    claim that you wrote the original software.
// 2. Altered source versions must be plainly marked as such, and must not be
//    misrepresented as being the original software.
// 3. This notice may not be removed or altered from any source distribution.
// 4. Any published work based on this code must include public acknowledgement
//    of the origin. This includes following when applicable:
//	   - Journal/Paper publications. Credited by reference to work in text & citation.
//	   - Public presentations. Credited in at least one slide.
//	   - Distributed Games/Apps. Credited as single line in game or app credit page.	 
//	 Retaining this additional license term is required in derivative works.
//	 Acknowledgement may be provided as:
//	   Publication version:  
//	      2012-2013, Hoetzlein, Rama C. Fluids v.3 - A Large-Scale, Open Source
//	 	  Fluid Simulator. Published online at: http://fluids3.com
//	   Single line (slides or app credits):
//	      GPU Fluids: Rama C. Hoetzlein (Fluids v3 2013)
//--------------------------------------------------------------------

//#pragma once

#ifndef DEF_FLUID_SYS
	#define DEF_FLUID_SYS


    #include <dirent.h> 
    #include <filesystem>
	#include <cstring>
	#include <iostream>
	#include <vector>
	#include <stdio.h>
	#include <stdlib.h>
	#include <math.h>
	#include <sys/stat.h>
    #include <sys/types.h> 
	#include "fluid.h"
	#include <filesystem>
	namespace fs = std::filesystem;

// 	#include <vtk-9.0/vtkCellArray.h>
//     #include <vtk-9.0/vtkPoints.h>
//     #include <vtk-9.0/vtkXMLPolyDataWriter.h>
//     #include <vtk-9.0/vtkPolyData.h>
//     #include <vtk-9.0/vtkSmartPointer.h>
//     #include <vtk-9.0/vtkLine.h>
//     #include <vtk-9.0/vtkDataSet.h>
//     #include <vtk-9.0/vtkUnsignedIntArray.h>
//     #include <vtk-9.0/vtkUnsignedCharArray.h>
//     #include <vtk-9.0/vtkFloatArray.h>
//     #include <vtk-9.0/vtkPointData.h>
//     #include <vtk-9.0/vtkCellData.h>

	extern bool gProfileRend;

    #define EPSILON			0.00001f			// for collision detection
    #define SCAN_BLOCKSIZE		128				// must match value in fluid_system_cuda.cu

	#define MAX_PARAM			50             // used for m_Param[], m_Vec[], m_Toggle[]
	#define GRID_UCHAR			0xFF           // used in void FluidSystem::InsertParticles (){.. memset(..); ...}
	#define GRID_UNDEF			4294967295	

	// Scalar params   "m_Param[]"  //  /*remove some of these lines ?*/   Need to check if/when each is used...
	#define PMODE				0
	#define PNUM				1
	#define PEXAMPLE			2   // 0=Regression test. N x N x N static grid, 1=Tower , 2=Wave pool , 3=Small dam break , 4=Dual-Wave pool , 5=Microgravity . See  FluidSystem::SetupExampleParams (). Used in void FluidSystem::Initialize ()  
	#define PSIMSIZE			3
	#define PSIMSCALE			4
	#define PGRID_DENSITY		5
	#define PGRIDSIZE			6
	#define PVISC				7
	#define PRESTDENSITY		8
	#define PMASS				9
	#define PRADIUS				10
	#define PDIST				11
	#define PSMOOTHRADIUS		12
	#define PINTSTIFF			13
	#define PEXTSTIFF			14
	#define PEXTDAMP			15
	#define PACCEL_LIMIT		16
	#define PVEL_LIMIT			17
	#define PSPACING			18
	#define PGROUND_SLOPE		19
	#define PFORCE_MIN			20
	#define PFORCE_MAX			21
	#define PGRAV				22 
	#define PFORCE_FREQ			23	
    #define PSURFACE_TENSION    24
    
    #define PACTUATION_FACTOR   25
    #define PACTUATION_PERIOD   26

	// Vector params   "m_Vec[]" 
	#define PVOLMIN				0
	#define PVOLMAX				1
	#define PBOUNDMIN			2
	#define PBOUNDMAX			3
	#define PINITMIN			4
	#define PINITMAX			5
	#define PPLANE_GRAV_DIR		6

    //  used for AllocateBuffer(  .... )
	#define GPU_OFF				0
	#define GPU_SINGLE			1
	#define GPU_TEMP			2
	#define GPU_DUAL			3
	#define CPU_OFF				4
	#define CPU_YES				5

	////RUNCL.h///////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
	#define PIXELS			0	// TODO Can these be #included from a common header for both host and device code?
	#define ROWS			1
	#define COLS			2
	#define LAYERS			3
	#define MAX_INV_DEPTH	4
	#define MIN_INV_DEPTH	5
	#define INV_DEPTH_STEP	6
	#define ALPHA_G			7
	#define BETA_G			8	//  __kernel void CacheG4
	//#define EPSILON 		9	//  __kernel void UpdateQD		// epsilon = 0.1		//This is a conflicting definition with fluid_system.h:6
	#define SIGMA_Q 		10									// sigma_q = 0.0559017
	#define SIGMA_D 		11
	#define THETA			12
	#define LAMBDA			13	//  __kernel void UpdateA2
	#define SCALE_EAUX		14

	#define FUNC_INSERT			0
	#define	FUNC_COUNTING_SORT_FULL	1
	#define FUNC_QUERY			2
	#define FUNC_COMPUTE_PRESS	3
	#define FUNC_COMPUTE_FORCE	4
	#define FUNC_ADVANCE		5
	#define FUNC_EMIT			6
	#define FUNC_RANDOMIZE		7
	#define FUNC_SAMPLE			8
	#define FUNC_FPREFIXSUM		9
	#define FUNC_FPREFIXUP	10
	#define FUNC_TALLYLISTS     11
	#define FUNC_COMPUTE_DIFFUSION          12
	#define FUNC_COUNT_SORT_DENSE_LISTS           13
	#define FUNC_COMPUTE_GENE_ACTION        14
	#define FUNC_TALLY_GENE_ACTION        35
	#define FUNC_COMPUTE_BOND_CHANGES       15

	#define FUNC_INSERT_CHANGES             16 //insertChanges
	#define FUNC_PREFIXUP_CHANGES           17 //prefixFixupChanges
	#define FUNC_PREFIXSUM_CHANGES          18 //prefixSumChanges
	#define FUNC_TALLYLISTS_CHANGES         19 //tally_changelist_lengths
	#define FUNC_COUNTING_SORT_CHANGES      20 //countingSortChanges
	#define FUNC_COMPUTE_NERVE_ACTION       21 //computeNerveActivation

	#define FUNC_COMPUTE_MUSCLE_CONTRACTION 22 //computeMuscleContraction
	#define FUNC_HEAL                       23 //heal
	#define FUNC_LENGTHEN_TISSUE            24 //lengthen_muscle
	#define FUNC_LENGTHEN_MUSCLE            25 //lengthen_tissue
	#define FUNC_SHORTEN_TISSUE             26 //shorten_muscle
	#define FUNC_SHORTEN_MUSCLE             27 //shorten_tissue

	#define FUNC_STRENGTHEN_TISSUE          28 //strengthen_muscle
	#define FUNC_STRENGTHEN_MUSCLE          29 //strengthen_tissue
	#define FUNC_WEAKEN_TISSUE              30 //weaken_muscle
	#define FUNC_WEAKEN_MUSCLE              31 //weaken_tissue

	#define FUNC_EXTERNAL_ACTUATION         32
	#define FUNC_FIXED                      33
	#define FUNC_CLEAN_BONDS                34

	#define FUNC_INIT_RANDOMCL         36
	#define FUNC_COUNTING_SORT_EPIGEN       37

	#define	FUNC_ASSEMBLE_MUSCLE_FIBRES_OUTGOING     38
	#define	FUNC_ASSEMBLE_MUSCLE_FIBRES_INCOMING     39
	#define	FUNC_INITIALIZE_BONDS                    40
	#define	FUNC_CONTRIBUTE_PRESSURE                    41
	#define	FUNC_MEMSET32D                    42
	#define FUNC_FPREFIXSUMCHANGES			43
	#define FUNC_MAX			            44


using namespace std;


	cl_float3 make_cl_float3(float x, float y, float z)
    {
        cl_float3 t; t.x = x; t.y = y; t.z = z; return t;
    }

    cl_int3 make_cl_int3(int x, int y, int z)
    {
        cl_int3 t; t.x = x; t.y = y; t.z = z; return t;
    }

	bool clCheck (cl_int launch_stat, const char* method, const char* apicall, const char* arg, bool bDebug);

	void setBuf(FBufs *fb, int n, char* buf) {
			fb->mcpu[n] = buf;
	}

	void setGpuBuf(FBufs *fb, int n, cl_mem buf) {
	fb->mgpu[n] = buf;
	}

	cl_mem gpuVar(FBufs *fb, int n) {
		return fb->mgpu[n];
	}

	cl_mem cl_gpuVar(cl_mem* mgpu, int n) {
		return mgpu[n];
	}

	cl_mem cl_cpuVar(cl_mem* mcpu, int n) {
		return mcpu[n];
	}

	cl_mem* gpuptr(FBufs *fb, int n) {
		return &fb->mgpu[n];

	}

	cl_float3 cl_int3_to_cl_float3(cl_int3 op) {
		cl_float3 result;
		result.x = (float) op.x;
		result.y = (float) op.y;
		result.z = (float) op.z;
		return result;
	}

	cl_float3 cl_float3_init_with_values(float xa, float ya, float za) {
		cl_float3 v;
		v.x = xa;
		v.y = ya;
		v.z = za;
		return v;
	}

	cl_float3 cl_float3_addFloat(cl_float3 *vector, float op) {
		cl_float3 result;
		result.x = vector->x + op;
		result.y = vector->y + op;
		result.z = vector->z + op;
		return result;
	}

	cl_float3 cl_float3_add_cl_float3(cl_float3 *vector, cl_float3 op) {
		cl_float3 result;
		result.x = vector->x + op.x;
		result.y = vector->y + op.y;
		result.z = vector->z + op.z;
		return result;
	}

	cl_float3 cl_float3_addDouble(cl_float3 *vector, double op) {
		cl_float3 result;
		result.x = vector->x + op;
		result.y = vector->y + op;
		result.z = vector->z + op;
		return result;
	}

	cl_float3 cl_float3_subtractFloat(cl_float3 *vector, float op) {
		cl_float3 result;
		result.x = vector->x - op;
		result.y = vector->y - op;
		result.z = vector->z - op;
		return result;
	}

	cl_float3 cl_float3_subtractDouble(cl_float3 *vector, double op) {
		cl_float3 result;
		result.x = vector->x - op;
		result.y = vector->y - op;
		result.z = vector->z - op;
		return result;
	}

	cl_float3 cl_float3_multiplyFloat(cl_float3 *vector, float op) {
		cl_float3 result;
		result.x = vector->x * op;
		result.y = vector->y * op;
		result.z = vector->z * op;
		return result;
	}

	cl_float3 *cl_float3_multiplyDouble(cl_float3 *v, double op) {
		v->x *= op;
		v->y *= op;
		v->z *= op;
		return v;
	}

	cl_float3 cl_float3_subtract_cl_float3(cl_float3 *vector, cl_float3 *op) {
		cl_float3 result;
		result.x = vector->x - op->x;
		result.y = vector->y - op->y;
		result.z = vector->z - op->z;
		return result;
	}

	cl_float3 cl_float3_devide_cl_float3(cl_float3 *vector, cl_float3 *op) {
		cl_float3 result;
		result.x = vector->x / op->x;
		result.y = vector->y / op->y;
		result.z = vector->z / op->z;
		return result;
	}

	cl_float3 *cl_float3_operator_equal_cl_float3 (cl_float3 *v, cl_float3 *op) {
		v->x = op->x;
		v->y = op->y;
		v->z = op->z;
		return v;
	}

	cl_float3 *cl_float3_operator_equal_cl_int3 (cl_float3 *v, cl_int3 *op) {
		v->x = (float) op->x;
		v->y = (float) op->y;
		v->z = (float) op->z;
		return v;
	}


	void Set(cl_float3* vec, float x, float y, float z) {
		vec->x = x;
		vec->y = y;
		vec->z = z;
	}

	void Set4(cl_float4* vec, float x, float y, float z) {
		vec->x = x;
		vec->y = y;
		vec->z = z;
		vec->w = 0;
	}

	cl_float3 Clamp(cl_float3* vec, float a, float b) {
		cl_float3 result;
		result.x = (vec->x < a) ? a : ((vec->x > b) ? b : vec->x);
		result.y = (vec->y < a) ? a : ((vec->y > b) ? b : vec->y);
		result.z = (vec->z < a) ? a : ((vec->z > b) ? b : vec->z);
		return result;
	}


		class FluidSystem {
	public:
		/////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
		Json::Value obj;
		int					verbosity;
		std::vector<cl_platform_id> m_platform_ids;
		cl_context			m_context;
		cl_device_id 		m_device;
		cl_command_queue	m_queue , upload_queue, dload_queue, track_queue;
		cl_program			m_program;
		cl_kernel			m_Kern[FUNC_MAX];
		cl_mem				m_FParamsDevice, m_FluidDevice, m_FluidTempDevice, m_FGenomeDevice, m_FPrefixDevice;		//GPU pointer containers
		size_t 				num_work_groups, global_work_size, local_work_size;
		bool 				gpu, amdPlatform;
		float 				params[16] = {0};
		std::map< std::string, std::filesystem::path > paths;


		FluidSystem(Json::Value obj_);
		void createFolders();	// Called by RunCL(..) constructor, above.
		void saveCostVols(float max_range);
		void updateQD(float epsilon, float theta, float sigma_q, float sigma_d);
		void updateA ( float lambda, float theta );

		void CleanUp();
		void allocatemem(FParams fparam, FBufs *fbuf, FBufs *ftemp, FGenome fgenome, FPrefix fprefix);
		void exit_(cl_int res);
		cl_int clMemsetD32(cl_mem buffer, int value, size_t count);
		void CalculateWorkGroupSizes(size_t maxWorkGroupSize, size_t numComputeUnits, size_t numItems, size_t &numGroups, size_t &numItemsPerGroup);
		~FluidSystem();

		//void DownloadAndSaveVolume_3Channel(cl_mem buffer, std::string count, boost::filesystem::path folder, size_t image_size_bytes, cv::Size size_mat, int type_mat, bool show );
		//void initializeCostVol(float* k2k, cv::Mat &baseImage, cv::Mat &image, float *cdata, float *hdata, float thresh, int layers);
		//void initializeAD();

		int  convertToString(const char *filename, std::string& s){
			size_t size;
			char*  str;
			std::fstream f(filename, (std::fstream::in | std::fstream::binary));
			if (f.is_open() ) {
				size_t fileSize;
				f.seekg(0, std::fstream::end);
				size = fileSize = (size_t)f.tellg();
				f.seekg(0, std::fstream::beg);
				str = new char[size + 1];
				if (!str) {
					f.close();
					return 0;
				}
				f.read(str, fileSize);
				f.close();
				str[size] = '\0';
				s = str;
				delete[] str;
				return 0;
			}
												cout << "Error: failed to open file\n:" << filename << flush;
			return 1;
		}

		int waitForEventAndRelease(cl_event *event){
												if(verbosity>0) cout << "\nwaitForEventAndRelease_chk0, event="<<event<<" *event="<<*event << flush;
			cl_int status = CL_SUCCESS;
			status = clWaitForEvents(1, event); if (status != CL_SUCCESS) { cout << "\nclWaitForEvents status=" << status << ", " <<  checkerror(status) <<"\n" << flush; exit_(status); }
			status = clReleaseEvent(*event); 	if (status != CL_SUCCESS) { cout << "\nclReleaseEvent status="  << status << ", " <<  checkerror(status) <<"\n" << flush; exit_(status); }
			return status;
		}

		string checkerror(int input) {
			int errorCode = input;
			switch (errorCode) {
			case -9999:											return "Illegal read or write to a buffer";		// NVidia error code
			case CL_DEVICE_NOT_FOUND:							return "CL_DEVICE_NOT_FOUND";
			case CL_DEVICE_NOT_AVAILABLE:						return "CL_DEVICE_NOT_AVAILABLE";
			case CL_COMPILER_NOT_AVAILABLE:						return "CL_COMPILER_NOT_AVAILABLE";
			case CL_MEM_OBJECT_ALLOCATION_FAILURE:				return "CL_MEM_OBJECT_ALLOCATION_FAILURE";
			case CL_OUT_OF_RESOURCES:							return "CL_OUT_OF_RESOURCES";
			case CL_OUT_OF_HOST_MEMORY:							return "CL_OUT_OF_HOST_MEMORY";
			case CL_PROFILING_INFO_NOT_AVAILABLE:				return "CL_PROFILING_INFO_NOT_AVAILABLE";
			case CL_MEM_COPY_OVERLAP:							return "CL_MEM_COPY_OVERLAP";
			case CL_IMAGE_FORMAT_MISMATCH:						return "CL_IMAGE_FORMAT_MISMATCH";
			case CL_IMAGE_FORMAT_NOT_SUPPORTED:					return "CL_IMAGE_FORMAT_NOT_SUPPORTED";
			case CL_BUILD_PROGRAM_FAILURE:						return "CL_BUILD_PROGRAM_FAILURE";
			case CL_MAP_FAILURE:								return "CL_MAP_FAILURE";
			case CL_MISALIGNED_SUB_BUFFER_OFFSET:				return "CL_MISALIGNED_SUB_BUFFER_OFFSET";
			case CL_EXEC_STATUS_ERROR_FOR_EVENTS_IN_WAIT_LIST:	return "CL_EXEC_STATUS_ERROR_FOR_EVENTS_IN_WAIT_LIST";
			case CL_INVALID_VALUE:								return "CL_INVALID_VALUE";
			case CL_INVALID_DEVICE_TYPE:						return "CL_INVALID_DEVICE_TYPE";
			case CL_INVALID_PLATFORM:							return "CL_INVALID_PLATFORM";
			case CL_INVALID_DEVICE:								return "CL_INVALID_DEVICE";
			case CL_INVALID_CONTEXT:							return "CL_INVALID_CONTEXT";
			case CL_INVALID_QUEUE_PROPERTIES:					return "CL_INVALID_QUEUE_PROPERTIES";
			case CL_INVALID_COMMAND_QUEUE:						return "CL_INVALID_COMMAND_QUEUE";
			case CL_INVALID_HOST_PTR:							return "CL_INVALID_HOST_PTR";
			case CL_INVALID_MEM_OBJECT:							return "CL_INVALID_MEM_OBJECT";
			case CL_INVALID_IMAGE_FORMAT_DESCRIPTOR:			return "CL_INVALID_IMAGE_FORMAT_DESCRIPTOR";
			case CL_INVALID_IMAGE_SIZE:							return "CL_INVALID_IMAGE_SIZE";
			case CL_INVALID_SAMPLER:							return "CL_INVALID_SAMPLER";
			case CL_INVALID_BINARY:								return "CL_INVALID_BINARY";
			case CL_INVALID_BUILD_OPTIONS:						return "CL_INVALID_BUILD_OPTIONS";
			case CL_INVALID_PROGRAM:							return "CL_INVALID_PROGRAM";
			case CL_INVALID_PROGRAM_EXECUTABLE:					return "CL_INVALID_PROGRAM_EXECUTABLE";
			case CL_INVALID_KERNEL_NAME:						return "CL_INVALID_KERNEL_NAME";
			case CL_INVALID_KERNEL_DEFINITION:					return "CL_INVALID_KERNEL_DEFINITION";
			case CL_INVALID_KERNEL:								return "CL_INVALID_KERNEL";
			case CL_INVALID_ARG_INDEX:							return "CL_INVALID_ARG_INDEX";
			case CL_INVALID_ARG_VALUE:							return "CL_INVALID_ARG_VALUE";
			case CL_INVALID_ARG_SIZE:							return "CL_INVALID_ARG_SIZE";
			case CL_INVALID_KERNEL_ARGS:						return "CL_INVALID_KERNEL_ARGS";
			case CL_INVALID_WORK_DIMENSION:						return "CL_INVALID_WORK_DIMENSION";
			case CL_INVALID_WORK_GROUP_SIZE:					return "CL_INVALID_WORK_GROUP_SIZE";
			case CL_INVALID_WORK_ITEM_SIZE:						return "CL_INVALID_WORK_ITEM_SIZE";
			case CL_INVALID_GLOBAL_OFFSET:						return "CL_INVALID_GLOBAL_OFFSET";
			case CL_INVALID_EVENT_WAIT_LIST:					return "CL_INVALID_EVENT_WAIT_LIST";
			case CL_INVALID_EVENT:								return "CL_INVALID_EVENT";
			case CL_INVALID_OPERATION:							return "CL_INVALID_OPERATION";
			case CL_INVALID_GL_OBJECT:							return "CL_INVALID_GL_OBJECT";
			case CL_INVALID_BUFFER_SIZE:						return "CL_INVALID_BUFFER_SIZE";
			case CL_INVALID_MIP_LEVEL:							return "CL_INVALID_MIP_LEVEL";
			case CL_INVALID_GLOBAL_WORK_SIZE:					return "CL_INVALID_GLOBAL_WORK_SIZE";
			case CL_SUCCESS:									return "CL_SUCCESS";
		#if CL_HPP_MINIMUM_OPENCL_VERSION >= 200
				case CL_INVALID_DEVICE_QUEUE:						return "CL_INVALID_DEVICE_QUEUE";
				case CL_INVALID_PIPE_SIZE:							return "CL_INVALID_PIPE_SIZE";
		#endif
				default:											return "unknown error code";
				}
			}

		void InitializeKernels(cl_program m_program) {
		// Setup the array of kernels
			cout << "\n----- InitializeKernels() started... -----\n" << flush;
			cl_int status;
			m_Kern[FUNC_INSERT] = 							clCreateKernel(m_program, "insertParticlesCL", &status);
			if(verbosity>0 && status!=CL_SUCCESS) cout << "\nclCreateKernel() Error: " << checkerror(status) << "\n" << flush;
			m_Kern[FUNC_COUNTING_SORT_FULL] = 					clCreateKernel(m_program, "countingSortFull", &status);
			if(verbosity>0 && status!=CL_SUCCESS) cout << "\nclCreateKernel() Error: " << checkerror(status) << "\n" << flush;
			/*m_Kern[FUNC_QUERY] = 							clCreateKernel(m_program, "computeQuery", &err);
			if(verbosity>0 && err!=CL_SUCCESS) cout << "\nclCreateKernel() Error: " << checkerror(err) << "\n" << flush;*/
			m_Kern[FUNC_COMPUTE_PRESS] = 					clCreateKernel(m_program, "computePressure", &status);
			if(verbosity>0 && status!=CL_SUCCESS) cout << "\nclCreateKernel() Error: " << checkerror(status) << "\n" << flush;
			m_Kern[FUNC_COMPUTE_FORCE] = 					clCreateKernel(m_program, "computeForce", &status);
 			m_Kern[FUNC_ADVANCE] = 							clCreateKernel(m_program, "advanceParticles", NULL);
// 			m_Kern[FUNC_EMIT] = 							clCreateKernel(m_program, "emitParticles", NULL);
// 			m_Kern[FUNC_RANDOMIZE] = 						clCreateKernel(m_program, "randomInit", NULL);
 			m_Kern[FUNC_FPREFIXSUM] = 						clCreateKernel(m_program, "prefixSum", &status);
			m_Kern[FUNC_FPREFIXSUMCHANGES] = 				clCreateKernel(m_program, "prefixSumChanges", &status);
 			m_Kern[FUNC_FPREFIXUP] = 						clCreateKernel(m_program, "prefixUp", &status);
 			m_Kern[FUNC_TALLYLISTS] = 						clCreateKernel(m_program, "tally_denselist_lengths", NULL);
// 			m_Kern[FUNC_COMPUTE_DIFFUSION] = 				clCreateKernel(m_program, "computeDiffusion", NULL);
 			m_Kern[FUNC_COUNT_SORT_DENSE_LISTS] = 				clCreateKernel(m_program, "countingSortDenseLists", NULL);
// 			m_Kern[FUNC_COMPUTE_GENE_ACTION] = 				clCreateKernel(m_program, "computeGeneAction", NULL);
// 			m_Kern[FUNC_TALLY_GENE_ACTION] = 				clCreateKernel(m_program, "tallyGeneAction", NULL);
// 			m_Kern[FUNC_COMPUTE_BOND_CHANGES] = 			clCreateKernel(m_program, "computeBondChanges", NULL);
// 			m_Kern[FUNC_COUNTING_SORT_CHANGES] = 			clCreateKernel(m_program, "countingSortChanges", NULL);
// 			m_Kern[FUNC_COMPUTE_NERVE_ACTION] = 			clCreateKernel(m_program, "computeNerveActivation", NULL);
// 			m_Kern[FUNC_COMPUTE_MUSCLE_CONTRACTION] = 		clCreateKernel(m_program, "computeMuscleContraction", NULL);
// 			m_Kern[FUNC_CLEAN_BONDS] = 						clCreateKernel(m_program, "cleanBonds", NULL);
// 			m_Kern[FUNC_HEAL] = 							clCreateKernel(m_program, "heal", NULL);
// 			m_Kern[FUNC_LENGTHEN_MUSCLE] = 					clCreateKernel(m_program, "lengthen_muscle", NULL);
// 			m_Kern[FUNC_LENGTHEN_TISSUE] = 					clCreateKernel(m_program, "lengthen_tissue", NULL);
// 			m_Kern[FUNC_SHORTEN_MUSCLE] = 					clCreateKernel(m_program, "shorten_muscle", NULL);
// 			m_Kern[FUNC_SHORTEN_TISSUE] = 					clCreateKernel(m_program, "shorten_tissue", NULL);
// 			m_Kern[FUNC_STRENGTHEN_MUSCLE] = 				clCreateKernel(m_program, "strengthen_muscle", NULL);
// 			m_Kern[FUNC_STRENGTHEN_TISSUE] = 				clCreateKernel(m_program, "strengthen_tissue", NULL);
// 			m_Kern[FUNC_WEAKEN_MUSCLE] = 					clCreateKernel(m_program, "weaken_muscle", NULL);
// 			m_Kern[FUNC_WEAKEN_TISSUE] = 					clCreateKernel(m_program, "weaken_tissue", NULL);
// 			m_Kern[FUNC_EXTERNAL_ACTUATION] = 				clCreateKernel(m_program, "externalActuation", NULL);
// 			m_Kern[FUNC_FIXED] = 							clCreateKernel(m_program, "fixedParticles", NULL);
 			m_Kern[FUNC_INIT_RANDOMCL] = 					clCreateKernel(m_program, "init_RandCL", &status);
// 			m_Kern[FUNC_ASSEMBLE_MUSCLE_FIBRES_OUTGOING] = 	clCreateKernel(m_program, "assembleMuscleFibresOutGoing", NULL);
// 			m_Kern[FUNC_ASSEMBLE_MUSCLE_FIBRES_INCOMING] = 	clCreateKernel(m_program, "assembleMuscleFibresInComing", NULL);
// 			m_Kern[FUNC_INITIALIZE_BONDS] = 				clCreateKernel(m_program, "initialize_bonds", NULL);
 			m_Kern[FUNC_MEMSET32D] = 						clCreateKernel(m_program, "memset32d_kernel", NULL);
			if(verbosity>0 && status!=CL_SUCCESS) cout << "\nclCreateKernel() Error: " << checkerror(status) << "\n" << flush;

		}
		//////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
		//FluidSystem (RunCL& runcl);
// 		bool clCheck(cl_int status, std::string method, std::string apicall, std::string arg, bool bDebug);
        bool clCheck (cl_int launch_stat, const char* method, const char* apicall, const char* arg, bool bDebug);
		//FluidSystem(RunCL& runCLInstance) : m_runCLInstance(runCLInstance) {}
		void LoadKernel (int id, std::string kname );
		void Initialize ();
        void InitializeOpenCL ();
		void initializeFBufs(FBufs* fluid);
        void InitializeCL ();                             // used for load_sim

		// Particle Utilities
			// Define a flag array to keep track of whether each buffer has been allocated
			bool bufferAllocated[MAX_BUF] = { false };
		void AllocateBuffer(int buf_id, int stride, int cpucnt, int gpucnt, int gpumode, int cpumode);		
        void AllocateBufferDenseLists ( int buf_id, int stride, int gpucnt, int lists );
        void AllocateParticles ( int cnt, int gpu_mode = GPU_DUAL, int cpu_mode = CPU_YES );
        void AddNullPoints ();
        int  AddParticleMorphogenesis2(cl_float3* Pos, cl_float3* Vel, uint Age, uint Clr, uint *_ElastIdxU, float *_ElastIdxF, uint *_Particle_Idx, uint Particle_ID, uint Mass_Radius, uint NerveIdx, float* _Conc, uint* _EpiGen );
        
        
		//void AddEmit ( float spacing );
		int NumPoints ()				{ return mNumPoints; }
		int MaxPoints ()                { return mMaxPoints; }
		int ActivePoints ()             { return mActivePoints; }
		cl_float3* getPos ( int n )	    { return &bufV3(&m_Fluid, FPOS)[n]; }
		cl_float3* getVel ( int n )	    { return &bufV3(&m_Fluid, FVEL)[n]; }
		cl_float3* getVeval ( int n )   { return &bufV3(&m_Fluid, FVEVAL)[n]; }
		cl_float3* getForce ( int n )   { return &bufV3(&m_Fluid, FFORCE)[n]; }
		
		float* getPres ( int n )        { return &bufF(&m_Fluid, FPRESS)[n];}
		float* getDensity ( int n )     { return &bufF(&m_Fluid, FDENSITY)[n];}
		
		uint* getClr ( int n )			{ return &bufI(&m_Fluid, FCOLOR)[n]; }
		uint* getAge ( int n )			{ return &bufI(&m_Fluid, FAGE)[n]; }
        uint* getElastIdx( int n )      { return &bufI(&m_Fluid, FELASTIDX)[n*(BONDS_PER_PARTICLE * DATA_PER_BOND)]; }        //note #define FELASTIDX   14
        uint* getParticle_Idx( int n )  { return &bufI(&m_Fluid, FPARTICLEIDX)[n*BONDS_PER_PARTICLE*2]; }
        uint* getParticle_ID(int n )    { return &bufI(&m_Fluid, FPARTICLE_ID)[n]; }
        uint* getMass_Radius(int n )    { return &bufI(&m_Fluid, FMASS_RADIUS)[n]; }
        uint* getNerveIdx( int n )      { return &bufI(&m_Fluid, FNERVEIDX)[n]; }              //#define FNERVEIDX        15    //# uint
        float* getConc(int tf)          { return &bufF(&m_Fluid, FCONC)[tf*mMaxPoints];}       //note #define FCONC       16    //# float[NUM_TF]        NUM_TF = num transcription factors & morphogens
        uint* getEpiGen(int gene)       { return &bufI(&m_Fluid, FEPIGEN)[gene*mMaxPoints];}   //note #define FEPIGEN     17    //# uint[NUM_GENES] // used in savePoints...
                                                                                             //NB int mMaxPoints is set even if FluidSetupCL(..) isn't called, e.g. in makedemo ..
		// Setup
		void SetupSPH_Kernels ();
		void SetupDefaultParams ();
		void SetupExampleParams (uint simSpace);
        void SetupExampleGenome();
		void SetupSpacing ();
        void SetupAddVolumeMorphogenesis2(cl_float3 min, cl_float3 max, float spacing, float offs, uint demoType );  // NB ony used in WriteDemoSimParams()
		void SetupGrid ( cl_float3 min, cl_float3 max, float sim_scale, float cell_size);
		void AllocateGrid ();
        void AllocateGrid(int gpu_mode, int cpu_mode);
        void SetupSimulation(int gpu_mode, int cpu_mode);

		// Simulation
		void Run ();	
        void Run( const char * relativePath, int frame, bool debug, bool gene_activity, bool remodelling );
        void RunSimulation ();
        
        void Run2PhysicalSort();
        void Run2InnerPhysicalLoop();
        void Run2GeneAction();
        void Run2Remodelling(uint steps_per_InnerPhysicalLoop);
        void Run2Simulation();
        
        void setFreeze(bool freeze);
        void Freeze ();
        void Freeze (const char * relativePath, int frame, bool debug, bool gene_activity, bool remodelling);
		void AdvanceTime ();
		
		void Exit ();
        void Exit_no_CL ();
		void CreateFluidBuffers();
		void TransferToCL ();
		void TransferFromCL ();
        //DT=deltaTime?
		double GetDT()		{ return m_DT; }
		
		// Acceleration Grid
		cl_float3 GetGridRes ()		{ return cl_int3_to_cl_float3(m_GridRes); }
		cl_float3 GetGridMin ()		{ return m_GridMin; }
		cl_float3 GetGridMax ()		{ return m_GridMax; }
		cl_float3 GetGridDelta ()	{ return m_GridDelta; }

		void FluidSetupCL ( int num, int gsrch, cl_int3 res, cl_float3 size, cl_float3 delta, cl_float3 gmin, cl_float3 gmax, int total, int chk );
		void FluidParamCL(float ss, float sr, float pr, float mass, float rest, cl_float3 bmin, cl_float3 bmax, float estiff, float istiff, float visc, float surface_tension, float damp, float fmin, float fmax, float ffreq, float gslope, float gx, float gy, float gz, float al, float vl, float a_f, float a_p);

        void Init_CLRand ();
		void InsertParticlesCL ( uint* gcell, uint* ccell, uint* gcnt );
		void PrefixSumCellsCL ( int zero_offsets );
		void PrefixSumCellsCL2 ( int zero_offsets );
		void CountingSortFullCL ( cl_float3* gpos );
        
        void InitializeBondsCL ();
        
        void InsertChangesCL ( /*uint* gcell, uint* gndx, uint* gcnt*/ );
        void PrefixSumChangesCL ( int zero_offsets );
        void CountingSortChangesCL ( );
        
		void ComputePressureCL ();
		void ComputeDiffusionCL();
		void ComputeForceCL ();
        void ComputeGenesCL ();
        void AssembleFibresCL ();
        void ComputeBondChangesCL (uint steps_per_InnerPhysicalLoop);
        void ComputeParticleChangesCL ();
        void CleanBondsCL ();                                         // Should this functionality be rolled into countingSortFull() ? OR should it be kept separate?
        
        void TransferToTempCL ( int buf_id, int sz );
        void TransferFromTempCL ( int buf_id, int sz );
		void TransferPosVelVeval ();                                    // Called B4 1st timestep, & B4 AdvanceCL thereafter.
        void TransferPosVelVevalFromTemp ();
        void ZeroVelCL ();
        
        void AdvanceCL ( float time, float dt, float ss );            // Writes to ftemp
        void SpecialParticlesCL (float tm, float dt, float ss);       // Reads fbuf, writes to ftemp, corects AdvanceCL().
		void EmitParticlesCL ( float time, int cnt );
        
		// I/O Files
        void SaveUintArray( uint* array, int numElem1, const char * relativePath, float count=0 );
		//void SaveUintArray( uint* array, int numElem1, stringstream /*const char * */  relativePath );
        void SaveUintArray_2Columns( uint* array, int numElem1, int buff_len, const char * relativePath , float count=0 ); /// Used to save DESNSE_LIST_CHANGES (particle,bondIdx) arrays to .csv for debugging.
        void SaveUintArray_2D ( uint* array, int numElem1, int numElem2, const char * relativePath , float count=0 );
        
        void SavePointsVTP2 ( const char * relativePath, int frame );
        void SavePointsCSV2 ( const char * relativePath, int frame );
        void ReadSimParams ( const char * relativePath );    // path to folder containing simparams and .csv files
        void WriteDemoSimParams ( const char * relativePath, int gpu_mode, int cpu_mode, uint num_particles, float spacing, float x_dim, float y_dim, float z_dim, uint demoType, uint simSpace, uint debug); // Write standard demo to file, as demonstration of file format. 
        void WriteSimParams ( const char * relativePath );
        void ReadPointsCSV2 ( const char * relativePath, int gpu_mode, int cpu_mode);
        //void ReadPointsCSV2_DEBUG ( const char * relativePath, int gpu_mode, int cpu_mode);

        void ReadSpecificationFile(const char* relativePath);
        void WriteExampleSpecificationFile ( const char * relativePath );
        void WriteSpecificationFile_fromLaunchParams ( const char * relativePath );
        void WriteResultsCSV ( const char * input_folder, const char * output_folder, uint num_particles_start );

        // Genome for Morphogenesis
        void UpdateGenome ();
        void SetGenome ( FGenome newGenome ){m_FGenome=newGenome;}
		int  mk_subdir(char* path);
        void ReadGenome( const char * relativePath);
        void Set_genome_tanh_param();
        void WriteGenome( const char * relativePath);
        FGenome	GetGenome();/*{
            FGenome tempGenome = m_FGenome;
            for (int i=0; i<3;i++)for(int j=0; j<12; j++)tempGenome.param[i][j]=m_FGenome.param[i][j];
            return tempGenome;
        }*/
        
        
		// Parameters
		void UpdateParams ();
		void SetParam(int p, float v);//     { m_Param[p] = v; }           // NB must call UpdateParams() afterwards, to call FluidParamCL
		//void SetParam (RunCL& runcl, int p, int v )		{ m_Param[p] = (float) v; }
		float GetParam ( int p )			{ return (float) m_Param[p]; }

		cl_float3 GetVec ( int p )			{ return m_Vec[p]; }
		void SetVec(int p, cl_float3 v);
		void SetDebug(uint b) { m_debug=b; verbosity=b; /*mbDebug = (bool)b;*/
            std::cout<<"\n\nSetDebug(uint b): b="<<b<<", verbosity = "<<verbosity<<", (verbosity>1)="<<(verbosity>1)<<"\n"<<std::flush;
        }
        uint GetDebug(){return verbosity;}
        
        struct {
            const char * relativePath;
            uint num_particles;
            //float spacing;
            float x_dim, y_dim, z_dim, pos_x, pos_y, pos_z;
            uint demoType, simSpace;
            char paramsPath[256];
            char pointsPath[256];
            char genomePath[256];
            char outPath[256];
            uint num_files=1, steps_per_file=1, freeze_steps=0, verbosity=0, steps_per_InnerPhysicalLoop=3;
            int file_num=0, file_increment=0;
            char save_ply='n', save_csv='n', save_vtp='n',  gene_activity='n', remodelling='n', read_genome='n';
            
            float m_Time, m_DT, gridsize, spacing, simscale, smoothradius, visc, surface_tension, mass, radius, /*dist,*/ intstiff, extstiff, extdamp, accel_limit, vel_limit, grav, ground_slope, force_min, force_max, force_freq;
            cl_float3 volmin, volmax, initmin, initmax;
            float actuation_factor, actuation_period;
        }launchParams ;
	
	private:
		bool						mbDebug;
        uint                        m_debug;            // 0=full speed, 1=current special output,  2=host cout, 3=device printf, 4=SaveUintArray, 5=save csv after each kernel

		// Time
		int							m_Frame;
        int                         m_Debug_file;
		float						m_DT;
		float						m_Time;	

		// Simulation Parameters                                //  NB MAX_PARAM = 50 
		float						m_Param [ MAX_PARAM ];	    // 0-47 used.  see defines above. NB m_Param[1] = maximum number of points.
		cl_float3					m_Vec   [ MAX_PARAM ];      // 0-12 used

		// SPH Kernel functions
		float						m_R2, m_Poly6Kern, m_LapKern, m_SpikyKern;		

		// Particle Buffers
		int						mNumPoints;
		int						mMaxPoints;
		int						mActivePoints;

		FBufs					m_Fluid;				// Fluid buffers - NB this is an array of pointers (in mPackBuf ?)
		FBufs					m_FluidTemp;			// Fluid buffers (temporary)
		FParams					m_FParams;				// Fluid parameters struct - that apply to all particles 
		FGenome					m_FGenome;				// Genome struct of arrays for genne params
/*
		cl_mem				clFBuf;					// GPU pointer containers
		cl_mem				clFTemp;
		cl_mem				clFParams;
		cl_mem				clFGenome;
		cl_device_idptr				clFBuf;					// GPU pointer containers
		cl_device_idptr				clFTemp;
		cl_device_idptr				clFParams;
		cl_device_idptr				clFGenome;*/


		// Acceleration Grid
		int						m_GridTotal;			// total # cells
		cl_int3					m_GridRes;				// resolution in each axis
		cl_float3				m_GridMin;				// volume of grid (may not match domain volume exactly)
		cl_float3				m_GridMax;
		cl_float3				m_GridSize;				// physical size in each axis
		cl_float3				m_GridDelta;
		int						m_GridSrch;
		int						m_GridAdjCnt;
		int						m_GridAdj[216];         // 216 => up to 8 particles per cell

		int*					mPackGrid;
	};	
#endif
