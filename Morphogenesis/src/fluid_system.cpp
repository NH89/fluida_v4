#include <assert.h>
#include <iostream>
#include <stdlib.h>
#include <unistd.h>
#include <chrono>
#include <cstring>
#include <stdio.h>
#include <iostream>
#include <fstream>
#include <vector>
#include "vector.h"
#include <math.h>
#include <regex>
#include <jsoncpp/json/json.h>
#include "fluid.h"
#include "fluid_system.h"
#include <thread>

#define CHECK_ERROR(status) if (status != CL_SUCCESS) { printf("Error: %d\n", status); exit(1); }
#define SDK_SUCCESS 0
#define SDK_FAILURE 1
#define UINT_MAXSIZE 65535
#define INT_MAX 2147483647

using namespace std;


std::string checkerror(int input) {
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
        #if CL_HPP_MINIMUM_OPENCL_VERSION >= 200
            case CL_INVALID_DEVICE_QUEUE:						return "CL_INVALID_DEVICE_QUEUE";
            case CL_INVALID_PIPE_SIZE:							return "CL_INVALID_PIPE_SIZE";
        #endif
            default:											return "unknown error code";
		}
	}

// bool clCheck(cl_int status, const char* method, const char* apicall, const char* arg, bool bDebug)
// {
//
//     // DEBUG IMPLEMENTATION MISSING!!!
// //         if (bDebug) {
// //         kern_stat = clFinish(m_queue);
// //     }
//
//     if (status != CL_SUCCESS) {
//         std::string errorMessage = checkerror(status);
//         std::cout << "OpenCL Error: " << errorMessage << std::endl;
//         std::cout << "Caller: " << method << std::endl;
//         std::cout << "Call: " << apicall << std::endl;
//         std::cout << "Args: " << arg << std::flush;
//         return false;
//     }
//     return true;
// }
/*
FluidSystem::FluidSystem (RunCL& runcl){
    if (verbosity>1)cout<<"\n\nFluidSystem ()"<<std::flush;
    memset ( &m_Fluid, 0,		sizeof(FBufs) );
    memset ( &m_FluidTemp, 0,	sizeof(FBufs) );
    memset ( &m_FParams, 0,		sizeof(FParams) );
    memset ( &m_FGenome, 0,		sizeof(FGenome) );
    mNumPoints = 0;
    mMaxPoints = 0;
    mPackGrid = 0x0;
    m_Frame = 0;
    //Kernels are get defined in RunCl.h!!!
    //for (int n=0; n < FUNC_MAX; n++ ) runcl.m_Kern[n] = (cl_kernel) -1;
}
*/

/*
bool FluidSystem::clCheck(cl_int status, std::string method, std::string apicall, std::string arg, bool bDebug) {


    // Check for null pointers and handle them appropriately
    const std::string methodString = (method == "NULL") ? "" : method;
    const std::string apicallString = (apicall == "NULL") ? "" : apicall;
    const std::string argString = (arg == "NULL") ? "" : arg;

    // Check for null pointers and handle them appropriately
    if (methodString.empty() || apicallString.empty() || argString.empty()) {
        std::cout << "WARING: method, apicall or arg is empty! " << std::endl;
        std::cout << "Caller: " << methodString << std::endl;
        std::cout << "Call: " << apicallString << std::endl;
        std::cout << "Args: " << argString << std::flush;
        return false;
    }

    // DEBUG IMPLEMENTATION MISSING!!!
    if (apicallString == "clEnqueueNDRangeKernel") {
        std::cout << "\n\n\n+ + + + + + + + + + +  K E R N E L   L A U N C H  + + + + + + + + + + +\n" << std::flush;
        if (argString != "FUNC_COUNTING_SORT_FULL 1") std::cout << "Kernel: " << argString << "\n" << std::flush;
        if (argString == "FUNC_COUNTING_SORT_FULL 1") std::cout << "THE Kernel: FUNC_COUNTING_SORT_FULL\n" << std::flush;
    }

    if (status != CL_SUCCESS) {
        std::string errorMessage = checkerror(status);
        std::cout << "\nOpenCL Error: " << errorMessage << std::endl;
        std::cout << "Caller: " << methodString << std::endl;
        std::cout << "Call: " << apicallString << std::endl;
        std::cout << "Args: " << argString << std::flush;
        return false;
    }
    return true;
}*/


bool FluidSystem::clCheck(cl_int status, const char* method, const char* apicall, const char* arg, bool bDebug) {

    std::cout << "                                                                                                                            clCheck() called for ";
    if (method != nullptr) {
        std::cout << method;
    }
    std::cout << ", ";
    if (apicall != nullptr) {
        std::cout << apicall;
    }
    std::cout << ", ";
    if (arg != nullptr) {
        std::cout << arg;
    }
    std::cout << ", + + +\n" << std::flush;
    if (status != CL_SUCCESS) {
        std::string errorMessage = checkerror(status);
        std::cout << "\nOpenCL Error: " << errorMessage << std::endl;
        std::cout << "Caller: " << method << std::endl;
        std::cout << "Call: " << apicall << std::endl;
        std::cout << "Args: " << arg << std::flush;
        return false;
    }
    return true;
}

// void FluidSystem::CalculateWorkGroupSizes(size_t maxWorkGroupSize, size_t numComputeUnits, size_t numItems, size_t &numGroups, size_t &numItemsPerGroup) {
//     // Calculate the ideal number of items per group
//     numItemsPerGroup = std::min(maxWorkGroupSize, numItems / numComputeUnits);
//
//     // Calculate the number of work groups
//     numGroups = (numItems + numItemsPerGroup - 1) / numItemsPerGroup;
// }

void FluidSystem::initializeFBufs(FBufs* fluid) {
    // Check if fluid is a valid pointer
    if (fluid == nullptr) {
        return; // Return if fluid is invalid
    }

    // Initialize all memory pointers to nullptr
    for (int i = 0; i < MAX_BUF; ++i) {

        fluid->mcpu[i] = nullptr;
        fluid->mgpu[i] = nullptr;

    }
}
///////////////////////////////////////////////////////////////// Initialize OpenCL /////////////

FluidSystem::FluidSystem(Json::Value obj_)
{
	obj = obj_;
	verbosity = obj["verbosity"].asInt();
																						if(verbosity>1) cout << "\nFluidSystem_chk 0\n" << flush;
	//createFolders( );																	/*Step1: Getting platforms and choose an available one.*/////////
	cl_uint 		numPlatforms;														//the NO. of platforms
	cl_platform_id 	platform 		= NULL;												//the chosen platform
	cl_int			status 			= clGetPlatformIDs(0, NULL, &numPlatforms);			if (status != CL_SUCCESS){ cout << "Error: Getting platforms!" << endl; exit_(status); }
	uint			conf_platform	= obj["opencl_platform"].asUInt();					if(verbosity>1) cout << "numPlatforms = " << numPlatforms << "\n" << flush;
	if (numPlatforms > conf_platform){																/*Choose the platform.*/
		cl_platform_id* platforms 	= (cl_platform_id*)malloc(numPlatforms * sizeof(cl_platform_id));
		status 	 					= clGetPlatformIDs(numPlatforms, platforms, NULL);	if (status != CL_SUCCESS){ cout << "Error: Getting platformsIDs" << endl; exit_(status); }
		platform 					= platforms[ conf_platform ];
		free(platforms);																if(verbosity>1) cout << "\nplatforms[0] = "<<platforms[0]<<", \nplatforms[1] = "<<platforms[1]\
																						<<"\nSelected platform number :"<<conf_platform<<",\n cl_platform_id platform = " << platform<<"\n"<<flush;
	} else {cout<<"Platform num "<<conf_platform<<" not available."<<flush; exit(0);}

	cl_uint				numDevices = 0;													/*Step 2:Query the platform.*//////////////////////////////////
	cl_device_id        *devices;
	status = clGetDeviceIDs(platform, CL_DEVICE_TYPE_GPU, 0, NULL, &numDevices);		if (status != CL_SUCCESS) {cout << "\nstatus = " << checkerror(status) <<"\n"<<flush; exit_(status);}
	uint conf_device = obj["opencl_device"].asUInt();

	if (numDevices > conf_device){
		devices = (cl_device_id*)malloc(numDevices * sizeof(cl_device_id));
		status  = clGetDeviceIDs(platform, CL_DEVICE_TYPE_GPU, numDevices, devices, NULL);
	}																					if (status != CL_SUCCESS) {cout << "\nstatus = " << checkerror(status) <<"\n"<<flush; exit_(status);}

																						if(verbosity>1) cout << "FluidSystem_chk 3\n" << flush;
                                                                                        if(verbosity>1) cout << "\ncl_device_id  devices = " << devices << "\n" << flush;

	cl_context_properties cps[3]={CL_CONTEXT_PLATFORM,(cl_context_properties)platform,0};/*Step 3: Create context.*////////////////////////////////////
	m_context = clCreateContextFromType( cps, CL_DEVICE_TYPE_GPU, NULL, NULL, &status); if(status!=0) 			{cout<<"\nstatus="<<checkerror(status)<<"\n"<<flush;exit_(status);}
	m_device  = devices[conf_device];													/*Step 4: Create command queue & associate context.*///////////
	cl_command_queue_properties prop[] = { 0 };											//  NB Device (GPU) queues are out-of-order execution -> need synchronization.
	m_queue =     clCreateCommandQueueWithProperties(m_context, m_device, prop, &status);	if(status!=CL_SUCCESS)	{cout<<"\nstatus="<<checkerror(status)<<"\n"<<flush;exit_(status);};
	upload_queue = clCreateCommandQueueWithProperties(m_context, m_device, prop, &status);	if(status!=CL_SUCCESS)	{cout<<"\nstatus="<<checkerror(status)<<"\n"<<flush;exit_(status);};
	dload_queue = clCreateCommandQueueWithProperties(m_context, m_device, prop, &status);	if(status!=CL_SUCCESS)	{cout<<"\nstatus="<<checkerror(status)<<"\n"<<flush;exit_(status);};
	track_queue = clCreateCommandQueueWithProperties(m_context, m_device, prop, &status);	if(status!=CL_SUCCESS)	{cout<<"\nstatus="<<checkerror(status)<<"\n"<<flush;exit_(status);};
																						if(verbosity>1) cout << "FluidSystem_chk 4: \n" << upload_queue << flush;

																						// Multiple queues for latency hiding: Upload, Download, Mapping, Tracking,... autocalibration, SIRFS, SPMP
																						// NB Might want to create command queues on multiple platforms & devices.
																						// NB might want to divde a task across multiple MPI Ranks on a multi-GPU WS or cluster.
    /*Step 5: Create program object*///////////////////////////////
    const char *filename = obj["kernel_filepath"].asCString();
    printf("\nKernel File Path: %s\n", filename);

    string sourceStr;
	status 						= convertToString(filename, sourceStr);					if(status!=CL_SUCCESS)	{cout<<"\nconvertToString status="<<checkerror(status)<<"\n"<<flush;exit_(status);}
	const char 	*source 		= sourceStr.c_str();
	size_t 		sourceSize[] 	= { strlen(source) };
    printf("Source Size: %zu bytes. \n", sourceSize[0]); //currently 26758 bytes
	m_program 	= clCreateProgramWithSource(m_context, 1, &source, sourceSize, &status);    if(status!=CL_SUCCESS)	{cout<<"\nstatus="<<checkerror(status)<<"\n"<<flush;exit_(status);};

    //TODO The apsolute path here needs to be read from the jason file
    cout << obj["includeOptions_filepath"].asCString();
    const char *includeOptions =  obj["includeOptions_filepath"].asCString();
    //const char includeOptions[] = "-I /lib/i386-linux-gnu -I \"/home/goldi/Documents/KDevelop Projects/Morphogenesis/Morphogenesis/src\""; // Paths to include directories ///// -I /usr/lib/gcc/x86_64-linux-gnu/11/include /////

	status = clBuildProgram(m_program, 1, devices, includeOptions, NULL, NULL);					/*Step 6: Build program.*/////////////////////
	if (status != CL_SUCCESS){
		printf("\nclBuildProgram failed: %d\n", status);
		char buf[0x10000];
		clGetProgramBuildInfo(m_program, m_device, CL_PROGRAM_BUILD_LOG, 0x10000, buf, NULL);
		printf("\n%s\n End of clBuildProgram error log..", buf);
		exit_(status);
	}

	//Initialize the Array of Kernels "m_Kern[]"
	InitializeKernels(m_program);

    //initializeFBufs(&m_Fluid);

	m_FParamsDevice=m_FluidDevice=m_FluidTempDevice=m_FGenomeDevice=0;		// set device pointers to zero
																						if(verbosity>0) cout << "\n-----FluidSystem::FluidSystem finished-----\n\n" << flush;
}

FluidSystem::~FluidSystem()
{
	cl_int status;

	status = clReleaseKernel(m_Kern[FUNC_INSERT]); 							if (status != CL_SUCCESS) {cout << "\nRelease Kernel FUNC_INSERT status = " << checkerror(status) << "\n" << flush;}
	status = clReleaseKernel(m_Kern[FUNC_COUNTING_SORT_FULL]); 					if (status != CL_SUCCESS) {cout << "\nRelease Kernel FUNC_COUNTING_SORT_FULL status = " << checkerror(status) << "\n" << flush;}
// 	status = clReleaseKernel(m_Kern[FUNC_QUERY]); 							if (status != CL_SUCCESS) {cout << "\nRelease Kernel FUNC_QUERY status = " << checkerror(status) << "\n" << flush;}
 	status = clReleaseKernel(m_Kern[FUNC_COMPUTE_PRESS]); 					if (status != CL_SUCCESS) {cout << "\nRelease Kernel FUNC_COMPUTE_PRESS status = " << checkerror(status) << "\n" << flush;}
 	status = clReleaseKernel(m_Kern[FUNC_COMPUTE_FORCE]); 					if (status != CL_SUCCESS) {cout << "\nRelease Kernel FUNC_COMPUTE_FORCE status = " << checkerror(status) << "\n" << flush;}
// 	status = clReleaseKernel(m_Kern[FUNC_ADVANCE]); 						if (status != CL_SUCCESS) {cout << "\nRelease Kernel FUNC_ADVANCE status = " << checkerror(status) << "\n" << flush;}
// 	status = clReleaseKernel(m_Kern[FUNC_EMIT]); 							if (status != CL_SUCCESS) {cout << "\nRelease Kernel FUNC_EMIT status = " << checkerror(status) << "\n" << flush;}
// 	status = clReleaseKernel(m_Kern[FUNC_RANDOMIZE]); 						if (status != CL_SUCCESS) {cout << "\nRelease Kernel FUNC_RANDOMIZE status = " << checkerror(status) << "\n" << flush;}
// 	status = clReleaseKernel(m_Kern[FUNC_SAMPLE]); 							if (status != CL_SUCCESS) {cout << "\nRelease Kernel FUNC_SAMPLE status = " << checkerror(status) << "\n" << flush;}
	status = clReleaseKernel(m_Kern[FUNC_FPREFIXSUM]); 						if (status != CL_SUCCESS) {cout << "\nRelease Kernel FUNC_FPREFIXSUM status = " << checkerror(status) << "\n" << flush;}
	status = clReleaseKernel(m_Kern[FUNC_FPREFIXSUMCHANGES]); 			    if (status != CL_SUCCESS) {cout << "\nRelease Kernel FUNC_FPREFIXSUMCHANGES status = " << checkerror(status) << "\n" << flush;}
	status = clReleaseKernel(m_Kern[FUNC_FPREFIXUP]); 					if (status != CL_SUCCESS) {cout << "\nRelease Kernel FUNC_FPREFIXUP status = " << checkerror(status) << "\n" << flush;}
	status = clReleaseKernel(m_Kern[FUNC_TALLYLISTS]);						if (status != CL_SUCCESS) {cout << "\nRelease Kernel FUNC_TALLYLISTS status = " << checkerror(status) << "\n" << flush;}
// 	status = clReleaseKernel(m_Kern[FUNC_COMPUTE_DIFFUSION]);				if (status != CL_SUCCESS) {cout << "\nRelease Kernel FUNC_COMPUTE_DIFFUSION status = " << checkerror(status) << "\n" << flush;}
 	status = clReleaseKernel(m_Kern[FUNC_COUNT_SORT_DENSE_LISTS]);				if (status != CL_SUCCESS) {cout << "\nRelease Kernel FUNC_COUNT_SORT_LISTS status = " << checkerror(status) << "\n" << flush;}
// 	status = clReleaseKernel(m_Kern[FUNC_COMPUTE_GENE_ACTION]);				if (status != CL_SUCCESS) {cout << "\nRelease Kernel FUNC_COMPUTE_GENE_ACTION status = " << checkerror(status) << "\n" << flush;}
// 	status = clReleaseKernel(m_Kern[FUNC_TALLY_GENE_ACTION]);				if (status != CL_SUCCESS) {cout << "\nRelease Kernel FUNC_TALLY_GENE_ACTION status = " << checkerror(status) << "\n" << flush;}
// 	status = clReleaseKernel(m_Kern[FUNC_COMPUTE_BOND_CHANGES]);			if (status != CL_SUCCESS) {cout << "\nRelease Kernel FUNC_COMPUTE_BOND_CHANGES status = " << checkerror(status) << "\n" << flush;}
// 	status = clReleaseKernel(m_Kern[FUNC_COUNTING_SORT_CHANGES]);			if (status != CL_SUCCESS) {cout << "\nRelease Kernel FUNC_COUNTING_SORT_CHANGES status = " << checkerror(status) << "\n" << flush;}
// 	status = clReleaseKernel(m_Kern[FUNC_COMPUTE_NERVE_ACTION]);			if (status != CL_SUCCESS) {cout << "\nRelease Kernel FUNC_COMPUTE_NERVE_ACTION status = " << checkerror(status) << "\n" << flush;}
// 	status = clReleaseKernel(m_Kern[FUNC_COMPUTE_MUSCLE_CONTRACTION]);		if (status != CL_SUCCESS) {cout << "\nRelease Kernel FUNC_COMPUTE_MUSCLE_CONTRACTION status = " << checkerror(status) << "\n" << flush;}
// 	status = clReleaseKernel(m_Kern[FUNC_CLEAN_BONDS]);						if (status != CL_SUCCESS) {cout << "\nRelease Kernel FUNC_CLEAN_BONDS status = " << checkerror(status) << "\n" << flush;}
// 	status = clReleaseKernel(m_Kern[FUNC_HEAL]);							if (status != CL_SUCCESS) {cout << "\nRelease Kernel FUNC_HEAL status = " << checkerror(status) << "\n" << flush;}
// 	status = clReleaseKernel(m_Kern[FUNC_LENGTHEN_MUSCLE]);					if (status != CL_SUCCESS) {cout << "\nRelease Kernel FUNC_LENGTHEN_MUSCLE status = " << checkerror(status) << "\n" << flush;}
// 	status = clReleaseKernel(m_Kern[FUNC_LENGTHEN_TISSUE]);					if (status != CL_SUCCESS) {cout << "\nRelease Kernel FUNC_LENGTHEN_TISSUE status = " << checkerror(status) << "\n" << flush;}
// 	status = clReleaseKernel(m_Kern[FUNC_SHORTEN_MUSCLE]);					if (status != CL_SUCCESS) {cout << "\nRelease Kernel FUNC_SHORTEN_MUSCLE status = " << checkerror(status) << "\n" << flush;}
// 	status = clReleaseKernel(m_Kern[FUNC_SHORTEN_TISSUE]);					if (status != CL_SUCCESS) {cout << "\nRelease Kernel FUNC_SHORTEN_TISSUE status = " << checkerror(status) << "\n" << flush;}
// 	status = clReleaseKernel(m_Kern[FUNC_STRENGTHEN_TISSUE]);				if (status != CL_SUCCESS) {cout << "\nRelease Kernel FUNC_STRENGTHEN_TISSUE status = " << checkerror(status) << "\n" << flush;}
// 	status = clReleaseKernel(m_Kern[FUNC_WEAKEN_MUSCLE]);					if (status != CL_SUCCESS) {cout << "\nRelease Kernel FUNC_WEAKEN_MUSCLE status = " << checkerror(status) << "\n" << flush;}
// 	status = clReleaseKernel(m_Kern[FUNC_WEAKEN_TISSUE]);					if (status != CL_SUCCESS) {cout << "\nRelease Kernel FUNC_WEAKEN_TISSUE status = " << checkerror(status) << "\n" << flush;}
// 	status = clReleaseKernel(m_Kern[FUNC_EXTERNAL_ACTUATION]);				if (status != CL_SUCCESS) {cout << "\nRelease Kernel FUNC_EXTERNAL_ACTUATION status = " << checkerror(status) << "\n" << flush;}
// 	status = clReleaseKernel(m_Kern[FUNC_FIXED]);							if (status != CL_SUCCESS) {cout << "\nRelease Kernel FUNC_FIXED status = " << checkerror(status) << "\n" << flush;}
 	status = clReleaseKernel(m_Kern[FUNC_INIT_RANDOMCL]);				if (status != CL_SUCCESS) {cout << "\nRelease Kernel FUNC_INIT_RANDOMCL status = " << checkerror(status) << "\n" << flush;}
// 	status = clReleaseKernel(m_Kern[FUNC_ASSEMBLE_MUSCLE_FIBRES_OUTGOING]);	if (status != CL_SUCCESS) {cout << "\nRelease Kernel FUNC_ASSEMBLE_MUSCLE_FIBRES_OUTGOING status = " << checkerror(status) << "\n" << flush;}
// 	status = clReleaseKernel(m_Kern[FUNC_ASSEMBLE_MUSCLE_FIBRES_INCOMING]);	if (status != CL_SUCCESS) {cout << "\nRelease Kernel FUNC_ASSEMBLE_MUSCLE_FIBRES_INCOMING status = " << checkerror(status) << "\n" << flush;}
// 	status = clReleaseKernel(m_Kern[FUNC_INITIALIZE_BONDS]);				if (status != CL_SUCCESS) {cout << "\nRelease Kernel FUNC_INITIALIZE_BONDS status = " << checkerror(status) << "\n" << flush;}
    // Release kernel
    status = clReleaseKernel(m_Kern[FUNC_MEMSET32D]);                   if (status != CL_SUCCESS) {cout << "\nRelease Kernel FUNC_INIT_RANDOMCL status = " << checkerror(status) << "\n" << flush;}

	status = clReleaseProgram(m_program);			if (status != CL_SUCCESS)	{ cout << "\nRelease Program status = " << checkerror(status) <<"\n"<<flush; }
	status = clReleaseCommandQueue(m_queue);		if (status != CL_SUCCESS)	{ cout << "\nRelease CQ1 status = " << checkerror(status) <<"\n"<<flush; }
	status = clReleaseCommandQueue(upload_queue);	if (status != CL_SUCCESS)	{ cout << "\nRelease CQ2 status = " << checkerror(status) <<"\n"<<flush; }
	status = clReleaseCommandQueue(dload_queue);	if (status != CL_SUCCESS)	{ cout << "\nRelease CQ3 status = " << checkerror(status) <<"\n"<<flush; }
	status = clReleaseCommandQueue(track_queue);	if (status != CL_SUCCESS)	{ cout << "\nRelease CQ4 status = " << checkerror(status) <<"\n"<<flush; }
	status = clReleaseContext(m_context);			if (status != CL_SUCCESS)	{ cout << "\nRelease Context status = " << checkerror(status) <<"\n"<<flush; }
}

void FluidSystem::InitializeOpenCL()
{
																				if(verbosity>0) cout << "-----FluidSystem::InitializeOpenCL() started -----\n" << flush;
	stringstream 		ss;
	ss << "allocatemem";
	cl_int status;
	cl_event writeEvt;

	int layerstep 		= m_FParams.szPnts;

    																				if(verbosity>0) cout << "FluidSystem::InitializeOpenCL_chk1\n" << flush;
	cl_int res;
    /*    cl_int status;
    m_FluidTemp.mgpu[buf_id] = clCreateBuffer(m_context, CL_MEM_READ_WRITE, sz, NULL, &status);

        if(status!=CL_SUCCESS)	{cout<<"\nTransferToTempCL: clCreateBuffer status="<<checkerror(status)<<"\n"<<flush;exit_(status);}
*/
	m_FParamsDevice		= clCreateBuffer(m_context, CL_MEM_READ_WRITE , sizeof(m_FParams), 0, &res);	if(res!=CL_SUCCESS){cout<<"\nres = "<<checkerror(res)<<"\n"<<flush;exit_(res);}
	m_FluidDevice		= clCreateBuffer(m_context, CL_MEM_READ_WRITE , sizeof(FBufs), 0, &res);	    if(res!=CL_SUCCESS){cout<<"\nres = "<<checkerror(res)<<"\n"<<flush;exit_(res);}
	m_FluidTempDevice	= clCreateBuffer(m_context, CL_MEM_READ_WRITE , sizeof(FBufs), 0, &res);	if(res!=CL_SUCCESS){cout<<"\nres = "<<checkerror(res)<<"\n"<<flush;exit_(res);}
	m_FGenomeDevice		= clCreateBuffer(m_context, CL_MEM_READ_WRITE , sizeof(m_FGenome), 0, &res);	if(res!=CL_SUCCESS){cout<<"\nres = "<<checkerror(res)<<"\n"<<flush;exit_(res);}
    m_FPrefixDevice		= clCreateBuffer(m_context, CL_MEM_READ_WRITE , sizeof(FPrefix), 0, &res);	if(res!=CL_SUCCESS){cout<<"\nres = "<<checkerror(res)<<"\n"<<flush;exit_(res);}


																				if(verbosity>1) {
																					cout << "m_FParamDevice = " 	<< m_FParamsDevice << flush;
																					cout << "m_FluidDevice = " 		<< m_FluidDevice << flush;
																					cout << "m_FluidTempDevice = " 	<< m_FluidTempDevice << flush;
																					cout << "m_FGenomeDevice = " 	<< m_FGenomeDevice << flush;
																				}

    																				if(verbosity>0) cout << "FluidSystem::InitializeOpenCL_chk3\n" << flush;


// 	cl_event readEvt;
//
//     // For m_FParamDevice
//     status = clEnqueueReadBuffer(uload_queue, m_FParamDevice, CL_TRUE, 0, sizeof(m_FParams), &m_FParams, 0, NULL, &readEvt);
//     if (status != CL_SUCCESS) {
//         cout << "\nstatus = " << checkerror(status) <<"\n"<<flush;
//         cout << "Error: ReadBuffer failed for m_FParamDevice.\n" << endl;
//         exit_(status);
//     }
//
//     // For m_FluidDevice
//     status = clEnqueueReadBuffer(uload_queue, m_FluidDevice, CL_TRUE, 0, sizeof(m_Fluid), &m_Fluid, 0, NULL, &readEvt);
//     if (status != CL_SUCCESS) {
//         cout << "\nstatus = " << checkerror(status) <<"\n"<<flush;
//         cout << "Error: ReadBuffer failed for m_FluidDevice.\n" << endl;
//         exit_(status);
//     }
//
//     // For m_FluidTempDevice
//     status = clEnqueueReadBuffer(uload_queue, m_FluidTempDevice, CL_TRUE, 0, sizeof(m_FluidTemp), &m_FluidTemp, 0, NULL, &readEvt);
//     if (status != CL_SUCCESS) {
//         cout << "\nstatus = " << checkerror(status) <<"\n"<<flush;
//         cout << "Error: ReadBuffer failed for m_FluidTempDevice.\n" << endl;
//         exit_(status);
//     }
//
//     // For m_FGenomeDevice
//     status = clEnqueueReadBuffer(uload_queue, m_FGenomeDevice, CL_TRUE, 0, sizeof(m_FGenome)*3, &m_FGenome, 0, NULL, &readEvt);
//     if (status != CL_SUCCESS) {
//         cout << "\nstatus = " << checkerror(status) <<"\n"<<flush;
//         cout << "Error: ReadBuffer failed for m_FGenomeDevice.\n" << endl;
//         exit_(status);
//     }
//
//     // Wait for the read command to complete.
//     status = clWaitForEvents(1, &readEvt);
//     if (status != CL_SUCCESS) {
//         cout << "\nclWaitForEvents(readEvt)=" << status << checkerror(status)<<"\n"<<flush;
//         exit_(status);
//     }
//
// //     // For m_FParams
// //     for (int i = 0; i < sizeof(m_FParams)/sizeof(m_FParams[0]); ++i) {
// //         cout << "m_FParams[" << i << "] = " << m_FParams[i] << endl;
// //     }
//
//     // For m_Fluid
//     for (int i = 0; i < MAX_BUF; ++i) {
//         cout << "m_Fluid.mcpu[" << i << "] = " << bufF(&m_Fluid, i) << endl;
//     }
// //
//     // For m_FluidTemp
//     for (int i = 0; i < MAX_BUF; ++i) {
//         cout << "m_FluidTemp.mcpu[" << i << "] = " << bufF(&m_FluidTemp, i) << endl;
//     }

//     // For m_FGenome
//     for (int i = 0; i < MAX_BUF; ++i) {
//         cout << "m_FGenome.mcpu[" << i << "] = " << bufF(&m_FGenome, i) << endl;
//     }
	//clFlush(uload_queue);
    //clFinish(uload_queue);
//     status = clFinish(uload_queue); 					if (status != CL_SUCCESS)	{ cout << "\nclFinish(uload_queue)="				<< status << checkerror(status)<<"\n"<<flush; exit_(status);}
                                                                            if(verbosity>0) cout << "-----FluidSystem::InitializeOpenCL() finished-----\n\n" << flush;
}


void FluidSystem::exit_(cl_int res)
{
	//CleanUp();
	//~RunCL(); Never need to call a destructor manually.
	exit(res);
}

bool isBufferAllZeros(const char* buffer, int stride, int cpucnt) {
    cout << "\n" << stride << ", " << cpucnt << flush;
    for (int i = 0; i < cpucnt * stride; ++i) {
        if (buffer[i] != 0) {
            cout << "FUCK" << flush;
            return false; // Non-zero byte found
        }
        cout << "(" << i << "), " << buffer[i] << flush;
    }
    cout << "\nHOORAY\n" << flush;
    return true; // All bytes are zero
}

/////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////

void FluidSystem::Initialize(){             //Left aside for now, implement by copying from InitializeOpenCLused for CPU only for "check_demo".
                                                                            if(verbosity>0) cout << "\n-------FluidSystem::Initialize()-------" << flush;
    // An FBufs struct holds an array of pointers.
    // Clear all buffers
    memset ( &m_Fluid,      0,      sizeof(FBufs) );
    memset ( &m_FluidTemp,  0,      sizeof(FBufs) );
    memset ( &m_FParams,    0,      sizeof(FParams) );
    memset ( &m_FGenome,    0,      sizeof(FGenome) );

    mNumPoints = 0;
	mMaxPoints = 0;
	mPackGrid = 0x0;
	m_Frame = 0;
    m_Time = 0; //TODO remove

    if (verbosity>1)std::cout << "\nChk1.4 ";

    AllocateBuffer ( FPARAMS,		sizeof(FParams),	1,	0,	 GPU_OFF,     CPU_YES ); //AllocateBuffer ( int buf_id, int stride,     int cpucnt, int gpucnt,    int gpumode,    int cpumode )

    if (verbosity>1)std::cout << "\nChk1.6 ";

                                                                            if(verbosity>0) cout << "\n-----FluidSystem::Initialize() successful-----\n" << flush;

}

void FluidSystem::UpdateGenome (){              // Update Genome on GPU

    if (verbosity>0) std::cout << "\n-----UpdateGenome() started... -----" << std::flush;

    clCheck ( clEnqueueWriteBuffer(m_queue, m_FGenomeDevice, CL_TRUE, 0, sizeof(FGenome), &m_FGenome, 0, NULL, NULL), "FluidUpdateGenome", "clEnqueueWriteBuffer", "m_FGenomeDevice", mbDebug);

    if (verbosity>0) std::cout << "\n-----UpdateGenome() finished-----\n" << std::flush;

}

FGenome	FluidSystem::GetGenome(){
            FGenome tempGenome = m_FGenome;
            for (int i=0; i<NUM_GENES; i++)tempGenome.mutability[i]=m_FGenome.mutability[i];
            for (int i=0; i<NUM_GENES; i++)tempGenome.delay[i]=m_FGenome.delay[i];
            for (int i=0; i<NUM_GENES; i++)for (int j=0; j<NUM_GENES; j++)tempGenome.sensitivity[i][j]=m_FGenome.sensitivity[i][j];

            for (int i=0; i<NUM_TF; i++)tempGenome.tf_diffusability[i]=m_FGenome.tf_diffusability[i];
            for (int i=0; i<NUM_TF; i++)tempGenome.tf_breakdown_rate[i]=m_FGenome.tf_breakdown_rate[i];

            for (int i=0; i<NUM_GENES; i++)for (int j=0; j<2*NUM_TF+1; j++)tempGenome.secrete[i][j]=m_FGenome.secrete[i][j];
            for (int i=0; i<NUM_GENES; i++)for (int j=0; j<2*NUM_GENES+1; j++)tempGenome.activate[i][j]=m_FGenome.activate[i][j];

            for (int i=0; i<3;i++)for(int j=0; j<12; j++)tempGenome.param[i][j]=m_FGenome.param[i][j];
            std::cout<<"\nGetGenome(): m_FGenome.delay[0]="<<m_FGenome.delay[0]<<"\ttempGenome.delay[0]="<<tempGenome.delay[0]<<std::flush;
            return tempGenome;
}

void FluidSystem::FluidParamCL (float ss, float sr, float pr, float mass, float rest, cl_float3 bmin, cl_float3 bmax, float estiff, float istiff, float visc, float surface_tension, float damp, float fmin, float fmax, float ffreq, float gslope, float gx, float gy, float gz, float al, float vl, float a_f, float a_p ){

    if (verbosity>0) std::cout << "\n-----FluidParamCL() started... -----" << std::flush;
    m_FParams.psimscale = ss;
    m_FParams.psmoothradius = sr;
    m_FParams.pradius = pr;
    m_FParams.r2 = sr * sr;
    m_FParams.pmass = mass;
    m_FParams.prest_dens = rest;
    m_FParams.pboundmin = bmin;
    m_FParams.pboundmax = bmax;
    m_FParams.pextstiff = estiff;
    m_FParams.pintstiff = istiff;
    m_FParams.pvisc = visc;
    m_FParams.psurface_t = surface_tension;
    m_FParams.pdamp = damp;
    m_FParams.pforce_min = fmin;
    m_FParams.pforce_max = fmax;
    m_FParams.pforce_freq = ffreq;
    m_FParams.pground_slope = gslope;
    m_FParams.pgravity = make_cl_float3( gx, gy, gz );
    m_FParams.AL = al;
    m_FParams.AL2 = al * al;
    m_FParams.VL = vl;
    m_FParams.VL2 = vl * vl;
    //m_FParams.pemit = emit;

    m_FParams.pdist = pow ( m_FParams.pmass / m_FParams.prest_dens, 1/3.0f );
                                                                                // Normalization constants.
    m_FParams.poly6kern = 315.0f / (64.0f * 3.141592f * pow( sr, 9.0f) );
    m_FParams.wendlandC2kern = 21 / (2 * 3.141592f );   // This is the value calculated in SymPy as per Wendland C2 as per (Dehnen & Aly 2012)
    // 16   // The  WC2 kernel in DualSPHysics assumes  values of 0<=q<=2 , hence the divisor 16pi in the normalisation constant for 3D.
    /* My notes from Sympy my notebook.
    Where Wendland C2 kernel:

        wc2 = (1-r*ss/2*sr)**4  * ((2*q) +1)

    Normalisation constant = 1/integrate( (wc2*(4*pi*r**2)), (r,0, 2*sr/ss)),  NB *(4*pi*r**2) area of a sphere, & 2=basis of wc2.

        =  1/ (288pi - 15552.0πss^2/sr^2 + 77760.0πss^3/sr^3 - 149965.714285714πss^4/sr^4 + 104976.0πss^5/sr^5  )

    */
    /* Notes from DualSPHysics Wiki
    // Normalization const = reciprocal of radial integral of (kernel * area of sphere), found using Sympy.
    // NB using W(r,h)=alpha_D (1-q/2)**4 *(2*q +1), 0<=q<=2, as per DualSPHysics Wiki. Where alpha_D is the normaliation constant.
    // * m_FParams.pmass * m_FParams.psimscale
    */
    m_FParams.spikykern = -45.0f / (3.141592f * pow( sr, 6.0f) );            // spikykern used for force due to pressure.
    m_FParams.lapkern = 45.0f / (3.141592f * pow( sr, 6.0f) );
    // NB Viscosity uses a different kernel, this is the constant portion of its Laplacian.
    // NB Laplacian is a scalar 2nd order differential, "The divergence of the gradient"
    // This Laplacian comes from Muller et al 2003, NB The kernel is defined by the properties of  its Laplacian, gradient and value at the basis (outer limit) of the kernel. The Laplacian is the form used in the code. The equation of the kernel in Muller et al seems to be wrong, but this does not matter.

/*
    // -32*(1 - r)**3 + 12*(1 - r)**2*(4*r + 1)  // the Laplacian of  WC2 = (1-r)**4 *(1+4*r)
//(15*r**2*(h/r**3 + 2/h**2 - 3*r/h**3)/(2*pi*h**3) + 15*r*(-h/(2*r**2) + 2*r/h**2 - 3*r**2/(2*h**3))/(pi*h**3))/r**2
//(45/pi*h^6)((h^2/12r^3)+(2h/3)-(3r/4))

//(r**2*(h/r**3 + 2/h**2 - 3*r/h**3) + 2*r*(-h/(2*r**2) + 2*r/h**2 - 3*r**2/(2*h**3) ) )/r**2
*/

    m_FParams.gausskern = 1.0f / pow(3.141592f * 2.0f*sr*sr, 3.0f/2.0f);     // Gaussian not currently used.

    m_FParams.H = m_FParams.psmoothradius / m_FParams.psimscale;
    m_FParams.d2 = m_FParams.psimscale * m_FParams.psimscale;
    m_FParams.rd2 = m_FParams.r2 / m_FParams.d2;
    m_FParams.vterm = m_FParams.lapkern * m_FParams.pvisc;

    m_FParams.actuation_factor = a_f;
    m_FParams.actuation_period = a_p;


    // Transfer sim params to device
    clCheck ( clEnqueueWriteBuffer(

        m_queue,
        m_FParamsDevice,
        CL_TRUE,
        0,
        sizeof(FParams),
        &m_FParams,
        0,
        NULL,
        NULL),
    "FluidParamCL", "clEnqueueWriteBuffer", "m_FParamDevice", mbDebug);

    if (verbosity>0) std::cout << "\n-------FluidParamCL() finished. ---------" << std::flush;


}

void FluidSystem::UpdateParams (){

    if (verbosity>0) std::cout << "\n-----UpdateParams() started... -----" << std::flush;

    // Update Params on GPU
    cl_float3 grav = cl_float3_multiplyFloat(&m_Vec[PPLANE_GRAV_DIR], m_Param[PGRAV]);
    FluidParamCL (m_Param[PSIMSCALE], m_Param[PSMOOTHRADIUS], m_Param[PRADIUS], m_Param[PMASS], m_Param[PRESTDENSITY],
                      *(cl_float3*)& m_Vec[PBOUNDMIN], *(cl_float3*)& m_Vec[PBOUNDMAX], m_Param[PEXTSTIFF], m_Param[PINTSTIFF],
                      m_Param[PVISC], m_Param[PSURFACE_TENSION], m_Param[PEXTDAMP], m_Param[PFORCE_MIN], m_Param[PFORCE_MAX], m_Param[PFORCE_FREQ],
                      m_Param[PGROUND_SLOPE], grav.x, grav.y, grav.z, m_Param[PACCEL_LIMIT], m_Param[PVEL_LIMIT],
                      m_Param[PACTUATION_FACTOR], m_Param[PACTUATION_PERIOD]);

    if (verbosity>0) std::cout << "\n-----UpdateParams finished()-----\n" << std::flush;



}

void FluidSystem::SetParam (int p, float v ){
    m_Param[p] = v;
    UpdateParams ();
}

void FluidSystem::SetVec (int p, cl_float3 v ){
    m_Vec[p] = v;
    UpdateParams ();
}

void FluidSystem::Exit (){

    // Free fluid buffers
    clCheck(clFinish(m_queue), "Exit ", "clFinish", "before cudaDeviceReset()", mbDebug);
    for (int n=0; n < MAX_BUF; n++ ) {
        if (verbosity>0)std::cout << "\n n = " << n << std::flush;
        if ( bufC(&m_Fluid, n) != 0x0 )
            free ( bufC(&m_Fluid, n) );
    exit(0);
    }
}

void FluidSystem::Exit_no_CL (){
    // Free fluid buffers
    for (int n=0; n < MAX_BUF; n++ ) {
        if (verbosity>0)std::cout << "\n n = " << n << std::flush;
        if (bufC(&m_Fluid, n) != 0x0 )
            free ( bufC(&m_Fluid, n) );
    }
    exit(0);
}

void FluidSystem::AllocateBuffer(int buf_id, int stride, int cpucnt, int gpucnt, int gpumode, int cpumode) {
    bool rtn = true;
    cl_int status;
    if (verbosity > 0) {
        std::cout << "\nAllocateBuffer ( int buf_id=" << buf_id <<  ", int stride=" << stride <<
                                      ", int cpucnt=" << cpucnt <<  ", int gpucnt=" << gpucnt <<
                                      ", int "        << gpumode << ", int "        << cpumode << " )\t" << std::flush;
    }

    // Initialize the mcpu array for buffer buf_id
    if (buf_id < MAX_BUF) {

        //Free existing buffer
        if (m_Fluid.mcpu[buf_id] != nullptr) {free(m_Fluid.mcpu[buf_id]);}

        // Allocate memory
        m_Fluid.mcpu[buf_id] = (char*)malloc(cpucnt * stride);
        m_FluidTemp.mcpu[buf_id] = (char*)malloc(cpucnt * stride);


        // Initialize the buffer with zeros
        if (m_Fluid.mcpu[buf_id] != nullptr) {
            memset(m_Fluid.mcpu[buf_id], 0, cpucnt * stride);
            memset(m_FluidTemp.mcpu[buf_id], 0, cpucnt * stride);

        } else {std::cout << "\nmcpu[" << buf_id << "] allocation failed!" << std::flush;}

    } else {std::cout << "\nInvalid buf_id: " << buf_id << std::flush;}


    //copy memory to dest_buf
    if (cpumode == CPU_YES) {

        char* src_buf = bufC(&m_Fluid, buf_id);
        char* dest_buf = (char*)malloc(cpucnt * stride);

        if (src_buf == nullptr) {cout << "\n-----src_buf is a nullpointer!-----" << flush;}

        m_Fluid.mcpu[buf_id] = dest_buf;
    }



    if (gpumode == GPU_SINGLE || gpumode == GPU_DUAL || gpumode == GPU_TEMP) {

        clFinish(m_queue);  // Ensure any previous operations on the command queue are completed

        size_t total;

        clGetDeviceInfo(m_device, CL_DEVICE_GLOBAL_MEM_SIZE, sizeof(total), &total, NULL);

        if (gpumode == GPU_SINGLE || gpumode == GPU_DUAL) {

            //this if-statement ensure that the clReleaseMemObject() function only runs once at the beginning of the program.
            if (bufferAllocated[buf_id]) {

                clCheck(clReleaseMemObject(*gpuptr(&m_Fluid, buf_id)), "AllocateBuffer", "clReleaseMemObject SINGLE/DUAL", "gpuVar", mbDebug);

                cl_mem buf_loc = clCreateBuffer(m_context, CL_MEM_READ_WRITE, stride * gpucnt, NULL, &status);



                                                                                            if(verbosity>0 && status!=CL_SUCCESS) cout << checkerror(status)  << "\nstride*gpucnt: " << stride*gpucnt << "\n" << flush;

                                                                                            if (verbosity > 1) { std::cout << "\t\t gpuptr(&m_Fluid, " << buf_id << ")'" << gpuptr(&m_Fluid, buf_id) << ",   gpu(&m_Fluid, " << buf_id << ")=" << gpuVar(&m_Fluid, buf_id) << "\n" << std::flush;}

                cout << "buf_loc= " << buf_loc << "\n" << flush;

                setGpuBuf(&m_Fluid, buf_id, buf_loc);

            }else{

                cl_mem buf_loc = clCreateBuffer(m_context, CL_MEM_READ_WRITE, stride * gpucnt, NULL, &status);



                                                                                            if(verbosity>0 && status!=CL_SUCCESS) cout << checkerror(status)  << "\nstride*gpucnt: " << stride*gpucnt << "\n" << flush;

                                                                                            if (verbosity > 1) { std::cout << "\t\t gpuptr(&m_Fluid, " << buf_id << ")'" << gpuptr(&m_Fluid, buf_id) << ",   gpu(&m_Fluid, " << buf_id << ")=" << gpuVar(&m_Fluid, buf_id) << "\n" << std::flush;}

                cout << "buf_loc= " << buf_loc << "\n" << flush;

                setGpuBuf(&m_Fluid, buf_id, buf_loc);

                bufferAllocated[buf_id] = true;

            }


        }

        if (gpumode == GPU_TEMP || gpumode == GPU_DUAL) {
            cout << "gpuptr(&m_FluidTemp, buf_id)= " << gpuVar(&m_FluidTemp, buf_id) << "\n" << flush;
            if (gpuptr(&m_FluidTemp, buf_id) != NULL) {

                clCheck(clReleaseMemObject(*gpuptr(&m_Fluid, buf_id)), "AllocateBuffer", "clReleaseMemObject TEMP/DUAL", "gpuVar", mbDebug);
            }

            setGpuBuf(&m_FluidTemp, buf_id, clCreateBuffer(m_context, CL_MEM_READ_WRITE, stride * gpucnt, NULL, &status));
                                                                                            if(verbosity>0) cout << checkerror(status)  << stride*gpucnt << "\n" << flush;

            if (status != CL_SUCCESS) {_exit(status);}
        }

        clFinish(m_queue);  // Ensure any new OpenCL operations are completed
        clGetDeviceInfo(m_device, CL_DEVICE_GLOBAL_MEM_SIZE, sizeof(total), &total, NULL);
    }
}

void FluidSystem::AllocateParticles ( int cnt, int gpu_mode, int cpu_mode ){ // calls AllocateBuffer(..) for each buffer.
// Defaults in header : int gpu_mode = GPU_DUAL, int cpu_mode = CPU_YES
// Called by FluidSystem::ReadPointsCSV(..), and FluidSystem::WriteDemoSimParams(...), cnt = mMaxPoints.
    //size_t threefloat = 3 * sizeof(float);



    if (verbosity>0)std::cout<<"\n\nAllocateParticles ( cnt= "<<cnt<<", gpu_mode "<<gpu_mode<<", cpu_mode "<<cpu_mode<<" ), debug="<<verbosity<<", launchParams.verbosity="<<launchParams.verbosity<<", m_FParams.szPnts:"<<m_FParams.szPnts<<"\n";//<<std::flush;
    if (verbosity>1)std::cout<<"\n\tGPU_OFF=0, GPU_SINGLE=1, GPU_TEMP=2, GPU_DUAL=3, CPU_OFF=4, CPU_YES=5"<<std::flush;
    AllocateBuffer ( FPOS,		sizeof(cl_float4),	cnt,	m_FParams.szPnts,	gpu_mode, cpu_mode );//
    AllocateBuffer ( FCOLOR,	sizeof(uint),		cnt,	m_FParams.szPnts,	gpu_mode, cpu_mode );//
    AllocateBuffer ( FVEL,		sizeof(cl_float4),	cnt,	m_FParams.szPnts,	gpu_mode, cpu_mode );//
    AllocateBuffer ( FVEVAL,	sizeof(cl_float4),	cnt,	m_FParams.szPnts,	gpu_mode, cpu_mode );//
    AllocateBuffer ( FAGE,		sizeof(uint),       cnt,    m_FParams.szPnts,	gpu_mode, cpu_mode );//
    AllocateBuffer ( FPRESS,	sizeof(float),		cnt,	m_FParams.szPnts,	gpu_mode, cpu_mode );//
    AllocateBuffer ( FDENSITY,	sizeof(float),		cnt, 	m_FParams.szPnts,	gpu_mode, cpu_mode );//
    AllocateBuffer ( FFORCE,	sizeof(cl_float4),	cnt,	m_FParams.szPnts,	gpu_mode, cpu_mode );//
    AllocateBuffer ( FCLUSTER,	sizeof(uint),		cnt,	m_FParams.szPnts,	gpu_mode, cpu_mode );//
    AllocateBuffer ( FGCELL,	sizeof(uint),		cnt,	m_FParams.szPnts,	gpu_mode, cpu_mode );//
    AllocateBuffer ( FGNDX,		sizeof(uint),		cnt,	m_FParams.szPnts,	gpu_mode, cpu_mode );//
    AllocateBuffer ( FGNEXT,	sizeof(uint),		cnt,	m_FParams.szPnts,	gpu_mode, cpu_mode );//
    AllocateBuffer ( FNBRNDX,	sizeof(uint),		cnt,	m_FParams.szPnts,	gpu_mode, cpu_mode );//
    AllocateBuffer ( FNBRCNT,	sizeof(uint),		cnt,	m_FParams.szPnts,	gpu_mode, cpu_mode );//

    // extra buffers for morphogenesis
    AllocateBuffer ( FELASTIDX,	    sizeof(uint[BOND_DATA]),             cnt,   m_FParams.szPnts,	gpu_mode, cpu_mode );//
    AllocateBuffer ( FPARTICLEIDX,	sizeof(uint[BONDS_PER_PARTICLE *2]), cnt,	m_FParams.szPnts,	gpu_mode, cpu_mode );//
    AllocateBuffer ( FPARTICLE_ID,	sizeof(uint),		                 cnt,	m_FParams.szPnts,	gpu_mode, cpu_mode );//
    AllocateBuffer ( FMASS_RADIUS,	sizeof(uint),		                 cnt,	m_FParams.szPnts,	gpu_mode, cpu_mode );//
    AllocateBuffer ( FNERVEIDX,	    sizeof(uint),		                 cnt,	m_FParams.szPnts,	gpu_mode, cpu_mode );//
    AllocateBuffer ( FCONC,	        sizeof(float[NUM_TF]),		         cnt,	m_FParams.szPnts,	gpu_mode, cpu_mode );//
    AllocateBuffer ( FEPIGEN,	    sizeof(uint[NUM_GENES]),	         cnt,	m_FParams.szPnts,	gpu_mode, cpu_mode );//

    // original buffers continued, TODO not part of the original AllocateParticles() though!!!
    AllocateBuffer ( FSTATE,	            sizeof(uint),		cnt,	m_FParams.szPnts,	gpu_mode, cpu_mode );
    //AllocateBuffer ( FPARAMS,	            sizeof(uint),		cnt,	m_FParams.szPnts,	gpu_mode, cpu_mode );

    //AllocateBuffer ( FPARTICLEIDX,	        sizeof(uint),		cnt,	m_FParams.szPnts,	gpu_mode, cpu_mode );
    //AllocateBuffer ( FPARTICLE_ID,	        sizeof(uint),		cnt,	m_FParams.szPnts,	gpu_mode, cpu_mode );
    //AllocateBuffer ( FMASS_RADIUS,          sizeof(uint),		cnt,	m_FParams.szPnts,	gpu_mode, cpu_mode );

    // CL-specific buffers TODO
    //AllocateBuffer ( FCURAND_STATE,	sizeof(curandState_t),	             cnt,	m_FParams.szPnts,	gpu_mode, cpu_mode );
    AllocateBuffer ( FCURAND_SEED,	sizeof(unsigned long long),	         cnt,	m_FParams.szPnts,	gpu_mode, cpu_mode );

    // Update GPU access pointers
    if (gpu_mode != GPU_OFF ) {
        cout << "\nUpdating GPU access pointers ..." << flush;
        clCheck( clEnqueueWriteBuffer(m_queue, m_FluidDevice, CL_TRUE, 0, sizeof(FBufs), &m_Fluid, 0, NULL, NULL), "AllocateParticles", "clEnqueueWriteBuffer", "m_FluidDevice", mbDebug);
        clCheck( clEnqueueWriteBuffer(m_queue, m_FluidTempDevice, CL_TRUE, 0, sizeof(FBufs), &m_FluidTemp, 0, NULL, NULL),	"AllocateParticles", "clEnqueueWriteBuffer", "m_FluidTempDevice", mbDebug);
        clCheck( clEnqueueWriteBuffer(m_queue, m_FParamsDevice, CL_TRUE, 0, sizeof(FParams), &m_FParams, 0, NULL, NULL),  "AllocateParticles", "clEnqueueWriteBuffer", "m_FParamDevice", mbDebug);
        clCheck( clEnqueueWriteBuffer(m_queue, m_FGenomeDevice, CL_TRUE, 0, sizeof(FGenome), &m_FGenome, 0, NULL, NULL),  "AllocateParticles", "clEnqueueWriteBuffer", "m_FGenomeDevice", mbDebug);

        clCheck(clFinish(m_queue), "AllocateParticles", "clFinish", "", mbDebug );
        cout << "\nUpdating GPU access pointers finished.\n" << flush;

    }

    // Allocate auxiliary buffers (prefix sums)
    int blockSize = SCAN_BLOCKSIZE << 1;
    int numElem1 = m_GridTotal;
    int numElem2 = int ( numElem1 / blockSize ) + 1;
    int numElem3 = int ( numElem2 / blockSize ) + 1;


    if (gpu_mode != GPU_OFF ) {
        AllocateBuffer ( FAUXARRAY1,	sizeof(uint),		0,	numElem2, GPU_SINGLE, CPU_OFF );
        AllocateBuffer ( FAUXSCAN1,	    sizeof(uint),		0,	numElem2, GPU_SINGLE, CPU_OFF );
        AllocateBuffer ( FAUXARRAY2,	sizeof(uint),		0,	numElem3, GPU_SINGLE, CPU_OFF );
        AllocateBuffer ( FAUXSCAN2,	    sizeof(uint),		0,	numElem3, GPU_SINGLE, CPU_OFF );
    }
																				if(verbosity>0) cout << "\n-----AllocateParticles() finished-----\n\n" << flush;


}

void FluidSystem::AllocateBufferDenseLists(int buf_id, int stride, int gpucnt, int lists) {
    // mallocs a buffer - called by FluidSystem::AllocateGrid(int gpu_mode, int cpu_mode)
    // Need to save "pointers to the allocated gpu buffers" in a cpu array, AND then clEnqueueWriteBuffer(...) that list of pointers into the device array.
    // also called by FluidSystem::....() to quadruple buffer as needed.
    clCheck(clFinish(m_queue), "AllocateBufferDenseLists ", "clFinish", "before 1st clGetDeviceInfo", mbDebug);
    cl_ulong total;
    clGetDeviceInfo(m_device, CL_DEVICE_GLOBAL_MEM_SIZE, sizeof(cl_ulong), &total, NULL);
    if (verbosity > 1) printf("\nOpenCL Memory: total=%lu.\t", total);

    cl_mem* listpointer = (cl_mem*)&bufC(&m_Fluid, lists)[buf_id * sizeof(cl_mem)];

    if (verbosity > 1) printf("\n*listpointer=%p, listpointer=%p,  lists=%i, buf_id=%i, \t", (cl_mem*)*listpointer, listpointer, lists, buf_id);

    if (verbosity > 1) printf("\nAllocateBufferDenseLists: buf_id=%i, stride=%i, gpucnt=%i, lists=%i,  .\t", buf_id, stride, gpucnt, lists);

    if (*listpointer != NULL) clCheck(clReleaseMemObject(*listpointer), "AllocateBufferDenseLists1", "clReleaseMemObject", "*listpointer", mbDebug);

    cl_int status;
    *listpointer = clCreateBuffer(m_context, CL_MEM_READ_WRITE, stride * gpucnt, NULL, &status);
    bool result = clCheck(status, "AllocateBufferDenseLists2", "clCreateBuffer", "listpointer", mbDebug);

    clCheck(clFinish(m_queue), "AllocateBufferDenseLists ", "clFinish", "after allocation", mbDebug);
    if (result == false) _exit(result);
}

void FluidSystem::AllocateGrid(int gpu_mode, int cpu_mode){ // NB void FluidSystem::AllocateBuffer (int buf_id, int stride, int cpucnt, int gpucnt, int gpumode, int cpumode)
    // Allocate grid
        if (verbosity>0)std::cout<<"\n\nAllocateGrid ( gpu_mode "<<gpu_mode<<", cpu_mode "<<cpu_mode<<" ), debug="<<verbosity<<", launchParams.verbosity="<<launchParams.verbosity<<", m_FParams.szPnts:"<<m_FParams.szPnts<<"\t"<<std::flush;

    int cnt = m_GridTotal;
    m_FParams.szGrid = (m_FParams.gridBlocks * m_FParams.gridThreads);
    if (verbosity>1)cout<<"\nAllocateGrid: m_FParams.szGrid = ("<<m_FParams.gridBlocks<<" * "<<m_FParams.gridThreads<<")"<<std::flush;
    AllocateBuffer ( FBIN,		sizeof(uint),		mMaxPoints,	m_FParams.szPnts,	gpu_mode, cpu_mode );    // # grid elements = number of points
    AllocateBuffer ( FBIN_COUNT,	sizeof(uint),		cnt,	    m_FParams.szGrid,	gpu_mode, cpu_mode );
    AllocateBuffer ( FBIN_OFFSET,	sizeof(uint),		cnt,	    m_FParams.szGrid,	gpu_mode, cpu_mode );
    AllocateBuffer ( FBIN_ACT,	sizeof(uint),		cnt,	    m_FParams.szGrid,	gpu_mode, cpu_mode );      // ?? not used ?? ... active bins i.e. containing particles ?
    // extra buffers for dense lists
    AllocateBuffer ( FBIN_COUNT_ACTIVE_GENES,  sizeof(uint[NUM_GENES]),       cnt,   m_FParams.szGrid,	gpu_mode, cpu_mode );
    AllocateBuffer ( FBIN_OFFSET_ACTIVE_GENES,  sizeof(uint[NUM_GENES]),       cnt,   m_FParams.szGrid,	gpu_mode, cpu_mode );
    AllocateBuffer ( FDENSE_LIST_LENGTHS,	 sizeof(uint),		      NUM_GENES,   NUM_GENES,	        gpu_mode, cpu_mode );
    AllocateBuffer ( FDENSE_LISTS,	         sizeof(cl_mem), NUM_GENES,   NUM_GENES,           gpu_mode, cpu_mode );             //was CUdeviceptr, is it = cl_mem?
    AllocateBuffer ( FDENSE_BUF_LENGTHS,	 sizeof(uint),            NUM_GENES,   NUM_GENES,           gpu_mode, cpu_mode );
    //AllocateBuffer ( __________,	         sizeof(uint),		            cnt,   m_FParams.szPnts,	gpu_mode, cpu_mode );                  //Nr.37 Macro missing


    AllocateBuffer ( FBIN_COUNT_CHANGES,               sizeof(uint[NUM_CHANGES]),          cnt,   m_FParams.szGrid,	    gpu_mode, cpu_mode );
    AllocateBuffer ( FBIN_OFFSET_CHANGES,               sizeof(uint[NUM_CHANGES]),          cnt,   m_FParams.szGrid,	    gpu_mode, cpu_mode );
    AllocateBuffer ( FDENSE_LIST_LENGTHS_CHANGES,	 sizeof(uint),		         NUM_CHANGES,        NUM_CHANGES,	    gpu_mode, cpu_mode );
    AllocateBuffer ( FDENSE_LISTS_CHANGES,	         sizeof(cl_mem),    NUM_CHANGES,        NUM_CHANGES,       gpu_mode, cpu_mode );             //was CUdeviceptr, is it = cl_mem?
    AllocateBuffer ( FDENSE_BUF_LENGTHS_CHANGES,	 sizeof(uint),               NUM_CHANGES,        NUM_CHANGES,       gpu_mode, cpu_mode );

    if (gpu_mode != GPU_OFF ) {
        /*if(gpu_mode == GPU_SINGLE || gpu_mode == GPU_DUAL )*/
        for(int i=0; i<NUM_GENES; i++){ //for each gene allocate intial buffer, write pointer and size to FDENSE_LISTS and FDENSE_LIST_LENGTHS
            cl_mem*  _listpointer = (cl_mem*) &bufC(&m_Fluid, FDENSE_LISTS)[i * sizeof(cl_mem)] ;
            *_listpointer = NULL;
            AllocateBufferDenseLists( i, sizeof(uint), INITIAL_BUFFSIZE_ACTIVE_GENES, FDENSE_LISTS);  // AllocateBuffer writes pointer to  gpuptr(&m_Fluid, buf_id).
            bufI(&m_Fluid, FDENSE_LIST_LENGTHS)[i] = 0;
            bufI(&m_Fluid, FDENSE_BUF_LENGTHS)[i]  = INITIAL_BUFFSIZE_ACTIVE_GENES;
        }
        /*if(gpu_mode == GPU_SINGLE || gpu_mode == GPU_DUAL )*/
        for(int i=0; i<NUM_CHANGES; i++){
            cl_mem*  _listpointer = (cl_mem*) &bufC(&m_Fluid, FDENSE_LISTS_CHANGES)[i * sizeof(cl_mem)] ;
            *_listpointer = NULL;
            AllocateBufferDenseLists( i, sizeof(uint), 2*INITIAL_BUFFSIZE_ACTIVE_GENES, FDENSE_LISTS_CHANGES); // NB buf[2][list_length] holding : particleIdx, bondIdx
            bufI(&m_Fluid, FDENSE_LIST_LENGTHS_CHANGES)[i] = 0;
            bufI(&m_Fluid, FDENSE_BUF_LENGTHS_CHANGES)[i]  = INITIAL_BUFFSIZE_ACTIVE_GENES;
        }

        clEnqueueWriteBuffer(m_queue, gpuVar(&m_Fluid, FDENSE_LISTS), CL_TRUE, 0, NUM_GENES * sizeof(cl_mem), bufC(&m_Fluid, FDENSE_LISTS), 0, NULL, NULL);
        clEnqueueWriteBuffer(m_queue, gpuVar(&m_Fluid, FDENSE_LISTS_CHANGES), CL_TRUE, 0, NUM_GENES * sizeof(cl_mem), bufC(&m_Fluid, FDENSE_LISTS_CHANGES), 0, NULL, NULL);
        clEnqueueWriteBuffer(m_queue, gpuVar(&m_Fluid, FDENSE_BUF_LENGTHS), CL_TRUE, 0, NUM_GENES * sizeof(cl_mem), bufC(&m_Fluid, FDENSE_BUF_LENGTHS), 0, NULL, NULL);
        clEnqueueWriteBuffer(m_queue, gpuVar(&m_Fluid, FDENSE_BUF_LENGTHS_CHANGES), CL_TRUE, 0, sizeof(uint[NUM_CHANGES]), bufI(&m_Fluid, FDENSE_BUF_LENGTHS_CHANGES), 0, NULL, NULL);

        clEnqueueWriteBuffer(m_queue, m_FluidDevice, CL_TRUE, 0, sizeof(FBufs), &m_Fluid, 0, NULL, NULL);

        clCheck(clFinish(m_queue), "AllocateParticles", "clFinish", "", mbDebug);

        																				if(verbosity>0) cout << "\n-----AllocateGrid() finished-----\n\n" << flush;

    }
}

int FluidSystem::AddParticleMorphogenesis2 (cl_float3* Pos, cl_float3* Vel, uint Age, uint Clr, uint *_ElastIdxU, float *_ElastIdxF, uint *_Particle_Idx, uint Particle_ID, uint Mass_Radius, uint NerveIdx, float* _Conc, uint* _EpiGen ){  // called by :ReadPointsCSV2 (...) where :    uint Particle_Idx[BONDS_PER_PARTICLE * 2];  AND SetupAddVolumeMorphogenesis2(....)
    //cout <<"\n-----AddParticleMorphogenesis2() started-----"<<flush;

    if ( mNumPoints >= mMaxPoints ) return -1;

    int n = mNumPoints;
    Set4(bufV4(&m_Fluid, FPOS) + n, Pos->x,Pos->y,Pos->z );
    printf("\n--->Assigned values at index %d: x=%f, y=%f, z=%f<---\n", n, Pos->x, Pos->y, Pos->z);
    Set(bufV3(&m_Fluid, FVEL) + n, Vel->x,Vel->y,Vel->z );
    Set(bufV3(&m_Fluid, FVEVAL) + n, 0,0,0 );
    Set(bufV3(&m_Fluid, FFORCE) + n, 0,0,0 );
    *(bufF(&m_Fluid, FPRESS) + n) = 0;
    *(bufF(&m_Fluid, FDENSITY) + n) = 0;
    *(bufI(&m_Fluid, FGNEXT) + n) = -1;
    *(bufI(&m_Fluid, FCLUSTER)  + n) = -1;
        //std::cout << "\n AddParticleMorphogenesis2() XXXXXXXXXXXXXXXX DEBUG 1 XXXXXXXXXXXXXXXXX \t" << flush; //TODO remove
    *(bufF(&m_Fluid, FSTATE) + n ) = (float) rand();
    *(bufI(&m_Fluid, FAGE) + n) = Age;
    *(bufI(&m_Fluid, FCOLOR) + n) = Clr;
  if (verbosity>0)printf("bufV3(&m_Fluid, FPOS)[n]=(%f,%f,%f), Pos->x=%f, Pos->y=%f, Pos->z=%f,\t",bufV3(&m_Fluid, FPOS)[n].x,bufV3(&m_Fluid, FPOS)[n].y,bufV3(&m_Fluid, FPOS)[n].z,Pos->x,Pos->y,Pos->z);

        //std::cout << "\n AddParticleMorphogenesis2() XXXXXXXXXXXXXXXX DEBUG 2 XXXXXXXXXXXXXXXXX \t" << flush; //TODO remove
    uint* ElastIdx = (bufI(&m_Fluid, FELASTIDX) + n * BOND_DATA );
    float* ElastIdxFlt = (bufF(&m_Fluid, FELASTIDX) + n * BOND_DATA );
    for (int i = 0; i<BONDS_PER_PARTICLE;i++){
        //if (verbosity>1)printf("ADD PARTICLE: \t%u",_ElastIdxU[i*DATA_PER_BOND+0]);  //=65535
        ElastIdx[i*DATA_PER_BOND+0] = _ElastIdxU[i*DATA_PER_BOND+0] ;
        ElastIdx[i*DATA_PER_BOND+5] = _ElastIdxU[i*DATA_PER_BOND+5] ;
        ElastIdx[i*DATA_PER_BOND+6] = _ElastIdxU[i*DATA_PER_BOND+6] ;
        ElastIdx[i*DATA_PER_BOND+8] = _ElastIdxU[i*DATA_PER_BOND+8] ;
        ElastIdxFlt[i*DATA_PER_BOND+1] = _ElastIdxF[i*DATA_PER_BOND+1] ;
        ElastIdxFlt[i*DATA_PER_BOND+2] = _ElastIdxF[i*DATA_PER_BOND+2] ;
        ElastIdxFlt[i*DATA_PER_BOND+3] = _ElastIdxF[i*DATA_PER_BOND+3] ;
        ElastIdxFlt[i*DATA_PER_BOND+4] = _ElastIdxF[i*DATA_PER_BOND+4] ;
        ElastIdxFlt[i*DATA_PER_BOND+7] = _ElastIdxF[i*DATA_PER_BOND+7] ;
    }
    uint* Particle_Idx = (bufI(&m_Fluid, FPARTICLEIDX) + n * BONDS_PER_PARTICLE *2 );     // index of incoming bonds
    for(int j=0; j<(BONDS_PER_PARTICLE *2); j++) {
        Particle_Idx[j] = _Particle_Idx[j] ;
    }
    *(bufI(&m_Fluid, FPARTICLE_ID) + n)   = Particle_ID;                                  // permanent ID of particle
    *(bufI(&m_Fluid, FMASS_RADIUS) + n)   = Mass_Radius;
    *(bufI(&m_Fluid, FNERVEIDX) + n)      = NerveIdx;

    for(int j=0; j<(NUM_TF); j++) {
        float* Conc = getConc(j);
        Conc[n] = _Conc[j];
    }
    for(int j=0; j<(NUM_GENES); j++) {
        uint* EpiGen = getEpiGen(j);
        EpiGen[n]    = _EpiGen[j];                                                      // NB 'n' is particle index, from start of this gene. Data order:  FEPIGEN[gene][particle]
    }
    mNumPoints++;
    return n;
    //cout <<"-----AddParticleMorphogenesis2() finished-----";

}

void FluidSystem::AddNullPoints (){// fills unallocated particles with null data upto mMaxPoints. These can then be used to "create" new particles.
    if (verbosity>1) std::cout<<"\n AddNullPoints ()\n"<<std::flush;
    cl_float3 Pos, Vel, binSize;
    uint Age, Clr;
    uint  ElastIdxU[BOND_DATA];
    float ElastIdxF[BOND_DATA];
    uint Particle_Idx[2*BONDS_PER_PARTICLE];
    uint Particle_ID, Mass_Radius, NerveIdx;
    float Conc[NUM_TF];
    uint EpiGen[NUM_GENES];

    //Pos.x = m_FParams.pboundmax.x; // does not work in makeDemo because no CL &=> no UpdateParams.
    //Pos.y = m_FParams.pboundmax.y;
    //Pos.z = m_FParams.pboundmax.z;

    binSize.x=1.0/m_GridDelta.x; binSize.y=1.0/m_GridDelta.y; binSize.z=1.0/m_GridDelta.z;
    /*
    Pos = GetVec(PVOLMAX);        // SetupSpacing() has been called => m_Vec[PBOUNDMAX] is correctly set.  PVOLMAX - m_Param [ PRADIUS ]
    //Pos.x -= m_GridDelta.x/2; Pos.y -= m_GridDelta.y/2; Pos.z -= m_GridDelta.z/2;  // Should place particle in centre of last bin.
    Pos.x -= binSize.x*1.5;
    Pos.y -= binSize.y*1.5;
    Pos.z -= binSize.z*1.5;
    */
    Pos = GetVec(PBOUNDMAX);

    Vel.x = 0;
    Vel.y = 0;
    Vel.z = 0;
    Age   = UINT_MAXSIZE; // oldest active particles have lowest "age".
    Clr   = 0;
    for (int j=0;j<BOND_DATA;j++)               ElastIdxU[j]     = UINT_MAXSIZE;
    ElastIdxU[8] = 0;
    for (int j=0;j<BOND_DATA;j++)               ElastIdxF[j]     = 0.0;
    for (int j=0;j<2*BONDS_PER_PARTICLE;j++)    Particle_Idx[j] = UINT_MAXSIZE;
    Particle_ID = UINT_MAXSIZE;
    Mass_Radius = 0;
    NerveIdx    = UINT_MAXSIZE;
    for (int j=0;j<NUM_TF;j++)      Conc[j]     = 0;
    for (int j=0;j<NUM_GENES;j++)   EpiGen[j]   = 0;

    // TODO FPARTICLE_ID   // should equal mNumPoints when created
    //if (verbosity>1)std::cout<<"\n AddNullPoints (): mNumPoints="<<mNumPoints<<", mMaxPoints="<<mMaxPoints<<"\n"<<std::flush;
    while (mNumPoints < mMaxPoints){
        AddParticleMorphogenesis2 (&Pos, &Vel, Age, Clr, ElastIdxU, ElastIdxF, Particle_Idx, Particle_ID, Mass_Radius,  NerveIdx, Conc, EpiGen );
        //if (verbosity>1)std::cout<<"\n AddNullPoints (): mNumPoints="<<mNumPoints<<", mMaxPoints="<<mMaxPoints<<"\n"<<std::flush;
    }
}


void FluidSystem::SetupAddVolumeMorphogenesis2(cl_float3 min, cl_float3 max, float spacing, float offs, uint demoType ){  // NB ony used in WriteDemoSimParams() called by make_demo.cpp . Creates a cuboid with all particle values definable.
    if (verbosity>1)std::cout << "\n SetupAddVolumeMorphogenesis2 \t" << std::flush;

    cl_float3 pos;
    float dx, dy, dz;
    int cntx, cntz, p, c2;
    cntx = (int) ceil( (max.x-min.x-offs) / spacing );
    cntz = (int) ceil( (max.z-min.z-offs) / spacing );
    int cnt = cntx * cntz;
    //min += offs;            // NB by default offs=0.1f, & min=m_Vec[PINITMIN], when called in WriteDemoSimParams(..)
    min = cl_float3_addFloat(&min, offs);
    //max -= offs;            // m_Vec[PINITMIN] is set in SetupExampleParams()
    max = cl_float3_subtractFloat(&max, offs);

    dx = max.x-min.x;       // m_Vec[PBOUNDMIN] is set in SetupSpacing
    dy = max.y-min.y;
    dz = max.z-min.z;
    cl_float3 rnd;
    c2 = cnt/2;
    cl_float3 Pos, Vel;
    uint Age, Clr, Particle_ID, Mass_Radius, NerveIdx;
    uint  ElastIdxU[BOND_DATA];
    float ElastIdxF[BOND_DATA];
    uint Particle_Idx[BONDS_PER_PARTICLE*2]; // FPARTICLE_IDX : other particles with incoming bonds attaching here.
    float Conc[NUM_TF];
    uint EpiGen[NUM_GENES]={0};
    Particle_ID = 0;                         // NB Particle_ID=0 means "no particle" in ElastIdx.
    cl_float3 volV3DF = cl_float3_subtract_cl_float3(&max, &min);
    int num_particles_to_make = 8 * int(volV3DF.x*volV3DF.y*volV3DF.z);// 27 * //int(volV3DF.x*volV3DF.y*volV3DF.z / spacing*spacing*spacing);
    srand((unsigned int)time(NULL));
    if (verbosity>1)cout<<"\nSetupAddVolumeMorphogenesis2: num_particles_to_make= "<<num_particles_to_make<<",   min=("<<min.x<<","<<min.y<<","<<min.z<<"), max=("<<max.x<<","<<max.y<<","<<max.z<<") "<<std::flush;
    for (int i=0; i<num_particles_to_make; i++){
        cout << "\n\nAddParticleMorphogenesis2(" << p+1 << ") running----------------------------------------------------------" << flush;
        cout << "\nParticle " << i << ",\n mMaxPoints: " << mMaxPoints << ",\n mNumPoints: " << mNumPoints << "\n\n\n" << flush;
        Pos.x =  min.x + (float(rand())/float((RAND_MAX)) * dx) ;
        Pos.y =  min.y + (float(rand())/float((RAND_MAX)) * dy) ;
        Pos.z =  min.z + (float(rand())/float((RAND_MAX)) * dz) ;

                Particle_ID ++;  // NB AddParticleMorphogenesis2(...) checks not to exceed max num particles
                Vel.x=0; Vel.y=0; Vel.z=0;
                Age =  0;
                // Colour of particles
                //cl_float3 clr ( (pos.x-min.x)/dx, 0.0f, (pos.z-min.z)/dz );
                cl_float3 clr = cl_float3_init_with_values((pos.x-min.x)/dx, 0, (pos.z-min.z)/dz );
                //clr *= 0.8;
                clr = *cl_float3_multiplyDouble(&clr, 0.8);
                //clr += 0.2;
                clr = cl_float3_addFloat(&clr, 0.2);
                clr = Clamp(&clr, 0, 1.0);
                Clr = COLORA( clr.x, clr.y, clr.z, 1);
                // Modulus & length of elastic bonds
                // 8bits log modulus + 24bit uid, with fixed length // but for now 16bit modulus and radius
                uint modulus, length, mod_len;
                modulus = uint(m_Param [ PINTSTIFF ]) ; // m_Param [ PINTSTIFF ] =		1.0f;
                length = uint(1000 * m_Param [ PSMOOTHRADIUS ]); // m_Param [ PSMOOTHRADIUS ] =	0.015f;	// m // related to spacing, but also max particle range i.e. ....
                mod_len = ( modulus <<16 | length ); // NB should mask length to prevent it exceeding 16bits, i.e. 255*255

                for (int j = 0; i<BONDS_PER_PARTICLE;i++){
                    for (int j = 0; j< DATA_PER_BOND; j++){ ElastIdxU[i*DATA_PER_BOND +j] = UINT_MAXSIZE; ElastIdxF[i*DATA_PER_BOND +j] = 0; }
                    ElastIdxU[i*DATA_PER_BOND +8] = 0;
                }
                //NB #define DATA_PER_BOND 6 //6 : [0]current index, [1]elastic limit, [2]restlength, [3]modulus, [4]damping coeff, [5]particle ID, [6]bond index
                for (int i = 0; i<BONDS_PER_PARTICLE*2;i++) { Particle_Idx[i] = UINT_MAXSIZE; }
                if (Particle_ID % 10 == 0){NerveIdx = Particle_ID/10;} else {NerveIdx = 0;} // Every 10th particle has nerve connection

                // Mass & radius of particles
                // 4bit mass + 4bit radius + 24bit uid // but for now, 16bit mass & radius
                // Note m_params[] is set in "FluidSystem::SetupDefaultParams ()" and "FluidSystem::SetupExampleParams ()"
                // mass = m_Param[PMASS]; // 0.00020543f; // kg
                // radius = m_Param[PRADIUS]; // 0.015f; // m
                Mass_Radius =  ( (uint(m_Param[PMASS]*255.0f*255.0f)<<16) | uint(m_Param[PRADIUS]*255.0f*255.0f) ) ; // mass=>13, radius=>975
                for (int i=0; i< NUM_TF; i++)    { Conc[i]   = 0 ;}     // morphogen & transcription factor concentrations
                for (int i=0; i< NUM_GENES; i++) { EpiGen[i] = 0 ;}     // epigenetic state of each gene in this particle
                uint fixedActive = INT_MAX;                             // FEPIGEN below INT_MAX will count down to inactivation. Count down is inactivated by adding INT_MAX.
                EpiGen[0] = fixedActive;                                        // active, i.e. not reserve
                EpiGen[1] = fixedActive;                                        // solid, i.e. have elastic bonds
                EpiGen[2] = fixedActive;                                        // living/telomere, i.e. has genes
                if(demoType == 1){                                                                    ////// Remodelling & actuation demo
                                                                                // Fixed base, bone, tendon, muscle, elastic, external actuation
                    if(Pos.z <= min.z+spacing)                                EpiGen[11]=fixedActive;   // fixed particle
                    if(Pos.z >= max.z-spacing)                                EpiGen[12]=fixedActive;   // external actuation particle

                    if(Pos.z >= min.z+5*spacing && Pos.z < min.z+10*spacing)  EpiGen[9] =fixedActive;   // bone
                    if(Pos.z >= min.z+10*spacing && Pos.z < min.z+15*spacing) EpiGen[6] =fixedActive;   // tendon
                    if(Pos.z >= min.z+15*spacing && Pos.z < min.z+20*spacing) EpiGen[7] =fixedActive;   // muscle
                    if(Pos.z >= min.z+20*spacing && Pos.z < min.z+25*spacing) EpiGen[10]=fixedActive;   // elastic tissue
                }else if (demoType == 2){                                                            ////// Diffusion & epigenetics demo
                                                                            // Fixed base, homogeneous particles (initially)
                    if(Pos.z == min.z) EpiGen[0]=fixedActive;                                           // fixed particle
                    EpiGen[2]=1;                                            // living particle NB set gene behaviour
                }                                                           // => (i) French flag, (ii) polartity, (iii) clock & wave front

        if (verbosity > 1) {
            std::cout << "\n Pos: \t(" << Pos.x << ", " << Pos.y << ", " << Pos.z << ")" << std::flush;
            std::cout << "\n Vel: \t(" << Vel.x << ", " << Vel.y << ", " << Vel.z << ")" << std::flush;
            std::cout << "\n Age:  \t" << Age << std::flush;
            std::cout << "\n Clr:  \t" << Clr << std::flush;
            std::cout << "\n ElastIdxU:  \t" << *ElastIdxU << std::flush;
            std::cout << "\n ElastIdxF:  \t" << *ElastIdxF << std::flush;
            std::cout << "\n Particle_Idx:  \t" << *Particle_Idx << std::flush;
            std::cout << "\n Particle_ID:  \t" << Particle_ID << std::flush;
            std::cout << "\n Mass_Radius:  \t" << Mass_Radius << std::flush;
            std::cout << "\n NerveIdx:  \t" << NerveIdx << std::flush;
            std::cout << "\n Conc:  \t" << *Conc << std::flush;
            std::cout << "\n EpiGen:  \t" << *EpiGen << std::flush;
        }

                p = AddParticleMorphogenesis2 (
                /* cl_float3* */ &Pos,
                /* cl_float3* */ &Vel,
                /* uint */ Age,
                /* uint */ Clr,
                /* uint *_*/ ElastIdxU,
                /* uint *_*/ ElastIdxF,
                /* unit * */ Particle_Idx,
                /* uint */ Particle_ID,
                /* uint */ Mass_Radius,
                /* uint */ NerveIdx,
                /* float* */ Conc,
                /* uint* */ EpiGen
                );
                if(p==-1){
                    if (verbosity>1){std::cout << "\n SetupAddVolumeMorphogenesis2 exited on p==-1, Pos=("<<Pos.x<<","<<Pos.y<<","<<Pos.z<<"), Particle_ID="<<Particle_ID<<",  EpiGen[0]="<<EpiGen[0]<<" \n " << std::flush ;}

                    return;
                }
     }
                            mActivePoints=mNumPoints; // Initial active points, used in make_demo2.cpp, by WriteResultsCSV()
                            AddNullPoints();
                            if (verbosity < 0) {cout << "---------------------------------------------------------------------------------------------- SetupAddVolumeMorphogenesis2 finished ----- "<< flush;}
}

/*void FluidSystem::Run (){   // deprecated, rather use: Run(const char * relativePath, int frame, bool debug)

std::cout << "\tFluidSystem::Run (),  "<<std::flush;

    //case RUN_GPU_FULL: // Full CL pathway, GRID-accelerted GPU, /w deep copy sort

//TransferFromCL ();

//std::cout << "\n\n Chk1 \n"<<std::flush;

    InsertParticlesCL ( 0x0, 0x0, 0x0 );
    clCheck(void FluidSystem::Run (){   // deprecated, rather use: Run(const char * relativePath, int frame, bool debug)

std::cout << "\tFluidSystem::Run (),  "<<std::flush;

    //case RUN_GPU_FULL: // Full CL pathway, GRID-accelerted GPU, /w deep copy sort

//TransferFromCL ();

//std::cout << "\n\n Chk1 \n"<<std::flush;

    InsertParticlesCL ( 0x0, 0x0, 0x0 );

    clCheck(clFinish(m_queue)), "Run", "cuCtxSynchronize", "After InsertParticlesCL", mbDebug);

//TransferFromCL ();

std::cout << "\n\n Chk2 \n"<<std::flush;

    PrefixSumCellsCL ( 1 );

    clCheck(clFinish(m_queue)), "Run", "cuCtxSynchronize", "After PrefixSumCellsCL", mbDebug);

//TransferFromCL ();

std::cout << "\n\n Chk3 \n"<<std::flush;

    CountingSortFullCL ( 0x0 );

    clCheck(clFinish(m_queue)), "Run", "cuCtxSynchronize", "After CountingSortFullCL", mbDebug);

//TransferFromCL ();

std::cout << "\n\n Chk4 \n"<<std::flush;



    ComputePressureCL();

    clCheck(clFinish(m_queue)), "Run", "cuCtxSynchronize", "After ComputePressureCL", mbDebug);

//TransferFromCL ();

std::cout << "\n\n Chk5 \n"<<std::flush;

    // FreezeCL ();                                   // makes the system plastic, ie the bonds keep reforming

//std::cout << "\n\n Chk6 \n"<<std::flush;

    ComputeForceCL ();                                // now includes the function of freeze

    clCheck(clFinish(m_queue)), "Run", "cuCtxSynchronize", "After ComputeForceCL", mbDebug);


    // TODO compute nerve activation ?





    // TODO compute muscle action ?



//std::cout << "\n\n Chk6 \n"<<std::flush;

    ComputeDiffusionCL();

    clCheck(clFinish(m_queue)), "Run", "cuCtxSynchronize", "After ComputeDiffusionCL", mbDebug);


//std::cout << "\n\n Chk7 \n"<<std::flush;

    ComputeGenesCL();

    clCheck(clFinish(m_queue)), "Run", "cuCtxSynchronize", "After ComputeGenesCL", mbDebug);



std::cout << "\n\n Chk8 \n"<<std::flush;

    ComputeBondChangesCL ();

    clCheck(clFinish(m_queue)), "Run", "cuCtxSynchronize", "After ComputeBondChangesCL", mbDebug);



    //  make dense lists of particle changes

    // insert changes

    // prefix sum changes, inc tally_changelist_lengths

    // counting sort changes

    //InsertChangesCL ( ); // done by ComputeBondChanges() above

    //clCheck(clFinish(m_queue)), "Run", "cuCtxSynchronize", "After InsertChangesCL", mbDebug);


std::cout << "\n\n Chk9 \n"<<std::flush;

    PrefixSumChangesCL ( 1 );

    clCheck(clFinish(m_queue)), "Run", "cuCtxSynchronize", "After PrefixSumChangesCL", mbDebug);



std::cout << "\n\n Chk10 \n"<<std::flush;

    CountingSortChangesCL (  );

    clCheck(clFinish(m_queue)), "Run", "cuCtxSynchronize", "After CountingSortChangesCL", mbDebug);





    //  execute particle changes // _should_ be able to run concurrently => no clFinish(m_queue))

    // => single fn ComputeParticleChangesCL ()

std::cout << "\n\n Chk11 \n"<<std::flush;

    ComputeParticleChangesCL ();

    clCheck(clFinish(m_queue)), "Run", "cuCtxSynchronize", "After ComputeParticleChangesCL", mbDebug);


std::cout << "\n\n Chk12 \n"<<std::flush;

    CleanBondsCL ();

    clCheck(clFinish(m_queue)), "Run", "cuCtxSynchronize", "After CleanBondsCL ", mbDebug);



std::cout << "\n\n Chk13 \n"<<std::flush;

    TransferPosVelVeval ();

    clCheck(clFinish(m_queue)), "Run", "cuCtxSynchronize", "After TransferPosVelVeval ", mbDebug);



    AdvanceCL ( m_Time, m_DT, m_Param[PSIMSCALE] );

    clCheck(clFinish(m_queue)), "Run", "cuCtxSynchronize", "After AdvanceCL", mbDebug);



    SpecialParticlesCL ( m_Time, m_DT, m_Param[PSIMSCALE] );

    clCheck(clFinish(m_queue)), "Run", "cuCtxSynchronize", "After SpecialParticlesCL", mbDebug);





//TransferFromCL ();

    //EmitParticlesCL ( m_Time, (int) m_Vec[PEMIT_RATE].x );

    TransferFromCL (); // return for rendering

//std::cout << "\n\n Chk7 \n"<<std::flush;


    AdvanceTime ();

    clCheck(clFinish(m_queue)), "Run", "cuCtxSynchronize", "After AdvanceTime", mbDebug);

//std::cout << " finished \n";

}*/


/*void FluidSystem::Run (const char * relativePath, int frame, bool debug, bool gene_activity, bool remodelling ){       // version to save data after each kernel

    m_FParams.frame = frame;                 // used by computeForceCuda( .. Args)

if (verbosity>1)std::cout << "\n\n###### FluidSystem::Run (.......) frame = "<<frame<<" #########################################################"<<std::flush;

    clCheck(clFinish(m_queue)), "Run", "cuCtxSynchronize", "begin Run", mbDebug);

    if(debug){
        TransferFromCL ();
        SavePointsCSV2 (  relativePath, frame );
        std::cout << "\n\nRun(relativePath,frame) Chk1, saved "<< frame <<".csv At start of Run(...) \n"<<std::flush;
    }

    InsertParticlesCL ( 0x0, 0x0, 0x0 );

    clCheck(clFinish(m_queue)), "Run", "cuCtxSynchronize", "After InsertParticlesCL", mbDebug);

    if(debug){
        TransferFromCL ();
        SavePointsCSV2 (  relativePath, frame+1 );
        std::cout << "\n\nRun(relativePath,frame) Chk2, saved "<< frame+1 <<".csv  After InsertParticlesCL\n"<<std::flush;
    }

    PrefixSumCellsCL ( 1 );

    clCheck(clFinish(m_queue)), "Run", "cuCtxSynchronize", "After PrefixSumCellsCL", mbDebug);

    if(debug){
        TransferFromCL ();
        SavePointsCSV2 (  relativePath, frame+2 );
        std::cout << "\n\nRun(relativePath,frame) Chk3, saved "<< frame+2 <<".csv  After PrefixSumCellsCL\n"<<std::flush;
    }

    CountingSortFullCL ( 0x0 );

    clCheck(clFinish(m_queue)), "Run", "cuCtxSynchronize", "After CountingSortFullCL", mbDebug);

    if(debug){
        TransferFromCL ();
        SavePointsCSV2 (  relativePath, frame+3 );
        std::cout << "\n\nRun(relativePath,frame) Chk4, saved "<< frame+3 <<".csv  After CountingSortFullCL\n"<<std::flush;
    }

    ComputePressureCL();

    clCheck(clFinish(m_queue)), "Run", "cuCtxSynchronize", "After ComputePressureCL", mbDebug);

    if(debug){
        TransferFromCL ();
        SavePointsCSV2 (  relativePath, frame+4 );
        std::cout << "\n\nRun(relativePath,frame) Chk5, saved "<< frame+4 <<".csv  After ComputePressureCL \n"<<std::flush;
    }

    ComputeForceCL ();

    clCheck(clFinish(m_queue)), "Run", "cuCtxSynchronize", "After ComputeForceCL", mbDebug);

    if(debug){
        TransferFromCL ();
        SavePointsCSV2 (  relativePath, frame+5 );
        std::cout << "\n\nRun(relativePath,frame) Chk6, saved "<< frame+5 <<".csv  After ComputeForceCL \n"<<std::flush;
    }
    // TODO compute nerve activation ?


    // TODO compute muscle action ?


    ComputeDiffusionCL();
    clCheck(clFinish(m_queue)), "Run", "cuCtxSynchronize", "After ComputeDiffusionCL", mbDebug);

    if(debug){
        TransferFromCL ();
        SavePointsCSV2 (  relativePath, frame+6 );
        std::cout << "\n\nRun(relativePath,frame) Chk7, saved "<< frame+6 <<".csv  After ComputeDiffusionCL \n"<<std::flush;
    }

    if(gene_activity){

        ComputeGenesCL();     // NB (i)Epigenetic countdown, (ii) GRN gene regulatory network sensitivity to TransciptionFactors (FCONC)
        clCheck(clFinish(m_queue)), "Run", "cuCtxSynchronize", "After ComputeGenesCL", mbDebug);

        if(debug){
            TransferFromCL ();
            SavePointsCSV2 (  relativePath, frame+7 );
            std::cout << "\n\nRun(relativePath,frame) Chk8, saved "<< frame+7 <<".csv  After ComputeGenesCL \n"<<std::flush;
        }
        clCheck(clFinish(m_queue)), "Run", "cuCtxSynchronize", "After SavePointsCSV2 after ComputeGenesCL", mbDebug); // wipes out FEPIGEN
    }

    if(remodelling){

        AssembleFibresCL ();

        clCheck(clFinish(m_queue)), "Run", "cuCtxSynchronize", "After AssembleFibresCL", mbDebug);

        if(debug){
            TransferFromCL ();
            SavePointsCSV2 (  relativePath, frame+8 );
            std::cout << "\n\nRun(relativePath,frame) Chk9.0, saved "<< frame+8 <<".csv  After AssembleFibresCL  \n"<<std::flush;
        }

        ComputeBondChangesCL ();

        clCheck(clFinish(m_queue)), "Run", "cuCtxSynchronize", "After ComputeBondChangesCL", mbDebug); // wipes out FEPIGEN ////////////////////////////////////

        if(debug){
            TransferFromCL ();
            SavePointsCSV2 (  relativePath, frame+9 );
            std::cout << "\n\nRun(relativePath,frame) Chk9, saved "<< frame+8 <<".csv  After ComputeBondChangesCL  \n"<<std::flush;
        }

        PrefixSumChangesCL ( 1 );
        clCheck(clFinish(m_queue)), "Run", "cuCtxSynchronize", "After PrefixSumChangesCL", mbDebug); // writes mangled (?original?) data to FEPIGEN - not anymore

        if(debug){
            TransferFromCL ();
            SavePointsCSV2 (  relativePath, frame+10 );
            std::cout << "\n\nRun(relativePath,frame) Chk10, saved "<< frame+9 <<".csv  After PrefixSumChangesCL \n"<<std::flush;
        }

        CountingSortChangesCL (  );
        clCheck(clFinish(m_queue)), "Run", "cuCtxSynchronize", "After CountingSortChangesCL", mbDebug);

        if(debug){
            TransferFromCL ();
            SavePointsCSV2 (  relativePath, frame+11 );
            std::cout << "\n\nRun(relativePath,frame) Chk11, saved "<< frame+10 <<".csv  After CountingSortChangesCL  \n"<<std::flush;
        }

        ComputeParticleChangesCL ();                                     // execute particle changes // _should_ be able to run concurrently => no clFinish(m_queue))

        clCheck(clFinish(m_queue)), "Run", "cuCtxSynchronize", "After ComputeParticleChangesCL", mbDebug);

        if(debug){
            TransferFromCL ();
            SavePointsCSV2 (  relativePath, frame+12 );
            std::cout << "\n\nRun(relativePath,frame) Chk12, saved "<< frame+11 <<".csv  After  ComputeParticleChangesCL.  mMaxPoints="<<mMaxPoints<<"\n"<<std::flush;
        }
        //CleanBondsCL ();

        //clCheck(clFinish(m_queue)), "Run", "cuCtxSynchronize", "After CleanBondsCL ", mbDebug);

    }

    TransferPosVelVeval ();

    clCheck(clFinish(m_queue)), "Run", "cuCtxSynchronize", "After TransferPosVelVeval ", mbDebug);

    if(debug){
        TransferFromCL ();
        SavePointsCSV2 (  relativePath, frame+13 );
        std::cout << "\n\nRun(relativePath,frame) Chk13, saved "<< frame+12 <<".csv  After  TransferPosVelVeval.  mMaxPoints="<<mMaxPoints<<"\n"<<std::flush;
    }

    AdvanceCL ( m_Time, m_DT, m_Param[PSIMSCALE] );
    clCheck(clFinish(m_queue)), "Run", "cuCtxSynchronize", "After AdvanceCL", mbDebug);

    if(debug){
        TransferFromCL ();
        SavePointsCSV2 (  relativePath, frame+14 );
        std::cout << "\n\nRun(relativePath,frame) Chk14, saved "<< frame+13 <<".csv  After  AdvanceCL\n"<<std::flush;
    }

    SpecialParticlesCL ( m_Time, m_DT, m_Param[PSIMSCALE]);
    clCheck(clFinish(m_queue)), "Run", "cuCtxSynchronize", "After SpecialParticlesCL", mbDebug);

    if(debug){
        TransferFromCL ();
        SavePointsCSV2 (  relativePath, frame+15 );
        std::cout << "\n\nRun(relativePath,frame) Chk15, saved "<< frame+14 <<".csv  After  SpecialParticlesCL\n"<<std::flush;
    }
    AdvanceTime ();*/
/*

    clCheck(clFinish(m_queue)), "Run", "cuCtxSynchronize", "After AdvanceTime", mbDebug);

    if(debug){

        TransferFromCL ();

        SavePointsCSV2 (  relativePath, frame+15 );

    }

*/

/*

//     if(debug){

//         TransferFromCL ();

//         SavePointsCSV2 (  relativePath, frame+18 );

//         std::cout << "\n\nRun(relativePath,frame) Chk16, saved "<< frame+6 <<".csv  After AdvanceCL \n"<<std::flush;

//     }

    //clCheck(clFinish(m_queue)), "Run", "cuCtxSynchronize", "After AdvanceCL", mbDebug);

    //EmitParticlesCL ( m_Time, (int) m_Vec[PEMIT_RATE].x );

    //TransferFromCL (); // return for rendering


//     if(debug){

//         TransferFromCL ();

//         SavePointsCSV2 (  relativePath, frame+19 );

//         std::cout << "Run(relativePath,frame) finished,  saved "<< frame+7 <<".csv  After AdvanceTime \n";

//     }

*/

/*}// 0:start, 1:InsertParticles, 2:PrefixSumCellsCL, 3:CountingSortFull, 4:ComputePressure, 5:ComputeForce, 6:Advance, 7:AdvanceTime
, "Run", "cuCtxSynchronize", "After InsertParticlesCL", mbDebug);

//TransferFromCL ();

std::cout << "\n\n Chk2 \n"<<std::flush;

    PrefixSumCellsCL ( 1 );
    clCheck(clFinish(m_queue)), "Run", "cuCtxSynchronize", "After PrefixSumCellsCL", mbDebug);

//TransferFromCL ();

std::cout << "\n\n Chk3 \n"<<std::flush;

    CountingSortFullCL ( 0x0 );
    clCheck(clFinish(m_queue)), "Run", "cuCtxSynchronize", "After CountingSortFullCL", mbDebug);

//TransferFromCL ();

std::cout << "\n\n Chk4 \n"<<std::flush;

    ComputePressureCL();
    clCheck(clFinish(m_queue)), "Run", "cuCtxSynchronize", "After ComputePressureCL", mbDebug);

//TransferFromCL ();

std::cout << "\n\n Chk5 \n"<<std::flush;

    // FreezeCL ();                                   // makes the system plastic, ie the bonds keep reforming

//std::cout << "\n\n Chk6 \n"<<std::flush;

    ComputeForceCL ();                                // now includes the function of freeze
    clCheck(clFinish(m_queue)), "Run", "cuCtxSynchronize", "After ComputeForceCL", mbDebug);


    // TODO compute nerve activation ?

    // TODO compute muscle action ?

//std::cout << "\n\n Chk6 \n"<<std::flush;

    ComputeDiffusionCL();
    clCheck(clFinish(m_queue)), "Run", "cuCtxSynchronize", "After ComputeDiffusionCL", mbDebug);

//std::cout << "\n\n Chk7 \n"<<std::flush;

    ComputeGenesCL();
    clCheck(clFinish(m_queue)), "Run", "cuCtxSynchronize", "After ComputeGenesCL", mbDebug);

std::cout << "\n\n Chk8 \n"<<std::flush;

    ComputeBondChangesCL ();
    clCheck(clFinish(m_queue)), "Run", "cuCtxSynchronize", "After ComputeBondChangesCL", mbDebug);

    //  make dense lists of particle changes
    // insert changes
    // prefix sum changes, inc tally_changelist_lengths
    // counting sort changes
    //InsertChangesCL ( ); // done by ComputeBondChanges() above
    //clCheck(clFinish(m_queue)), "Run", "cuCtxSynchronize", "After InsertChangesCL", mbDebug);

std::cout << "\n\n Chk9 \n"<<std::flush;

    PrefixSumChangesCL ( 1 );
    clCheck(clFinish(m_queue)), "Run", "cuCtxSynchronize", "After PrefixSumChangesCL", mbDebug);

std::cout << "\n\n Chk10 \n"<<std::flush;

    CountingSortChangesCL (  );
    clCheck(clFinish(m_queue)), "Run", "cuCtxSynchronize", "After CountingSortChangesCL", mbDebug);

    //  execute particle changes // _should_ be able to run concurrently => no clFinish(m_queue))
    // => single fn ComputeParticleChangesCL ()

std::cout << "\n\n Chk11 \n"<<std::flush;

    ComputeParticleChangesCL ();
    clCheck(clFinish(m_queue)), "Run", "cuCtxSynchronize", "After ComputeParticleChangesCL", mbDebug);

std::cout << "\n\n Chk12 \n"<<std::flush;

    CleanBondsCL ();
    clCheck(clFinish(m_queue)), "Run", "cuCtxSynchronize", "After CleanBondsCL ", mbDebug);

std::cout << "\n\n Chk13 \n"<<std::flush;

    TransferPosVelVeval ();
    clCheck(clFinish(m_queue)), "Run", "cuCtxSynchronize", "After TransferPosVelVeval ", mbDebug);

    AdvanceCL ( m_Time, m_DT, m_Param[PSIMSCALE] );
    clCheck(clFinish(m_queue)), "Run", "cuCtxSynchronize", "After AdvanceCL", mbDebug);

    SpecialParticlesCL ( m_Time, m_DT, m_Param[PSIMSCALE] );
    clCheck(clFinish(m_queue)), "Run", "cuCtxSynchronize", "After SpecialParticlesCL", mbDebug);

//TransferFromCL ();

    //EmitParticlesCL ( m_Time, (int) m_Vec[PEMIT_RATE].x );

    TransferFromCL (); // return for rendering

//std::cout << "\n\n Chk7 \n"<<std::flush;

    AdvanceTime ();

    clCheck(clFinish(m_queue)), "Run", "cuCtxSynchronize", "After AdvanceTime", mbDebug);
    //std::cout << " finished \n";
}*/

/*
void FluidSystem::Run (const char * relativePath, int frame, bool debug, bool gene_activity, bool remodelling ){       // version to save data after each kernel

    m_FParams.frame = frame;                 // used by computeForceCuda( .. Args)

if (verbosity>1)std::cout << "\n\n###### FluidSystem::Run (.......) frame = "<<frame<<" #########################################################"<<std::flush;

    clCheck(clFinish(m_queue)), "Run", "cuCtxSynchronize", "begin Run", mbDebug);

    if(debug){

        TransferFromCL ();
        SavePointsCSV2 (  relativePath, frame );
        std::cout << "\n\nRun(relativePath,frame) Chk1, saved "<< frame <<".csv At start of Run(...) \n"<<std::flush;
    }

    InsertParticlesCL ( 0x0, 0x0, 0x0 );
    clCheck(clFinish(m_queue)), "Run", "cuCtxSynchronize", "After InsertParticlesCL", mbDebug);

    if(debug){

        TransferFromCL ();
        SavePointsCSV2 (  relativePath, frame+1 );
        std::cout << "\n\nRun(relativePath,frame) Chk2, saved "<< frame+1 <<".csv  After InsertParticlesCL\n"<<std::flush;
    }

    PrefixSumCellsCL ( 1 );
    clCheck(clFinish(m_queue)), "Run", "cuCtxSynchronize", "After PrefixSumCellsCL", mbDebug);
    if(debug){

        TransferFromCL ();
        SavePointsCSV2 (  relativePath, frame+2 );
        std::cout << "\n\nRun(relativePath,frame) Chk3, saved "<< frame+2 <<".csv  After PrefixSumCellsCL\n"<<std::flush;
    }

    CountingSortFullCL ( 0x0 );
    clCheck(clFinish(m_queue)), "Run", "cuCtxSynchronize", "After CountingSortFullCL", mbDebug);

    if(debug){

        TransferFromCL ();
        SavePointsCSV2 (  relativePath, frame+3 );
        std::cout << "\n\nRun(relativePath,frame) Chk4, saved "<< frame+3 <<".csv  After CountingSortFullCL\n"<<std::flush;
    }

    ComputePressureCL();
    clCheck(clFinish(m_queue)), "Run", "cuCtxSynchronize", "After ComputePressureCL", mbDebug);

    if(debug){

        TransferFromCL ();
        SavePointsCSV2 (  relativePath, frame+4 );
        std::cout << "\n\nRun(relativePath,frame) Chk5, saved "<< frame+4 <<".csv  After ComputePressureCL \n"<<std::flush;
    }

    ComputeForceCL ();
    clCheck(clFinish(m_queue)), "Run", "cuCtxSynchronize", "After ComputeForceCL", mbDebug);

    if(debug){

        TransferFromCL ();
        SavePointsCSV2 (  relativePath, frame+5 );
        std::cout << "\n\nRun(relativePath,frame) Chk6, saved "<< frame+5 <<".csv  After ComputeForceCL \n"<<std::flush;
    }

    // TODO compute nerve activation ?

    // TODO compute muscle action ?

    ComputeDiffusionCL();
    clCheck(clFinish(m_queue)), "Run", "cuCtxSynchronize", "After ComputeDiffusionCL", mbDebug);

    if(debug){

        TransferFromCL ();
        SavePointsCSV2 (  relativePath, frame+6 );
        std::cout << "\n\nRun(relativePath,frame) Chk7, saved "<< frame+6 <<".csv  After ComputeDiffusionCL \n"<<std::flush;
    }

    if(gene_activity){

        ComputeGenesCL();     // NB (i)Epigenetic countdown, (ii) GRN gene regulatory network sensitivity to TransciptionFactors (FCONC)
        clCheck(clFinish(m_queue)), "Run", "cuCtxSynchronize", "After ComputeGenesCL", mbDebug);

        if(debug){

            TransferFromCL ();
            SavePointsCSV2 (  relativePath, frame+7 );
            std::cout << "\n\nRun(relativePath,frame) Chk8, saved "<< frame+7 <<".csv  After ComputeGenesCL \n"<<std::flush;
        }

        clCheck(clFinish(m_queue)), "Run", "cuCtxSynchronize", "After SavePointsCSV2 after ComputeGenesCL", mbDebug); // wipes out FEPIGEN
    }

    if(remodelling){

        AssembleFibresCL ();
        clCheck(clFinish(m_queue)), "Run", "cuCtxSynchronize", "After AssembleFibresCL", mbDebug);

        if(debug){

            TransferFromCL ();
            SavePointsCSV2 (  relativePath, frame+8 );
            std::cout << "\n\nRun(relativePath,frame) Chk9.0, saved "<< frame+8 <<".csv  After AssembleFibresCL  \n"<<std::flush;
        }

        ComputeBondChangesCL ();
        clCheck(clFinish(m_queue)), "Run", "cuCtxSynchronize", "After ComputeBondChangesCL", mbDebug); // wipes out FEPIGEN ////////////////////////////////////

        if(debug){

            TransferFromCL ();
            SavePointsCSV2 (  relativePath, frame+9 );
            std::cout << "\n\nRun(relativePath,frame) Chk9, saved "<< frame+8 <<".csv  After ComputeBondChangesCL  \n"<<std::flush;
        }

        PrefixSumChangesCL ( 1 );
        clCheck(clFinish(m_queue)), "Run", "cuCtxSynchronize", "After PrefixSumChangesCL", mbDebug); // writes mangled (?original?) data to FEPIGEN - not anymore

        if(debug){

            TransferFromCL ();
            SavePointsCSV2 (  relativePath, frame+10 );
            std::cout << "\n\nRun(relativePath,frame) Chk10, saved "<< frame+9 <<".csv  After PrefixSumChangesCL \n"<<std::flush;
        }

        CountingSortChangesCL (  );
        clCheck(clFinish(m_queue)), "Run", "cuCtxSynchronize", "After CountingSortChangesCL", mbDebug);

        if(debug){

            TransferFromCL ();
            SavePointsCSV2 (  relativePath, frame+11 );
            std::cout << "\n\nRun(relativePath,frame) Chk11, saved "<< frame+10 <<".csv  After CountingSortChangesCL  \n"<<std::flush;

        }

        ComputeParticleChangesCL ();                                     // execute particle changes // _should_ be able to run concurrently => no clFinish(m_queue))
        clCheck(clFinish(m_queue)), "Run", "cuCtxSynchronize", "After ComputeParticleChangesCL", mbDebug);

        if(debug){

            TransferFromCL ();
            SavePointsCSV2 (  relativePath, frame+12 );
            std::cout << "\n\nRun(relativePath,frame) Chk12, saved "<< frame+11 <<".csv  After  ComputeParticleChangesCL.  mMaxPoints="<<mMaxPoints<<"\n"<<std::flush;

        }

        //CleanBondsCL ();
        //clCheck(clFinish(m_queue)), "Run", "cuCtxSynchronize", "After CleanBondsCL ", mbDebug);
    }

    TransferPosVelVeval ();
    clCheck(clFinish(m_queue)), "Run", "cuCtxSynchronize", "After TransferPosVelVeval ", mbDebug);

    if(debug){

        TransferFromCL ();
        SavePointsCSV2 (  relativePath, frame+13 );
        std::cout << "\n\nRun(relativePath,frame) Chk13, saved "<< frame+12 <<".csv  After  TransferPosVelVeval.  mMaxPoints="<<mMaxPoints<<"\n"<<std::flush;

    }

        AdvanceCL ( m_Time, m_DT, m_Param[PSIMSCALE] );
        clCheck(clFinish(m_queue)), "Run", "cuCtxSynchronize", "After AdvanceCL", mbDebug);

    if(debug){

        TransferFromCL ();
        SavePointsCSV2 (  relativePath, frame+14 );
        std::cout << "\n\nRun(relativePath,frame) Chk14, saved "<< frame+13 <<".csv  After  AdvanceCL\n"<<std::flush;
    }

        SpecialParticlesCL ( m_Time, m_DT, m_Param[PSIMSCALE]);
        clCheck(clFinish(m_queue)), "Run", "cuCtxSynchronize", "After SpecialParticlesCL", mbDebug);

    if(debug){

        TransferFromCL ();
        SavePointsCSV2 (  relativePath, frame+15 );
        std::cout << "\n\nRun(relativePath,frame) Chk15, saved "<< frame+14 <<".csv  After  SpecialParticlesCL\n"<<std::flush;

    }

    AdvanceTime ();

/*

    clCheck(clFinish(m_queue)), "Run", "cuCtxSynchronize", "After AdvanceTime", mbDebug);

    if(debug){

        TransferFromCL ();

        SavePointsCSV2 (  relativePath, frame+15 );

    }

*/

/*

//     if(debug){

//         TransferFromCL ();

//         SavePointsCSV2 (  relativePath, frame+18 );

//         std::cout << "\n\nRun(relativePath,frame) Chk16, saved "<< frame+6 <<".csv  After AdvanceCL \n"<<std::flush;

//     }

    //clCheck(clFinish(m_queue)), "Run", "cuCtxSynchronize", "After AdvanceCL", mbDebug);

    //EmitParticlesCL ( m_Time, (int) m_Vec[PEMIT_RATE].x );

    //TransferFromCL (); // return for rendering


//     if(debug){

//         TransferFromCL ();

//         SavePointsCSV2 (  relativePath, frame+19 );

//         std::cout << "Run(relativePath,frame) finished,  saved "<< frame+7 <<".csv  After AdvanceTime \n";

//     }

*/

/*}*/// 0:start, 1:InsertParticles, 2:PrefixSumCellsCL, 3:CountingSortFull, 4:ComputePressure, 5:ComputeForce, 6:Advance, 7:AdvanceTime


void FluidSystem::Run2PhysicalSort(){ // beginning of every time step, sorrting the particles

    if(verbosity>0)std::cout<<"-----starting Run2PhysicalSort-----\n\n";
    InsertParticlesCL ( 0x0, 0x0, 0x0 );
    clCheck(clFinish(m_queue), "Run", "clFinish", "After InsertParticlesCL", mbDebug);

    if(launchParams.verbosity>0){
        std::cout<<"\nchk a"<<std::flush;
        TransferFromCL ();
        m_Debug_file++;
        std::cout<<"\nchk b: launchParams.outPath="<<launchParams.outPath<<",  m_Frame+m_Debug_file=0;="<<m_Frame+m_Debug_file<<"\t"<<std::flush;
        SavePointsCSV2 (  launchParams.outPath, m_Frame+m_Debug_file );
        //TransferFromTempCL(int buf_id, int sz );
    }
    cout<<"\nX X X X X X before PrefixSumCells(), m_Param [PNUM] = " << m_Param [PNUM] <<std::flush; //TODO Remove this debug line
    PrefixSumCellsCL ( 1 );

    clCheck(clFinish(m_queue), "Run", "clFinish", "After PrefixSumCellsCL", mbDebug);

    if(launchParams.verbosity>0){
        TransferFromCL ();
        m_Debug_file++;
        SavePointsCSV2 (  launchParams.outPath, m_Frame+m_Debug_file );
        std::cout << "\n\nRun2PhysicalSort() Chk2, saved "<<launchParams.outPath<< m_Frame+m_Debug_file <<".csv  _after_  PrefixSumCellsCL\n"<<std::flush;
        //TransferFromTempCL(int buf_id, int sz );
    }
    cout << "\nCHECK 1.1 Run2Simulation() \n" << flush;
    CountingSortFullCL ( 0x0 );
    clCheck(clFinish(m_queue), "Run", "clFinish", "After CountingSortFullCL", mbDebug);

    printf("\nIndices: spacer");
    printf("\nCHECK 1.2 Run2Simulation() \n");
    if(verbosity>0)std::cout<<"\n####\nRun2PhysicalSort()end";
}

void FluidSystem::Run2InnerPhysicalLoop(){ //
    cout << "\n####\n------- starting Run2InnerPhysicalLoop()... -------" << flush;
    printf("\n####\n------- starting Run2InnerPhysicalLoop()... -------");//     if(m_FParams.freeze==true){
//         InitializeBondsCL ();
//         clCheck(clFinish(m_queue), "Run", "clFinish", "After InitializeBondsCL ", mbDebug);
//     }
    printf("\n####\n------- Run2InnerPhysicalLoop() chk1 -------");
cout << flush;
    ComputePressureCL();
    clCheck(clFinish(m_queue), "Run", "clFinish", "After ComputePressureCL", mbDebug);
    printf("\n####\n------- Run2InnerPhysicalLoop() chk2 -------");
cout << flush;
    ComputeForceCL ();
    clCheck(clFinish(m_queue), "Run", "clFinish", "After ComputeForceCL", mbDebug);
    printf("\n####\n------- Run2InnerPhysicalLoop() chk3 -------");
cout << flush;
    if(launchParams.verbosity>1){
        TransferFromCL ();
        launchParams.file_increment++;
        cout << "----------------------------------------------------launchParams.file_num: " << launchParams.file_num << flush;
        cout << "----------------------------------------------------launchParams.file_increment: " << launchParams.file_increment << flush;

        SavePointsCSV2 (  launchParams.outPath, launchParams.file_num+launchParams.file_increment );
        std::cout << "\n\nRun(relativePath,frame) Chk4, saved "<< launchParams.file_num+3 <<".csv  After CountingSortFullCL\n"<<std::flush;
    }
    printf("\n####\n------- Run2InnerPhysicalLoop() chk4 -------");
cout << flush;
    TransferPosVelVeval ();
    clFinish(m_queue);
    printf("\n####\n------- Run2InnerPhysicalLoop() chk5 -------");

    AdvanceCL ( m_Time, m_DT, m_Param[PSIMSCALE] );
    clFinish(m_queue);
    printf("\n####\n------- Run2InnerPhysicalLoop() chk6 -------");

//     SpecialParticlesCL ( m_Time, m_DT, m_Param[PSIMSCALE]); //TODO THIS CRASHES!!!!!!
//     clFinish(m_queue);
//     printf("\n####\n------- Run2InnerPhysicalLoop() chk7 -------");

    TransferPosVelVevalFromTemp ();
    printf("\n####\n------- Run2InnerPhysicalLoop() chk7 -------");

     if(launchParams.verbosity>0){
        TransferFromCL ();
        m_Debug_file++;
        SavePointsCSV2 (  launchParams.outPath, m_Frame+m_Debug_file );
        std::cout << "\n\nRun2InnerPhysicalLoop() Chk1, saved "<<launchParams.outPath<< m_Frame+m_Debug_file <<".csv  After  TransferPosVelVevalFromTemp ();\n"<<std::flush;
        //TransferFromTempCL(int buf_id, int sz );
    }
    printf("\n####\n------- Run2InnerPhysicalLoop() chk8 -------");
    AdvanceTime ();
    if(verbosity>0)    printf("\n####\n------- Run2InnerPhysicalLoop() finished.  -------");

}

void FluidSystem::Run2GeneAction(){//NB gene sorting occurs within Run2PhysicalSort()
    if(verbosity>1)std::cout<<"\n####\nRun2GeneAction()start";
    ComputeDiffusionCL();
    clCheck(clFinish(m_queue), "Run", "clFinish", "After ComputeDiffusionCL", mbDebug);

    ComputeGenesCL(); // NB (i)Epigenetic countdown, (ii) GRN gene regulatory network sensitivity to TransciptionFactors (FCONC)
    clCheck(clFinish(m_queue), "Run", "clFinish", "After ComputeGenesCL", mbDebug);
    if(verbosity>1)std::cout<<"\n####\nRun2GeneAction()end";
}

void FluidSystem::AdvanceTime () {  // may need to prune unused details from this fn.
    m_Time += m_DT;
}

void FluidSystem::setFreeze(bool freeze){
    m_FParams.freeze = freeze;

    std::cout<<"\n-------setFreeze() started... -------"<<std::flush;

    if (freeze == true) {
            m_FParams.freezeBoolToInt = 1;
    }else{
            m_FParams.freezeBoolToInt = 0;
    }
    //clCheck ( cuMemcpyHtoD ( clFParams,	&m_FParams,		sizeof(FParams) ), "FluidParamCL", "cuMemcpyHtoD", "clFParams", mbDebug);
    clCheck ( clEnqueueWriteBuffer(m_queue, m_FParamsDevice, CL_TRUE, 0 , sizeof(FParams), &m_FParams, 0, NULL, NULL ), "FluidParamCL", "cuMemcpyHtoD", "clFParams", mbDebug);
    std::cout<<"\n-------setFreeze() finished.-------"<<std::flush;


}

void FluidSystem::Run2Remodelling(uint steps_per_InnerPhysicalLoop){
    if(verbosity>1){std::cout<<"\n####\nRun2Remodelling()start";}
    AssembleFibresCL ();
    clCheck(clFinish(m_queue), "Run", "clFinish", "After AssembleFibresCL", mbDebug);

    ComputeBondChangesCL (steps_per_InnerPhysicalLoop);
    clCheck(clFinish(m_queue), "Run", "clFinish", "After ComputeBondChangesCL", mbDebug);

    PrefixSumChangesCL ( 1 );
    clCheck(clFinish(m_queue), "Run", "clFinish", "After PrefixSumChangesCL", mbDebug);

    CountingSortChangesCL (  );
    clCheck(clFinish(m_queue), "Run", "clFinish", "After CountingSortChangesCL", mbDebug);

    if(launchParams.verbosity>0){
        TransferFromCL ();
        m_Debug_file++;
        SavePointsCSV2 (  launchParams.outPath, m_Frame+m_Debug_file );
        std::cout << "\n\nRun2Remodelling() Chk1, saved "<<launchParams.outPath<< m_Frame+m_Debug_file <<".csv  After  CountingSortChangesCL ();\n"<<std::flush;
        //TransferFromTempCL(int buf_id, int sz );
    }

    ComputeParticleChangesCL ();
    clCheck(clFinish(m_queue), "Run", "clFinish", "After ComputeParticleChangesCL", mbDebug);

    if(launchParams.verbosity>0){
        TransferFromCL ();
        m_Debug_file++;
        SavePointsCSV2 (  launchParams.outPath, m_Frame+m_Debug_file );
        std::cout << "\n\nRun2Remodelling() Chk1, saved "<<launchParams.outPath<< m_Frame+m_Debug_file <<".csv  After  ComputeParticleChangesCL ();\n"<<std::flush;
        //TransferFromTempCL(int buf_id, int sz );
    }

    if(verbosity>1)std::cout<<"\n####\nRun2Remodelling()end";
}

void FluidSystem::SetupGrid(cl_float3 min, cl_float3 max, float sim_scale, float cell_size) {
    if (verbosity > 0) {
        std::cout << "\n-------SetupGrid(min(" << min.x << "," << min.y << "," << min.z << "), "
                  <<                    "max(" << max.x << "," << max.y << "," << max.z << "), "
                  <<              "sim_scale " << sim_scale << ", "
                  <<              "cell_size " << cell_size << ") started... -------" << std::flush;
    }

    float world_cellsize = cell_size / sim_scale;
m_GridMin = min;
cout << "m_GridMin: (" << m_GridMin.x << ", " << m_GridMin.y << ", " << m_GridMin.z << ")" << endl;

m_GridMax = max;
cout << "m_GridMin: (" << m_GridMax.x << ", " << m_GridMax.y << ", " << m_GridMax.z << ")" << endl;
m_GridSize = m_GridMax;
cout << "m_GridSize: (" << m_GridSize.x << ", " << m_GridSize.y << ", " << m_GridSize.z << ")" << endl;

m_GridSize = cl_float3_subtract_cl_float3(&m_GridSize,&m_GridMin);
cout << "m_GridSize after subtraction: (" << m_GridSize.x << ", " << m_GridSize.y << ", " << m_GridSize.z << ")" << endl;
m_GridRes.x = (int) ceil ( m_GridSize.x / world_cellsize );       // Determine grid resolution
m_GridRes.y = (int) ceil ( m_GridSize.y / world_cellsize );
m_GridRes.z = (int) ceil ( m_GridSize.z / world_cellsize );
cout << "m_GridRes: (" << m_GridRes.x << ", " << m_GridRes.y << ", " << m_GridRes.z << ")" << endl;
m_GridSize.x = m_GridRes.x * cell_size / sim_scale;               // Adjust grid size to multiple of cell size
m_GridSize.y = m_GridRes.y * cell_size / sim_scale;
m_GridSize.z = m_GridRes.z * cell_size / sim_scale;
cout << "m_GridSize after adjustment: (" << m_GridSize.x << ", " << m_GridSize.y << ", " << m_GridSize.z << ")" << endl;
m_GridDelta = *cl_float3_operator_equal_cl_int3(&m_GridDelta, &m_GridRes);       // delta = translate from world space to cell #
m_GridDelta = cl_float3_devide_cl_float3(&m_GridDelta, &m_GridSize);
cout << "m_GridDelta: (" << m_GridDelta.x << ", " << m_GridDelta.y << ", " << m_GridDelta.z << ")" << endl;

m_GridTotal = (int)(m_GridRes.x * m_GridRes.y * m_GridRes.z);
cout << "m_GridTotal: " << m_GridTotal << endl;

    cout << "\n m_GridTotal: " << m_GridTotal << flush;

    // Number of cells to search:
    // n = (2r / w) +1,  where n = 1D cell search count, r = search radius, w = world cell width
    m_GridSrch = (int) (floor(2.0f*(m_Param[PSMOOTHRADIUS]/sim_scale) / world_cellsize) + 1.0f);
    if ( m_GridSrch < 2 ) m_GridSrch = 2;
    m_GridAdjCnt = m_GridSrch * m_GridSrch * m_GridSrch ;			// 3D search count = n^3, e.g. 2x2x2=8, 3x3x3=27, 4x4x4=64

    if ( m_GridSrch > 6 ) {
        //if (verbosity>1)nvprintf ( "ERROR: Neighbor search is n > 6. \n " );
        exit(-1);
    }

    int cell = 0;
    for (int y=0; y < m_GridSrch; y++ )
        for (int z=0; z < m_GridSrch; z++ )
            for (int x=0; x < m_GridSrch; x++ )
                m_GridAdj[cell++] = ( y*m_GridRes.z + z )*m_GridRes.x +  x ;			// -1 compensates for ndx 0=empty

    if ( mPackGrid != 0x0 ) free ( mPackGrid );
    mPackGrid = (int*) malloc ( sizeof(int) * m_GridTotal );

    if (verbosity > 0) std::cout<<"\n-------SetupGrid() finished.-------\n\n"<<std::flush;

}

void FluidSystem::SetupSPH_Kernels (){
    std::cout<<"\n-------SetupSPH_Kernels() started... -------"<<std::flush;
    m_Param [ PDIST ] = pow ( (float) m_Param[PMASS] / m_Param[PRESTDENSITY], 1.0f/3.0f );
    m_R2 = m_Param [PSMOOTHRADIUS] * m_Param[PSMOOTHRADIUS];
    m_Poly6Kern = 315.0f / (64.0f * 3.141592f * pow( m_Param[PSMOOTHRADIUS], 9.0f) );	// Wpoly6 kernel (denominator part) - 2003 Muller, p.4
    m_SpikyKern = -45.0f / (3.141592f * pow( m_Param[PSMOOTHRADIUS], 6.0f) );			// Laplacian of viscocity (denominator): PI h^6
    m_LapKern = 45.0f / (3.141592f * pow( m_Param[PSMOOTHRADIUS], 6.0f) );
    std::cout<<"\n-------SetupSPH_Kernels() finished.-------\n"<<std::flush;
}

void FluidSystem::SetupDefaultParams (){
    //{//  Range = +/- 10.0 * 0.006 (r) =	   0.12			m (= 120 mm = 4.7 inch)
    //  Container Volume (Vc) =			   0.001728		m^3
    //  Rest Density (D) =				1000.0			kg / m^3
    //  Particle Mass (Pm) =			   0.00020543	kg						(mass = vol * density)
    //  Number of Particles (N) =		4000.0
    //  Water Mass (M) =				   0.821		kg (= 821 grams)
    //  Water Volume (V) =				   0.000821     m^3 (= 3.4 cups, .21 gals)
    //  Smoothing Radius (R) =             0.02			m (= 20 mm = ~3/4 inch)
    //  Particle Radius (Pr) =			   0.00366		m (= 4 mm  = ~1/8 inch)
    //  Particle Volume (Pv) =			   2.054e-7		m^3	(= .268 milliliters)
    //  Rest Distance (Pd) =			   0.0059		m
    //
    //  Given: D, Pm, N
    //    Pv = Pm / D			0.00020543 kg / 1000 kg/m^3 = 2.054e-7 m^3
    //    Pv = 4/3*pi*Pr^3    cuberoot( 2.054e-7 m^3 * 3/(4pi) ) = 0.00366 m
    //     M = Pm * N			0.00020543 kg * 4000.0 = 0.821 kg
    //     V =  M / D              0.821 kg / 1000 kg/m^3 = 0.000821 m^3
    //     V = Pv * N			 2.054e-7 m^3 * 4000 = 0.000821 m^3
    //    Pd = cuberoot(Pm/D)    cuberoot(0.00020543/1000) = 0.0059 m
    //
    // Ideal grid cell size (gs) = 2 * smoothing radius = 0.02*2 = 0.04
    // Ideal domain size = k*gs/d = k*0.02*2/0.005 = k*8 = {8, 16, 24, 32, 40, 48, ..}
    //    (k = number of cells, gs = cell size, d = simulation scale)

    // "The viscosity coefficient is the dynamic viscosity, visc > 0 (units Pa.s),
    // and to include a reasonable damping contribution, it should be chosen
    // to be approximately a factor larger than any physical correct viscosity
    // coefficient that can be looked up in the literature. However, care should
    // be taken not to exaggerate the viscosity coefficient for fluid materials.
    // If the contribution of the viscosity force density is too large, the net effect
    // of the viscosity term will introduce energy into the system, rather than
    // draining the system from energy as intended."
    //    Actual visocity of water = 0.001 Pa.s    // viscosity of water at 20 deg C.
    //}

    m_Time = 0.0f;							// Start at T=0
    m_DT = 0.003f;

    m_Param [ PSIMSCALE ] =		0.005f;			// unit size
    m_Param [ PVISC ] =			0.50f;			// pascal-second (Pa.s) = 1 kg m^-1 s^-1  (see wikipedia page on viscosity)
    m_Param [ PSURFACE_TENSION ] = 0.1f;
    m_Param [ PRESTDENSITY ] =	400.0f;			// kg / m^3
    m_Param [ PSPACING ]	=	0.0f;			// spacing will be computed automatically from density in most examples (set to 0 for autocompute)
    m_Param [ PMASS ] =			0.00020543f;		// kg
    m_Param [ PRADIUS ] =		0.015f;			// m
    m_Param [ PDIST ] =			0.0059f;			// m
    m_Param [ PSMOOTHRADIUS ] =	0.015f;			// m
    m_Param [ PINTSTIFF ] =		1.0f;
    m_Param [ PEXTSTIFF ] =		50000.0f;
    m_Param [ PEXTDAMP ] =		100.0f;
    m_Param [ PACCEL_LIMIT ] =	150.0f;			// m / s^2
    m_Param [ PVEL_LIMIT ] =	3.0f;			// m / s
    m_Param [ PGRAV ] =			1.0f;

    m_Param [ PGROUND_SLOPE ] = 0.0f;
    m_Param [ PFORCE_MIN ] =	0.0f;
    m_Param [ PFORCE_MAX ] =	0.0f;
    m_Param [ PFORCE_FREQ ] =	16.0f;
    m_Vec [ PPLANE_GRAV_DIR ] = cl_float3_init_with_values( 0, -9.8f, 0 );

    // Default sim config
    m_Param [PGRIDSIZE] = m_Param[PSMOOTHRADIUS] * 2;
    //cout << "\n\n\n++++++++++ m_Param [PGRIDSIZE] = " << m_Param[PGRIDSIZE] << flush; TODO remove
// 	m_Param [PDRAWMODE] = 1;				// Sprite drawing
// 	m_Param [PDRAWGRID] = 0;				// No grid
// 	m_Param [PDRAWTEXT] = 0;				// No text

    m_Param [ PACTUATION_FACTOR ] = 0;
    m_Param [ PACTUATION_PERIOD ] = 1;
}

void FluidSystem::SetupExampleParams (uint spacing){
    cl_float3 pos;
    cl_float3 min, max;
    m_Param [ PSPACING ] = spacing;
        cout << "\nXXXXXXXXXXXXX XXXXXXXXXXXX XXXXXXXXXXXXX XXXXXXXXXXXX PSPACING: " << m_Param[PSPACING] << flush;



    std::cout<<"\nSetupExampleParams()1: m_Param[PEXAMPLE] = "<<m_Param[PEXAMPLE]<<", SetupExampleParams()2: launchParams.genomePath = "<<launchParams.genomePath<<"\n"<<std::flush;
    //cout << "\nX X X X X m_Param[EXAMPLE]: " <<   m_Param[PEXAMPLE]  << flush; //TODO remove this debugging line

    switch ( (int) m_Param[PEXAMPLE] ) {

    case 0:	{	// Regression test. N x N x N static grid

        int k = (int) ceil ( pow ( (float) m_Param[PNUM], (float) 1.0f/3.0f ) );
        m_Vec [ PVOLMIN ] = cl_float3_init_with_values( 0, 0, 0 );
        m_Vec [ PVOLMAX ] = cl_float3_init_with_values( 2.0f+(k/2), 2.0f+(k/2), 2.0f+(k/2) );
        m_Vec [ PINITMIN ] = cl_float3_init_with_values( 1.0f, 1.0f, 1.0f );
        m_Vec [ PINITMAX ] = cl_float3_init_with_values( 1.0f+(k/2), 1.0f+(k/2), 1.0f+(k/2) );

        m_Param [ PGRAV ] = 0.0;
        m_Vec [ PPLANE_GRAV_DIR ] = cl_float3_init_with_values( 0.0, 0.0, 0.0 );
        //m_Param [ PSPACING ] = spacing;//0.5;				// Fixed spacing		Dx = x-axis density
        m_Param [ PSMOOTHRADIUS ] =	m_Param [PSPACING];		// Search radius
        //m_Toggle [ PRUN ] = false;				// Do NOT run sim. Neighbors only.
        //m_Param [PDRAWMODE] = 1;				// Point drawing
        //m_Param [PDRAWGRID] = 1;				// Grid drawing
        //m_Param [PDRAWTEXT] = 1;				// Text drawing
        m_Param [PSIMSCALE ] = 1.0f;
        launchParams.read_genome = 'y';
    }
    break;
    case 1:		// Tower
        m_Vec [ PVOLMIN ] = cl_float3_init_with_values(   0,   0,   0 );
        m_Vec [ PVOLMAX ] = cl_float3_init_with_values(  256, 128, 256 );
        m_Vec [ PINITMIN ] = cl_float3_init_with_values(  5,   5,  5 );
        m_Vec [ PINITMAX ] = cl_float3_init_with_values( 256*0.3, 128*0.9, 256*0.3 );
        break;
    case 2:		// Wave pool
        m_Vec [ PVOLMIN ] = cl_float3_init_with_values(   0,   0,   0 );
        m_Vec [ PVOLMAX ] = cl_float3_init_with_values(  400, 200, 400 );
        m_Vec [ PINITMIN ] = cl_float3_init_with_values( 100, 80,  100 );
        m_Vec [ PINITMAX ] = cl_float3_init_with_values( 300, 190, 300 );
        m_Param [ PFORCE_MIN ] = 100.0f;
        m_Param [ PFORCE_FREQ ] = 6.0f;
        m_Param [ PGROUND_SLOPE ] = 0.10f;
        break;
    case 3:		// Small dam break
        m_Vec [ PVOLMIN ] = cl_float3_init_with_values( -40, 0, -40  );
        m_Vec [ PVOLMAX ] = cl_float3_init_with_values( 40, 60, 40 );
        m_Vec [ PINITMIN ] = cl_float3_init_with_values( 0, 8, -35 );
        m_Vec [ PINITMAX ] = cl_float3_init_with_values( 35, 55, 35 );
        m_Param [ PFORCE_MIN ] = 0.0f;
        m_Param [ PFORCE_MAX ] = 0.0f;
        m_Vec [ PPLANE_GRAV_DIR ] = cl_float3_init_with_values( 0.0f, -9.8f, 0.0f );
        break;
    case 4:		// Dual-Wave pool
        m_Vec [ PVOLMIN ] = cl_float3_init_with_values( -100, 0, -15 );
        m_Vec [ PVOLMAX ] = cl_float3_init_with_values( 100, 100, 15 );
        m_Vec [ PINITMIN ] = cl_float3_init_with_values( -80, 8, -10 );
        m_Vec [ PINITMAX ] = cl_float3_init_with_values( 80, 90, 10 );
        m_Param [ PFORCE_MIN ] = 20.0;
        m_Param [ PFORCE_MAX ] = 20.0;
        m_Vec [ PPLANE_GRAV_DIR ] = cl_float3_init_with_values( 0.0f, -9.8f, 0.0f );
        break;
    case 5:		// Microgravity
        m_Vec [ PVOLMIN ] = cl_float3_init_with_values( -80, 0, -80 );
        m_Vec [ PVOLMAX ] = cl_float3_init_with_values( 80, 100, 80 );
        m_Vec [ PINITMIN ] = cl_float3_init_with_values( -60, 40, -60 );
        m_Vec [ PINITMAX ] = cl_float3_init_with_values( 60, 80, 60 );
        m_Vec [ PPLANE_GRAV_DIR ] = cl_float3_init_with_values( 0, -1, 0 );
        m_Param [ PGROUND_SLOPE ] = 0.1f;
        break;
    case 6:     // Morphogenesis small demo
        m_Param [ PSIMSCALE ] = 1.0f;
        m_Param [ PRADIUS ] = 1.0f;
        m_Param [ PSMOOTHRADIUS ] = 1.0f;
        m_Param [ PVISC ] = 0.1f;

        m_Vec [ PVOLMIN ] = cl_float3_init_with_values( 0, 0, 0 );
        m_Vec [ PVOLMAX ] = cl_float3_init_with_values( 10, 20, 50 ); //( 80, 50, 80 );
        m_Vec [ PINITMIN ] = cl_float3_init_with_values( m_Vec [ PVOLMIN ].x,  m_Vec [ PVOLMIN ].y, m_Vec [ PVOLMIN ].z );// will be reset to m_Vec[PBOUNDMIN].
        m_Vec [ PINITMAX ] = cl_float3_init_with_values( 10, 20, 30 );

        m_Param [ PGRAV ] = 2.000000f;
        m_Vec [ PPLANE_GRAV_DIR ] = cl_float3_init_with_values( 0, -1, 0 );
        m_Param [ PGROUND_SLOPE ] = 0.1f;
        break;
    case 7:     // From SpecificationFile.txt
        m_Time = launchParams.m_Time;
        m_DT = launchParams.m_DT;
        m_Param [ PGRIDSIZE ] = launchParams.gridsize;
            //cout << "\n\n\n++++++++++ m_Param [PGRIDSIZE] = " << m_Param[PGRIDSIZE] << flush;//TODO Remove this debug line


        m_Param [ PSPACING ] = launchParams.spacing;
        m_Param [ PSIMSCALE ] = launchParams.simscale;
            //std::cout<<"\nX X X X X X X X X launchParams.simscale = " << launchParams.simscale <<std::flush; //TODO Remove this debug line

        m_Param [ PSMOOTHRADIUS ] = launchParams.smoothradius;
            //cout << "\nXXXXXXXXXXXXXXXXXXXXXXX PSMOOTHRADIUS: " << m_Param[PSMOOTHRADIUS] << flush;//TODO Remove this debug line


        m_Param [ PVISC ] = launchParams.visc;
        m_Param [ PSURFACE_TENSION ] = launchParams.surface_tension;
        m_Param [ PMASS ] = launchParams.mass;
        m_Param [ PRADIUS ] = launchParams.radius;
        /*m_Param [ PDIST ] = launchParams.dist;*/
        m_Param [ PINTSTIFF ] = launchParams.intstiff;
        m_Param [ PEXTSTIFF ] = launchParams.extstiff;
        m_Param [ PEXTDAMP ] = launchParams.extdamp;
        m_Param [ PACCEL_LIMIT ] = launchParams.accel_limit;
        m_Param [ PVEL_LIMIT ] = launchParams.vel_limit;
        m_Param [ PGRAV ] = launchParams.grav;
        m_Param [ PGROUND_SLOPE ] = launchParams.ground_slope;
        m_Param [ PFORCE_MIN ] = launchParams.force_min;
        m_Param [ PFORCE_MAX ] = launchParams.force_max;
        m_Param [ PFORCE_FREQ ] = launchParams.force_freq;

        m_Vec [ PVOLMIN ] = launchParams.volmin;
        m_Vec [ PVOLMAX ] = launchParams.volmax;
        m_Vec [ PINITMIN ] = launchParams.initmin;
        m_Vec [ PINITMAX ] = launchParams.initmax;

        m_Param [ PACTUATION_FACTOR ] = launchParams.actuation_factor;
        m_Param [ PACTUATION_PERIOD ] = launchParams.actuation_period;

        break;
    case 8:  // default demo for parameter sweeps
        launchParams.num_particles = 4000;
        launchParams.demoType = 0;
        launchParams.simSpace = 7;

        m_Time = 0.000000;
        m_DT = 0.003000;

        m_Param [ PGRIDSIZE ] = 1.0;
        m_Param [ PSPACING ] = 1.000000;

        m_Param [ PSIMSCALE ] = 1.0;
        m_Param [ PSMOOTHRADIUS ] = 1.0;

        m_Param [ PVISC ] = 0.500000;
        m_Param [ PSURFACE_TENSION ] = 1.000000;

        m_Param [ PMASS ] = 0.00205;
        m_Param [ PRADIUS ] = 1.0000;

        m_Param [ PINTSTIFF ] = 2.000000;
        m_Param [ PEXTSTIFF ] = 50000.000000;
        m_Param [ PEXTDAMP ] = 100.000000;

        m_Param [ PACCEL_LIMIT ] = 150.000000;
        m_Param [ PVEL_LIMIT ] = 3.000000;

        m_Param [ PGRAV ] = 10.000000;
        m_Param [ PGROUND_SLOPE ] = 0.100000;

        m_Param [ PFORCE_MIN ] = 0.000000;
        m_Param [ PFORCE_MAX ] = 0.000000;
        m_Param [ PFORCE_FREQ ] = 16.000000;

        launchParams.x_dim = 10.000000;
        launchParams.y_dim = 10.000000;
        launchParams.z_dim = 3.000000;

        launchParams.pos_x = 0.000000;
        launchParams.pos_y = 0.000000;
        launchParams.pos_z = 0.000000;

        m_Vec [ PVOLMIN ] = cl_float3_init_with_values( 0.0, 0.0, 0.0 );
        m_Vec [ PVOLMAX ] = cl_float3_init_with_values( 10.0, 20.0, 20.0 );
        m_Vec [ PINITMIN ] = cl_float3_init_with_values( 2.0, 2.0, 2.0 );
        m_Vec [ PINITMAX ] = cl_float3_init_with_values( 10.0, 20.0, 10.0 );

        launchParams.num_files = 4000;
        launchParams.steps_per_InnerPhysicalLoop = 3;
        launchParams.steps_per_file = 6;
        launchParams.freeze_steps = 1;

        launchParams.verbosity = 0;
        launchParams.file_num = 0;

        launchParams.save_ply = 'n';
        launchParams.save_csv = 'n';
        launchParams.save_vtp = 'n';

        launchParams.gene_activity = 'n';
        launchParams.remodelling = 'n';
        launchParams.read_genome = 'y';

        m_Param [ PACTUATION_FACTOR ] = 0;
        m_Param [ PACTUATION_PERIOD ] = 1;

        break;
    }
    //std::cout<<"\nSetupExampleParams()3: launchParams.genomePath = "<<launchParams.genomePath<<"\n"<<std::flush;
}

void FluidSystem::SetupExampleGenome()  {   // need to set up a demo genome
    // Null genome
    for(int i=0; i< NUM_GENES; i++) m_FGenome.mutability[i] = 0;
    for(int i=0; i< NUM_GENES; i++) m_FGenome.delay[i] = 1;
    for(int i=0; i< NUM_GENES; i++) for(int j=0; j< NUM_GENES; j++) m_FGenome.sensitivity[i][j] = j;

    for(int i=0; i< NUM_TF/2; i++)      m_FGenome.tf_diffusability[i]    = 0;           // 1st half of TFs are non-diffusible.
    for(int i=NUM_TF/2; i< NUM_TF; i++) m_FGenome.tf_diffusability[i]    = 1;
    for(int i=0; i< NUM_TF; i++)        m_FGenome.tf_breakdown_rate[i]  = 1;

    for(int i=0; i< NUM_GENES; i++) for(int j=0; j< 2*NUM_TF+1; j++) m_FGenome.secrete[i][j]=0;     // 1st zero arrays.
    for(int i=0; i< NUM_GENES; i++) for(int j=0; j< 2*NUM_TF+1; j++) m_FGenome.activate[i][j]=0;

    m_FGenome.secrete[0][2*NUM_TF] = 2; // gene [0] secretes TF 1 & 3, at rates 1 & 4. // minimal test case.
    m_FGenome.secrete[0][2*0] = 1;
    m_FGenome.secrete[0][2*0+1] = 1;
    m_FGenome.secrete[0][2*1] = 3;
    m_FGenome.secrete[0][2*1+1] = 4;

    m_FGenome.activate[0][2*NUM_TF] = 1; // gene [0] activates TF 5, at rates 6. // minimal test case.
    m_FGenome.activate[0][2*0] = 5;
    m_FGenome.activate[0][2*0+1] = 6;

    //FBondParams *params_ =  &m_FGenome.fbondparams[0];
    //Particle remodelling, bond defaults & limits  m_FGenome.param[3][12]
    //0=elastin
    m_FGenome.param[0][m_FGenome.elongation_threshold]   = 0.5  ;
    m_FGenome.param[0][m_FGenome.elongation_factor]      = 0.002  ;
    m_FGenome.param[0][m_FGenome.strength_threshold]     = 1.0  ;
    m_FGenome.param[0][m_FGenome.strengthening_factor]   = 0.002  ;

    m_FGenome.param[0][m_FGenome.max_rest_length]        = 1.0  ;
    m_FGenome.param[0][m_FGenome.min_rest_length]        = 0.3  ;
    m_FGenome.param[0][m_FGenome.max_modulus]            = 100000;
    m_FGenome.param[0][m_FGenome.min_modulus]            = 10  ;

    m_FGenome.param[0][m_FGenome.elastLim]               = 8  ;
    m_FGenome.param[0][m_FGenome.default_rest_length]    = 0.5  ;
    m_FGenome.param[0][m_FGenome.default_modulus]        = 1000;
    m_FGenome.param[0][m_FGenome.default_damping]        = 10;

    //1=collagen
    m_FGenome.param[1][m_FGenome.elongation_threshold]   = 4.0  ;
    m_FGenome.param[1][m_FGenome.elongation_factor]      = 0.01 ;
    m_FGenome.param[1][m_FGenome.strength_threshold]     = 4.1  ;
    m_FGenome.param[1][m_FGenome.strengthening_factor]   = 0.01 ;

    m_FGenome.param[1][m_FGenome.max_rest_length]        = 1.0  ;
    m_FGenome.param[1][m_FGenome.min_rest_length]        = 0.3  ;
    m_FGenome.param[1][m_FGenome.max_modulus]            = 1000000000;//0.8  ;
    m_FGenome.param[1][m_FGenome.min_modulus]            = 1000000;//0.3  ;

    m_FGenome.param[1][m_FGenome.elastLim]               = 0.55  ;
    m_FGenome.param[1][m_FGenome.default_rest_length]    = 0.5  ;
    m_FGenome.param[1][m_FGenome.default_modulus]        = 10000000  ;
    m_FGenome.param[1][m_FGenome.default_damping]        = 100  ;

    //2=apatite
    m_FGenome.param[2][m_FGenome.elongation_threshold]   = 1.0  ;
    m_FGenome.param[2][m_FGenome.elongation_factor]      = 0.001  ;
    m_FGenome.param[2][m_FGenome.strength_threshold]     = 1.0  ;
    m_FGenome.param[2][m_FGenome.strengthening_factor]   = 0.001  ;

    m_FGenome.param[2][m_FGenome.max_rest_length]        = 1.0  ;
    m_FGenome.param[2][m_FGenome.min_rest_length]        = 0.3  ;
    m_FGenome.param[2][m_FGenome.max_modulus]            = 1000000000;//0.8  ;
    m_FGenome.param[2][m_FGenome.min_modulus]            = 1000000;//0.3  ;

    m_FGenome.param[2][m_FGenome.elastLim]               = 0.05  ;
    m_FGenome.param[2][m_FGenome.default_rest_length]    = 0.5  ;
    m_FGenome.param[2][m_FGenome.default_modulus]        = 10000000  ;
    m_FGenome.param[2][m_FGenome.default_damping]        = 1000  ;

    //Bond remodelling  m_FGenome.tanh_param[3][8];                 // lengthening/shortening
    m_FGenome.tanh_param[m_FGenome.elastin][m_FGenome.l_a] = 0.0;   // y-shift
    m_FGenome.tanh_param[m_FGenome.elastin][m_FGenome.l_b] = 0.0;   // y-scaling
    m_FGenome.tanh_param[m_FGenome.elastin][m_FGenome.l_c] = 0.0;   // x-scaling
    m_FGenome.tanh_param[m_FGenome.elastin][m_FGenome.l_d] = 0.0;   // x-shift

    m_FGenome.tanh_param[m_FGenome.elastin][m_FGenome.s_a] = 1.008; // strengthening/weakening  // mod_mul = 1.008+0.01*np.tanh(10*si2[i+1]-7)
    m_FGenome.tanh_param[m_FGenome.elastin][m_FGenome.s_b] = 0.01;
    m_FGenome.tanh_param[m_FGenome.elastin][m_FGenome.s_c] = 10.0;
    m_FGenome.tanh_param[m_FGenome.elastin][m_FGenome.s_d] = 7.0;

    m_FGenome.tanh_param[m_FGenome.collagen][m_FGenome.l_a] = 0.0;  // lengthening/shortening
    m_FGenome.tanh_param[m_FGenome.collagen][m_FGenome.l_b] = 0.0;
    m_FGenome.tanh_param[m_FGenome.collagen][m_FGenome.l_c] = 0.0;
    m_FGenome.tanh_param[m_FGenome.collagen][m_FGenome.l_d] = 0.0;

    m_FGenome.tanh_param[m_FGenome.collagen][m_FGenome.s_a] = 0.0;  // strengthening/weakening
    m_FGenome.tanh_param[m_FGenome.collagen][m_FGenome.s_b] = 0.0;
    m_FGenome.tanh_param[m_FGenome.collagen][m_FGenome.s_c] = 0.0;
    m_FGenome.tanh_param[m_FGenome.collagen][m_FGenome.s_d] = 0.0;

    m_FGenome.tanh_param[m_FGenome.apatite][m_FGenome.l_a] = 0.0;   // lengthening/shortening
    m_FGenome.tanh_param[m_FGenome.apatite][m_FGenome.l_b] = 0.0;
    m_FGenome.tanh_param[m_FGenome.apatite][m_FGenome.l_c] = 0.0;
    m_FGenome.tanh_param[m_FGenome.apatite][m_FGenome.l_d] = 0.0;

    m_FGenome.tanh_param[m_FGenome.apatite][m_FGenome.s_a] = 0.0;   // strengthening/weakening
    m_FGenome.tanh_param[m_FGenome.apatite][m_FGenome.s_b] = 0.0;
    m_FGenome.tanh_param[m_FGenome.apatite][m_FGenome.s_c] = 0.0;
    m_FGenome.tanh_param[m_FGenome.apatite][m_FGenome.s_d] = 0.0;

}

void FluidSystem::SetupSpacing (){

                                                                                                                            if(verbosity>0)std::cout<<"\n-----SetupSpacing() started... -----\n";

    m_Param [ PSIMSIZE ] = m_Param [ PSIMSCALE ] * (m_Vec[PVOLMAX].z - m_Vec[PVOLMIN].z);

    if ( m_Param[PSPACING] == 0 ) {
        // Determine spacing from density
        m_Param [PDIST] = pow ( (float) m_Param[PMASS] / m_Param[PRESTDENSITY], 1/3.0f );
        m_Param [PSPACING] = m_Param [ PDIST ]*0.87f / m_Param[ PSIMSCALE ];
    } else {
        // Determine density from spacing
        m_Param [PDIST] = m_Param[PSPACING] * m_Param[PSIMSCALE] / 0.87f;
        m_Param [PRESTDENSITY] = m_Param[PMASS] / pow ( (float) m_Param[PDIST], 3.0f );
    }
                                                                                                                        if (verbosity>0)printf ( "\nSetupSpacing: Density=,%f, Spacing=,%f, PDist=,%f\n", m_Param[PRESTDENSITY], m_Param[PSPACING], m_Param[PDIST] );

    // Particle Boundaries
    m_Vec[PBOUNDMIN] = m_Vec[PVOLMIN];
    m_Vec[PBOUNDMIN] = cl_float3_addDouble(&m_Vec[PBOUNDMIN],  2.0*(m_Param[PGRIDSIZE] / m_Param[PSIMSCALE])  );
    m_Vec[PBOUNDMAX] = m_Vec[PVOLMAX];
    m_Vec[PBOUNDMAX] = cl_float3_subtractDouble(&m_Vec[PBOUNDMAX], 2.0*(m_Param[PGRIDSIZE] / m_Param[PSIMSCALE])  );

                                                                                                                        if(verbosity>0)std::cout<<"\n-----SetupSpacing() finished-----\n";

}

void FluidSystem::SetupSimulation(int gpu_mode, int cpu_mode){ // const char * relativePath, int gpu_mode, int cpu_mode
     // Allocate buffers for points
    if (verbosity>1) std::cout<<"\n----- SetupSimulation() started,,, -----\n" << std::flush;
    if (verbosity>1) std::cout<<"\nSetupSimulation chk1 // verbosity="<<verbosity<< " // launchParams.num_particles= " << launchParams.num_particles << std::flush;

    m_Param [PNUM] = launchParams.num_particles;                             // NB there is a line of text above the particles, hence -1.

    //std::cout<<"\nX X X X X X X X X m_Param[PNUM] = " << m_Param[PNUM] <<std::flush; //TODO Remove this debug line
    mMaxPoints = m_Param[PNUM];
    std::cout<<"\nX X X X X X X X X X X X X X X m_Param [PNUM] = " << m_Param [PNUM] <<std::flush; //TODO Remove this debug line

    m_Param [PGRIDSIZE] = 2*m_Param[PSMOOTHRADIUS] / m_Param[PGRID_DENSITY];
    //std::cout<<"\nX X X X X X X X X m_Param [PGRIDSIZE] = " << m_Param [PGRIDSIZE] <<std::flush; //TODO Remove this debug line
    //std::cout<<"\nX X X X X X X X X m_Param m_Param[PSMOOTHRADIUS] = " << m_Param[PSMOOTHRADIUS] <<std::flush; //TODO Remove this debug line
    //std::cout<<"\nX X X X X X X X X m_Param m_Param[PGRID_DENSITY] = " << m_Param[PGRID_DENSITY] <<std::flush; //TODO Remove this debug line
    std::cout<<"\nSetupSimulation chk2" <<std::flush;

    SetupSPH_Kernels ();
    SetupSpacing ();

    std::cout << "\nmin: (" << m_Vec[PVOLMIN].x << ", " << m_Vec[PVOLMIN].y << ", " << m_Vec[PVOLMIN].z << ")" << std::flush;
    std::cout << "max: (" << m_Vec[PVOLMAX].x << ", " << m_Vec[PVOLMAX].y << ", " << m_Vec[PVOLMAX].z << ")" << std::flush;
    std::cout << "sim_scale: " << m_Param[PSIMSCALE] << std::flush;
    std::cout << "cell_size: " << m_Param[PGRIDSIZE] << std::flush;

    SetupGrid ( m_Vec[PVOLMIN]/*bottom corner*/, m_Vec[PVOLMAX]/*top corner*/, m_Param[PSIMSCALE], m_Param[PGRIDSIZE]);

    std::cout << "\nSetupSimulation chk3, verbosity=" << verbosity << "\n" <<std::flush;

    if (gpu_mode != GPU_OFF) {     // create CL instance etc..
        cout << "+OpenCL+ Setup started ...." << flush;
        cout << "\n\n\n\nRUNNING FLUIDSETUPCL() WITH mMaxPoints SET TO: " << mMaxPoints << ".\n\n\n\n" << flush;
        FluidSetupCL ( mMaxPoints, m_GridSrch, *(cl_int3*)& m_GridRes, *(cl_float3*)& m_GridSize, *(cl_float3*)& m_GridDelta, *(cl_float3*)& m_GridMin, *(cl_float3*)& m_GridMax, m_GridTotal, 0 );
        UpdateParams();            //  sends simulation params to device.
        UpdateGenome();            //  sends genome to device.              // NB need to initialize genome from file, or something.
        cout << "+OpenCL+ Setup finished." << flush;
    }
    if (verbosity>1)std::cout<<"\nSetupSimulation chk4, mMaxPoints="<<mMaxPoints<<", gpu_mode="<<gpu_mode<<", cpu_mode="<<cpu_mode<<", verbosity="<<verbosity<<std::flush;
    cout << "\n\n\n\nRUNNING ALLOCATEPARTICLES() WITH mMaxPoints SET TO: " << mMaxPoints << ".\n\n\n\n" << flush;
    AllocateParticles ( mMaxPoints, gpu_mode, cpu_mode );  // allocates only cpu buffer for particles
    if (verbosity>1)std::cout<<"\nSetupSimulation chk5 "<<std::flush;

    AllocateGrid(gpu_mode, cpu_mode);
    if (verbosity>1) std::cout<<"\n----- SetupSimulation() finished. -----\n" << std::flush;


}

void FluidSystem::Run2Simulation(){
        std::cout<<"\n\n-----Run2Simulation() started... -----"<<"\n"<<std::flush;
    Init_CLRand();
    auto old_begin = std::chrono::steady_clock::now();

    TransferPosVelVeval ();
    clCheck(clFinish(m_queue), "Run", "clFinish", "After TransferPosVelVeval, before 1st timestep", mbDebug);
    setFreeze(true);
    m_Debug_file=0;
    if (verbosity>0)std::cout<<"\n\nFreeze() "<<-1<<"\n"<<std::flush;
    Run2PhysicalSort();
//     InitializeBondsCL(); //TODO Implement again!

    if(launchParams.save_csv=='y'||launchParams.save_vtp=='y') TransferFromCL ();
    clCheck(clFinish(m_queue), "Run", "clFinish", "Run2Simulation After TransferFromCL", mbDebug);
    if(launchParams.save_csv=='y') SavePointsCSV2 ( launchParams.outPath, launchParams.file_num+90);
    if(launchParams.save_vtp=='y') SavePointsVTP2 ( launchParams.outPath, launchParams.file_num+90);
    if (verbosity>0)cout << "\n File# " << launchParams.file_num << ". " << std::flush;
    launchParams.file_num+=100;
    /*
    for (int k=0; k<launchParams.freeze_steps; k++){
      m_Debug_file=0;
      if (verbosity>0)std::cout<<"\n\nFreeze()"<<k<<"\n"<<std::flush;
      //Run2PhysicalSort();
      //InitializeBondsCL();
      //Run (launchParams.outPath, launchParams.file_num, (launchParams.verbosity>4), (launchParams.gene_activity=='y'), (launchParams.remodelling=='y') );
      for (int k=0; k<launchParams.steps_per_InnerPhysicalLoop*2; k++) Run2InnerPhysicalLoop();
      Run2PhysicalSort();
      ZeroVelCL ();                                                                                       // remove velocity, kinetic energy and momentum
      //TransferPosVelVeval ();
      if(launchParams.save_csv=='y'||launchParams.save_vtp=='y') TransferFromCL ();
      if(launchParams.save_csv=='y') SavePointsCSV2 ( launchParams.outPath, launchParams.file_num+90);
      if(launchParams.save_vtp=='y') SavePointsVTP2 ( launchParams.outPath, launchParams.file_num+90);
      launchParams.file_num+=100;
      m_Frame=launchParams.file_num;
    }
    */
    /////////
    printf("\nCHECK 2 Run2Simulation() \n");
    fflush(stdout);
    printf("launchParams.file_num: %d\n", launchParams.file_num);
    printf("\nlaunchParams.freeze_steps: %d\n", launchParams.freeze_steps);
    printf("\nlaunchParams.steps_per_InnerPhysicalLoop: %d\n", launchParams.steps_per_InnerPhysicalLoop);


    for ( ; launchParams.file_num<=launchParams.freeze_steps*100; launchParams.file_num+=100 ) {
            printf("\nCHECK 1.LOOP Run2Simulation() \n");

        std::cout<<"\n\nfile_num="<<launchParams.file_num<<", of "<<launchParams.num_files<<"\n"<<std::flush;
        m_Debug_file=0;
        m_Frame=launchParams.file_num;
        launchParams.file_increment=0;                                                                      // used within Run2InnerPhysicalLoop();
        for ( int j=0; j<launchParams.steps_per_file; j++ ) {
                        printf("\nCHECK 2.LOOP Run2Simulation() \n");

            for (int k=0; k<launchParams.steps_per_InnerPhysicalLoop; k++) {
                std::cout<<"\tk="<<k<<"\n"<<flush;
                Run2InnerPhysicalLoop();                                                                    // Run2InnerPhysicalLoop();
                printf("\nCHECK 3.LOOP Run2Simulation() \n");
            }
            if(launchParams.gene_activity=='y') Run2GeneAction();                                           // Run2GeneAction();
            if(launchParams.remodelling=='y') Run2Remodelling(launchParams.steps_per_InnerPhysicalLoop);                                          // Run2Remodelling();
            Run2PhysicalSort();                                                                             // Run2PhysicalSort();                // sort required for SavePointsVTP2
            ZeroVelCL ();                                                                                 // remove velocity, kinetic energy and momentum
        }
        if(launchParams.save_csv=='y'||launchParams.save_vtp=='y') TransferFromCL ();
        clCheck(clFinish(m_queue), "Run", "clFinish", "Run2Simulation After TransferFromCL", mbDebug);
        if(launchParams.save_csv=='y') SavePointsCSV2 ( launchParams.outPath, launchParams.file_num+90);
        if(launchParams.save_vtp=='y') SavePointsVTP2 ( launchParams.outPath, launchParams.file_num+90);
        if (verbosity>0)cout << "\n File# " << launchParams.file_num << ". " << std::flush;
    }
    setFreeze(false);                                                                                       // freeze=false => bonds can be broken now.
    printf("\n\nFreeze finished, starting normal Run ##############################################\n\n");
    //Run2PhysicalSort();

    for ( ; launchParams.file_num<launchParams.num_files; launchParams.file_num+=100 ) {
        std::cout<<"\n\nfile_num="<<launchParams.file_num<<", of "<<launchParams.num_files<<"\n"<<std::flush;
        m_Debug_file=0;
        m_Frame=launchParams.file_num;
        launchParams.file_increment=0;                                                                      // used within Run2InnerPhysicalLoop();
        for ( int j=0; j<launchParams.steps_per_file; j++ ) {
            for (int k=0; k<launchParams.steps_per_InnerPhysicalLoop; k++) {
                std::cout<<"\tk="<<k;
                Run2InnerPhysicalLoop();                                                                    // Run2InnerPhysicalLoop();
            }
            if(launchParams.gene_activity=='y') Run2GeneAction();                                           // Run2GeneAction();
            if(launchParams.remodelling=='y') Run2Remodelling(launchParams.steps_per_InnerPhysicalLoop);                                          // Run2Remodelling();
            Run2PhysicalSort();                                                                             // Run2PhysicalSort();                // sort required for SavePointsVTP2
        }
        auto begin = std::chrono::steady_clock::now();
        if(launchParams.save_csv=='y'||launchParams.save_vtp=='y') TransferFromCL ();
        clCheck(clFinish(m_queue), "Run", "clFinish", "Run2Simulation After TransferFromCL", mbDebug);
        if(launchParams.save_csv=='y') SavePointsCSV2 ( launchParams.outPath, launchParams.file_num+90);
        if(launchParams.save_vtp=='y') SavePointsVTP2 ( launchParams.outPath, launchParams.file_num+90);
        if (verbosity>0)cout << "\n File# " << launchParams.file_num << ". " << std::flush;

        auto end = std::chrono::steady_clock::now();
        std::chrono::duration<double> time = end - begin;
        std::chrono::duration<double> begin_dbl = begin - old_begin;
        if(launchParams.verbosity>0) std::cout << "\nLoop duration : "
                    << begin_dbl.count() <<" seconds. Time taken to write files for "
                    << NumPoints() <<" particles : "
                    << time.count() << " seconds. launchParams.num_files="
                    << launchParams.num_files << "\n" << std::endl;
        old_begin = begin;

        //if (mActivePoints < 500 ){std::cout<<"\n(mActivePoints < 500) stopping. chk why I am loosing particles?"<<std::flush;  Exit();}        // temp chk for why I am loosing particles.
    }
    //launchParams.file_num++;

    TransferFromCL ();
    SavePointsCSV2 ( launchParams.outPath, launchParams.file_num+99);   // save "end condition", even if not saving the series.
    SavePointsVTP2 ( launchParams.outPath, launchParams.file_num+99);

    WriteSimParams ( launchParams.outPath );
    WriteGenome( launchParams.outPath );
    WriteSpecificationFile_fromLaunchParams( launchParams.outPath );
}
