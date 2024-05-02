#include <assert.h>
#include <CL/cl.h>
#include <unistd.h>
#include "fluid_system.h"
#include "fluid_system.cpp"
#include "fileCL_IO.cpp"
#include <CL/cl.h>
#include <CL/cl_ext.h>
#include "randCL_well512.cl"        // for RandCL
#include <unordered_set>
#include <algorithm> // Include algorithm library for std::max_element
#include <numeric> // Include numeric library for std::accumulate

#define UINT_MAXSIZE 65535

cl_int FluidSystem::clMemsetD32(cl_mem buffer, int value, size_t count) {
    cl_int status;

    // Set kernel arguments

    status  = clSetKernelArg(m_Kern[FUNC_MEMSET32D], 0, sizeof(cl_mem), &buffer);
    //std::cout << "\nCountingSortFullCL()2: chk -------clMemsetD32: clSetKernelArg 1 done ---------"<< std::flush;

    status |= clSetKernelArg(m_Kern[FUNC_MEMSET32D], 1, sizeof(int), &value);
    clCheck(status, "clMemsetD32", "clSetKernelArg", NULL, mbDebug);
    //std::cout << "\nCountingSortFullCL()2: chk -------clMemsetD32: clSetKernelArg 2 done ---------"<< std::flush;

    // Enqueue kernel
    clCheck(clEnqueueNDRangeKernel(m_queue, m_Kern[FUNC_MEMSET32D], 1, NULL, &count, NULL, 0, NULL, NULL), "clMemsetD32", "clEnqueueNDRangeKernel", NULL, mbDebug);
    //std::cout << "\nCountingSortFullCL()2: chk -------clMemsetD32: clEnqueueNDRangeKernel done ---------"<< std::flush;

    // Ensure all commands have finished
    clFlush(m_queue);
    clFinish(m_queue);

    return CL_SUCCESS;
}

void FluidSystem::FluidSetupCL ( int num, int gsrch, cl_int3 res, cl_float3 size, cl_float3 delta, cl_float3 gmin, cl_float3 gmax, int total, int chk ){
	cl_int status;
	cl_event writeEvt;

    if (verbosity>0) std::cout << "\n-----FluidSetupCL()  started... -----\n" << std::flush;

    m_FParams.pnum = num;
    m_FParams.maxPoints = num;
    m_FParams.freeze = false;
    m_FParams.gridRes = res;
    m_FParams.gridSize = size;
    m_FParams.gridDelta = delta;
    m_FParams.gridMin = gmin;
    m_FParams.gridMax = gmax;
    m_FParams.gridTotal = total;
    m_FParams.gridSrch = gsrch;
    m_FParams.gridAdjCnt = gsrch*gsrch*gsrch;
    m_FParams.gridScanMax = res;
    cl_int3 temp = make_cl_int3(m_FParams.gridSrch, m_FParams.gridSrch, m_FParams.gridSrch);
    m_FParams.gridScanMax.x -= temp.x;
    m_FParams.gridScanMax.y -= temp.y;
    m_FParams.gridScanMax.z -= temp.z;
    m_FParams.chk = chk;

    std::cout << "Particle Number (pnum): " << m_FParams.pnum << "\n"
          << "Maximum Points (maxPoints): " << m_FParams.maxPoints << "\n"
          << "Freeze: " << m_FParams.freeze << "\n"
          << "Grid Resolution (gridRes): (" << m_FParams.gridRes.x << ", " << m_FParams.gridRes.y << ", " << m_FParams.gridRes.z << ")\n"
          << "Grid Size (gridSize): (" << m_FParams.gridSize.x << ", " << m_FParams.gridSize.y << ", " << m_FParams.gridSize.z << ")\n"
          << "Grid Delta (gridDelta): (" << m_FParams.gridDelta.x << ", " << m_FParams.gridDelta.y << ", " << m_FParams.gridDelta.z << ")\n"
          << "Grid Min (gridMin): (" << m_FParams.gridMin.x << ", " << m_FParams.gridMin.y << ", " << m_FParams.gridMin.z << ")\n"
          << "Grid Max (gridMax): (" << m_FParams.gridMax.x << ", " << m_FParams.gridMax.y << ", " << m_FParams.gridMax.z << ")\n"
          << "Grid Total (gridTotal): " << m_FParams.gridTotal << "\n"
          << "Grid Search (gridSrch): " << m_FParams.gridSrch << "\n"
          << "Grid Adjusted Count (gridAdjCnt): " << m_FParams.gridAdjCnt << "\n"
          << "Grid Scan Max (gridScanMax): (" << m_FParams.gridScanMax.x << ", " << m_FParams.gridScanMax.y << ", " << m_FParams.gridScanMax.z << ")\n"
          << "Check: " << m_FParams.chk << std::flush;

	clCheck(clEnqueueWriteBuffer(upload_queue, m_FluidTempDevice, 	CL_FALSE, 0, sizeof(m_FluidTemp), 		&m_FluidTemp, 		0, NULL, &writeEvt), "FluidSetupCL", "clEnqueueWriteBuffer", "m_FluidTempDevice", mbDebug);

	clCheck(clEnqueueWriteBuffer(upload_queue, m_FGenomeDevice, 	CL_FALSE, 0, sizeof(m_FGenome), 		&m_FGenome,         0, NULL, &writeEvt), "FluidSetupCL", "clEnqueueWriteBuffer", "m_FGenomeDevice",   mbDebug);

    //set m_FParamsDevice and m_FluidDevice(Temp) buffers as arguments to kernels
/*  TODO TODO Wohin damit?


    status = clSetKernelArg(m_Kern[FUNC_COUNT_SORT_DENSE_LISTS], 0, sizeof(cl_mem),     &m_FParamsDevice);                   if (status != CL_SUCCESS)	{ cout << "\nstatus = " << checkerror(status) <<"\n"<<flush; cout << "Error: allocatemem_chk1.4, clSetKernelArg(FUNC_COUNT_SORT_LISTS, 0)\n" << endl; exit_(status);}

    status = clSetKernelArg(m_Kern[FUNC_COUNT_SORT_DENSE_LISTS], 1, sizeof(cl_mem),     &m_FluidDevice);                     if (status != CL_SUCCESS)	{ cout << "\nstatus = " << checkerror(status) <<"\n"<<flush; cout << "Error: allocatemem_chk1.4, clSetKernelArg(FUNC_COUNT_SORT_LISTS, 0)\n" << endl; exit_(status);}

    status = clSetKernelArg(m_Kern[FUNC_COMPUTE_PRESS], 0, sizeof(cl_mem),              &m_FParamsDevice);                   if (status != CL_SUCCESS)	{ cout << "\nstatus = " << checkerror(status) <<"\n"<<flush; cout << "Error: allocatemem_chk1.4, clSetKernelArg(FUNC_COMPUTE_PRESS, 0)\n" << endl; exit_(status);}

    status = clSetKernelArg(m_Kern[FUNC_COMPUTE_PRESS], 1, sizeof(cl_mem),              &m_FluidDevice);                     if (status != CL_SUCCESS)	{ cout << "\nstatus = " << checkerror(status) <<"\n"<<flush; cout << "Error: allocatemem_chk1.4, clSetKernelArg(FUNC_COMPUTE_PRESS, 0)\n" << endl; exit_(status);}

    status = clSetKernelArg(m_Kern[FUNC_COMPUTE_FORCE], 0, sizeof(cl_mem),              &m_FParamsDevice);                   if (status != CL_SUCCESS)	{ cout << "\nstatus = " << checkerror(status) <<"\n"<<flush; cout << "Error: allocatemem_chk1.4, clSetKernelArg(FUNC_COMPUTE_FORCE, 0)\n" << endl; exit_(status);}

    status = clSetKernelArg(m_Kern[FUNC_COMPUTE_FORCE], 1, sizeof(cl_mem),              &m_FluidDevice);                     if (status != CL_SUCCESS)	{ cout << "\nstatus = " << checkerror(status) <<"\n"<<flush; cout << "Error: allocatemem_chk1.4, clSetKernelArg(FUNC_COMPUTE_FORCE, 0)\n" << endl; exit_(status);}

*/



    // Build Adjacency Lookup
    int cell = 0;
    for (int y=0; y < gsrch; y++ )
        for (int z=0; z < gsrch; z++ )
            for (int x=0; x < gsrch; x++ )
                m_FParams.gridAdj [ cell++]  = ( y * m_FParams.gridRes.z+ z )*m_FParams.gridRes.x +  x ;

    //Calculate the work group sizes
    status = clGetKernelWorkGroupInfo(m_Kern[FUNC_INSERT], m_device, CL_KERNEL_WORK_GROUP_SIZE, sizeof(local_work_size), &local_work_size, NULL); 	if (status != CL_SUCCESS)	{ cout << "\nstatus = " << checkerror(status) <<"\n"<<flush; exit_(status);}
    cout << "\nclGetKernelWorkGroupInfo: CL_KERNEL_WORK_GROUP_SIZE = " << local_work_size << "\n\n" << flush;

    num_work_groups = (mMaxPoints + local_work_size - 1) / local_work_size;
    global_work_size = num_work_groups * local_work_size;


    // Print global_work_size

    std::cout << "\n+--------------------------------------+" << std::endl;
    std::cout << "|           total_num_work_items       |" << std::endl;
    std::cout << "+--------------------------------------+" << std::endl;
    std::cout << "total_num_work_items = " <<  mMaxPoints << std::endl;

    std::cout << "+----------------------------------+" << std::endl;
    std::cout << "|           global_work_size       |" << std::endl;
    std::cout << "+----------------------------------+" << std::endl;
    std::cout << "global_work_size = " << num_work_groups * local_work_size << std::endl;

    // Print num_work_groups
    std::cout << "+----------------------------------+" << std::endl;
    std::cout << "|           num_work_groups        |" << std::endl;
    std::cout << "+----------------------------------+" << std::endl;
    std::cout << "num_work_groups = " << num_work_groups << std::endl;

    // Print local_work_size
    std::cout << "+----------------------------------+" << std::endl;
    std::cout << "|           local_work_size        |" << std::endl;
    std::cout << "+----------------------------------+" << std::endl;
    std::cout << "local_work_size = " << local_work_size << "\n" <<std::endl;

    // Compute number of blocks and threads
    m_FParams.itemsPerGroup = 256;                    //TODO probe hardware to set m_FParams.itemsPerGroup

    computeNumBlocks ( m_FParams.pnum, local_work_size, m_FParams.numGroups, m_FParams.numItems);				// particles

    // num_work_groups = m_FParams.numGroups

    std::cout << "+-------------------------------------+" << std::endl;
    std::cout << "|           m_FParams.numGroups       |" << std::endl;
    std::cout << "+-------------------------------------+" << std::endl;
    std::cout << "m_FParams.numGroups = " << m_FParams.numGroups << std::endl;

    std::cout << "+------------------------------------+" << std::endl;
    std::cout << "|           m_FParams.numItems       |" << std::endl;
    std::cout << "+------------------------------------+" << std::endl;
    std::cout << "m_FParams.numItems = " << m_FParams.numItems << std::endl;

    computeNumBlocks ( m_FParams.gridTotal, local_work_size, m_FParams.gridBlocks, m_FParams.gridThreads);		// grid cell

    // Compute particle buffer & grid dimensions

    cout << "\n\n++++++++++++++++++++++++++++++++++++++++++\n" <<
                    "\nNumber of Particles:" << m_FParams.pnum <<
                    "\nLocal Work Size:" << local_work_size <<
                    "\nItem per Group (hardcoded): " << m_FParams.itemsPerGroup <<
                    "\nNumber of Groups:" << num_work_groups <<
                    "\nm_FParams.numItems:" << m_FParams.numItems <<
            "\n\n++++++++++++++++++++++++++++++++++++++++++\n\n\n" << std::flush;

    m_FParams.szPnts = (m_FParams.numGroups  * m_FParams.numItems);
    if (verbosity > 1) cout << "m_FParams.szPnts: " << m_FParams.szPnts;

    if (verbosity>0) std::cout << "\n-----FluidSetupCL() finished----- \n" << std::flush;

}

void FluidSystem::TransferToTempCL ( int buf_id, int sz ){

//     printf("\n------- TransferToTempCL() started... -------\n");
//     fflush(stdout);
    clFlush (m_queue);
    clFinish (m_queue);

    clCheck ( clEnqueueCopyBuffer(m_queue, gpuVar(&m_Fluid, buf_id), gpuVar(&m_FluidTemp, buf_id), 0, 0, sz, 0, NULL, NULL), "TransferToTempCL", "clEnqueueCopyBuffer", "m_FluidTemp", mbDebug);

    clFlush (m_queue);
    clFinish (m_queue);

/*
    printf("------- TransferToTempCL() finished. -------\n");
    fflush(stdout);*/
}

void FluidSystem::TransferFromTempCL ( int buf_id, int sz ){

    clCheck ( clEnqueueCopyBuffer(m_queue, gpuVar(&m_FluidTemp, buf_id), gpuVar(&m_Fluid, buf_id), 0, 0, sz, 0, NULL, NULL), "TransferFromTempCL", "clEnqueueCopyBuffer", "m_Fluid", mbDebug);

}

void FluidSystem::TransferPosVelVeval (){

    printf("\n-------TransferPosVelVeval started... -------\n");
    fflush(stdout);
    TransferToTempCL ( FPOS,		 mMaxPoints *sizeof(cl_float3) );    // NB if some points have been removed, then the existing list is no longer dense,
    TransferToTempCL ( FVEL,		 mMaxPoints *sizeof(cl_float3) );    // hence must use mMaxPoints, not mNumPoints
    TransferToTempCL ( FVEVAL,	     mMaxPoints *sizeof(cl_float3) );
    printf("-------TransferPosVelVeval finished. -------\n");
    fflush(stdout);
}

void FluidSystem::TransferPosVelVevalFromTemp (){

    printf("------- TransferPosVelVevalFromTemp started... -------\n"); fflush(stdout);

    clFlush (m_queue);
    clFinish (m_queue);

    TransferFromTempCL ( FPOS,	     mMaxPoints *sizeof(cl_float3) );    // NB if some points have been removed, then the existing list is no longer dense,
    TransferFromTempCL ( FVEL,	     mMaxPoints *sizeof(cl_float3) );    // hence must use mMaxPoints, not mNumPoints
    TransferFromTempCL ( FVEVAL,	 mMaxPoints *sizeof(cl_float3) );

    clFlush (m_queue);
    clFinish (m_queue);

    printf("------- TransferPosVelVevalFromTemp finished.-------\n"); fflush(stdout);

}

void FluidSystem::InsertParticlesCL(uint* gcell, uint* gndx, uint* gcnt) { //bin sorting
    cl_int status;

    if (verbosity>1) printf("\n-----InsertParticles() started... -----");

        printf("\nm_GridTotal = %d\n", m_GridTotal);

    m_Fluid.mgpu[FBIN_COUNT] = clCreateBuffer(m_context, CL_MEM_READ_WRITE, m_GridTotal * sizeof(int),NULL, &status);

    if(status!=CL_SUCCESS)	{cout<<"\nclCreateBuffer status="<<checkerror(status)<<"\n"<<flush;exit_(status);}

    printf("\nMemory address of buffer: %p\n", (void*)&m_Fluid.mgpu[FBIN_COUNT]);


    // check if buffer is valid
    if   (m_Fluid.mgpu[FBIN_COUNT] == NULL)
         {printf("\nMemory object FBIN_COUNT is NOT valid\n\n");}
    else {printf("\nMemory object FBIN_COUNT IS valid\n\n");}

    printf("Size of m_GridTotal: %d ", m_GridTotal);

    size_t bufferSize;

    clGetMemObjectInfo(m_Fluid.mgpu[FBIN_COUNT], CL_MEM_SIZE, sizeof(size_t), &bufferSize, NULL);
    printf("\nAVAILABLE Buffer size: %zu bytes", bufferSize);

    // set offset and fill size
    size_t offset = 0;
    size_t fillSize = m_GridTotal * sizeof(int);

    // get needed buffer size and compare to available buffer size
    printf("\nBuffer size NEEDED: %zu bytes\n", fillSize);
    //printf("Size of int: %zu\n", sizeof(int)); // = 4

    if (offset + fillSize > bufferSize) {
         printf("\n--------------------------BUFFER TOO SMALL!-------------------------------------------");
         exit_(status);
    }
    /*DEBUGGING: else {
        // Perform the fill operation
        printf("\nBUFFER SIZE SUFFICES\n");
        //clEnqueueFillBuffer(m_queue, m_Fluid.mgpu[FBIN_COUNT], 0, sizeof(int), offset, fillSize, 0, NULL, NULL);
    }*/


    // first zero the counters
    int zero = 0;

    int eins = 1;
    clCheck(clEnqueueFillBuffer(m_queue,    gpuVar(&m_Fluid, FBIN_COUNT),                   &zero, sizeof(int),             0, m_GridTotal * sizeof(int),               0, NULL, NULL),   "InsertParticlesCL", "clEnqueueFillBuffer 1", NULL, mbDebug);
    clCheck(clEnqueueFillBuffer(m_queue,    gpuVar(&m_Fluid, FBIN_OFFSET),                  &zero, sizeof(int),             0, m_GridTotal * sizeof(int),               0, NULL, NULL),   "InsertParticlesCL", "clEnqueueFillBuffer 2", NULL, mbDebug);
    clCheck(clEnqueueFillBuffer(m_queue,    gpuVar(&m_Fluid, FBIN_COUNT_ACTIVE_GENES),      &zero, sizeof(uint[NUM_GENES]), 0, m_GridTotal * sizeof(uint[NUM_GENES]),   0, NULL, NULL),   "InsertParticlesCL", "clEnqueueFillBuffer 3", NULL, mbDebug);
    clCheck(clEnqueueFillBuffer(m_queue,    gpuVar(&m_Fluid, FBIN_OFFSET_ACTIVE_GENES),     &zero, sizeof(uint[NUM_GENES]), 0, m_GridTotal * sizeof(uint[NUM_GENES]),   0, NULL, NULL),   "InsertParticlesCL", "clEnqueueFillBuffer 4", NULL, mbDebug);

    clFlush(m_queue);
    clFinish(m_queue);

    clCheck(clEnqueueReadBuffer(m_queue, gpuVar(&m_Fluid, FBIN_COUNT), CL_TRUE, 0, (m_GridTotal - 1) * sizeof(int), bufI(&m_Fluid, FBIN_COUNT), 0, NULL, NULL), "InsertParticlesCL", "clEnqueueReadBuffer", "NULL", mbDebug);

    int t_blockSize = SCAN_BLOCKSIZE << 1;
    cout << "\n\nt_blockSize = " << t_blockSize << flush;

    size_t allElements = static_cast<size_t>(mNumPoints);
    size_t numItemsPerGroup = static_cast<size_t>(((allElements) / (t_blockSize)) + 1);


    cout << "\n+++++++++++++++++++++++++++ num_work_groups: " << num_work_groups << " +++++++++++++++++++++++" << flush;
    cout << "\n+++++++++++++++++++++++++++ mMaxPoints: " << mMaxPoints << " +++++++++++++++++++++++\n\n" << flush;


    clFlush(m_queue);
    clFinish(m_queue);

//     clEnqueueReadBuffer(m_queue, gpuVar(&m_Fluid, FPOS), CL_TRUE, 0, mMaxPoints*sizeof(cl_float3), bufV3(&m_Fluid, FPOS), 0, NULL, NULL);
//
//
//     printf("\n############################################################################################### FPOS B E F O R E InsertParticles():\n");
//     for (int i = 0; i < mMaxPoints; ++i) {
//         cl_float3 value = bufV3(&m_Fluid, FPOS)[i];
//         printf("Index: %d, Value: (%f, %f, %f)\n", i, value.s[0], value.s[1], value.s[2]);
//     }
//     fflush(stdout);


    clFlush(m_queue);
    clFinish(m_queue);

    // set arguments and launch kernel "InsertParticles"
    //void* args[1] = {&mMaxPoints}; //&mNumPoints
    clCheck(clSetKernelArg(m_Kern[FUNC_INSERT], 0, sizeof(cl_mem),   &m_FParamsDevice),                         "InsertParticlesCL", "clSetKernelArg 0", "FUNC_INSERT", mbDebug);
    clCheck(clSetKernelArg(m_Kern[FUNC_INSERT], 1, sizeof(int),      &mMaxPoints),                              "InsertParticlesCL", "clSetKernelArg 1", "FUNC_INSERT", mbDebug);
    clCheck(clSetKernelArg(m_Kern[FUNC_INSERT], 2, sizeof(cl_mem),   &m_Fluid.mgpu[FPOS]),                      "InsertParticlesCL", "clSetKernelArg 2", "FUNC_INSERT", mbDebug);
    clCheck(clSetKernelArg(m_Kern[FUNC_INSERT], 3, sizeof(cl_mem),   &m_Fluid.mgpu[FGCELL]),                    "InsertParticlesCL", "clSetKernelArg 3", "FUNC_INSERT", mbDebug);
    clCheck(clSetKernelArg(m_Kern[FUNC_INSERT], 4, sizeof(cl_mem),   &m_Fluid.mgpu[FGNDX]),                     "InsertParticlesCL", "clSetKernelArg 4", "FUNC_INSERT", mbDebug);
    clCheck(clSetKernelArg(m_Kern[FUNC_INSERT], 5, sizeof(cl_mem),   &m_Fluid.mgpu[FEPIGEN]),                   "InsertParticlesCL", "clSetKernelArg 5", "FUNC_INSERT", mbDebug);
    clCheck(clSetKernelArg(m_Kern[FUNC_INSERT], 6, sizeof(cl_mem),   &m_Fluid.mgpu[FBIN_COUNT]),                "InsertParticlesCL", "clSetKernelArg 6", "FUNC_INSERT", mbDebug);
    clCheck(clSetKernelArg(m_Kern[FUNC_INSERT], 7, sizeof(cl_mem),   &m_Fluid.mgpu[FBIN_COUNT_ACTIVE_GENES]),   "InsertParticlesCL", "clSetKernelArg 7", "FUNC_INSERT", mbDebug);




    if (verbosity > 0)
        cout << "\n#######\nCalling InsertParticles kernel: args[1] = {" << mNumPoints << "}, mMaxPoints=" << mMaxPoints
             << "\t m_FParams.numGroups=" << m_FParams.numGroups << ", m_FParams.numItems=" << m_FParams.numItems << " \t" << std::flush;

    clFlush(m_queue);
    clFinish(m_queue);

    //Running kernel
    cl_event func_insert_event;
    clCheck(                clEnqueueNDRangeKernel(m_queue,             // cl_command_queue     command_queue,
                                 m_Kern[FUNC_INSERT],                   // cl_kernel            kernel,
                                 1,                                     // cl_uint              work_dim,
                                 NULL,                                  // const size_t         *global_work_offset,
                                 &allElements,                      // const size_t         *global_work_size,
                                 &numItemsPerGroup,                       // const size_t         *local_work_size,
                                 0,                                     // cl_uint              num_events_in_wait_list,
                                 NULL,                                  // const cl_event       *event_wait_list,
                                 &func_insert_event),                   // cl_event             *event

    "InsertParticlesCL", "clEnqueueNDRangeKernel", "FUNC_INSERT", mbDebug);

    clWaitForEvents(1, &func_insert_event);


    clFlush(m_queue);
    clFinish(m_queue);


    clEnqueueReadBuffer(m_queue, gpuVar(&m_Fluid, FBIN_COUNT), CL_TRUE, 0, (m_GridTotal - 1) * sizeof(int), bufI(&m_Fluid, FBIN_COUNT), 0, NULL, NULL);


    clEnqueueReadBuffer(m_queue, gpuVar(&m_Fluid, FPOS), CL_TRUE, 0, mMaxPoints*sizeof(cl_float3), bufV3(&m_Fluid, FPOS), 0, NULL, NULL);


    printf("\n############################################################################################### FPOS A F T E R InsertParticles():\n");
    for (int i = 0; i < mMaxPoints; ++i) {
        cl_float3 value = bufV3(&m_Fluid, FPOS)[i];
        printf("Index: %d, Value: (%f, %f, %f)\n", i, value.s[0], value.s[1], value.s[2]);
    }
    fflush(stdout);



    clFlush(m_queue);
    clFinish(m_queue);


    // Transfer data back if requested (for validation)
    if (gcell != 0x0) {
        status = clEnqueueReadBuffer(m_queue,
                                  gpuVar(&m_Fluid, FGCELL),
                                  CL_TRUE,
                                  0,
                                  mNumPoints * sizeof(uint),
                                  gcell,
                                  0,
                                  NULL,
                                  NULL);
        status = clEnqueueReadBuffer(m_queue,
                                  gpuVar(&m_Fluid, FGNDX),
                                  CL_TRUE,
                                  0,
                                  mNumPoints * sizeof(uint),
                                  gndx,
                                  0,
                                  NULL,
                                  NULL);
        status = clEnqueueReadBuffer(m_queue,
                                  gpuVar(&m_Fluid, FBIN_COUNT),
                                  CL_TRUE,
                                  0,
                                  m_GridTotal * sizeof(uint),
                                  gcnt,
                                  0,
                                  NULL,
                                  NULL);

        //clean up TODO is something missing?
        clReleaseMemObject(m_Fluid.mgpu[FBIN_COUNT]);
        clReleaseMemObject(m_Fluid.mgpu[FBIN_OFFSET]);
    }
    if (verbosity > 0) {
        if (verbosity > 1)
            cout << "\nSaving (FGCELL) InsertParticlesCL: (particleIdx, cell) , mMaxPoints=" << mMaxPoints << "\t" << std::flush;
        status = clEnqueueReadBuffer(m_queue,
                                  gpuVar(&m_Fluid, FGCELL),
                                  CL_TRUE,
                                  0,
                                  sizeof(uint[mMaxPoints]),
                                  bufI(&m_Fluid, FGCELL),
                                  0,
                                  NULL,
                                  NULL);
        // TODO Has to be implemented again. CUCLCUCL
        SaveUintArray(bufI(&m_Fluid, FGCELL), mMaxPoints, "InsertParticlesCL__bufI(&m_Fluid, FGCELL).csv");
    }
                                                                            if(verbosity>1) cout << "\n-----InsertParticlesCL finished-----\n\n" << flush;
}

 void FluidSystem::PrefixSumCellsCL (int zero_offsets) {
    if (verbosity>1) cout << "-------------------------------------------------------------------------------------------" << flush;
    if (verbosity>1) printf("\n-----PrefixSumCellsCL() started... -----\n");

    cl_int status;
    cl_event writeEvt;
    // Prefix Sum - determine grid offsets

    // Doing this, because different variable types of the same values are needed as parameters.
    int t_blockSize = SCAN_BLOCKSIZE << 1;
    int t_blockSize_t = SCAN_BLOCKSIZE << 1;
    size_t t_blockSize2 = SCAN_BLOCKSIZE;
    size_t sizeValue0 = static_cast<size_t>(t_blockSize);
    const size_t* blockSize = &sizeValue0;
    cout << "\n\nt_blockSize = " << t_blockSize << flush;

    size_t t_fixedNumElem = static_cast<size_t>(1);
    const size_t* fixedNumElem = &t_fixedNumElem;

    // Calculate the smallest multiple of 128 greater than 10000
    size_t globalNumThreads = ((m_GridTotal / t_blockSize) +1) * SCAN_BLOCKSIZE;
    cout << "\nnglobalNumThreads = " << globalNumThreads << flush;          //allElements = 10000 = m_GridTotal
    size_t globalNumThreads2 = ((globalNumThreads / t_blockSize_t) +1) * SCAN_BLOCKSIZE;
    cout << "\nglobalNumThreads2 = " << globalNumThreads2 << flush;          //allElements = 10000 = m_GridTotal

    int t_numElem1 = m_GridTotal;
    size_t allElements = static_cast<size_t>(t_numElem1);
    if (verbosity>1) printf("\nmNumPoints before PrefixSum: %d\n", mNumPoints);
    cout << "\nnumElem1/allElements = " << allElements << flush;          //allElements = 10000 = m_GridTotal


    int t_numElem2 = int (t_numElem1 / t_blockSize) + 1;
    size_t numItemsPerGroup2 = static_cast<size_t>(t_numElem2);
    cout << "\nnumElem2/numItemsPerGroup = " << numItemsPerGroup2 << flush;          //numItemsPerGroup = 20


    int t_numElem3 = int (t_numElem2/ t_blockSize) + 1;
    size_t numItemsPerGroup3 = static_cast<size_t>(t_numElem3);
    cout << "\nnumElem3/numItemsPerGroup3 = " << numItemsPerGroup3 << flush;          //numItemsPerGroup2 = 1

    int t_threads = SCAN_BLOCKSIZE;
    size_t sizeValue4 = static_cast<size_t>(t_threads);
    const size_t* threads = &sizeValue4;
    cout << "\nthreads = " << *threads << flush;

    int t_numItems = t_threads * t_numElem2;
    size_t numItems = static_cast<size_t>(t_numItems);
    cout << "\nnumItems= " << numItems << flush;          //numItems =

    int t_numItems2 = t_threads * t_numElem3;
    size_t numItems2 = static_cast<size_t>(t_numItems2);
    cout << "\nnumItems2 = " << numItems2 << "\n" << flush;          //numItems2 =


    int zon = 1;

    cl_mem array1  = gpuVar(&m_Fluid, FBIN_COUNT);		// input
    cl_mem scan1   = gpuVar(&m_Fluid, FBIN_OFFSET);		// output
    cl_mem array2  = gpuVar(&m_Fluid, FAUXARRAY1);
    cl_mem scan2   = gpuVar(&m_Fluid, FAUXSCAN1);
    cl_mem array3  = gpuVar(&m_Fluid, FAUXARRAY2);
    cl_mem scan3   = gpuVar(&m_Fluid, FAUXSCAN2);
/*

    clEnqueueReadBuffer(m_queue, gpuVar(&m_Fluid, FBIN_COUNT), CL_TRUE, 0, (m_GridTotal-1)*sizeof(int), bufI(&m_Fluid, FBIN_COUNT), 0, NULL, NULL);

    cout << "\n############################################################################################### FBIN_COUNT B E F O R E PrefixSumCellsCL():\n";
    for (int i = 0; i < 10000; ++i) {
        if (bufI(&m_Fluid, FBIN_COUNT)[i] != 0) {
            cout << "Index: " << i << ", Value: " << bufI(&m_Fluid, FBIN_COUNT)[i] << endl;
        }
    }

    clEnqueueReadBuffer(m_queue, gpuVar(&m_Fluid, FBIN_OFFSET), CL_TRUE, 0, (m_GridTotal-1)*sizeof(int), bufI(&m_Fluid, FBIN_OFFSET), 0, NULL, NULL);

    cout << "\n############################################################################################### FBIN_OFFSET B E F O R E PrefixSumCellsCL():\n";
    for (int i = 0; i < 10000; ++i) {
        if (bufI(&m_Fluid, FBIN_OFFSET)[i] != 0) {
            cout << "Index: " << i << ", Value: " << bufI(&m_Fluid, FBIN_OFFSET)[i] << endl;
        }
    }*/
/*
//     clEnqueueReadBuffer(m_queue, gpuVar(&m_Fluid, FPOS), CL_TRUE, 0, mMaxPoints*sizeof(cl_float3), bufV3(&m_Fluid, FPOS), 0, NULL, NULL);
//
//
//     printf("\n############################################################################################### FPOS B E F O R E PrefixSumCells():\n");
//     for (int i = 0; i < mMaxPoints; ++i) {
//         cl_float3 value = bufV3(&m_Fluid, FPOS)[i];
//         printf("Index: %d, Value: (%f, %f, %f)\n", i, value.s[0], value.s[1], value.s[2]);
//     }
//     fflush(stdout);
//
*/

    clFlush(m_queue);
    clFinish(m_queue);


    #ifndef xlong
        typedef unsigned long long	xlong;		// 64-bit integer
    #endif
    if ( t_numElem1 > SCAN_BLOCKSIZE*xlong(SCAN_BLOCKSIZE)*SCAN_BLOCKSIZE) { if (m_FParams.debug>1)printf ( "\nERROR: Number of elements exceeds prefix sum max. Adjust SCAN_BLOCKSIZE.\n" );  }

            clSetKernelArg(m_Kern[FUNC_FPREFIXSUM], 0, sizeof(cl_mem), &array1);
            clSetKernelArg(m_Kern[FUNC_FPREFIXSUM], 1, sizeof(cl_mem), &scan1);
            clSetKernelArg(m_Kern[FUNC_FPREFIXSUM], 2, sizeof(cl_mem), &array2);
            clSetKernelArg(m_Kern[FUNC_FPREFIXSUM], 3, sizeof(int), &t_numElem1); // size_t  allElements
            clSetKernelArg(m_Kern[FUNC_FPREFIXSUM], 4, sizeof(int), &zero_offsets);


    //clCheck(clEnqueueNDRangeKernel(m_queue, m_Kern[FUNC_FPREFIXSUM], 1, NULL, numItemsPerGroup, threads, 0, NULL, NULL), "PrefixSumCellsCL", "clEnqueueNDRangeKernel", "FUNC_FPREFIXSUM_1", mbDebug);
    clCheck(clEnqueueNDRangeKernel( //----------------------------------------------FUNC_FPREFIXSUM--------------------------------------------------------------------------------------

        m_queue, m_Kern[FUNC_FPREFIXSUM],
        1,
        NULL,
        &globalNumThreads,
        &t_blockSize2,
        0,
        NULL,
        NULL),

    "PrefixSumCellsCL", "clEnqueueNDRangeKernel", "FUNC_FPREFIXSUM 1.1", mbDebug);

    clFlush(m_queue);
    clFinish(m_queue);

    /// TODO    insert if(...) here , wrapping 2nd & 3rd preficsum(..) calls .

            clSetKernelArg(m_Kern[FUNC_FPREFIXSUM], 0, sizeof(cl_mem), &array2);
            clSetKernelArg(m_Kern[FUNC_FPREFIXSUM], 1, sizeof(cl_mem), &scan2);
            clSetKernelArg(m_Kern[FUNC_FPREFIXSUM], 2, sizeof(cl_mem), &array3);
            clSetKernelArg(m_Kern[FUNC_FPREFIXSUM], 3, sizeof(int), &numItemsPerGroup2);
            clSetKernelArg(m_Kern[FUNC_FPREFIXSUM], 4, sizeof(int), &zon);

    clCheck(clEnqueueNDRangeKernel( //----------------------------------------------FUNC_FPREFIXSUM

        m_queue,
        m_Kern[FUNC_FPREFIXSUM],
        1,
        NULL,
        &globalNumThreads2,
        &t_blockSize2,
        0,
        NULL,
        NULL),
    "PrefixSumCellsCL", "clEnqueueNDRangeKernel", "FUNC_FPREFIXSUM 1.2", mbDebug);

/*
//     clFlush(m_queue);
//     clFinish(m_queue);
//
//     // Define the path to the desktop
//     const std::string desktopPath = "/home/goldi/Documents/KDevelop Projects/Morphogenesis/";
//
//     // Define the file name for the .csv file
//     const std::string filename = desktopPath + "output_scan1.csv";
//
//     // Open the file for writing
//     std::ofstream outputFile(filename);
//
//     // Check if the file is open
//     if (!outputFile.is_open()) {
//         std::cerr << "Error: Unable to open file for writing." << std::endl;
//         return; // or handle the error accordingly
//     }
//
//     // Write the contents of the scan1 buffer to the file in CSV format
//     outputFile << "Index, Value" << std::endl;
//     for (int i = 0; i < m_GridTotal; ++i) {
//         if(bufI(&m_Fluid, FBIN_OFFSET)[i] != 0) outputFile << i << ", " << bufI(&m_Fluid, FBIN_OFFSET)[i] << std::endl;
//     }
//
//     // Close the file
//     outputFile.close();
*/
    clFlush(m_queue);
    clFinish(m_queue);


      if (t_numElem3 > 1) {
        cl_mem nptr_cl = NULL;
        //void* argsC[5] = { &array3, &scan3, &nptr, &numItemsPerGroup2, &zon };	        // sum array3. output -> scan3                  i.e. FAUXARRAY2 -> FAUXSCAN2, &nptr
                clSetKernelArg(m_Kern[FUNC_FPREFIXSUM], 0, sizeof(cl_mem), &array3);
                clSetKernelArg(m_Kern[FUNC_FPREFIXSUM], 1, sizeof(cl_mem), &scan3);
                clSetKernelArg(m_Kern[FUNC_FPREFIXSUM], 2, sizeof(cl_mem), &nptr_cl);
                clSetKernelArg(m_Kern[FUNC_FPREFIXSUM], 3, sizeof(int), &numItemsPerGroup3);
                clSetKernelArg(m_Kern[FUNC_FPREFIXSUM], 4, sizeof(int), &zon);

        clCheck(clEnqueueNDRangeKernel( //----------------------------------------------FUNC_FPREFIXSUM


            m_queue, m_Kern[FUNC_FPREFIXSUM],
            1,
            NULL,
            &t_blockSize2,
            &t_blockSize2,
            0,
            NULL,
            NULL),

        "PrefixSumCellsCL", "clEnqueueNDRangeKernel", "FUNC_FPREFIXSUM 1.3", mbDebug);
        //clCheck ( clLaunchKernel ( m_Kern[FUNC_FPREFIXSUM], 1, 1, 1, threads, 1, 1, 0, NULL, argsC, NULL ), "PrefixSumCellsCL", "cuLaunch", "FUNC_PREFIXFIXUP", mbDebug);
        clFlush(m_queue);
        clFinish(m_queue);

        //void* argsD[3] = { &scan2, &scan3, &numItemsPerGroup };	                        // merge scan3 into scan2. output -> scan2      i.e. FAUXSCAN2, FAUXSCAN1 -> FAUXSCAN1
                clSetKernelArg(m_Kern[FUNC_FPREFIXUP], 0, sizeof(cl_mem), &scan2);
                clSetKernelArg(m_Kern[FUNC_FPREFIXUP], 1, sizeof(cl_mem), &scan3);
                clSetKernelArg(m_Kern[FUNC_FPREFIXUP], 2, sizeof(int), &numItemsPerGroup2);

        //clCheck ( clLaunchKernel ( m_Kern[FUNC_FPREFIXUP], numItemsPerGroup2, 1, 1, threads, 1, 1, 0, NULL, argsD, NULL ), "PrefixSumCellsCL", "cuLaunch", "FUNC_PREFIXFIXUP", mbDebug);
        clCheck(clEnqueueNDRangeKernel( //----------------------------------------------FUNC_FPREFIXUP


            m_queue,
            m_Kern[FUNC_FPREFIXUP],
            1,
            NULL,
            &globalNumThreads2,
            &t_blockSize2,
            0,
            NULL,
            NULL),

        "PrefixSumCellsCL", "clEnqueueNDRangeKernel", "FUNC_FPREFIXUP 1.4", mbDebug);

     }

    clFlush(m_queue);
    clFinish(m_queue);

    //void* argsE[3] = { &scan1, &scan2, &numElem1 };		                        // merge scan2 into scan1. output -> scan1      i.e. FAUXSCAN1, FBIN_OFFSET -> FBIN_OFFSET
            clSetKernelArg(m_Kern[FUNC_FPREFIXUP], 0, sizeof(cl_mem), &scan1);
            clSetKernelArg(m_Kern[FUNC_FPREFIXUP], 1, sizeof(cl_mem), &scan2);
            clSetKernelArg(m_Kern[FUNC_FPREFIXUP], 2, sizeof(int), &t_numElem1);    //allElements

            // TODO end of if(..) block for 2nd & 3rd prefixsum(..) calls


/*
    //clCheck ( clLaunchKernel ( m_Kern[FUNC_FPREFIXUP], numItemsPerGroup, 1, 1, threads, 1, 1, 0, NULL, argsE, NULL ), "PrefixSumCellsCL", "cuLaunch", "FUNC_PREFIXFIXUP", mbDebug);

    cout << "\n+#+#+#+#+#+#+#+#+#+#+#+#+#+#+#+#+#+#+#+#+#+#+#+#+#+#+#+#+#+#+#+# globalNumThreads = " << globalNumThreads << flush;          //allElements = 10000 = m_GridTotal
    cout << "\n+#+#+#+#+#+#+#+#+#+#+#+#+#+#+#+#+#+#+#+#+#+#+#+#+#+#+#+#+#+#+#+# t_blockSize2 = " << t_blockSize2 << flush;          //allElements = 10000 = m_GridTotal
*/
    clCheck(clEnqueueNDRangeKernel( //----------------------------------------------FUNC_FPREFIXUP


        m_queue,
        m_Kern[FUNC_FPREFIXUP],
        1,
        NULL,
        &globalNumThreads,
        &t_blockSize2,
        0,
        NULL,
        NULL),

    "PrefixSumCellsCL", "clEnqueueNDRangeKernel", "FUNC_FPREFIXUP 2.1", mbDebug);

    clFlush(m_queue);
    clFinish(m_queue);
/*
//     clEnqueueReadBuffer(m_queue, gpuVar(&m_Fluid, FBIN_OFFSET), CL_TRUE, 0, (m_GridTotal-1)*sizeof(int), bufI(&m_Fluid, FBIN_OFFSET), 0, NULL, NULL);
//
//     cout << "\n############################################################################################### FBIN_OFFSET A F T E R  PrefixSumCellsCL():\n";
//     for (int i = 0; i < 10000; ++i) {
//         if (bufI(&m_Fluid, FBIN_OFFSET)[i] != 0) {
//             cout << "Index: " << i << ", Value: " << bufI(&m_Fluid, FBIN_OFFSET)[i] << endl;
//         }
//     }
*/
    clEnqueueReadBuffer(m_queue, gpuVar(&m_Fluid, FPOS), CL_TRUE, 0, mMaxPoints*sizeof(cl_float3), bufV3(&m_Fluid, FPOS), 0, NULL, NULL);

    clFlush(m_queue);
    clFinish(m_queue);

    printf("\n############################################################################################### FPOS A F T E R PrefixSumCells():\n");
    for (int i = 0; i < mMaxPoints; ++i) {
        cl_float3 value = bufV3(&m_Fluid, FPOS)[i];
        uint x = (uint)value.s[0];
        uint y = (uint)value.s[1];
        uint z = (uint)value.s[2];
        printf("Index: %d, Value: (%u, %u, %u)\n", i, x, y, z);
    }
    fflush(stdout);


    // Loop to PrefixSum the Dense Lists - NB by doing one gene at a time, we reuse the FAUX* arrays & scans.
    // For each gene, input FBIN_COUNT_ACTIVE_GENES[gene*m_GridTotal], output FBIN_OFFSET_ACTIVE_GENES[gene*m_GridTotal]
    cl_mem array0 = gpuVar(&m_Fluid,FBIN_COUNT_ACTIVE_GENES);
    cl_mem scan0 = gpuVar(&m_Fluid,FBIN_OFFSET_ACTIVE_GENES);
    /*
    cout << "\n____\n\n";

    for (int gene = 0; gene < NUM_GENES; gene++) {
    size_t offset_array0 = gene * t_numElem1 * sizeof(int);
    size_t offset_scan0 = gene * t_numElem1 * sizeof(int);

    status = clEnqueueWriteBuffer(m_queue, m_FPrefixDevice, CL_FALSE, 0, 16 * sizeof(offset_array0), &offset_array0, 0, NULL, &writeEvt);						// WriteBuffer param_buf ##########


    int pattern = 0;


        // Set kernel arguments
        clCheck(clSetKernelArg(m_Kern[FUNC_FPREFIXSUM], 0, sizeof(cl_mem), &array0),            "YourFunction", "clSetKernelArg", "FUNC_FPREFIXSUM1", mbDebug);
        clCheck(clSetKernelArg(m_Kern[FUNC_FPREFIXSUM], 1, sizeof(cl_mem), &scan0),             "YourFunction", "clSetKernelArg", "FUNC_FPREFIXSUM2", mbDebug);
        clCheck(clSetKernelArg(m_Kern[FUNC_FPREFIXSUM], 2, sizeof(cl_mem), &array2),            "YourFunction", "clSetKernelArg", "FUNC_FPREFIXSUM3", mbDebug);
        clCheck(clSetKernelArg(m_Kern[FUNC_FPREFIXSUM], 3, sizeof(size_t), &offset_array0),     "YourFunction", "clSetKernelArg", "FUNC_FPREFIXSUM6", mbDebug);
        clCheck(clSetKernelArg(m_Kern[FUNC_FPREFIXSUM], 4, sizeof(size_t), &offset_scan0),     "YourFunction", "clSetKernelArg", "FUNC_FPREFIXSUM7", mbDebug);
        clCheck(clSetKernelArg(m_Kern[FUNC_FPREFIXSUM], 5, sizeof(int), &t_numElem1),           "YourFunction", "clSetKernelArg", "FUNC_FPREFIXSUM4", mbDebug);
        clCheck(clSetKernelArg(m_Kern[FUNC_FPREFIXSUM], 6, sizeof(int), &zero_offsets),         "YourFunction", "clSetKernelArg", "FUNC_FPREFIXSUM5", mbDebug);



    //         clCheck(clEnqueueFillBuffer(m_queue, scan0, &pattern, sizeof(pattern), 0, offset_scan0 * sizeof(int), 0, NULL, NULL), "PrefixSumCellsCL", "clEnqueueFillBuffer", "FBIN_COUNT", mbDebug);
    //         clCheck(clEnqueueFillBuffer(m_queue, array0, &pattern, sizeof(pattern), 0, offset_array0 * sizeof(int), 0, NULL, NULL), "PrefixSumCellsCL", "clEnqueueFillBuffer", "FBIN_COUNT", mbDebug);
    //         clCheck(clEnqueueFillBuffer(m_queue, scan2, &pattern, sizeof(pattern), 0, t_numElem2 * sizeof(int), 0, NULL, NULL), "PrefixSumCellsCL", "clEnqueueFillBuffer", "FBIN_COUNT", mbDebug);
    //         clCheck(clEnqueueFillBuffer(m_queue, array3, &pattern, sizeof(pattern), 0, t_numElem3 * sizeof(int), 0, NULL, NULL), "PrefixSumCellsCL", "clEnqueueFillBuffer", "FBIN_COUNT", mbDebug);
    //         clCheck(clEnqueueFillBuffer(m_queue, scan3, &pattern, sizeof(pattern), 0, t_numElem3 * sizeof(int), 0, NULL, NULL), "PrefixSumCellsCL", "clEnqueueFillBuffer", "FBIN_COUNT", mbDebug);

    //void* argsA[5] = {&array1, &scan1, &array2, &allElements, &zero_offsets};

    status = clFlush(m_queue); 				if (status != CL_SUCCESS)	{ cout << "\nclFlush(m_queue) status = " << checkerror(status) <<"\n"<<flush; exit_(status);}
	status = clFinish(m_queue); 			if (status != CL_SUCCESS)	{ cout << "\nclFinish(m_queue) = "		 <<checkerror(status)  <<"\n"<<flush; exit_(status);}
    clCheck(clEnqueueNDRangeKernel(

        m_queue,
        m_Kern[FUNC_FPREFIXSUM],
        1,
        0,
        &global_work_size,
        0,
        0,
        NULL,
        NULL),

    "PrefixSumCellsCL", "clEnqueueNDRangeKernel", "FUNC_PREFIXSUM", mbDebug);


        printf("\nPREFIX SUM done.\n");
    //     //void* argsB[5] = {&array2, &scan2, &array3, &numItemsPerGroup, &zon};
    //     clCheck(clEnqueueNDRangeKernel(m_queue, m_Kern[FUNC_FPREFIXSUM], 1, NULL, numItemsPerGroup2, threads, 0, NULL, NULL), "PrefixSumCellsCL", "clEnqueueNDRangeKernel", "FUNC_PREFIXSUM", mbDebug);
    //
    //     if ((*numItemsPerGroup2) > 1) {
    //         cl_mem nptr = {0};
    //         //void* argsC[5] = {&array3, &scan3, &nptr, &numItemsPerGroup2, &zon};
    //         clCheck(clEnqueueNDRangeKernel(m_queue, m_Kern[FUNC_FPREFIXSUM], 1, NULL, fixedNumElem, threads, 0, NULL, NULL), "PrefixSumCellsCL", "clEnqueueNDRangeKernel", "FUNC_PREFIXFIXUP", mbDebug);
    //
    //         //void* argsD[3] = {&scan2,&scan3,&numItemsPerGroup};
    //         clCheck(clEnqueueNDRangeKernel(m_queue,m_Kern[FUNC_FPREFIXUP],1, NULL, numItemsPerGroup2, threads,0,NULL,NULL),"PrefixSumCellsCL","clEnqueueNDRangeKernel","FUNC_PREFIXFIXUP",mbDebug);
    //     }
    //
    // //     void* argsE[3] = {&scan1,&scan2,&allElements};
    //     clCheck(clEnqueueNDRangeKernel(m_queue,m_Kern[FUNC_FPREFIXUP],1,NULL, numItemsPerGroup, threads,0,NULL,NULL),"PrefixSumCellsCL","clEnqueueNDRangeKernel","FUNC_PREFIXFIXUP",mbDebug);

        // Loop to PrefixSum the Dense Lists - NB by doing one gene at a time,
        // we reuse the FAUX* arrays & scans.
        // For each gene,
        // input FBIN_COUNT_ACTIVE_GENES[gene*m_GridTotal],
        // output FBIN_OFFSET_ACTIVE_GENES[gene*m_GridTotal]



            //         void* argsA[5] = {&array1,&scan1,&array2,&allElements,&zero_offsets};
                    clCheck(clEnqueueNDRangeKernel(m_queue,m_Kern[FUNC_FPREFIXSUM],1,NULL,numItemsPerGroup,threads,0,NULL,NULL),"PrefixSumCellsCL","clEnqueueNDRangeKernel","FUNC_PREFIXSUM",mbDebug);

            //         void* argsB[5] = {&array2,&scan2,&array3,&numItemsPerGroup,&zon};
                    clCheck(clEnqueueNDRangeKernel(m_queue,m_Kern[FUNC_FPREFIXSUM],1,NULL,numItemsPerGroup2,threads,0,NULL,NULL),"PrefixSumCellsCL","clEnqueueNDRangeKernel","FUNC_PREFIXSUM",mbDebug);

                    if ((*numItemsPerGroup2) > 1) {
                        cl_mem nptr = {0};
            //             void* argsC[5] = {&array3,&scan3,&nptr,&numItemsPerGroup2,&zon};
                        clCheck(clEnqueueNDRangeKernel(m_queue,m_Kern[FUNC_FPREFIXSUM],1,NULL,fixedNumElem,threads,0,NULL,NULL),"PrefixSumCellsCL","clEnqueueNDRangeKernel","FUNC_PREFIXFIXUP",mbDebug);

            //             void* argsD[3] = {&scan2,&scan3,&numItemsPerGroup};
                        clCheck(clEnqueueNDRangeKernel(m_queue,m_Kern[FUNC_FPREFIXUP],1,NULL,numItemsPerGroup2,threads,0,NULL,NULL),"PrefixSumCellsCL","clEnqueueNDRangeKernel","FUNC_PREFIXFIXUP",mbDebug);
                    }

            //         void* argsE[3] = {&scan1,&scan2,&allElements};
                    //clCheck(clEnqueueNDRangeKernel(m_queue,m_Kern[FUNC_FPREFIXUP],1,NULL,numItemsPerGroup,threads,0,NULL,NULL),"PrefixSumCellsCL","clEnqueueNDRangeKernel","FUNC_PREFIXFIXUP",mbDebug);
    }
*/
/*
    printf("\nPhase 3 done.\n");

    int num_lists = NUM_GENES,
    length = FDENSE_LIST_LENGTHS,
    fgridcnt = FBIN_COUNT_ACTIVE_GENES,
    fgridoff = FBIN_OFFSET_ACTIVE_GENES;
    size_t t_NUM_GENES_CONST = static_cast<size_t>(NUM_GENES);
    const size_t* NUM_GENES_CONST = &t_NUM_GENES_CONST;

//     void* argsF[4] = {&num_lists,&length,&fgridcnt,&fgridoff};
    status = clFlush(m_queue); 				if (status != CL_SUCCESS)	{ cout << "\nclFlush(m_queue) status = " << checkerror(status) <<"\n"<<flush; exit_(status);}
	status = clFinish(m_queue); 			if (status != CL_SUCCESS)	{ cout << "\nclFinish(m_queue)="		 <<checkerror(status)  <<"\n"<<flush; exit_(status);}

    clCheck(    clEnqueueNDRangeKernel(m_queue,m_Kern[FUNC_TALLYLISTS], NUM_GENES, 0, &global_work_size, 0, 0, NULL, NULL), "PrefixSumCellsCL","clEnqueueNDRangeKernel","FUNC_TALLYLISTS",mbDebug);


    status = clFlush(m_queue); 				if (status != CL_SUCCESS)	{ cout << "\nclFlush(m_queue) status = " << checkerror(status) <<"\n"<<flush; exit_(status);}
	status = clFinish(m_queue); 			if (status != CL_SUCCESS)	{ cout << "\nclFinish(m_queue)="		 <<checkerror(status)  <<"\n"<<flush; exit_(status);}

    clCheck(    clEnqueueReadBuffer(m_queue,gpuVar(&m_Fluid,FDENSE_LIST_LENGTHS),CL_TRUE,0,sizeof(uint[NUM_GENES]),bufI(&m_Fluid, FDENSE_LIST_LENGTHS),0,NULL,NULL),    "PrefixSumCellsCL","clEnqueueReadBuffer","FDENSE_LIST_LENGTHS",mbDebug);

    uint* densebuff_len = bufI(&m_Fluid, FDENSE_BUF_LENGTHS);
    uint* denselist_len = bufI(&m_Fluid, FDENSE_LIST_LENGTHS);

    for (int gene = 0; gene < NUM_GENES; gene++) {
        if (denselist_len[gene] > densebuff_len[gene]) {
            if (verbosity > 1)
                printf("\n\nPrefixSumCellsCL: enlarging densebuff_len[%u], gpuptr(&m_Fluid, FDENSE_LIST_LENGTHS)[gene]=%p .\t", gene, gpuptr(&m_Fluid, FDENSE_LIST_LENGTHS)[gene]);
            while (denselist_len[gene] > densebuff_len[gene])
                densebuff_len[gene] *= 4;
            //AllocateBufferDenseLists(gene, sizeof(uint), bufI(&m_Fluid, FDENSE_LIST_LENGTHS)[gene], FDENSE_LISTS);
        }
    }

    clCheck(clEnqueueWriteBuffer(m_queue, gpuVar(&m_Fluid, FDENSE_LISTS), CL_TRUE, 0, NUM_GENES * sizeof(cl_mem), bufC(&m_Fluid, FDENSE_LISTS), 0, NULL, NULL), "PrefixSumCellsCL", "clEnqueueWriteBuffer", "FDENSE_LISTS", mbDebug);
    clCheck(clEnqueueWriteBuffer(m_queue, gpuVar(&m_Fluid, FDENSE_BUF_LENGTHS), CL_TRUE, 0, NUM_GENES * sizeof(cl_mem), bufC(&m_Fluid, FDENSE_BUF_LENGTHS), 0, NULL, NULL), "PrefixSumCellsCL", "clEnqueueWriteBuffer", "FDENSE_BUF_LENGTHS", mbDebug);

    ////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
    //if (verbosity > 1) {                                                                                                      //
    //    std::cout << "\nChk: PrefixSumCellsCL 4" << std::flush;                                                                     //
    //    for (int gene = 0; gene < NUM_GENES; gene++) {                                                                              //
    //        std::cout << "\ngene list_length[" << gene << "]=" << bufI(&m_Fluid, FDENSE_LIST_LENGTHS)[gene] << "\t" << std::flush;  //
    ////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
    //}
    //}
*/
    if (verbosity>1) printf("\n-----PrefixSumCellsCL() finished-----\n");

}

void FluidSystem::PrefixSumChangesCL ( int zero_offsets ){
    // Prefix Sum - determine grid offsets
    cl_int status;
    cl_event writeEvt;
    // Prefix Sum - determine grid offsets
    int t_blockSize = SCAN_BLOCKSIZE << 1;         // NB 1024 = 512 << 1.  NB SCAN_BLOCKSIZE is the number of


    size_t t_fixedNumElem = static_cast<size_t>(1);
    const size_t* fixedNumElem = &t_fixedNumElem;
    size_t sizeValue0 = static_cast<size_t>(t_blockSize);
    const size_t* blockSize = &sizeValue0;

    int t_numElem1 = m_GridTotal;
    size_t sizeValue1 = static_cast<size_t>(t_numElem1);
    const size_t* numElem1 = &sizeValue1;

    int t_numElem2 = int ((*numElem1) / (*blockSize)) + 1;
    size_t sizeValue2 = static_cast<size_t>(t_numElem2);
    const size_t* numElem2 = &sizeValue2;

    int t_numElem3 = int ((*numElem2) / (*blockSize)) + 1;
    size_t sizeValue3 = static_cast<size_t>(t_numElem3);
    const size_t* numElem3 = &sizeValue3;

    int t_threads = SCAN_BLOCKSIZE;
    size_t sizeValue4 = static_cast<size_t>(t_threads);
    const size_t* threads = &sizeValue4;



    int zon=1;
    cl_mem array1  ;		// input
    cl_mem scan1   ;		// output
    cl_mem array2  = gpuVar(&m_Fluid, FAUXARRAY1);
    cl_mem scan2   = gpuVar(&m_Fluid, FAUXSCAN1);
    cl_mem array3  = gpuVar(&m_Fluid, FAUXARRAY2);
    cl_mem scan3   = gpuVar(&m_Fluid, FAUXSCAN2);

    cl_mem array0  = gpuVar(&m_Fluid, FBIN_COUNT_CHANGES);
    cl_mem scan0   = gpuVar(&m_Fluid, FBIN_OFFSET_CHANGES);

    if(m_debug>3){
        // debug chk
        cout<<"\nSaving (FBIN_COUNT_CHANGES): (bin,#particles) , numElem1="<<numElem1<<"\t"<<std::flush;
        //clCheck( cuMemcpyDtoH ( bufI(&m_Fluid, FBIN_COUNT_CHANGES), gpuVar(&m_Fluid, FBIN_COUNT_CHANGES),	sizeof(uint[t_numElem1]) ), "PrefixSumChangesCL", "cuMemcpyDtoH", "FBIN_COUNT_CHANGES", mbDebug); // NUM_CHANGES*
        clCheck( clEnqueueReadBuffer(m_queue, gpuVar(&m_Fluid,FBIN_COUNT_CHANGES), CL_TRUE, 0, sizeof(uint[t_numElem1]), bufI(&m_Fluid, FBIN_COUNT_CHANGES), 0, NULL, NULL),    "PrefixSumChangesCL","clEnqueueReadBuffer","FBIN_COUNT_CHANGES",mbDebug);

        //### print to a csv file   AND do the same afterwards for FBIN_OFFSET_CHANGES ###
        SaveUintArray( bufI(&m_Fluid, FBIN_COUNT_CHANGES), t_numElem1, "bufI(&m_Fluid, FBIN_COUNT_CHANGES).csv" );
        //
        cout<<"\nSaving (FGCELL): (particleIdx, cell) , mMaxPoints="<<mMaxPoints<<"\t"<<std::flush;
        //clCheck( cuMemcpyDtoH ( bufI(&m_Fluid, FGCELL), gpuVar(&m_Fluid, FGCELL),	sizeof(uint[mMaxPoints]) ), "PrefixSumChangesCL", "cuMemcpyDtoH", "FGCELL", mbDebug);
        clCheck( clEnqueueReadBuffer(m_queue, gpuVar(&m_Fluid,FGCELL), CL_TRUE, 0, sizeof(uint[mMaxPoints]), bufI(&m_Fluid, FGCELL), 0, NULL, NULL),    "PrefixSumChangesCL","clEnqueueReadBuffer","FBIN_COUNT_CHANGES",mbDebug);

        SaveUintArray( bufI(&m_Fluid, FGCELL), mMaxPoints, "bufI(&m_Fluid, FGCELL).csv" );
        //
        //   clCheck( cuMemcpyDtoH ( bufI(&m_Fluid, FDENSE_LISTS), gpuVar(&m_Fluid, FDENSE_LISTS),	sizeof(uint[mMaxPoints]) ), "PrefixSumChangesCL", "cuMemcpyDtoH", "FDENSE_LISTS", mbDebug);
        //   SaveUintArray( bufI(&m_Fluid, FDENSE_LISTS), numElem1, "bufI(&m_Fluid, FDENSE_LISTS).csv" );
    }

    clFlush(m_queue);
    clFinish(m_queue);


    for(int change_list=0; change_list<NUM_CHANGES; change_list++){
        //array1  = array0 + change_list*t_numElem1*sizeof(int); //gpuVar(&m_Fluid, FBIN_COUNT_ACTIVE_GENES);//[change_list*numElem1]   ;      // cl_device_idptr to change_list within gpuVar(&m_Fluid, FBIN_COUNT_CHANGES), for start of prefix-sum.
        //scan1   = scan0 + change_list*t_numElem1*sizeof(int);
        size_t offset_array0 = change_list*t_numElem1*sizeof(int); //gpuVar(&m_Fluid, FBIN_COUNT_ACTIVE_GENES);//[change_list*numElem1]   ;      // cl_device_idptr to change_list within gpuVar(&m_Fluid, FBIN_COUNT_CHANGES), for start of prefix-sum.
        size_t offset_scan0 = change_list*t_numElem1*sizeof(int);

        clCheck ( clEnqueueFillBuffer(m_queue, scan1,  0, sizeof(cl_mem), 0, t_numElem1*sizeof(cl_mem), 0, NULL, NULL),    "PrefixSumChangesCL", "clEnqueueFillBuffer", "for FUNC_FPREFIXSUMCHANGES", mbDebug );
        clCheck ( clEnqueueFillBuffer(m_queue, array2, 0, sizeof(cl_mem), 0, t_numElem2*sizeof(cl_mem), 0, NULL, NULL),   "PrefixSumChangesCL", "clEnqueueFillBuffer", "for FUNC_FPREFIXSUMCHANGES", mbDebug );
        clCheck ( clEnqueueFillBuffer(m_queue, scan2,  0, sizeof(cl_mem), 0, t_numElem2*sizeof(cl_mem), 0, NULL, NULL),    "PrefixSumChangesCL", "clEnqueueFillBuffer", "for FUNC_FPREFIXSUMCHANGES", mbDebug );
        clCheck ( clEnqueueFillBuffer(m_queue, array3, 0, sizeof(cl_mem), 0, t_numElem3*sizeof(cl_mem), 0, NULL, NULL),   "PrefixSumChangesCL", "clEnqueueFillBuffer", "for FUNC_FPREFIXSUMCHANGES", mbDebug );
        clCheck ( clEnqueueFillBuffer(m_queue, scan3,  0, sizeof(cl_mem), 0, t_numElem3*sizeof(cl_mem), 0, NULL, NULL),     "PrefixSumChangesCL", "clEnqueueFillBuffer", "for FUNC_FPREFIXSUMCHANGES", mbDebug );

        const size_t* global_work_size = numElem2;
        const size_t* global_work_size3 = numElem3;
        const size_t* local_work_size = threads;

        // void* argsA[5] = {&array1, &scan1, &array2, &numElem1, &zero_offsets };     // sum array1. output -> scan1, array2.         i.e. FBIN_COUNT -> FBIN_OFFSET, FAUXARRAY1

        clSetKernelArg(m_Kern[FUNC_FPREFIXSUMCHANGES],  0, sizeof(cl_mem),  &array0);
        clSetKernelArg(m_Kern[FUNC_FPREFIXSUMCHANGES],  1, sizeof(cl_mem),  &scan0);
        clSetKernelArg(m_Kern[FUNC_FPREFIXSUMCHANGES],  2, sizeof(cl_mem),  &array2);
        clSetKernelArg(m_Kern[FUNC_FPREFIXSUMCHANGES],  3, sizeof(int),     &numElem1);
        clSetKernelArg(m_Kern[FUNC_FPREFIXSUMCHANGES],  4, sizeof(cl_mem),  &zero_offsets);
        clSetKernelArg(m_Kern[FUNC_FPREFIXSUMCHANGES],  5, sizeof(size_t),  &offset_array0);
        clSetKernelArg(m_Kern[FUNC_FPREFIXSUMCHANGES],  6, sizeof(size_t),  &offset_scan0);

        // clCheck ( cuLaunchKernel ( m_Kern[FUNC_FPREFIXSUM], numElem2, 1, 1, threads, 1, 1, 0, NULL, argsA, NULL ), "PrefixSumChangesCL", "cuLaunch", "FUNC_PREFIXSUM", mbDebug);
        clCheck ( clEnqueueNDRangeKernel(m_queue, m_Kern[FUNC_FPREFIXSUMCHANGES], 1, NULL, global_work_size , local_work_size , 0 , NULL , NULL),    "PrefixSumChangesCL", "clEnqueueNDRangeKernel", "FUNC_FPREFIXSUMCHANGES", mbDebug );
        clFlush(m_queue);
        clFinish(m_queue);

        //void* argsB[5] = { &array2, &scan2, &array3, &numElem2, &zon };             // sum array2. output -> scan2, array3.         i.e. FAUXARRAY1 -> FAUXSCAN1, FAUXARRAY2
        clSetKernelArg(m_Kern[FUNC_FPREFIXSUM],  0, sizeof(cl_mem), &array2);
        clSetKernelArg(m_Kern[FUNC_FPREFIXSUM],  1, sizeof(cl_mem), &scan2);
        clSetKernelArg(m_Kern[FUNC_FPREFIXSUM],  2, sizeof(cl_mem), &array3);
        clSetKernelArg(m_Kern[FUNC_FPREFIXSUM],  3, sizeof(int), &numElem2);
        clSetKernelArg(m_Kern[FUNC_FPREFIXSUM],  4, sizeof(cl_mem), &zon);

        //clCheck ( cuLaunchKernel ( m_Kern[FUNC_FPREFIXSUM], numElem3, 1, 1, threads, 1, 1, 0, NULL, argsB, NULL ), "PrefixSumChangesCL", "cuLaunch", "FUNC_PREFIXSUM", mbDebug);
        clEnqueueNDRangeKernel(m_queue, m_Kern[FUNC_FPREFIXSUM], 1, NULL, global_work_size3 , local_work_size , 0 , NULL , NULL);
        clFlush(m_queue);
        clFinish(m_queue);

        if ( t_numElem3 > 1 ) {
            cl_mem nptr = {0};
            //void* argsC[5] = { &array3, &scan3, &nptr, &numElem3, &zon };	        // sum array3. output -> scan3                  i.e. FAUXARRAY2 -> FAUXSCAN2, &nptr
            clSetKernelArg(m_Kern[FUNC_FPREFIXSUM], 0, sizeof(cl_mem), &array3);
            clSetKernelArg(m_Kern[FUNC_FPREFIXSUM], 1, sizeof(cl_mem), &scan3);
            clSetKernelArg(m_Kern[FUNC_FPREFIXSUM], 2, sizeof(cl_mem), &nptr);
            clSetKernelArg(m_Kern[FUNC_FPREFIXSUM], 3, sizeof(cl_mem), &numElem3);
            clSetKernelArg(m_Kern[FUNC_FPREFIXSUM], 4, sizeof(cl_mem), &zon);

            size_t global_one = 1;
            //clCheck ( cuLaunchKernel ( m_Kern[FUNC_FPREFIXSUM], 1, 1, 1, threads, 1, 1, 0, NULL, argsC, NULL ), "PrefixSumChangesCL", "cuLaunch", "FUNC_PREFIXFIXUP", mbDebug);
            clEnqueueNDRangeKernel(m_queue, m_Kern[FUNC_FPREFIXSUM], 1, NULL, &global_one, local_work_size , 0 , NULL , NULL);
            clFlush(m_queue);
            clFinish(m_queue);
            //void* argsD[3] = { &scan2, &scan3, &numElem2 };	                        // merge scan3 into scan2. output -> scan2      i.e. FAUXSCAN2, FAUXSCAN1 -> FAUXSCAN1
            clSetKernelArg(m_Kern[FUNC_FPREFIXSUM], 0, sizeof(cl_mem), &scan2);
            clSetKernelArg(m_Kern[FUNC_FPREFIXSUM], 1, sizeof(cl_mem), &scan3);
            clSetKernelArg(m_Kern[FUNC_FPREFIXSUM], 2, sizeof(cl_mem), &numElem2);
            //clCheck ( cuLaunchKernel ( m_Kern[FUNC_FPREFIXUP], numElem3, 1, 1, threads, 1, 1, 0, NULL, argsD, NULL ), "PrefixSumChangesCL", "cuLaunch", "FUNC_PREFIXFIXUP", mbDebug);
            clEnqueueNDRangeKernel(m_queue, m_Kern[FUNC_FPREFIXUP], 1, NULL, global_work_size3 , local_work_size , 0 , NULL , NULL);
            clFlush(m_queue);
            clFinish(m_queue);

        }
        //void* argsE[3] = { &scan1, &scan2, &numElem1 };		                        // merge scan2 into scan1. output -> scan1      i.e. FAUXSCAN1, FBIN_OFFSET -> FBIN_OFFSET
        //ORIGINAL:
        //clSetKernelArg(m_Kern[FUNC_FPREFIXSUM], 0, sizeof(cl_mem), &scan1);
        //clSetKernelArg(m_Kern[FUNC_FPREFIXSUM], 1, sizeof(cl_mem), &scan2);
        //clSetKernelArg(m_Kern[FUNC_FPREFIXSUM], 2, sizeof(int), &numElem1);

        clSetKernelArg(m_Kern[FUNC_FPREFIXSUM], 0, sizeof(cl_mem), &scan0);
        clSetKernelArg(m_Kern[FUNC_FPREFIXSUM], 1, sizeof(cl_mem), &scan2);
        clSetKernelArg(m_Kern[FUNC_FPREFIXSUM], 2, sizeof(int), &numElem1);
        clSetKernelArg(m_Kern[FUNC_FPREFIXSUM], 3, sizeof(int), &offset_scan0);

        //clCheck ( cuLaunchKernel ( m_Kern[FUNC_FPREFIXUP], numElem2, 1, 1, threads, 1, 1, 0, NULL, argsE, NULL ), "PrefixSumChangesCL", "cuLaunch", "FUNC_PREFIXFIXUP", mbDebug);
        clEnqueueNDRangeKernel(m_queue, m_Kern[FUNC_FPREFIXUP], 1, NULL, numElem2 , threads , 0 , NULL , NULL);
        clFlush(m_queue);
        clFinish(m_queue);

    }

    int num_lists = NUM_CHANGES, length = FDENSE_LIST_LENGTHS_CHANGES, fgridcnt = FBIN_COUNT_CHANGES, fgridoff = FBIN_OFFSET_CHANGES;
    void* argsF[4] = {&num_lists, &length,&fgridcnt,&fgridoff};
    clSetKernelArg(m_Kern[FUNC_TALLYLISTS], 0, sizeof(int), &num_lists);
    clSetKernelArg(m_Kern[FUNC_TALLYLISTS], 1, sizeof(int), &length);
    clSetKernelArg(m_Kern[FUNC_TALLYLISTS], 2, sizeof(int), &fgridcnt);
    clSetKernelArg(m_Kern[FUNC_TALLYLISTS], 3, sizeof(int), &fgridoff);

    size_t global_work_size = NUM_CHANGES;
    size_t local_work_size = NUM_CHANGES;

    //clCheck ( cuLaunchKernel ( m_Kern[FUNC_TALLYLISTS], NUM_CHANGES, 1, 1, NUM_CHANGES, 1, 1, 0, NULL, argsF, NULL ), "PrefixSumChangesCL", "cuLaunch", "FUNC_TALLYLISTS", mbDebug); //256 threads launched
    clEnqueueNDRangeKernel(m_queue, m_Kern[FUNC_TALLYLISTS], 1, NULL, &global_work_size , &local_work_size , 0 , NULL , NULL);
    clFlush(m_queue);
    clFinish(m_queue);

    //clCheck( cuMemcpyDtoH ( bufI(&m_Fluid, FUNC_TALLYLISTS), gpuVar(&m_Fluid, FUNC_TALLYLISTS),	sizeof(uint[NUM_CHANGES]) ), "PrefixSumChangesCL", "cuMemcpyDtoH", "FDENSE_LIST_LENGTHS_CHANGES", mbDebug);

    clCheck( clEnqueueReadBuffer(m_queue, gpuVar(&m_Fluid,FUNC_TALLYLISTS), CL_TRUE, 0, sizeof(uint[NUM_CHANGES]), bufI(&m_Fluid, FUNC_TALLYLISTS), 0, NULL, NULL),    "PrefixSumChangesCL","clEnqueueReadBuffer","FDENSE_LIST_LENGTHS_CHANGES",mbDebug);


                                                                                                                    // If active particles for change_list > existing buff, then enlarge buff.
    for(int change_list=0;change_list<NUM_CHANGES;change_list++){                                                   // Note this calculation could be done by a kernel,
        uint * densebuff_len = bufI(&m_Fluid, FDENSE_BUF_LENGTHS_CHANGES);                                            // and only bufI(&m_Fluid, FDENSE_LIST_LENGTHS); copied to host.
        uint * denselist_len = bufI(&m_Fluid, FDENSE_LIST_LENGTHS_CHANGES);                                           // For each change_list allocate intial buffer,
        //if (verbosity>1)printf("\nPrefixSumChangesCL: change_list=%u,  densebuff_len[change_list]=%u, denselist_len[change_list]=%u ,",change_list, densebuff_len[change_list], denselist_len[change_list] );
        if (denselist_len[change_list] > densebuff_len[change_list]) {                                              // write pointer and size to FDENSE_LISTS and FDENSE_LIST_LENGTHS
            while(denselist_len[change_list] >  densebuff_len[change_list])   densebuff_len[change_list] *=4;       // bufI(&m_Fluid, FDENSE_BUF_LENGTHS)[i].
                                                                                                                    // NB Need 2*densebuff_len[change_list] for particle & bond
            if (verbosity>1)printf("\nPrefixSumChangesCL: ## enlarging buffer## change_list=%u,  densebuff_len[change_list]=%u, denselist_len[change_list]=%u ,",change_list, densebuff_len[change_list], denselist_len[change_list] );
            AllocateBufferDenseLists( change_list, sizeof(uint), 2*densebuff_len[change_list], FDENSE_LISTS_CHANGES );// NB frees previous buffer &=> clears data
        }                                                                                                           // NB buf[2][list_length] holding : particleIdx, bondIdx
    }
    //clCheck( cuMemcpyHtoD ( gpuVar(&m_Fluid, FDENSE_BUF_LENGTHS_CHANGES), bufC(&m_Fluid, FDENSE_BUF_LENGTHS_CHANGES),	sizeof(uint[NUM_CHANGES]) ), "PrefixSumChangesCL", "cuMemcpyHtoD", "FDENSE_BUF_LENGTHS_CHANGES", mbDebug);

    clCheck(clEnqueueWriteBuffer(m_queue, gpuVar(&m_Fluid, FDENSE_BUF_LENGTHS_CHANGES), CL_TRUE, 0, NUM_CHANGES * sizeof(uint[NUM_CHANGES]), bufC(&m_Fluid, FDENSE_BUF_LENGTHS_CHANGES), 0, NULL, NULL), "PrefixSumChangesCL", "clEnqueueWriteBuffer", "FDENSE_BUF_LENGTHS_CHANGES", mbDebug);

    //cuMemcpyHtoD(gpuVar(&m_Fluid, FDENSE_LISTS_CHANGES), bufC(&m_Fluid, FDENSE_LISTS_CHANGES),  NUM_CHANGES * sizeof(cl_device_idptr)  );                      // update pointers to lists on device

    clCheck( clEnqueueWriteBuffer(m_queue, gpuVar(&m_Fluid, FDENSE_LISTS_CHANGES), CL_TRUE, 0, NUM_CHANGES * sizeof(cl_mem), bufC(&m_Fluid, FDENSE_LISTS_CHANGES), 0, NULL, NULL), "TransferToCL", "clEnqueueWriteBuffer", "FPOS", mbDebug);


    if (verbosity>1) {
        std::cout << "\nChk: PrefixSumChangesCL 4"<<std::flush;
        for(int change_list=0;change_list<NUM_CHANGES;change_list++){
            std::cout<<"\nPrefixSumChangesCL: change list_length["<<change_list<<"]="<<bufI(&m_Fluid, FDENSE_LIST_LENGTHS_CHANGES)[change_list]<<"\t"<<std::flush;
        }
    }
}

void FluidSystem::CountingSortFullCL ( cl_float3* ppos ){
    if (verbosity>-1) std::cout << "\n-----CountingSortFullCL() started... -----\tCountingSortFullCL()1: mMaxPoints="<<mMaxPoints<<", mNumPoints="<<mNumPoints<<",\tmActivePoints="<<mActivePoints<<".\n"<<std::flush;

    // get number of active particles & set short lists for later kernels
    int grid_ScanMax = (m_FParams.gridScanMax.y * m_FParams.gridRes.z + m_FParams.gridScanMax.z) * m_FParams.gridRes.x + m_FParams.gridScanMax.x;

    std::cout << "\nCountingSortFullCL()1: chk ----> 2"<< std::flush;

        clCheck(
            clEnqueueReadBuffer(m_queue, gpuVar(&m_Fluid, FBIN_OFFSET), CL_TRUE, (m_GridTotal-1)*sizeof(int), sizeof(int), &mNumPoints, 0, NULL, NULL), "CountingSortFullCL1", "clEnqueueReadBuffer", "FBIN_OFFSET", mbDebug);

    std::cout << "\nCountingSortFullCL()1: chk ----> 3"<< std::flush;

    cout << "\nCHECK 1.1 Run2Simulation() \n" << flush;

        clCheck(
            clEnqueueReadBuffer(m_queue, gpuVar(&m_Fluid, FBIN_OFFSET), CL_TRUE, grid_ScanMax*sizeof(int), sizeof(int), &mActivePoints, 0, NULL, NULL), "CountingSortFullCL2", "clEnqueueReadBuffer", "FBIN_OFFSET", mbDebug);

    if (verbosity>-1) std::cout<<"\nCountingSortFullCL()2: mMaxPoints="<<mMaxPoints<<" mNumPoints="<<mNumPoints<<",\tmActivePoints="<<mActivePoints<<",  m_GridTotal="<<m_GridTotal<<", grid_ScanMax="<<grid_ScanMax<<"\n"<<std::flush;

    clFlush (m_queue);
    clFinish (m_queue);

    /*
    int totalPoints = 0;
    clCheck( cuMemcpyDtoH ( &totalPoints,  gpuVar(&m_Fluid, FBIN_OFFSET)+(m_GridTotal)*sizeof(int), sizeof(int) ), "CountingSortFullCL3", "cuMemcpyDtoH", "FBIN_OFFSET", mbDebug);
    std::cout<<"\nCountingSortFullCL(): totalPoints="<<totalPoints<<std::flush;
    */

    m_FParams.pnumActive = mActivePoints;                                     // TODO eliminate duplication of information & variables between fluid.h and fluid_system.h
    //clCheck ( cuMemcpyHtoD ( clFParams,	&m_FParams, sizeof(FParams) ), "CountingSortFullCL3", "cuMemcpyHtoD", "clFParams", mbDebug); // seems the safest way to update fparam.pnumActive on device.
    clCheck( clEnqueueWriteBuffer(m_queue, m_FParamsDevice, CL_TRUE, 0, sizeof(FParams), &m_FParams, 0, NULL, NULL), "CountingSortFullCL3", "clEnqueueWriteBuffer", "clFParams", mbDebug);
    //std::cout << "\nCountingSortFullCL()1: chk ----> 4"<< std::flush;



    // Transfer particle data to temp buffers
    //  (gpu-to-gpu copy, no sync needed)
    //TransferToTempCL ( FPOS,		mMaxPoints *sizeof(cl_float3) );    // NB if some points have been removed, then the existing list is no longer dense,
    //TransferToTempCL ( FVEL,		mMaxPoints *sizeof(cl_float3) );    // hence must use mMaxPoints, not mNumPoints
    //TransferToTempCL ( FVEVAL,	mMaxPoints *sizeof(cl_float3) );    // { Could potentially use (old_mNumPoints + mNewPoints) instead of mMaxPoints}
    TransferToTempCL ( FFORCE,	     mMaxPoints *sizeof(cl_float3)   );    // NB buffers are declared and initialized on mMaxPoints.
    TransferToTempCL ( FPRESS,	     mMaxPoints *sizeof(float)       );
    TransferToTempCL ( FDENSITY,	 mMaxPoints *sizeof(float)       );
    TransferToTempCL ( FAGE,		 mMaxPoints *sizeof(uint)        );
    TransferToTempCL ( FCOLOR,		 mMaxPoints *sizeof(uint)        );
    TransferToTempCL ( FGCELL,	     mMaxPoints *sizeof(uint)        );
    TransferToTempCL ( FGNDX,		 mMaxPoints *sizeof(uint)        );
    //std::cout << "\nCountingSortFullCL()2: chk ----> 1"<< std::flush;

    // extra data for morphogenesis
    TransferToTempCL ( FELASTIDX,		mMaxPoints *sizeof(uint[BOND_DATA]) );
    TransferToTempCL ( FPARTICLEIDX,	mMaxPoints *sizeof(uint[BONDS_PER_PARTICLE *2]) );
    TransferToTempCL ( FPARTICLE_ID,	mMaxPoints *sizeof(uint) );
    TransferToTempCL ( FMASS_RADIUS,	mMaxPoints *sizeof(uint) );
    TransferToTempCL ( FNERVEIDX,		mMaxPoints *sizeof(uint) );
    TransferToTempCL ( FCONC,		    mMaxPoints *sizeof(float[NUM_TF]) );
    TransferToTempCL ( FEPIGEN,	        mMaxPoints *sizeof(uint[NUM_GENES]) );
    //std::cout << "\nCountingSortFullCL()2: chk ----> 2"<< std::flush;

    // debug chk
    //clCheck( cuMemcpyDtoH ( bufI(&m_Fluid, FEPIGEN), gpuVar(&m_FluidTemp, FEPIGEN),	mMaxPoints *sizeof(uint[NUM_GENES]) ), "CountingSortFullCL4", "cuMemcpyDtoH", "FBIN_COUNT", mbDebug);
    //SaveUintArray_2D( bufI(&m_Fluid, FEPIGEN), mMaxPoints, NUM_GENES, "CountingSortFullCL__m_FluidTemp.bufI(FEPIGEN)2.csv" );

    // reset bonds and forces in fbuf FELASTIDX, FPARTICLEIDX and FFORCE, required to prevent interference between time steps,
    // because these are not necessarily overwritten by the FUNC_COUNTING_SORT_FULL kernel.
    clFinish (m_queue);    // needed to prevent colision with previous operations

//     clEnqueueReadBuffer(m_queue, gpuVar(&m_Fluid, FPOS), CL_TRUE, 0, mMaxPoints*sizeof(cl_float3), bufV3(&m_Fluid, FPOS), 0, NULL, NULL);
//     clFinish (m_queue);
//
//     printf("\n############################################################################################### FPOS BEFORE MEMSET: FLUID-Buffer, CountingSortFullCL():\n");
//     for (int i = 0; i < mMaxPoints; ++i) {
//         cl_float3 value = bufV3(&m_Fluid, FPOS)[i];
//         printf("Index: %d, Value: (%f, %f, %f)\n", i, value.s[0], value.s[1], value.s[2]);
//     }
//     fflush(stdout);
//
//     clEnqueueReadBuffer(m_queue, gpuVar(&m_FluidTemp, FPOS), CL_TRUE, 0, mMaxPoints*sizeof(cl_float3), bufV3(&m_FluidTemp, FPOS), 0, NULL, NULL);
//     clFinish (m_queue);
//
//     printf("\n############################################################################################### FPOS BEFORE MEMSET: FluidTEMP-Buffer, CountingSortFullCL():\n");
//     fflush(stdout);
//     for (int i = 0; i < mMaxPoints; ++i) {
//         cl_float3 value2 = bufV3(&m_FluidTemp, FPOS)[i];
//         printf("Index: %d, Value: (%f, %f, %f)\n", i, value2.s[0], value2.s[1], value2.s[2]);
//     }
//     fflush(stdout);
//
//     clFlush (m_queue);
//     clFinish (m_queue);

    float max_pos = max(max(m_Vec[PVOLMAX].x, m_Vec[PVOLMAX].y), m_Vec[PVOLMAX].z);

    // Reset FPOS buffer
    size_t zero = 0;
    clCheck ( clEnqueueFillBuffer(m_queue, m_Fluid.mgpu[FPOS], &max_pos, sizeof(uint), zero, mMaxPoints * sizeof(cl_float3), 0, NULL, NULL),  "CountingSortFullCL", "clMemsetD32", "FELASTIDX",   mbDebug);

    clFlush (m_queue);
    clFinish (m_queue);

    clEnqueueReadBuffer(m_queue, gpuVar(&m_Fluid, FPOS), CL_TRUE, 0, mMaxPoints*sizeof(cl_float3), bufV3(&m_Fluid, FPOS), 0, NULL, NULL);


    printf("\n############################################################################################### FPOS AFTER MEMSET:\n");
    for (int i = 0; i < mMaxPoints; ++i) {
        cl_float3 value = bufV3(&m_Fluid, FPOS)[i];
        uint x = (uint)value.s[0];
        uint y = (uint)value.s[1];
        uint z = (uint)value.s[2];
        printf("Index: %d, Value: (%u, %u, %u)\n", i, x, y, z);
    }
    fflush(stdout);

    clFlush (m_queue);
    clFinish (m_queue);

    clEnqueueReadBuffer(m_queue, gpuVar(&m_FluidTemp, FPOS), CL_TRUE, 0, mMaxPoints*sizeof(cl_float3), bufV3(&m_FluidTemp, FPOS), 0, NULL, NULL);

    printf("\n############################################################################################### FPOS_TEMP AFTER MEMSET:\n");
    fflush(stdout);
    for (int i = 0; i < mMaxPoints; ++i) {
        cl_float3 value = bufV3(&m_FluidTemp, FPOS)[i];
//         uint x = (uint)value.s[0];
//         uint y = (uint)value.s[1];
//         uint z = (uint)value.s[2];
//         printf("Index: %d, Value: (%u, %u, %u)\n", i, x, y, z);
        printf("Index: %d, Value: (%f, %f, %f)\n", i, value.s[0], value.s[1], value.s[2]);

    }
    fflush(stdout);

    clFlush (m_queue);
    clFinish (m_queue);
    //cout<<"\nCountingSortFullCL: m_Vec[PVOLMAX]=("<<m_Vec[PVOLMAX].x<<", "<<m_Vec[PVOLMAX].y<<", "<<m_Vec[PVOLMAX].z<<"),  max_pos = "<< max_pos <<std::flush;
    // NB resetting  gpuVar(&m_Fluid, FPOS)  ensures no zombie particles. ?hopefully?

    clCheck ( clMemsetD32 ( gpuVar(&m_Fluid, FELASTIDX),    UINT_MAX,  mMaxPoints * BOND_DATA              ),  "CountingSortFullCL ", "clMemsetD32", "FELASTIDX",    mbDebug);
    clCheck ( clMemsetD32 ( gpuVar(&m_Fluid, FPARTICLEIDX), UINT_MAX,  mMaxPoints * BONDS_PER_PARTICLE *2  ),  "CountingSortFullCL ", "clMemsetD32", "FPARTICLEIDX", mbDebug);
    clCheck ( clMemsetD32 ( gpuVar(&m_Fluid, FPARTICLE_ID), UINT_MAX,  mMaxPoints                          ),  "CountingSortFullCL ", "clMemsetD32", "FPARTICLEIDX", mbDebug);
    clCheck ( clMemsetD32 ( gpuVar(&m_Fluid, FFORCE),      (uint)0.0,  mMaxPoints * 3 /* ie num elements */),  "CountingSortFullCL ", "clMemsetD32", "FFORCE",       mbDebug);
    clCheck ( clMemsetD32 ( gpuVar(&m_Fluid, FCONC),             0.0,  mMaxPoints * NUM_TF                 ),  "CountingSortFullCL ", "clMemsetD32", "FCONC",        mbDebug);
    clCheck ( clMemsetD32 ( gpuVar(&m_Fluid, FEPIGEN),     (uint)0.0,  mMaxPoints * NUM_GENES              ),  "CountingSortFullCL ", "clMemsetD32", "FEPIGEN",      mbDebug);

    clFlush (m_queue);
    clFinish (m_queue);    // needed to prevent colision with previous operations
    //std::cout << "\nCountingSortFullCL()2: chk ----> 5"<< std::flush;

    clEnqueueReadBuffer(m_queue, gpuVar(&m_Fluid, FPOS), CL_TRUE, 0, mMaxPoints*sizeof(cl_float3), bufV3(&m_Fluid, FPOS), 0, NULL, NULL);


//     clCheck ( clEnqueueCopyBuffer(m_queue, gpuVar(&m_FluidTemp, FPOS), cl_gpuVar(&m_FluidTempDevice, FPOS), 0, 0,  mMaxPoints*sizeof(cl_float3), 0, NULL, NULL), "TransferToTempCL", "clEnqueueCopyBuffer", "m_FluidTempDevice", mbDebug);
//
//     clFlush (m_queue);
//     clFinish (m_queue);

// 	clCheck(clEnqueueWriteBuffer(upload_queue, m_FluidDevice, 	    CL_FALSE, 0, sizeof(m_Fluid), 		    &m_Fluid, 		    0, NULL, NULL), "FluidSetupCL", "clEnqueueWriteBuffer", "m_FluidTempDevice", mbDebug);
// 	clCheck(clEnqueueWriteBuffer(upload_queue, m_FluidTempDevice, 	CL_FALSE, 0, sizeof(m_FluidTemp), 		&m_FluidTemp, 		0, NULL, NULL), "FluidSetupCL", "clEnqueueWriteBuffer", "m_FluidTempDevice", mbDebug);
//
//     clFlush (upload_queue);
//     clFinish (upload_queue);

    //Calculate the work group sizes
    computeNumBlocks ( mMaxPoints, local_work_size, m_FParams.numGroups, m_FParams.numItems); // particles

    std::cout << "\nFUNC_COUNTING_SORT_FULL  mMaxPoints: "          << mMaxPoints          << std::flush;
    std::cout << "\nFUNC_COUNTING_SORT_FULL local_work_size: "      << local_work_size     << std::flush;
    std::cout << "\nFUNC_COUNTING_SORT_FULL  m_FParams.numGroups: " << m_FParams.numGroups << std::flush;
    std::cout << "\nFUNC_COUNTING_SORT_FULL  m_FParams.numItems: "  << m_FParams.numItems  << std::flush;

    cl_int status;

    // Reset grid cell IDs
    // clCheck(clMemsetD32(gpu(&m_Fluid, FGCELL), GRID_UNDEF, numPoints ), "clMemsetD32(Sort)");

    //void* args[1] = { &mMaxPoints };

    clCheck(clSetKernelArg(m_Kern[FUNC_COUNTING_SORT_FULL], 0,  sizeof(cl_mem),  &m_FParamsDevice),                    "CountingSortFullCL", "clSetKernelArg 0",  "FUNC_COUNTING_SORT_FULL", mbDebug);
    clCheck(clSetKernelArg(m_Kern[FUNC_COUNTING_SORT_FULL], 1,  sizeof(int),     &mMaxPoints),                         "CountingSortFullCL", "clSetKernelArg 1",  "FUNC_COUNTING_SORT_FULL", mbDebug);
    /////////////////////////////////
    clCheck(clSetKernelArg(m_Kern[FUNC_COUNTING_SORT_FULL], 2,  sizeof(cl_mem),  &m_Fluid.mgpu[FBIN]),                 "CountingSortFullCL", "clSetKernelArg 2",  "FUNC_COUNTING_SORT_FULL", mbDebug);
    clCheck(clSetKernelArg(m_Kern[FUNC_COUNTING_SORT_FULL], 3,  sizeof(cl_mem),  &m_Fluid.mgpu[FBIN_OFFSET]),          "CountingSortFullCL", "clSetKernelArg 3",  "FUNC_COUNTING_SORT_FULL", mbDebug);
    clCheck(clSetKernelArg(m_Kern[FUNC_COUNTING_SORT_FULL], 4,  sizeof(cl_mem),  &m_Fluid.mgpu[FPOS]),                 "CountingSortFullCL", "clSetKernelArg 4",  "FUNC_COUNTING_SORT_FULL", mbDebug);
    clCheck(clSetKernelArg(m_Kern[FUNC_COUNTING_SORT_FULL], 5,  sizeof(cl_mem),  &m_Fluid.mgpu[FVEL]),                 "CountingSortFullCL", "clSetKernelArg 5",  "FUNC_COUNTING_SORT_FULL", mbDebug);
    clCheck(clSetKernelArg(m_Kern[FUNC_COUNTING_SORT_FULL], 6,  sizeof(cl_mem),  &m_Fluid.mgpu[FVEVAL]),               "CountingSortFullCL", "clSetKernelArg 6",  "FUNC_COUNTING_SORT_FULL", mbDebug);
    clCheck(clSetKernelArg(m_Kern[FUNC_COUNTING_SORT_FULL], 7,  sizeof(cl_mem),  &m_Fluid.mgpu[FFORCE]),               "CountingSortFullCL", "clSetKernelArg 7",  "FUNC_COUNTING_SORT_FULL", mbDebug);
    clCheck(clSetKernelArg(m_Kern[FUNC_COUNTING_SORT_FULL], 8,  sizeof(cl_mem),  &m_Fluid.mgpu[FPRESS]),               "CountingSortFullCL", "clSetKernelArg 8",  "FUNC_COUNTING_SORT_FULL", mbDebug);
    clCheck(clSetKernelArg(m_Kern[FUNC_COUNTING_SORT_FULL], 9,  sizeof(cl_mem),  &m_Fluid.mgpu[FDENSITY]),             "CountingSortFullCL", "clSetKernelArg 9",  "FUNC_COUNTING_SORT_FULL", mbDebug);
    clCheck(clSetKernelArg(m_Kern[FUNC_COUNTING_SORT_FULL], 10, sizeof(cl_mem),  &m_Fluid.mgpu[FAGE]),                 "CountingSortFullCL", "clSetKernelArg 10", "FUNC_COUNTING_SORT_FULL", mbDebug);
    clCheck(clSetKernelArg(m_Kern[FUNC_COUNTING_SORT_FULL], 11, sizeof(cl_mem),  &m_Fluid.mgpu[FCOLOR]),               "CountingSortFullCL", "clSetKernelArg 11", "FUNC_COUNTING_SORT_FULL", mbDebug);
    clCheck(clSetKernelArg(m_Kern[FUNC_COUNTING_SORT_FULL], 12, sizeof(cl_mem),  &m_Fluid.mgpu[FGCELL]),               "CountingSortFullCL", "clSetKernelArg 12", "FUNC_COUNTING_SORT_FULL", mbDebug);
    clCheck(clSetKernelArg(m_Kern[FUNC_COUNTING_SORT_FULL], 13, sizeof(cl_mem),  &m_Fluid.mgpu[FGNDX]),                "CountingSortFullCL", "clSetKernelArg 13", "FUNC_COUNTING_SORT_FULL", mbDebug);
    clCheck(clSetKernelArg(m_Kern[FUNC_COUNTING_SORT_FULL], 14, sizeof(cl_mem),  &m_Fluid.mgpu[FELASTIDX]),            "CountingSortFullCL", "clSetKernelArg 14", "FUNC_COUNTING_SORT_FULL", mbDebug);
    clCheck(clSetKernelArg(m_Kern[FUNC_COUNTING_SORT_FULL], 15, sizeof(cl_mem),  &m_Fluid.mgpu[FPARTICLEIDX]),         "CountingSortFullCL", "clSetKernelArg 15", "FUNC_COUNTING_SORT_FULL", mbDebug);
    clCheck(clSetKernelArg(m_Kern[FUNC_COUNTING_SORT_FULL], 16, sizeof(cl_mem),  &m_Fluid.mgpu[FPARTICLE_ID]),         "CountingSortFullCL", "clSetKernelArg 16", "FUNC_COUNTING_SORT_FULL", mbDebug);
    clCheck(clSetKernelArg(m_Kern[FUNC_COUNTING_SORT_FULL], 17, sizeof(cl_mem),  &m_Fluid.mgpu[FMASS_RADIUS]),         "CountingSortFullCL", "clSetKernelArg 17", "FUNC_COUNTING_SORT_FULL", mbDebug);
    clCheck(clSetKernelArg(m_Kern[FUNC_COUNTING_SORT_FULL], 18, sizeof(cl_mem),  &m_Fluid.mgpu[FNERVEIDX]),            "CountingSortFullCL", "clSetKernelArg 18", "FUNC_COUNTING_SORT_FULL", mbDebug);
    clCheck(clSetKernelArg(m_Kern[FUNC_COUNTING_SORT_FULL], 19, sizeof(cl_mem),  &m_Fluid.mgpu[FEPIGEN]),              "CountingSortFullCL", "clSetKernelArg 19", "FUNC_COUNTING_SORT_FULL", mbDebug);
    clCheck(clSetKernelArg(m_Kern[FUNC_COUNTING_SORT_FULL], 20, sizeof(cl_mem),  &m_Fluid.mgpu[FCONC]),                "CountingSortFullCL", "clSetKernelArg 20", "FUNC_COUNTING_SORT_FULL", mbDebug);
    clCheck(clSetKernelArg(m_Kern[FUNC_COUNTING_SORT_FULL], 21, sizeof(cl_mem),  &m_FluidTemp.mgpu[FPOS]),             "CountingSortFullCL", "clSetKernelArg 21", "FUNC_COUNTING_SORT_FULL", mbDebug);
    clCheck(clSetKernelArg(m_Kern[FUNC_COUNTING_SORT_FULL], 22,  sizeof(cl_mem), &m_FluidTemp.mgpu[FVEL]),             "CountingSortFullCL", "clSetKernelArg 22", "FUNC_COUNTING_SORT_FULL", mbDebug);
    clCheck(clSetKernelArg(m_Kern[FUNC_COUNTING_SORT_FULL], 23,  sizeof(cl_mem), &m_FluidTemp.mgpu[FVEVAL]),           "CountingSortFullCL", "clSetKernelArg 23", "FUNC_COUNTING_SORT_FULL", mbDebug);
    clCheck(clSetKernelArg(m_Kern[FUNC_COUNTING_SORT_FULL], 24,  sizeof(cl_mem), &m_FluidTemp.mgpu[FFORCE]),           "CountingSortFullCL", "clSetKernelArg 24", "FUNC_COUNTING_SORT_FULL", mbDebug);
    clCheck(clSetKernelArg(m_Kern[FUNC_COUNTING_SORT_FULL], 25,  sizeof(cl_mem), &m_FluidTemp.mgpu[FPRESS]),           "CountingSortFullCL", "clSetKernelArg 25", "FUNC_COUNTING_SORT_FULL", mbDebug);
    clCheck(clSetKernelArg(m_Kern[FUNC_COUNTING_SORT_FULL], 26,  sizeof(cl_mem), &m_FluidTemp.mgpu[FDENSITY]),         "CountingSortFullCL", "clSetKernelArg 26", "FUNC_COUNTING_SORT_FULL", mbDebug);
    clCheck(clSetKernelArg(m_Kern[FUNC_COUNTING_SORT_FULL], 27, sizeof(cl_mem),  &m_FluidTemp.mgpu[FAGE]),             "CountingSortFullCL", "clSetKernelArg 27", "FUNC_COUNTING_SORT_FULL", mbDebug);
    clCheck(clSetKernelArg(m_Kern[FUNC_COUNTING_SORT_FULL], 28, sizeof(cl_mem),  &m_FluidTemp.mgpu[FCOLOR]),           "CountingSortFullCL", "clSetKernelArg 28", "FUNC_COUNTING_SORT_FULL", mbDebug);
    clCheck(clSetKernelArg(m_Kern[FUNC_COUNTING_SORT_FULL], 29, sizeof(cl_mem),  &m_FluidTemp.mgpu[FGCELL]),           "CountingSortFullCL", "clSetKernelArg 29", "FUNC_COUNTING_SORT_FULL", mbDebug);
    clCheck(clSetKernelArg(m_Kern[FUNC_COUNTING_SORT_FULL], 30, sizeof(cl_mem),  &m_FluidTemp.mgpu[FGNDX]),            "CountingSortFullCL", "clSetKernelArg 30", "FUNC_COUNTING_SORT_FULL", mbDebug);
    clCheck(clSetKernelArg(m_Kern[FUNC_COUNTING_SORT_FULL], 31, sizeof(cl_mem),  &m_FluidTemp.mgpu[FELASTIDX]),        "CountingSortFullCL", "clSetKernelArg 31", "FUNC_COUNTING_SORT_FULL", mbDebug);
    clCheck(clSetKernelArg(m_Kern[FUNC_COUNTING_SORT_FULL], 32, sizeof(cl_mem),  &m_FluidTemp.mgpu[FPARTICLEIDX]),     "CountingSortFullCL", "clSetKernelArg 32", "FUNC_COUNTING_SORT_FULL", mbDebug);
    clCheck(clSetKernelArg(m_Kern[FUNC_COUNTING_SORT_FULL], 33, sizeof(cl_mem),  &m_FluidTemp.mgpu[FPARTICLE_ID]),     "CountingSortFullCL", "clSetKernelArg 33", "FUNC_COUNTING_SORT_FULL", mbDebug);
    clCheck(clSetKernelArg(m_Kern[FUNC_COUNTING_SORT_FULL], 34, sizeof(cl_mem),  &m_FluidTemp.mgpu[FMASS_RADIUS]),     "CountingSortFullCL", "clSetKernelArg 34", "FUNC_COUNTING_SORT_FULL", mbDebug);
    clCheck(clSetKernelArg(m_Kern[FUNC_COUNTING_SORT_FULL], 35, sizeof(cl_mem),  &m_FluidTemp.mgpu[FNERVEIDX]),        "CountingSortFullCL", "clSetKernelArg 35", "FUNC_COUNTING_SORT_FULL", mbDebug);
    clCheck(clSetKernelArg(m_Kern[FUNC_COUNTING_SORT_FULL], 36, sizeof(cl_mem),  &m_FluidTemp.mgpu[FEPIGEN]),          "CountingSortFullCL", "clSetKernelArg 36", "FUNC_COUNTING_SORT_FULL", mbDebug);
    clCheck(clSetKernelArg(m_Kern[FUNC_COUNTING_SORT_FULL], 37, sizeof(cl_mem),  &m_FluidTemp.mgpu[FCONC]),            "CountingSortFullCL", "clSetKernelArg 37", "FUNC_COUNTING_SORT_FULL", mbDebug);
    /*
    clCheck(clSetKernelArg(m_Kern[FUNC_COUNTING_SORT_FULL], 0,  sizeof(cl_mem),  &m_FParamsDevice),                    "CountingSortFullCL", "clSetKernelArg 0",  "FUNC_COUNTING_SORT_FULL", mbDebug);
    clCheck(clSetKernelArg(m_Kern[FUNC_COUNTING_SORT_FULL], 1,  sizeof(int),     &mMaxPoints),                         "CountingSortFullCL", "clSetKernelArg 1",  "FUNC_COUNTING_SORT_FULL", mbDebug);
    clCheck(clSetKernelArg(m_Kern[FUNC_COUNTING_SORT_FULL], 2,  sizeof(cl_mem),  &m_FluidTemp.mgpu[FPOS]),             "CountingSortFullCL", "clSetKernelArg 21", "FUNC_COUNTING_SORT_FULL", mbDebug);*/
    ////////////////////////////////////
    clFlush (m_queue);
    clFinish (m_queue);

    clCheck ( clEnqueueNDRangeKernel(

        m_queue,
        m_Kern[FUNC_COUNTING_SORT_FULL],
        1,
        NULL,
        &m_FParams.numItems,
        &m_FParams.numGroups,
        0,
        NULL,
        NULL),

        "CountingSortFullCL", "clEnqueueNDRangeKernel 1", "FUNC_COUNTING_SORT_FULL", mbDebug );


    // Having sorted the particle data, we can start using a shortened list of particles.
    // NB have to reset to long list at start of time step.
    //std::cout << "\nCountingSortFullCL()2: chk ----> 6"<< std::flush;

    clFlush (m_queue);
    clFinish (m_queue);

//     clCheck ( clEnqueueCopyBuffer(m_queue, cl_gpuVar(&m_FluidDevice, FPOS), gpuVar(&m_FluidTemp, FPOS), 0, 0,  mMaxPoints*sizeof(cl_float3), 0, NULL, NULL), "TransferToTempCL", "clEnqueueCopyBuffer", "m_FluidTempDevice", mbDebug);
//
    clFlush (m_queue);
    clFinish (m_queue);

    clEnqueueReadBuffer(m_queue, gpuVar(&m_Fluid, FPOS), CL_TRUE, 0, mMaxPoints*sizeof(cl_float3), bufV3(&m_Fluid, FPOS), 0, NULL, NULL);


    printf("\n############################################################################################### m_Fluid after CountingSortFullCL():\n");
    for (int i = 0; i < mMaxPoints; ++i) {
        cl_float3 value = bufV3(&m_Fluid, FPOS)[i];
        printf("Index: %d, Value: (%f, %f, %f)\n", i, value.s[0], value.s[1], value.s[2]);
    }
    fflush(stdout);

    clFlush (m_queue);
    clFinish (m_queue);

    clEnqueueReadBuffer(m_queue, gpuVar(&m_FluidTemp, FPOS), CL_TRUE, 0, mMaxPoints*sizeof(cl_float3), bufV3(&m_FluidTemp, FPOS), 0, NULL, NULL);

    printf("\n############################################################################################### m_FluidTemp after CountingSortFullCL():\n");
    fflush(stdout);
    for (int i = 0; i < mMaxPoints; ++i) {
        cl_float3 value = bufV3(&m_FluidTemp, FPOS)[i];
        printf("Index: %d, Value: (%f, %f, %f)\n", i, value.s[0], value.s[1], value.s[2]);
    }
    fflush(stdout);

    clFlush (m_queue);
    clFinish (m_queue);

    computeNumBlocks ( m_FParams.pnumActive, local_work_size, m_FParams.numGroups, m_FParams.numItems); // particles

    std::cout << "\nFUNC_COUNT_SORT_DENSE_LISTS  m_FParams.pnumActive: " <<  m_FParams.pnumActive   << std::flush;
    std::cout << "\nFUNC_COUNT_SORT_DENSE_LISTS  local_work_size: "      << local_work_size         << std::flush;
    std::cout << "\nFUNC_COUNT_SORT_DENSE_LISTS  m_FParams.numGroups: "  << m_FParams.numGroups     << std::flush;
    std::cout << "\nFUNC_COUNT_SORT_DENSE_LISTS  m_FParams.numItems: "   << m_FParams.numItems      << std::flush;


    if (verbosity>1) std::cout<<"\n CountingSortDenseListsCL : FUNC_COUNT_SORT_DENSE_LISTS\n"<<std::flush;
    // countingSortDenseLists ( int pnum ) // NB launch on bins not particles.
    // Calculate the smallest multiple of 128 greater than 10000
    size_t t_blockSize  = SCAN_BLOCKSIZE << 1;
    size_t t_numElem1   = m_GridTotal;
    size_t t_numElem2   = int (t_numElem1 / t_blockSize) + 1;
    clFinish (m_queue);    // needed to prevent colision with previous operations

    clCheck(clSetKernelArg(m_Kern[FUNC_COUNT_SORT_DENSE_LISTS], 0, sizeof(cl_mem),  &m_FParamsDevice),                          "CountingSortDenseListsCL", "clSetKernelArg 0", "FUNC_INSERT", mbDebug);
    clCheck(clSetKernelArg(m_Kern[FUNC_COUNT_SORT_DENSE_LISTS], 1, sizeof(cl_mem),  &m_Fluid.mgpu[FBIN_OFFSET]),                "CountingSortDenseListsCL", "clSetKernelArg 1", "FUNC_INSERT", mbDebug);
    clCheck(clSetKernelArg(m_Kern[FUNC_COUNT_SORT_DENSE_LISTS], 2, sizeof(cl_mem),  &m_Fluid.mgpu[FBIN_COUNT]),                 "CountingSortDenseListsCL", "clSetKernelArg 2", "FUNC_INSERT", mbDebug);
    clCheck(clSetKernelArg(m_Kern[FUNC_COUNT_SORT_DENSE_LISTS], 3, sizeof(cl_mem),  &m_Fluid.mgpu[FDENSE_LISTS]),               "CountingSortDenseListsCL", "clSetKernelArg 3", "FUNC_INSERT", mbDebug);
    clCheck(clSetKernelArg(m_Kern[FUNC_COUNT_SORT_DENSE_LISTS], 4, sizeof(cl_mem),  &m_Fluid.mgpu[FBIN_COUNT_ACTIVE_GENES]),    "CountingSortDenseListsCL", "clSetKernelArg 4", "FUNC_INSERT", mbDebug);
    clCheck(clSetKernelArg(m_Kern[FUNC_COUNT_SORT_DENSE_LISTS], 5, sizeof(cl_mem),  &m_Fluid.mgpu[FBIN_OFFSET_ACTIVE_GENES]),   "CountingSortDenseListsCL", "clSetKernelArg 4", "FUNC_INSERT", mbDebug);
    clCheck(clSetKernelArg(m_Kern[FUNC_COUNT_SORT_DENSE_LISTS], 6, sizeof(cl_mem),  &m_Fluid.mgpu[FEPIGEN]),                    "CountingSortDenseListsCL", "clSetKernelArg 5", "FUNC_INSERT", mbDebug);
    clCheck(clSetKernelArg(m_Kern[FUNC_COUNT_SORT_DENSE_LISTS], 7, sizeof(cl_mem),  &m_Fluid.mgpu[FPARTICLE_ID]),               "CountingSortDenseListsCL", "clSetKernelArg 6", "FUNC_INSERT", mbDebug);
    clCheck(clSetKernelArg(m_Kern[FUNC_COUNT_SORT_DENSE_LISTS], 8, sizeof(int),     &mMaxPoints),                               "CountingSortDenseListsCL", "clSetKernelArg 7", "FUNC_INSERT", mbDebug);


    // Enqueue kernel
    clCheck(clEnqueueNDRangeKernel(

        m_queue,
        m_Kern[FUNC_COUNT_SORT_DENSE_LISTS],
        1,
        NULL,
        &m_FParams.numItems,
        &m_FParams.numGroups,
        0,
        NULL,
        NULL),
    "CountingSortDenseListsCL", "clEnqueueNDRangeKernel", "FUNC_COUNT_SORT_LISTS", mbDebug);

    clFinish (m_queue);

    if(verbosity>3){// debug chk
        std::cout<<"\n### Saving UintArray .csv files."<<std::flush;

        //clCheck( cuMemcpyDtoH ( bufI(&m_Fluid, FEPIGEN), gpuVar(&m_FluidTemp, FEPIGEN),	mMaxPoints *sizeof(uint[NUM_GENES]) ), "CountingSortFullCL8", "cuMemcpyDtoH", "FBIN_COUNT", mbDebug);
        clCheck( clEnqueueReadBuffer(m_queue, gpuVar(&m_FluidTemp, FEPIGEN), CL_TRUE, mMaxPoints *sizeof(uint[NUM_GENES]), sizeof(int), bufI(&m_Fluid, FEPIGEN), 0, NULL, NULL), "CountingSortFullCL8", "clEnqueueReadBuffer", "FBIN_COUNT", mbDebug);
        SaveUintArray_2D( bufI(&m_Fluid, FEPIGEN), mMaxPoints, NUM_GENES, "CountingSortFullCL__m_FluidTemp.bufI(FEPIGEN)3.csv" );

        //clCheck( cuMemcpyDtoH ( bufI(&m_Fluid, FEPIGEN), gpuVar(&m_Fluid, FEPIGEN),	/*mMaxPoints*/mNumPoints *sizeof(uint[NUM_GENES]) ), "PrefixSumChangesCL", "cuMemcpyDtoH", "FBIN_COUNT", mbDebug);
        clCheck( clEnqueueReadBuffer(m_queue, gpuVar(&m_Fluid, FEPIGEN), CL_TRUE, mNumPoints *sizeof(uint[NUM_GENES]), sizeof(int), bufI(&m_Fluid, FEPIGEN), 0, NULL, NULL), "PrefixSumChangesCL", "clEnqueueReadBuffer", "FBIN_COUNT", mbDebug);
        SaveUintArray_2D( bufI(&m_Fluid, FEPIGEN), mMaxPoints, NUM_GENES, "CountingSortFullCL__bufI(&m_Fluid, FEPIGEN)3.csv" );

        //clCheck( cuMemcpyDtoH ( bufI(&m_Fluid, FBIN_COUNT), gpuVar(&m_Fluid, FBIN_COUNT),	sizeof(uint[m_GridTotal]) ), "CountingSortFullCL9", "cuMemcpyDtoH", "FBIN_COUNT", mbDebug);
        clCheck( clEnqueueReadBuffer(m_queue, gpuVar(&m_Fluid, FBIN_COUNT), CL_TRUE, sizeof(uint[m_GridTotal]), sizeof(int), bufI(&m_Fluid, FBIN_COUNT), 0, NULL, NULL), "PrefixSumChangesCL", "clEnqueueReadBuffer", "FBIN_COUNT", mbDebug);
        SaveUintArray( bufI(&m_Fluid, FBIN_COUNT), m_GridTotal, "CountingSortFullCL__bufI(&m_Fluid, FBIN_COUNT).csv" );

        //clCheck( cuMemcpyDtoH ( bufI(&m_Fluid, FBIN_OFFSET), gpuVar(&m_Fluid, FBIN_OFFSET),	sizeof(uint[m_GridTotal]) ), "CountingSortFullCL10", "cuMemcpyDtoH", "FBIN_OFFSET", mbDebug);
        clCheck( clEnqueueReadBuffer(m_queue, gpuVar(&m_Fluid, FBIN_OFFSET), CL_TRUE, sizeof(uint[m_GridTotal] ), sizeof(int), bufI(&m_Fluid, FBIN_OFFSET), 0, NULL, NULL), "PrefixSumChangesCL", "clEnqueueReadBuffer", "FBIN_COUNT", mbDebug);
        SaveUintArray( bufI(&m_Fluid, FBIN_OFFSET), m_GridTotal, "CountingSortFullCL__bufI(&m_Fluid, FBIN_OFFSET).csv" );

       // uint fDenseList2[100000];
       // cl_device_idptr*  _list2pointer = (cl_device_idptr*) &bufC(&m_Fluid, FDENSE_LISTS)[2 * sizeof(cl_device_idptr)];
       // clCheck( cuMemcpyDtoH ( fDenseList2, *_list2pointer,	sizeof(uint[ bufI(&m_Fluid, FDENSE_LIST_LENGTHS)[2] ])/*sizeof(uint[2000])*/ ), "CountingSortFullCL11", "cuMemcpyDtoH", "FBIN_COUNT", mbDebug);
       // SaveUintArray( fDenseList2, bufI(&m_Fluid, FDENSE_LIST_LENGTHS)[2], "CountingSortFullCL__m_Fluid.bufII(FDENSE_LISTS)[2].csv" );

        if (verbosity>1) std::cout << "\n-----CountingSortFullCL() finished. -----\n" <<std::flush;
    }
    clFlush (m_queue);
    clFinish (m_queue);

}

void FluidSystem::CountingSortChangesCL ( ){
    //std::cout<<"\n\n#### CountingSortChangesCL ( ):   verbosity = "<< verbosity <<"\n";
    if (verbosity>1) {std::cout<<"\n\n#### CountingSortChangesCL ( )"<<std::flush;}
    /* ////////
    clCheck( cuMemcpyDtoH ( bufI(&m_Fluid, FDENSE_LIST_LENGTHS_CHANGES), gpuVar(&m_Fluid, FDENSE_LIST_LENGTHS_CHANGES),	sizeof(uint[NUM_CHANGES]) ), "PrefixSumCellsCL", "cuMemcpyDtoH", "FDENSE_LIST_LENGTHS_CHANGES", mbDebug);

                                                                                                                    // If active particles for change_list > existing buff, then enlarge buff.
    for(int change_list=0;change_list<NUM_CHANGES;change_list++){                                                   // Note this calculation could be done by a kernel,
        uint * densebuff_len = bufI(&m_Fluid, FDENSE_BUF_LENGTHS_CHANGES);                                            // and only bufI(&m_Fluid, FDENSE_LIST_LENGTHS); copied to host.
        uint * denselist_len = bufI(&m_Fluid, FDENSE_LIST_LENGTHS_CHANGES);                                           // For each change_list allocate intial buffer,
        if (verbosity>1)printf("\nCountingSortChangesCL1: change_list=%u,  densebuff_len[change_list]=%u, denselist_len[change_list]=%u ,",change_list, densebuff_len[change_list], denselist_len[change_list] );
    }
    *//////////
    cl_int status;
    int blockSize = SCAN_BLOCKSIZE/2 << 1;

    int t_numElem1 = m_GridTotal;
    cout << m_GridTotal;
    size_t sizeValue1 = static_cast<size_t>(t_numElem1);
    const size_t* numElem1 = &sizeValue1;

    int t_numElem2 = 2* int (t_numElem1 / blockSize) + 1;
    size_t sizeValue2 = static_cast<size_t>(t_numElem2);
    const size_t* numElem2 = &sizeValue2;


    int t_threads = SCAN_BLOCKSIZE/2;
    size_t sizeValue4 = static_cast<size_t>(t_threads);
    const size_t* threads = &sizeValue4;
    //void* args[1] = { &mActivePoints };

    status  = clSetKernelArg(m_Kern[FUNC_COUNTING_SORT_CHANGES], 0, sizeof(int), &mActivePoints);

    //clCheck ( cuLaunchKernel ( m_Kern[FUNC_COUNTING_SORT_CHANGES], numElem2, 1, 1, threads , 1, 1, 0, NULL, args, NULL), "CountingSortChangesCL", "cuLaunch", "FUNC_COUNTING_SORT_CHANGES", mbDebug );
    status = clEnqueueNDRangeKernel(m_queue, m_Kern[FUNC_COUNTING_SORT_CHANGES], 1, NULL, &global_work_size, &local_work_size, 0, NULL, NULL);


     /////////
    clCheck(clFinish(m_queue), "CountingSortChangesCL()", "clFinish", "After FUNC_COUNTING_SORT_CHANGES", mbDebug);

    //clCheck( cuMemcpyDtoH ( bufI(&m_Fluid, FDENSE_LIST_LENGTHS_CHANGES), gpuVar(&m_Fluid, FDENSE_LIST_LENGTHS_CHANGES),	sizeof(uint[NUM_CHANGES]) ), "PrefixSumCellsCL", "cuMemcpyDtoH", "FDENSE_LIST_LENGTHS_CHANGES", mbDebug);
    clCheck( clEnqueueReadBuffer(m_queue, gpuVar(&m_Fluid, FDENSE_LIST_LENGTHS_CHANGES), CL_TRUE, sizeof(uint[NUM_CHANGES]), sizeof(int), bufI(&m_Fluid, FDENSE_LIST_LENGTHS_CHANGES), 0, NULL, NULL), "CountingSortChangesCL", "clEnqueueReadBuffer", "FBIN_COUNT", mbDebug);

                                                                                                                    // If active particles for change_list > existing buff, then enlarge buff.
//     for(int change_list=0;change_list<NUM_CHANGES;change_list++){                                                   // Note this calculation could be done by a kernel,
//         uint * densebuff_len = bufI(&m_Fluid, FDENSE_BUF_LENGTHS_CHANGES);                                            // and only bufI(&m_Fluid, FDENSE_LIST_LENGTHS); copied to host.
//         uint * denselist_len = bufI(&m_Fluid, FDENSE_LIST_LENGTHS_CHANGES);                                           // For each change_list allocate intial buffer,
//         if (verbosity>1)printf("\nCountingSortChangesCL2: change_list=%u,  densebuff_len[change_list]=%u, denselist_len[change_list]=%u ,\t\t threads=%u, numElem2=%u,  m_GridTotal=%u \t",
//                change_list, densebuff_len[change_list], denselist_len[change_list], threads, numElem2,  m_GridTotal );
//         clFinish ();
//         if(verbosity>0){
//             uint fDenseList2[1000000] = {UINT_MAXSIZE};//TODO make this array size safe!  NB 10* num particles.
//             cl_device_idptr*  _list2pointer = (cl_device_idptr*) &bufC(&m_Fluid, FDENSE_LISTS_CHANGES)[change_list*sizeof(cl_device_idptr)];
//                                                                                                                 // Get device pointer to FDENSE_LISTS_CHANGES[change_list].
//             clCheck( cuMemcpyDtoH ( fDenseList2, *_list2pointer,	2*sizeof(uint[densebuff_len[change_list]]) ), "CountingSortChangesCL", "cuMemcpyDtoH", "FBIN_COUNT", mbDebug);
//             char filename[256];
//             sprintf(filename, "CountingSortChangesCL__m_Fluid.bufII(FDENSE_LISTS_CHANGES)[%u].csv", change_list);
//             SaveUintArray_2Columns( fDenseList2, denselist_len[change_list], densebuff_len[change_list], filename );
//             ///
//             printf("\n\n*_list2pointer=%llu",*_list2pointer);
//
//         }
//     }
    // Assuming these variables are defined elsewhere:
    // NUM_CHANGES, m_Fluid, FDENSE_BUF_LENGTHS_CHANGES, FDENSE_LIST_LENGTHS_CHANGES,
    // verbosity, threads, numElem2, m_GridTotal, command_queue
                                                                                             // If active particles for change_list > existing buff, then enlarge buff.
    for (int change_list = 0; change_list < NUM_CHANGES; change_list++) {                    // Note this calculation could be done by a kernel,
        // Note: In OpenCL, you should use cl_mem objects instead of bufI and bufC functions
        // to access memory buffers.
        cl_int status;
        uint densebuff_len = bufI(&m_Fluid, FDENSE_BUF_LENGTHS_CHANGES)[change_list];        // and only bufI(&m_Fluid, FDENSE_LIST_LENGTHS); copied to host.
        uint denselist_len = bufI(&m_Fluid, FDENSE_LIST_LENGTHS_CHANGES)[change_list];       // For each change_list allocate intial buffer,

        if (verbosity > 1)
            printf("\nCountingSortChangesCL2: change_list=%d, densebuff_len=%u, denselist_len=%u, threads=%u, numElem2=%u, m_GridTotal=%u\t",
                change_list, densebuff_len, denselist_len, t_threads, t_numElem2, m_GridTotal);

        clFinish(m_queue);

        if (verbosity > 0) {
            uint fDenseList2[1000000] = {UINT_MAX}; // TODO: Adjust array size as needed.
            cl_mem list2buffer = (cl_mem)&bufC(&m_Fluid, FDENSE_LISTS_CHANGES)[change_list * sizeof(cl_mem)];

            // Map the buffer to read data from it.
            cl_event map_event;
            uint *mapped_ptr = (uint *)clEnqueueMapBuffer(m_queue, list2buffer, CL_TRUE, CL_MAP_READ, 0, 2 * sizeof(uint) * densebuff_len, 0, NULL, &map_event, &status);

            // Copy data to the host buffer.
            clCheck(clEnqueueReadBuffer(m_queue, list2buffer, CL_TRUE, 0, 2 * sizeof(uint) * densebuff_len, fDenseList2, 0, NULL, NULL), "CountingSortChangesCL", "clEnqueueReadBuffer", "FBIN_COUNT", mbDebug);

            // Unmap the buffer.
            clCheck(clEnqueueUnmapMemObject(m_queue, list2buffer, mapped_ptr, 0, NULL, NULL), "CountingSortChangesCL", "clEnqueueUnmapMemObject", "FBIN_COUNT", mbDebug);

            clWaitForEvents(1, &map_event);
            clReleaseEvent(map_event);      //good practice but no longer needed

            char filename[256];
            sprintf(filename, "CountingSortChangesCL__m_Fluid.bufII(FDENSE_LISTS_CHANGES)[%d].csv", change_list);
            SaveUintArray_2Columns(fDenseList2, denselist_len, densebuff_len, filename);
            printf("\n\nlist2buffer=%p", list2buffer);
        }
    }

}

void FluidSystem::CreateFluidBuffers() {
    cl_int status;

    m_Fluid.mgpu[FPOS] =                clCreateBuffer(m_context, CL_MEM_READ_WRITE, mMaxPoints *   sizeof(float) * 4,                  NULL, &status);
    m_Fluid.mgpu[FVEL] =                clCreateBuffer(m_context, CL_MEM_READ_WRITE, mMaxPoints *   sizeof(float) * 4,                  NULL, &status);
    m_Fluid.mgpu[FVEVAL] =              clCreateBuffer(m_context, CL_MEM_READ_WRITE, mMaxPoints *   sizeof(float) * 4,                  NULL, &status);
    m_Fluid.mgpu[FFORCE] =              clCreateBuffer(m_context, CL_MEM_READ_WRITE, mMaxPoints *   sizeof(float) * 4,                  NULL, &status);
    m_Fluid.mgpu[FPRESS] =              clCreateBuffer(m_context, CL_MEM_READ_WRITE, mMaxPoints *   sizeof(float),                      NULL, &status);
    m_Fluid.mgpu[FDENSITY] =            clCreateBuffer(m_context, CL_MEM_READ_WRITE, mMaxPoints *   sizeof(float),                      NULL, &status);
    m_Fluid.mgpu[FAGE] =                clCreateBuffer(m_context, CL_MEM_READ_WRITE, mMaxPoints *   sizeof(uint),                       NULL, &status);
    m_Fluid.mgpu[FCOLOR] =              clCreateBuffer(m_context, CL_MEM_READ_WRITE, mMaxPoints *   sizeof(uint)  * 3,                  NULL, &status); //TODO maybe need to adapt all of them to be "*4", as cl_float3 is actually = cl_float4

    m_Fluid.mgpu[FGNDX] =               clCreateBuffer(m_context, CL_MEM_READ_WRITE, mMaxPoints *  sizeof(uint)  * 4,                  NULL, &status);
    m_Fluid.mgpu[FGCELL] =              clCreateBuffer(m_context, CL_MEM_READ_WRITE, mMaxPoints *  sizeof(uint)  * 4,                  NULL, &status);

    m_Fluid.mgpu[FELASTIDX] =           clCreateBuffer(m_context, CL_MEM_READ_WRITE, mMaxPoints*    sizeof(uint[BOND_DATA]),            NULL, &status);
    m_Fluid.mgpu[FPARTICLEIDX] =        clCreateBuffer(m_context, CL_MEM_READ_WRITE, mMaxPoints*    sizeof(uint[BONDS_PER_PARTICLE*2]), NULL, &status);
    m_Fluid.mgpu[FPARTICLE_ID] =        clCreateBuffer(m_context, CL_MEM_READ_WRITE, mMaxPoints *   sizeof(uint)  * 3,                  NULL, &status);
    m_Fluid.mgpu[FMASS_RADIUS] =        clCreateBuffer(m_context, CL_MEM_READ_WRITE, mMaxPoints *   sizeof(uint)  * 3,                  NULL, &status);
    m_Fluid.mgpu[FNERVEIDX] =           clCreateBuffer(m_context, CL_MEM_READ_WRITE, mMaxPoints*    sizeof(uint),                       NULL, &status);
    m_Fluid.mgpu[FCONC] =               clCreateBuffer(m_context, CL_MEM_READ_WRITE, mMaxPoints*    sizeof(float[NUM_TF]),              NULL, &status);
    m_Fluid.mgpu[FEPIGEN] =             clCreateBuffer(m_context, CL_MEM_READ_WRITE, mMaxPoints*    sizeof(uint[NUM_GENES]),            NULL, &status);


    m_FluidTemp.mgpu[FPOS] =            clCreateBuffer(m_context, CL_MEM_READ_WRITE, mMaxPoints *   sizeof(float) * 4,                  NULL, &status);
    m_FluidTemp.mgpu[FVEL] =            clCreateBuffer(m_context, CL_MEM_READ_WRITE, mMaxPoints *   sizeof(float) * 4,                  NULL, &status);
    m_FluidTemp.mgpu[FVEVAL] =          clCreateBuffer(m_context, CL_MEM_READ_WRITE, mMaxPoints *   sizeof(float) * 4,                  NULL, &status);
    m_FluidTemp.mgpu[FFORCE] =          clCreateBuffer(m_context, CL_MEM_READ_WRITE, mMaxPoints *   sizeof(float) * 4,                  NULL, &status);
    m_FluidTemp.mgpu[FPRESS] =          clCreateBuffer(m_context, CL_MEM_READ_WRITE, mMaxPoints *   sizeof(float),                      NULL, &status);
    m_FluidTemp.mgpu[FDENSITY] =        clCreateBuffer(m_context, CL_MEM_READ_WRITE, mMaxPoints *   sizeof(float),                      NULL, &status);
    m_FluidTemp.mgpu[FAGE] =            clCreateBuffer(m_context, CL_MEM_READ_WRITE, mMaxPoints *   sizeof(uint),                       NULL, &status);
    m_FluidTemp.mgpu[FCOLOR] =          clCreateBuffer(m_context, CL_MEM_READ_WRITE, mMaxPoints *   sizeof(uint)  * 3,                  NULL, &status);

    m_FluidTemp.mgpu[FGNDX] =           clCreateBuffer(m_context, CL_MEM_READ_WRITE, mMaxPoints *  sizeof(uint)  * 4,                  NULL, &status);
    m_FluidTemp.mgpu[FGCELL] =          clCreateBuffer(m_context, CL_MEM_READ_WRITE, mMaxPoints *  sizeof(uint)  * 4,                  NULL, &status);

    m_FluidTemp.mgpu[FELASTIDX] =       clCreateBuffer(m_context, CL_MEM_READ_WRITE, mMaxPoints*    sizeof(uint[BOND_DATA]),            NULL, &status);
    m_FluidTemp.mgpu[FPARTICLEIDX] =    clCreateBuffer(m_context, CL_MEM_READ_WRITE, mMaxPoints*    sizeof(uint[BONDS_PER_PARTICLE*2]), NULL, &status);
    m_FluidTemp.mgpu[FPARTICLE_ID] =    clCreateBuffer(m_context, CL_MEM_READ_WRITE, mMaxPoints *   sizeof(uint)  * 3,                  NULL, &status);
    m_FluidTemp.mgpu[FMASS_RADIUS] =    clCreateBuffer(m_context, CL_MEM_READ_WRITE, mMaxPoints *   sizeof(uint)  * 3,                  NULL, &status);
    m_FluidTemp.mgpu[FNERVEIDX] =       clCreateBuffer(m_context, CL_MEM_READ_WRITE, mMaxPoints*    sizeof(uint),                       NULL, &status);
    m_FluidTemp.mgpu[FCONC] =           clCreateBuffer(m_context, CL_MEM_READ_WRITE, mMaxPoints*    sizeof(float[NUM_TF]),              NULL, &status);
    m_FluidTemp.mgpu[FEPIGEN] =         clCreateBuffer(m_context, CL_MEM_READ_WRITE, mMaxPoints*    sizeof(uint[NUM_GENES]),            NULL, &status);


/*
    m_FluidDevice->mgpu[FPOS] =               clCreateBuffer(m_context, CL_MEM_READ_WRITE, mMaxPoints *   sizeof(float) * 4,                  NULL, &status);
    m_FluidDevice.mgpu[FVEL] =                clCreateBuffer(m_context, CL_MEM_READ_WRITE, mMaxPoints *   sizeof(float) * 4,                  NULL, &status);
    m_FluidDevice.mgpu[FVEVAL] =              clCreateBuffer(m_context, CL_MEM_READ_WRITE, mMaxPoints *   sizeof(float) * 4,                  NULL, &status);
    m_FluidDevice.mgpu[FFORCE] =              clCreateBuffer(m_context, CL_MEM_READ_WRITE, mMaxPoints *   sizeof(float) * 4,                  NULL, &status);
    m_FluidDevice.mgpu[FPRESS] =              clCreateBuffer(m_context, CL_MEM_READ_WRITE, mMaxPoints *   sizeof(float),                      NULL, &status);
    m_FluidDevice.mgpu[FDENSITY] =            clCreateBuffer(m_context, CL_MEM_READ_WRITE, mMaxPoints *   sizeof(float),                      NULL, &status);
    m_FluidDevice.mgpu[FCLR] =                clCreateBuffer(m_context, CL_MEM_READ_WRITE, mMaxPoints *   sizeof(uint)  * 3,                  NULL, &status); //TODO maybe need to adapt all of them to be "*4", as cl_float3 is actually = cl_float4

    m_FluidDevice.mgpu[FGNDX] =               clCreateBuffer(m_context, CL_MEM_READ_WRITE, mMaxPoints *  sizeof(uint)  * 4,                  NULL, &status);
    m_FluidDevice.mgpu[FGCELL] =              clCreateBuffer(m_context, CL_MEM_READ_WRITE, mMaxPoints *  sizeof(uint)  * 4,                  NULL, &status);

    m_FluidDevice.mgpu[FELASTIDX] =           clCreateBuffer(m_context, CL_MEM_READ_WRITE, mMaxPoints*    sizeof(uint[BOND_DATA]),            NULL, &status);
    m_FluidDevice.mgpu[FPARTICLEIDX] =        clCreateBuffer(m_context, CL_MEM_READ_WRITE, mMaxPoints*    sizeof(uint[BONDS_PER_PARTICLE*2]), NULL, &status);
    m_FluidDevice.mgpu[FPARTICLE_ID] =        clCreateBuffer(m_context, CL_MEM_READ_WRITE, mMaxPoints *   sizeof(uint)  * 3,                  NULL, &status);
    m_FluidDevice.mgpu[FMASS_RADIUS] =        clCreateBuffer(m_context, CL_MEM_READ_WRITE, mMaxPoints *   sizeof(uint)  * 3,                  NULL, &status);
    m_FluidDevice.mgpu[FNERVEIDX] =           clCreateBuffer(m_context, CL_MEM_READ_WRITE, mMaxPoints*    sizeof(uint),                       NULL, &status);
    m_FluidDevice.mgpu[FCONC] =               clCreateBuffer(m_context, CL_MEM_READ_WRITE, mMaxPoints*    sizeof(float[NUM_TF]),              NULL, &status);
    m_FluidDevice.mgpu[FEPIGEN] =             clCreateBuffer(m_context, CL_MEM_READ_WRITE, mMaxPoints*    sizeof(uint[NUM_GENES]),            NULL, &status);


    m_FluidTempDevice.mgpu[FPOS] =            clCreateBuffer(m_context, CL_MEM_READ_WRITE, mMaxPoints *   sizeof(float) * 4,                  NULL, &status);
    m_FluidTempDevice.mgpu[FVEL] =            clCreateBuffer(m_context, CL_MEM_READ_WRITE, mMaxPoints *   sizeof(float) * 4,                  NULL, &status);
    m_FluidTempDevice.mgpu[FVEVAL] =          clCreateBuffer(m_context, CL_MEM_READ_WRITE, mMaxPoints *   sizeof(float) * 4,                  NULL, &status);
    m_FluidTempDevice.mgpu[FFORCE] =          clCreateBuffer(m_context, CL_MEM_READ_WRITE, mMaxPoints *   sizeof(float) * 4,                  NULL, &status);
    m_FluidTempDevice.mgpu[FPRESS] =          clCreateBuffer(m_context, CL_MEM_READ_WRITE, mMaxPoints *   sizeof(float),                      NULL, &status);
    m_FluidTempDevice.mgpu[FDENSITY] =        clCreateBuffer(m_context, CL_MEM_READ_WRITE, mMaxPoints *   sizeof(float),                      NULL, &status);
    m_FluidTempDevice.mgpu[FCLR] =            clCreateBuffer(m_context, CL_MEM_READ_WRITE, mMaxPoints *   sizeof(uint)  * 3,                  NULL, &status); //TODO maybe need to adapt all of them to be "*4", as cl_float3 is actually = cl_float4

    m_FluidTempDevice.mgpu[FGNDX] =           clCreateBuffer(m_context, CL_MEM_READ_WRITE, mMaxPoints *  sizeof(uint)  * 4,                  NULL, &status);
    m_FluidTempDevice.mgpu[FGCELL] =          clCreateBuffer(m_context, CL_MEM_READ_WRITE, mMaxPoints *  sizeof(uint)  * 4,                  NULL, &status);

    m_FluidTempDevice.mgpu[FELASTIDX] =       clCreateBuffer(m_context, CL_MEM_READ_WRITE, mMaxPoints*    sizeof(uint[BOND_DATA]),            NULL, &status);
    m_FluidTempDevice.mgpu[FPARTICLEIDX] =    clCreateBuffer(m_context, CL_MEM_READ_WRITE, mMaxPoints*    sizeof(uint[BONDS_PER_PARTICLE*2]), NULL, &status);
    m_FluidTempDevice.mgpu[FPARTICLE_ID] =    clCreateBuffer(m_context, CL_MEM_READ_WRITE, mMaxPoints *   sizeof(uint)  * 3,                  NULL, &status);
    m_FluidTempDevice.mgpu[FMASS_RADIUS] =    clCreateBuffer(m_context, CL_MEM_READ_WRITE, mMaxPoints *   sizeof(uint)  * 3,                  NULL, &status);
    m_FluidTempDevice.mgpu[FNERVEIDX] =       clCreateBuffer(m_context, CL_MEM_READ_WRITE, mMaxPoints*    sizeof(uint),                       NULL, &status);
    m_FluidTempDevice.mgpu[FCONC] =           clCreateBuffer(m_context, CL_MEM_READ_WRITE, mMaxPoints*    sizeof(float[NUM_TF]),              NULL, &status);
    m_FluidTempDevice.mgpu[FEPIGEN] =         clCreateBuffer(m_context, CL_MEM_READ_WRITE, mMaxPoints*    sizeof(uint[NUM_GENES]),            NULL, &status);*/

}

void FluidSystem::TransferToCL (){
    if (verbosity>0) std::cout<<"\n-----TransferToCL() started... -----\n"<<std::flush;

    clCheck( clEnqueueWriteBuffer(m_queue, gpuVar(&m_Fluid, FPOS), CL_TRUE, 0,      mMaxPoints * sizeof(float) * 3,                 bufC(&m_Fluid, FPOS), 0, NULL, NULL),       "TransferToCL", "clEnqueueWriteBuffer", "FPOS", mbDebug);
    clCheck( clEnqueueWriteBuffer(m_queue, gpuVar(&m_Fluid, FVEL), CL_TRUE, 0,      mMaxPoints * sizeof(float) * 3,                 bufC(&m_Fluid, FVEL), 0, NULL, NULL),       "TransferToCL", "clEnqueueWriteBuffer", "FVEL", mbDebug);
    clCheck( clEnqueueWriteBuffer(m_queue, gpuVar(&m_Fluid, FVEVAL), CL_TRUE, 0,    mMaxPoints * sizeof(float) * 3,                 bufC(&m_Fluid, FVEVAL), 0, NULL, NULL),     "TransferToCL", "clEnqueueWriteBuffer", "FVEVAL", mbDebug);
    clCheck( clEnqueueWriteBuffer(m_queue, gpuVar(&m_Fluid, FFORCE), CL_TRUE, 0,    mMaxPoints * sizeof(float) * 3,                 bufC(&m_Fluid, FFORCE), 0, NULL, NULL),     "TransferToCL", "clEnqueueWriteBuffer", "FFORCE", mbDebug);
    clCheck( clEnqueueWriteBuffer(m_queue, gpuVar(&m_Fluid, FPRESS), CL_TRUE, 0,    mMaxPoints * sizeof(float),                     bufC(&m_Fluid, FPRESS), 0, NULL, NULL),     "TransferToCL", "clEnqueueWriteBuffer", "FPRESS", mbDebug);
    clCheck( clEnqueueWriteBuffer(m_queue, gpuVar(&m_Fluid, FDENSITY), CL_TRUE, 0,  mMaxPoints * sizeof(float),                     bufC(&m_Fluid, FDENSITY), 0,NULL,NULL),     "TransferToCL","clEnqueueWriteBuffer","FDENSITY",mbDebug);
    clCheck( clEnqueueWriteBuffer(m_queue, gpuVar(&m_Fluid,FCOLOR),CL_TRUE ,0,        mMaxPoints * sizeof(uint),                      bufC(&m_Fluid,FCOLOR),0,NULL,NULL),           "TransferToCL","clEnqueueWriteBuffer","FCLR",mbDebug);

    // add extra data for morphogenesis
    clCheck( clEnqueueWriteBuffer(m_queue,gpuVar(&m_Fluid,FELASTIDX),CL_TRUE ,0,    mMaxPoints*sizeof(uint[BOND_DATA]),             bufC(&m_Fluid,FELASTIDX),0,NULL,NULL),      "TransferToCL","clEnqueueWriteBuffer","FELASTIDX",mbDebug);
    clCheck( clEnqueueWriteBuffer(m_queue,gpuVar(&m_Fluid,FPARTICLEIDX),CL_TRUE ,0, mMaxPoints*sizeof(uint[BONDS_PER_PARTICLE*2]),  bufC(&m_Fluid,FPARTICLEIDX),0,NULL,NULL),   "TransferToCL","clEnqueueWriteBuffer","FPARTICLEIDX",mbDebug);
    clCheck( clEnqueueWriteBuffer(m_queue,gpuVar(&m_Fluid,FPARTICLE_ID),CL_TRUE ,0, mMaxPoints*sizeof(uint),                        bufC(&m_Fluid,FPARTICLE_ID),0,NULL,NULL),   "TransferToCL","clEnqueueWriteBuffer","FPARTICLE_ID",mbDebug);
    clCheck( clEnqueueWriteBuffer(m_queue,gpuVar(&m_Fluid,FMASS_RADIUS),CL_TRUE ,0, mMaxPoints*sizeof(uint),                        bufC(&m_Fluid,FMASS_RADIUS),0,NULL,NULL),   "TransferToCL","clEnqueueWriteBuffer","FMASS_RADIUS",mbDebug);
    clCheck( clEnqueueWriteBuffer(m_queue,gpuVar(&m_Fluid,FNERVEIDX),CL_TRUE ,0,    mMaxPoints*sizeof(uint),                        bufC(&m_Fluid,FNERVEIDX),0,NULL,NULL),      "TransferToCL","clEnqueueWriteBuffer","FNERVEIDX",mbDebug);
    clCheck( clEnqueueWriteBuffer(m_queue,gpuVar(&m_Fluid,FCONC),CL_TRUE ,0,        mMaxPoints*sizeof(float[NUM_TF]),               bufC(&m_Fluid,FCONC),0,NULL,NULL),          "TransferToCL","clEnqueueWriteBuffer","FCONC",mbDebug);
    clCheck( clEnqueueWriteBuffer(m_queue,gpuVar(&m_Fluid,FEPIGEN),CL_TRUE ,0,      mMaxPoints*sizeof(uint[NUM_GENES]),             bufC(&m_Fluid,FEPIGEN),0,NULL,NULL),        "TransferToCL","clEnqueueWriteBuffer","FEPIGEN",mbDebug);

    if (verbosity>0) std::cout<<"\n-----TransferToCL() finished. -----\n"<<std::flush;



    if (verbosity>0) std::cout<<"TransferToCL ()  finished\n"<<std::flush;

}

void FluidSystem::TransferFromCL() {

    if (verbosity>0) std::cout<<"\n-----TransferFromCL() started... -----\n"<<std::flush;

    // Return particle buffers
    clFlush (m_queue);
    clFinish (m_queue);

    clCheck( clEnqueueReadBuffer(m_queue, gpuVar(&m_Fluid, FPOS), CL_TRUE, 0,         mMaxPoints *sizeof(cl_float3), bufC(&m_Fluid, FPOS), 0, NULL, NULL),         "TransferFromCL", "clEnqueueReadBuffer", "FPOS",     mbDebug);
    clCheck( clEnqueueReadBuffer(m_queue, gpuVar(&m_Fluid, FVEL), CL_TRUE, 0,         mMaxPoints *sizeof(cl_float3), bufC(&m_Fluid, FVEL), 0, NULL, NULL),         "TransferFromCL", "clEnqueueReadBuffer", "FVEL",     mbDebug);
    clCheck( clEnqueueReadBuffer(m_queue, gpuVar(&m_Fluid, FVEVAL), CL_TRUE, 0,       mMaxPoints *sizeof(cl_float3), bufC(&m_Fluid, FVEVAL), 0, NULL, NULL),       "TransferFromCL", "clEnqueueReadBuffer", "FVEVAL",   mbDebug);
    clCheck( clEnqueueReadBuffer(m_queue, gpuVar(&m_Fluid, FFORCE), CL_TRUE, 0,       mMaxPoints *sizeof(cl_float3), bufC(&m_Fluid, FFORCE), 0, NULL, NULL),       "TransferFromCL", "clEnqueueReadBuffer", "FFORCE",   mbDebug);
    //NB PhysicalSort zeros gpuVar(&m_FluidTemp, FFORCE)
    clCheck( clEnqueueReadBuffer(m_queue, gpuVar(&m_Fluid, FPRESS), CL_TRUE, 0,       mMaxPoints *sizeof(float),     bufC(&m_Fluid, FPRESS), 0, NULL, NULL),       "TransferFromCL", "clEnqueueReadBuffer", "FPRESS",   mbDebug);
    clCheck( clEnqueueReadBuffer(m_queue, gpuVar(&m_Fluid, FDENSITY), CL_TRUE, 0,     mMaxPoints *sizeof(float),     bufC(&m_Fluid, FDENSITY), 0, NULL,NULL),      "TransferFromCL", "clEnqueueReadBuffer", "FDENSITY", mbDebug);
    clCheck( clEnqueueReadBuffer(m_queue,gpuVar(&m_Fluid,FAGE), CL_TRUE ,0 ,          mMaxPoints*sizeof(uint) ,       bufC(&m_Fluid,FAGE) ,0 ,NULL ,NULL) ,        "TransferFromCL", "clEnqueueReadBuffer" ,"FAGE" ,    mbDebug);
    clCheck( clEnqueueReadBuffer(m_queue,gpuVar(&m_Fluid,FCOLOR), CL_TRUE ,0 ,          mMaxPoints*sizeof(uint) ,       bufC(&m_Fluid,FCOLOR) ,0 ,NULL ,NULL) ,        "TransferFromCL", "clEnqueueReadBuffer" ,"FCLR" ,    mbDebug);

    // add extra data for morphogenesis
    clCheck( clEnqueueReadBuffer(m_queue,gpuVar(&m_Fluid,FELASTIDX), CL_TRUE ,0 ,     mMaxPoints*sizeof(uint[BOND_DATA]) ,                bufC(&m_Fluid,FELASTIDX) ,0 ,NULL ,NULL) ,            "TransferFromCL","clEnqueueReadBuffer", "FELASTIDX" ,   mbDebug);
    clCheck( clEnqueueReadBuffer(m_queue,gpuVar(&m_Fluid,FPARTICLEIDX), CL_TRUE ,0 ,  mMaxPoints*sizeof(uint[BONDS_PER_PARTICLE *2]),     bufC(&m_Fluid,FPARTICLEIDX) ,0 ,NULL ,NULL) ,         "TransferFromCL","clEnqueueReadBuffer", "FPARTICLEIDX" ,mbDebug);
    clCheck( clEnqueueReadBuffer(m_queue,gpuVar(&m_Fluid,FPARTICLE_ID), CL_TRUE ,0 ,  mMaxPoints*sizeof(uint) ,                           bufC(&m_Fluid,FPARTICLE_ID) ,0 ,NULL,NULL) ,          "TransferFromCL","clEnqueueReadBuffer", "FPARTICLE_ID" ,mbDebug);
    clCheck( clEnqueueReadBuffer(m_queue,gpuVar(&m_Fluid,FMASS_RADIUS), CL_TRUE ,0 ,  mMaxPoints*sizeof(uint) ,                           bufC(&m_Fluid,FMASS_RADIUS) ,0 ,NULL,NULL) ,          "TransferFromCL","clEnqueueReadBuffer", "FMASS_RADIUS" ,mbDebug);
    clCheck( clEnqueueReadBuffer(m_queue,gpuVar(&m_Fluid,FNERVEIDX), CL_TRUE ,0 ,     mMaxPoints*sizeof(uint) ,                           bufC(&m_Fluid,FNERVEIDX) ,0 ,NULL,NULL) ,             "TransferFromCL","clEnqueueReadBuffer", "FNERVEIDX" ,   mbDebug);
    clCheck( clEnqueueReadBuffer(m_queue,gpuVar(&m_Fluid, FCONC), CL_TRUE ,0 ,        mMaxPoints*sizeof(float[NUM_TF]) ,                  bufC(&m_Fluid,FCONC) ,0,NULL,NULL),                   "TransferFromCL","clEnqueueReadBuffer", "FCONC",        mbDebug);
    clCheck( clEnqueueReadBuffer(m_queue,gpuVar(&m_Fluid,FEPIGEN), CL_TRUE ,0,        mMaxPoints*sizeof(uint[NUM_GENES]),                 bufC(&m_Fluid,FEPIGEN),0,NULL,NULL),                  "TransferFromCL","clEnqueueReadBuffer", "FEPIGEN",      mbDebug);

    clFlush (m_queue);
    clFinish (m_queue);

    if (verbosity>0) std::cout<<"----- TransferFromCL () finished. -----\n"<<std::flush;

}

void FluidSystem::Init_CLRand (){

                                                                                                                                                        if(verbosity>2) cout << "-----Init_CLRand() started... -----\n\n" << flush;
                                                                                                                                                        if(verbosity>2) cout << "nNumPoints = " << mNumPoints << "\n\n";
    cl_int status;

    // Initialize RNG
    unsigned long long* seeds = (unsigned long long*)malloc(sizeof(unsigned long long) * mNumPoints);

    for (unsigned int i = 0; i < mNumPoints; ++i) {
        seeds[i] = rand(); // Replace with your preferred method of generating seeds
    }
    if(verbosity>2) cout << "\nmNumPoints = " << mNumPoints << "\n" << flush;
    if(verbosity>2) cout << "\nsizeof(unsigned long long) = " << sizeof(unsigned long long) << "\n" << flush;

    // Create buffers
    cl_mem seed_buf =   clCreateBuffer(m_context, CL_MEM_READ_ONLY | CL_MEM_COPY_HOST_PTR, sizeof(unsigned long long) * mNumPoints, seeds, &status);   if(status!=CL_SUCCESS){cout<<"\nres1 = "<<checkerror(status)<<"\n"<<flush;exit_(status);}
    cl_mem res_buf =    clCreateBuffer(m_context, CL_MEM_WRITE_ONLY, sizeof(unsigned int) * mNumPoints, NULL, &status);                                if(status!=CL_SUCCESS){cout<<"\nres2 = "<<checkerror(status)<<"\n"<<flush;exit_(status);}


    // Set up the kernel arguments
    clCheck(clSetKernelArg(m_Kern[FUNC_INIT_RANDOMCL], 0, sizeof(cl_int) * mNumPoints, &mNumPoints), "Init_CLRand", "clSetKernelArg", "mNumPoints", mbDebug);
    clCheck(clSetKernelArg(m_Kern[FUNC_INIT_RANDOMCL], 1, sizeof(cl_mem), &seed_buf), "Init_CLRand", "clSetKernelArg", "seed_buf", mbDebug);
    clCheck(clSetKernelArg(m_Kern[FUNC_INIT_RANDOMCL], 2, sizeof(cl_mem), &res_buf), "Init_CLRand", "clSetKernelArg", "res_buf", mbDebug);

    // Launch the kernel
    size_t global_size_work = mNumPoints;
    clEnqueueNDRangeKernel(m_queue, m_Kern[FUNC_INIT_RANDOMCL], 1, NULL, &global_size_work, NULL, 0, NULL, NULL);

    status = clFlush(m_queue); 				if (status != CL_SUCCESS)	{ cout << "\nclFlush(m_queue) status = " << checkerror(status) <<"\n"<<flush; exit_(status);}
	status = clFinish(m_queue); 			if (status != CL_SUCCESS)	{ cout << "\nclFinish(m_queue)="		 <<checkerror(status)  <<"\n"<<flush; exit_(status);}

    // Create a host buffer to hold the results
    unsigned int* randBuf = (unsigned int*)malloc(sizeof(unsigned int) * mNumPoints);


    // Read the results back to host memory
    clEnqueueReadBuffer(m_queue, res_buf, CL_TRUE, 0, sizeof(unsigned int) * mNumPoints, randBuf, 0, NULL, NULL);                                       if(status!=CL_SUCCESS){cout<<"\nres3 = "<<checkerror(status)<<"\n"<<flush;exit_(status);}

    // Print out the results
    printf("Note: randBuf results are not printed (host_CL.cpp, Init_CLRand() )");                                                                                                                                                    //if(verbosity>1) {for (unsigned int i = 0; i < mNumPoints; ++i) {printf("randBuf[%u] = %u\n", i, randBuf[i]);} }

    // Free buffers
    free(randBuf);
    free(seeds);
                                                                                                                                                        if(verbosity>0) cout << "\n-----Init_CLRand() finished-----\n\n" << flush;
}
//     unsigned long long  seed=0;
//     srand (time(NULL));
//     for (int i=0;i<mNumPoints;i++){
//         //seed = seed << 32;
//         //seed += rand();
//         //seed = clock();
//         bufI(&m_Fluid, FCURAND_SEED)[i] = seed;
//         //curand_init(seed, sequence, offset, &m_Fluid.bufCuRNDST(FCURAND_STATE)[i]);
//         if (verbosity>1)printf("\n(2:seed=%llu,(FCURAND_SEED)[i]=%llu, rand()=%u), ",seed, bufULL(&m_Fluid, FCURAND_SEED)[i], rand() );
//         if (verbosity>1) cout<<"\t(seed="<<seed<<",(FCURAND_SEED)[i]="<<bufI(&m_Fluid, FCURAND_SEED)[i]<<"), "<<std::flush;
//     }
//
//     // transfer to cuda
//     //clCheck( cuMemcpyDtoH ( gcell,	gpuVar(&m_Fluid, FGCELL),	mNumPoints *sizeof(uint) ), "InsertParticlesCL", "cuMemcpyDtoH", "FGCELL", mbDebug );
//     //clCheck( cuMemcpyDtoH ( m_Fluid.bufCuRNDST(FCURAND_STATE),	gpuVar(&m_Fluid, FCURAND_STATE),	mNumPoints *sizeof(curandState_t) ),
//     //         "Init_FCURAND_STATE_CL", "cuMemcpyDtoH", "FCURAND_STATE", mbDebug );
//
//     clCheck( clEnqueueWriteBuffer(m_queue, gpuVar(&m_Fluid, FCURAND_SEED), CL_TRUE, 0, mNumPoints * sizeof(uint), bufC(&m_Fluid, FCURAND_SEED), 0, NULL, NULL), "Init_FCURAND_STATE_CL", "clEnqueueWriteBuffer", "FCURAND_SEED", mbDebug );
//
//     if (verbosity>1) std::cout <<"\nInit_FCURAND_STATE_CL_2.0\n\n"<<std::flush;

    /*
    int n=0;
    void* args[1] = {&n};
    int numGroups_=1, numItems_=1;
    if (verbosity>1) std::cout <<"\nInit_FCURAND_STATE_CL_1.0\t n="<<n<<",  pow(256,n)="<<pow(256,n)<<",  mNumPoints/256="<<mNumPoints/256<<",\t mNumPoints="<<mNumPoints<<", mMaxPoints="<<mMaxPoints<<"  \n"<<std::flush;

    do {
        computeNumBlocks ( pow(256,n), m_FParams.itemsPerGroup, numGroups_, numItems_);

        if (verbosity>1) std::cout <<"\nInit_FCURAND_STATE_CL_2.0\t n="<<n<<",  pow(256,n)="<<pow(256,n)<<",  mNumPoints/256="<<mNumPoints/256<<
        "\t numGroups_="<<numGroups_<<", numItems_="<<numItems_<<"  \n"<<std::flush;

        clCheck(clFinish(), "Init_FCURAND_STATE_CL", "clFinish", "Before m_Kern[FUNC_INIT_RANDOMCL], in do-while loop", 1);

        clCheck ( cuLaunchKernel ( m_Kern[FUNC_INIT_RANDOMCL],  numGroups_, 1, 1, numItems_, 1, 1, 0, NULL, args, NULL), "Init_FCURAND_STATE_CL", "cuLaunch", "FUNC_INIT_RANDOMCL", mbDebug);

        n++;
    } while (pow(256,n) < mNumPoints/256) ;

    if (verbosity>1) std::cout <<"\nInit_FCURAND_STATE_CL_3.0\t n="<<n<<",  pow(256,n)="<<pow(256,n)<<",  mNumPoints/256="<<mNumPoints/256<<
    "\t m_FParams.numGroups="<<m_FParams.numGroups<<",  m_FParams.numItems="<<m_FParams.numItems<<".  \n"<<std::flush;

    */
//     size_t streamBufferSize;

//     size_t global_size_work = mNumPoints;


//     clrngMrg31k3pStream* streams = clrngMrg31k3pCreateStreams (NULL, mNumPoints,&streamBufferSize, NULL);

    // Create an input buffer
//     cl_mem buf_in = clCreateBuffer (m_context, CL_MEM_READ_ONLY | CL_MEM_COPY_HOST_PTR, streamBufferSize, streams,  NULL);

    // Create buffer to transfer output back from the device.
//     cl_mem buf_out = clCreateBuffer (m_context, CL_MEM_WRITE_ONLY | CL_MEM_HOST_READ_ONLY, mNumPoints * sizeof (cl_double), NULL, NULL);

    // The kernel takes two arguments; set them to buf_in, buf_out.
//     clCheck( clSetKernelArg(m_Kern[FUNC_INIT_RANDOMCL], 0, sizeof (buf_in), &buf_in), "Init_FCURAND_STATE_CL", "clSetKernelArg", "buf_in", 1);
//     clCheck( clSetKernelArg(m_Kern[FUNC_INIT_RANDOMCL], 1, sizeof (buf_out), &buf_out), "Init_FCURAND_STATE_CL", "clSetKernelArg", "buf_out", 1);
/*
    // Enqueue the kernel on device.
    cl_event ev;
    clCheck( clEnqueueNDRangeKernel (m_queue, m_Kern[FUNC_INIT_RANDOMCL], 1, NULL, &global_size_work, NULL, 0, NULL, &ev), "Init_FCURAND_STATE_CL", "clEnqueueNDRangeKernel", "-", 1);

    // Wait for all work items to finish.
    clCheck( clWaitForEvents (1, &ev), "Init_FCURAND_STATE_CL", "clFinish", "After clEnqueueWriteBuffer FCURAND_STATE, before 1st timestep", 1);

     // Retrieve the contents of the output buffer from the device.
    double* out = (double*) malloc (mNumPoints * sizeof (double));
//     clCheck( clEnqueueReadBuffer(m_queue, buf_out, CL_TRUE, 0, mNumPoints * sizeof (out [0]), out, 0, NULL, NULL), "Init_FCURAND_STATE_CL", "clFinish", "After clEnqueueWriteBuffer FCURAND_STATE, before 1st timestep", 1);

    clCheck( clFinish(m_queue), "Init_FCURAND_STATE_CL", "clFinish", "After clEnqueueWriteBuffer FCURAND_STATE, before 1st timestep", 1);
    if (verbosity>1) std::cout <<"\nInit_FCURAND_STATE_CL_4.0\n"<<std::flush;
}*/

/* TODO
void FluidSystem::PrefixSumChangesCL ( int zero_offsets ){
    // Prefix Sum - determine grid offsets
    int blockSize = SCAN_BLOCKSIZE << 1;                // NB 1024 = 512 << 1.  NB SCAN_BLOCKSIZE is the number of threads per block
    int numElem1 = m_GridTotal;                         // tot num bins, computed in SetupGrid()
    int numElem2 = int ( numElem1 / blockSize ) + 1;    // num sheets of bins? NB not spatial, but just dividing the linear array of bins, by a factor of 512*2
    int numElem3 = int ( numElem2 / blockSize ) + 1;    // num rows of bins?
    int threads = SCAN_BLOCKSIZE;
    int zon=1;
    cl_device_idptr array1  ;		// input
    cl_device_idptr scan1   ;		// output
    cl_device_idptr array2  = gpuVar(&m_Fluid, FAUXARRAY1);
    cl_device_idptr scan2   = gpuVar(&m_Fluid, FAUXSCAN1);
    cl_device_idptr array3  = gpuVar(&m_Fluid, FAUXARRAY2);
    cl_device_idptr scan3   = gpuVar(&m_Fluid, FAUXSCAN2);

    // Loop to PrefixSum the Dense Lists - NB by doing one change_list at a time, we reuse the FAUX* arrays & scans.
    // For each change_list, input FBIN_COUNT_ACTIVE_GENES[change_list*m_GridTotal], output FBIN_OFFSET_ACTIVE_GENES[change_list*m_GridTotal]
    cl_device_idptr array0  = gpuVar(&m_Fluid, FBIN_COUNT_CHANGES);
    cl_device_idptr scan0   = gpuVar(&m_Fluid, FBIN_OFFSET_CHANGES);

    if(m_debug>3){
        // debug chk
        cout<<"\nSaving (FBIN_COUNT_CHANGES): (bin,#particles) , numElem1="<<numElem1<<"\t"<<std::flush;
        clCheck( cuMemcpyDtoH ( bufI(&m_Fluid, FBIN_COUNT_CHANGES), gpuVar(&m_Fluid, FBIN_COUNT_CHANGES),	sizeof(uint[numElem1]) ), "PrefixSumChangesCL", "cuMemcpyDtoH", "FBIN_COUNT_CHANGES", mbDebug); // NUM_CHANGES*
        //### print to a csv file   AND do the same afterwards for FBIN_OFFSET_CHANGES ###
        SaveUintArray( bufI(&m_Fluid, FBIN_COUNT_CHANGES), numElem1, "bufI(&m_Fluid, FBIN_COUNT_CHANGES).csv" );
        //
        cout<<"\nSaving (FGCELL): (particleIdx, cell) , mMaxPoints="<<mMaxPoints<<"\t"<<std::flush;
        clCheck( cuMemcpyDtoH ( bufI(&m_Fluid, FGCELL), gpuVar(&m_Fluid, FGCELL),	sizeof(uint[mMaxPoints]) ), "PrefixSumChangesCL", "cuMemcpyDtoH", "FGCELL", mbDebug);
        SaveUintArray( bufI(&m_Fluid, FGCELL), mMaxPoints, "bufI(&m_Fluid, FGCELL).csv" );
        //
        //   clCheck( cuMemcpyDtoH ( bufI(&m_Fluid, FDENSE_LISTS), gpuVar(&m_Fluid, FDENSE_LISTS),	sizeof(uint[mMaxPoints]) ), "PrefixSumChangesCL", "cuMemcpyDtoH", "FDENSE_LISTS", mbDebug);
        //   SaveUintArray( bufI(&m_Fluid, FDENSE_LISTS), numElem1, "bufI(&m_Fluid, FDENSE_LISTS).csv" );
    }
    clFinish ();

    for(int change_list=0; change_list<NUM_CHANGES; change_list++){
        array1  = array0 + change_list*numElem1*sizeof(int); //gpuVar(&m_Fluid, FBIN_COUNT_ACTIVE_GENES);//[change_list*numElem1]   ;      // cl_device_idptr to change_list within gpuVar(&m_Fluid, FBIN_COUNT_CHANGES), for start of prefix-sum.
        scan1   = scan0 + change_list*numElem1*sizeof(int);
        clCheck ( cuMemsetD8 ( scan1,  0,	numElem1*sizeof(int) ), "PrefixSumChangesCL", "cuMemsetD8", "FBIN_COUNT", mbDebug );
        clCheck ( cuMemsetD8 ( array2, 0,	numElem2*sizeof(int) ), "PrefixSumChangesCL", "cuMemsetD8", "FBIN_COUNT", mbDebug );
        clCheck ( cuMemsetD8 ( scan2,  0,	numElem2*sizeof(int) ), "PrefixSumChangesCL", "cuMemsetD8", "FBIN_COUNT", mbDebug );
        clCheck ( cuMemsetD8 ( array3, 0,	numElem3*sizeof(int) ), "PrefixSumChangesCL", "cuMemsetD8", "FBIN_COUNT", mbDebug );
        clCheck ( cuMemsetD8 ( scan3,  0,	numElem3*sizeof(int) ), "PrefixSumChangesCL", "cuMemsetD8", "FBIN_COUNT", mbDebug );

        void* argsA[5] = {&array1, &scan1, &array2, &numElem1, &zero_offsets };     // sum array1. output -> scan1, array2.         i.e. FBIN_COUNT -> FBIN_OFFSET, FAUXARRAY1
        clCheck ( cuLaunchKernel ( m_Kern[FUNC_FPREFIXSUM], numElem2, 1, 1, threads, 1, 1, 0, NULL, argsA, NULL ),
                  "PrefixSumChangesCL", "cuLaunch", "FUNC_PREFIXSUM", mbDebug);
        void* argsB[5] = { &array2, &scan2, &array3, &numElem2, &zon };             // sum array2. output -> scan2, array3.         i.e. FAUXARRAY1 -> FAUXSCAN1, FAUXARRAY2
        clCheck ( cuLaunchKernel ( m_Kern[FUNC_FPREFIXSUM], numElem3, 1, 1, threads, 1, 1, 0, NULL, argsB, NULL ), "PrefixSumChangesCL", "cuLaunch", "FUNC_PREFIXSUM", mbDebug);
        if ( numElem3 > 1 ) {
            cl_device_idptr nptr = {0};
            void* argsC[5] = { &array3, &scan3, &nptr, &numElem3, &zon };	        // sum array3. output -> scan3                  i.e. FAUXARRAY2 -> FAUXSCAN2, &nptr
            clCheck ( cuLaunchKernel ( m_Kern[FUNC_FPREFIXSUM], 1, 1, 1, threads, 1, 1, 0, NULL, argsC, NULL ), "PrefixSumChangesCL", "cuLaunch", "FUNC_PREFIXFIXUP", mbDebug);
            void* argsD[3] = { &scan2, &scan3, &numElem2 };	                        // merge scan3 into scan2. output -> scan2      i.e. FAUXSCAN2, FAUXSCAN1 -> FAUXSCAN1
            clCheck ( cuLaunchKernel ( m_Kern[FUNC_FPREFIXUP], numElem3, 1, 1, threads, 1, 1, 0, NULL, argsD, NULL ), "PrefixSumChangesCL", "cuLaunch", "FUNC_PREFIXFIXUP", mbDebug);
        }
        void* argsE[3] = { &scan1, &scan2, &numElem1 };		                        // merge scan2 into scan1. output -> scan1      i.e. FAUXSCAN1, FBIN_OFFSET -> FBIN_OFFSET
        clCheck ( cuLaunchKernel ( m_Kern[FUNC_FPREFIXUP], numElem2, 1, 1, threads, 1, 1, 0, NULL, argsE, NULL ), "PrefixSumChangesCL", "cuLaunch", "FUNC_PREFIXFIXUP", mbDebug);
    }

    int num_lists = NUM_CHANGES, length = FDENSE_LIST_LENGTHS_CHANGES, fgridcnt = FBIN_COUNT_CHANGES, fgridoff = FBIN_OFFSET_CHANGES;
    void* argsF[4] = {&num_lists, &length,&fgridcnt,&fgridoff};
    clCheck ( cuLaunchKernel ( m_Kern[FUNC_TALLYLISTS], NUM_CHANGES, 1, 1, NUM_CHANGES, 1, 1, 0, NULL, argsF, NULL ), "PrefixSumChangesCL", "cuLaunch", "FUNC_TALLYLISTS", mbDebug); //256 threads launched
    clCheck( cuMemcpyDtoH ( bufI(&m_Fluid, FDENSE_LIST_LENGTHS_CHANGES), gpuVar(&m_Fluid, FDENSE_LIST_LENGTHS_CHANGES),	sizeof(uint[NUM_CHANGES]) ), "PrefixSumChangesCL", "cuMemcpyDtoH", "FDENSE_LIST_LENGTHS_CHANGES", mbDebug);

                                                                                                                    // If active particles for change_list > existing buff, then enlarge buff.
    for(int change_list=0;change_list<NUM_CHANGES;change_list++){                                                   // Note this calculation could be done by a kernel,
        uint * densebuff_len = bufI(&m_Fluid, FDENSE_BUF_LENGTHS_CHANGES);                                            // and only bufI(&m_Fluid, FDENSE_LIST_LENGTHS); copied to host.
        uint * denselist_len = bufI(&m_Fluid, FDENSE_LIST_LENGTHS_CHANGES);                                           // For each change_list allocate intial buffer,
        //if (verbosity>1)printf("\nPrefixSumChangesCL: change_list=%u,  densebuff_len[change_list]=%u, denselist_len[change_list]=%u ,",change_list, densebuff_len[change_list], denselist_len[change_list] );
        if (denselist_len[change_list] > densebuff_len[change_list]) {                                              // write pointer and size to FDENSE_LISTS and FDENSE_LIST_LENGTHS
            while(denselist_len[change_list] >  densebuff_len[change_list])   densebuff_len[change_list] *=4;       // bufI(&m_Fluid, FDENSE_BUF_LENGTHS)[i].
                                                                                                                    // NB Need 2*densebuff_len[change_list] for particle & bond
            if (verbosity>1)printf("\nPrefixSumChangesCL: ## enlarging buffer## change_list=%u,  densebuff_len[change_list]=%u, denselist_len[change_list]=%u ,",change_list, densebuff_len[change_list], denselist_len[change_list] );
            AllocateBufferDenseLists( change_list, sizeof(uint), 2*densebuff_len[change_list], FDENSE_LISTS_CHANGES );// NB frees previous buffer &=> clears data
        }                                                                                                           // NB buf[2][list_length] holding : particleIdx, bondIdx
    }
    clCheck( cuMemcpyHtoD ( gpuVar(&m_Fluid, FDENSE_BUF_LENGTHS_CHANGES), bufC(&m_Fluid, FDENSE_BUF_LENGTHS_CHANGES),	sizeof(uint[NUM_CHANGES]) ), "PrefixSumChangesCL", "cuMemcpyHtoD", "FDENSE_BUF_LENGTHS_CHANGES", mbDebug);
    cuMemcpyHtoD(gpuVar(&m_Fluid, FDENSE_LISTS_CHANGES), bufC(&m_Fluid, FDENSE_LISTS_CHANGES),  NUM_CHANGES * sizeof(cl_device_idptr)  );                      // update pointers to lists on device

    if (verbosity>1) {
        std::cout << "\nChk: PrefixSumChangesCL 4"<<std::flush;
        for(int change_list=0;change_list<NUM_CHANGES;change_list++){
            std::cout<<"\nPrefixSumChangesCL: change list_length["<<change_list<<"]="<<bufI(&m_Fluid, FDENSE_LIST_LENGTHS_CHANGES)[change_list]<<"\t"<<std::flush;
        }
    }
}*/

// void FluidSystem::CountingSortChangesCL ( ){
//     //std::cout<<"\n\n#### CountingSortChangesCL ( ):   verbosity = "<< verbosity <<"\n";
//     if (verbosity>1) {std::cout<<"\n\n#### CountingSortChangesCL ( )"<<std::flush;}
//     /* ////////
//     clCheck( cuMemcpyDtoH ( bufI(&m_Fluid, FDENSE_LIST_LENGTHS_CHANGES), gpuVar(&m_Fluid, FDENSE_LIST_LENGTHS_CHANGES),	sizeof(uint[NUM_CHANGES]) ), "PrefixSumCellsCL", "cuMemcpyDtoH", "FDENSE_LIST_LENGTHS_CHANGES", mbDebug);
//
//                                                                                                                     // If active particles for change_list > existing buff, then enlarge buff.
//     for(int change_list=0;change_list<NUM_CHANGES;change_list++){                                                   // Note this calculation could be done by a kernel,
//         uint * densebuff_len = bufI(&m_Fluid, FDENSE_BUF_LENGTHS_CHANGES);                                            // and only bufI(&m_Fluid, FDENSE_LIST_LENGTHS); copied to host.
//         uint * denselist_len = bufI(&m_Fluid, FDENSE_LIST_LENGTHS_CHANGES);                                           // For each change_list allocate intial buffer,
//         if (verbosity>1)printf("\nCountingSortChangesCL1: change_list=%u,  densebuff_len[change_list]=%u, denselist_len[change_list]=%u ,",change_list, densebuff_len[change_list], denselist_len[change_list] );
//     }
//     *//////////
//     int blockSize = SCAN_BLOCKSIZE/2 << 1;
//     int numElem1 = m_GridTotal;
//     int numElem2 = 2* int( numElem1 / blockSize ) + 1;
//     int threads = SCAN_BLOCKSIZE/2;
//     void* args[1] = { &mActivePoints };
//     clCheck ( cuLaunchKernel ( m_Kern[FUNC_COUNTING_SORT_CHANGES], numElem2, 1, 1, threads , 1, 1, 0, NULL, args, NULL),
//               "CountingSortChangesCL", "cuLaunch", "FUNC_COUNTING_SORT_CHANGES", mbDebug );
//      /////////
//     clCheck(clFinish(), "CountingSortChangesCL()", "clFinish", "After FUNC_COUNTING_SORT_CHANGES", mbDebug);
//
//     clCheck( cuMemcpyDtoH ( bufI(&m_Fluid, FDENSE_LIST_LENGTHS_CHANGES), gpuVar(&m_Fluid, FDENSE_LIST_LENGTHS_CHANGES),	sizeof(uint[NUM_CHANGES]) ), "PrefixSumCellsCL", "cuMemcpyDtoH", "FDENSE_LIST_LENGTHS_CHANGES", mbDebug);
//                                                                                                                     // If active particles for change_list > existing buff, then enlarge buff.
//     for(int change_list=0;change_list<NUM_CHANGES;change_list++){                                                   // Note this calculation could be done by a kernel,
//         uint * densebuff_len = bufI(&m_Fluid, FDENSE_BUF_LENGTHS_CHANGES);                                            // and only bufI(&m_Fluid, FDENSE_LIST_LENGTHS); copied to host.
//         uint * denselist_len = bufI(&m_Fluid, FDENSE_LIST_LENGTHS_CHANGES);                                           // For each change_list allocate intial buffer,
//         if (verbosity>1)printf("\nCountingSortChangesCL2: change_list=%u,  densebuff_len[change_list]=%u, denselist_len[change_list]=%u ,\t\t threads=%u, numElem2=%u,  m_GridTotal=%u \t",
//                change_list, densebuff_len[change_list], denselist_len[change_list], threads, numElem2,  m_GridTotal );
//         clFinish ();
//         if(verbosity>0){
//             uint fDenseList2[1000000] = {UINT_MAXSIZE};//TODO make this array size safe!  NB 10* num particles.
//             cl_device_idptr*  _list2pointer = (cl_device_idptr*) &bufC(&m_Fluid, FDENSE_LISTS_CHANGES)[change_list*sizeof(cl_device_idptr)];
//                                                                                                                 // Get device pointer to FDENSE_LISTS_CHANGES[change_list].
//             clCheck( cuMemcpyDtoH ( fDenseList2, *_list2pointer,	2*sizeof(uint[densebuff_len[change_list]]) ), "PrefixSumChangesCL", "cuMemcpyDtoH", "FBIN_COUNT", mbDebug);
//             char filename[256];
//             sprintf(filename, "CountingSortChangesCL__m_Fluid.bufII(FDENSE_LISTS_CHANGES)[%u].csv", change_list);
//             SaveUintArray_2Columns( fDenseList2, denselist_len[change_list], densebuff_len[change_list], filename );
//             ///
//             printf("\n\n*_list2pointer=%llu",*_list2pointer);
//
//         }
//     }
// }
//
void FluidSystem::InitializeBondsCL (){

    cl_int status;
    if (verbosity>1)cout << "\n\nInitializeBondsCL ()\n"<<std::flush;
    uint gene           = 1;                                                            // solid  (has springs)
    uint list_length    = bufI(&m_Fluid, FDENSE_LIST_LENGTHS)[gene];
    //void* args[3]       = { &m_FParams.pnumActive, &list_length, &gene};                //initialize_bonds (int ActivePoints, uint list_length, uint gene)

    clCheck(clSetKernelArg(m_Kern[FUNC_INITIALIZE_BONDS], 0, sizeof(int), &m_FParams.pnumActive), "InitializeBondsCL", "clSetKernelArg", "FUNC_INITIALIZE_BONDS 0", mbDebug);
    clCheck(clSetKernelArg(m_Kern[FUNC_INITIALIZE_BONDS], 1, sizeof(uint), &list_length), "InitializeBondsCL", "clSetKernelArg", "FUNC_INITIALIZE_BONDS 1", mbDebug);
    clCheck(clSetKernelArg(m_Kern[FUNC_INITIALIZE_BONDS], 2, sizeof(uint), &gene), "InitializeBondsCL", "clSetKernelArg", "FUNC_INITIALIZE_BONDS 2", mbDebug);


    if (verbosity>1)cout << "\nInitializeBondsCL (): list_length="<<list_length<<", m_FParams.itemsPerGroup="<<m_FParams.itemsPerGroup<<", numGroups="<<m_FParams.numGroups<<", numItems="<<m_FParams.numItems<<" \t args{m_FParams.pnumActive="<<m_FParams.pnumActive<<", list_length="<<list_length<<", gene="<<gene<<"}"<<std::flush;

    size_t numGroups, numItems;
    computeNumBlocks ( list_length , local_work_size, numGroups, numItems);

    //clCheck ( cuLaunchKernel ( m_Kern[FUNC_INITIALIZE_BONDS],  m_FParams.numGroups, 1, 1, m_FParams.numItems, 1, 1, 0, NULL, args, NULL), "ComputePressureCL", "cuLaunch", "FUNC_COMPUTE_PRESS", mbDebug);
    clCheck(clEnqueueNDRangeKernel(

        m_queue,
        m_Kern[FUNC_INITIALIZE_BONDS],
        1,
        NULL,
        &numGroups,
        &numItems,
        0,
        NULL,
        NULL),

    "InitializeBondsCL", "clEnqueueNDRangeKernel", "FUNC_INITIALIZE_BONDS", mbDebug);
}

void FluidSystem::ComputePressureCL (){
    cout <<"\nFluidSystem::ComputePressureCL () : chk 1 "<< flush;
    cl_int status;
    //void* args[1] = { &mActivePoints };
    status  = clSetKernelArg(m_Kern[FUNC_COMPUTE_PRESS], 0, sizeof(int), &mActivePoints);

    size_t global_work_size = m_FParams.numGroups * m_FParams.numItems;
    size_t local_work_size = m_FParams.numItems;

    cout <<"\nFluidSystem::ComputePressureCL () : chk 2  global_work_size = " <<global_work_size<<",  local_work_size =  " << global_work_size << ", m_FParams.numGroups = "<<m_FParams.numGroups<<" , m_FParams.numItems = "<< m_FParams.numItems << flush;

    //cout<<"\nComputePressureCL: mActivePoints="<<mActivePoints<<std::flush;
    //clCheck ( cuLaunchKernel ( m_Kern[FUNC_COMPUTE_PRESS],  m_FParams.numGroups, 1, 1, m_FParams.numItems, 1, 1, 0, NULL, args, NULL), "ComputePressureCL", "cuLaunch", "FUNC_COMPUTE_PRESS", mbDebug);
    status = clEnqueueNDRangeKernel(m_queue, m_Kern[FUNC_COMPUTE_PRESS], 1, NULL, &global_work_size, &local_work_size, 0, NULL, NULL);

    cout <<"\nFluidSystem::ComputePressureCL () : finished "<< flush;
}

void FluidSystem::ComputeDiffusionCL(){
    //if (verbosity>1) std::cout << "\n\nRunning ComputeDiffusionCL()" << std::endl;
    cl_int status;
    //void* args[1] = { &mActivePoints };
    status  = clSetKernelArg(m_Kern[FUNC_COMPUTE_DIFFUSION], 0, sizeof(int), &mActivePoints);

    size_t global_work_size = m_FParams.numGroups * m_FParams.numItems;
    size_t local_work_size = m_FParams.numItems;

    //clCheck ( cuLaunchKernel ( m_Kern[FUNC_COMPUTE_DIFFUSION],  m_FParams.numGroups, 1, 1, m_FParams.numItems, 1, 1, 0, NULL, args, NULL), "ComputeDiffusionCL", "cuLaunch", "FUNC_COMPUTE_DIFFUSION", mbDebug);
    status = clEnqueueNDRangeKernel(m_queue, m_Kern[FUNC_COMPUTE_DIFFUSION], 1, NULL, &global_work_size, &local_work_size, 0, NULL, NULL);

}

void FluidSystem::ComputeForceCL (){
    //if (verbosity>1)printf("\n\nFluidSystem::ComputeForceCL (),  m_FParams.freeze=%s",(m_FParams.freeze==true) ? "true" : "false");
    cl_int status;
    //void* args[3] = { &m_FParams.pnumActive ,  &m_FParams.freeze, &m_FParams.frame};
    status  = clSetKernelArg(m_Kern[FUNC_COMPUTE_FORCE], 0, sizeof(int), &mActivePoints);
    status  = clSetKernelArg(m_Kern[FUNC_COMPUTE_FORCE], 1, sizeof(bool), &mActivePoints);
    status  = clSetKernelArg(m_Kern[FUNC_COMPUTE_FORCE], 2, sizeof(uint), &mActivePoints);

    size_t global_work_size = m_FParams.numGroups * m_FParams.numItems;
    size_t local_work_size = m_FParams.numItems;

    //clCheck ( cuLaunchKernel ( m_Kern[FUNC_COMPUTE_FORCE],  m_FParams.numGroups, 1, 1, m_FParams.numItems, 1, 1, 0, NULL, args, NULL), "ComputeForceCL", "cuLaunch", "FUNC_COMPUTE_FORCE", mbDebug);
    status = clEnqueueNDRangeKernel(m_queue, m_Kern[FUNC_COMPUTE_FORCE], 1, NULL, &global_work_size, &local_work_size, 0, NULL, NULL);
}

void FluidSystem::ComputeGenesCL (){  // for each gene, call a kernel wih the dese list for that gene
    // NB Must zero ftemp.bufI(FEPIGEN) and ftemp.bufI(FCONC) before calling kernel. ftemp is used to collect changes before FUNC_TALLY_GENE_ACTION.
    cl_int status;

    //clCheck ( cuMemsetD8 ( gpuVar(&m_FluidTemp, FCONC),   0,	m_FParams.szPnts *sizeof(float[NUM_TF])   ), "ComputeGenesCL", "cuMemsetD8", "gpuVar(&m_FluidTemp, FCONC)",   mbDebug );
    status = clEnqueueFillBuffer(m_queue, gpuVar(&m_Fluid, FCONC), 0, sizeof(float[NUM_TF]), 0, m_GridTotal * sizeof(float[NUM_TF]), 0, NULL, NULL);

    //clCheck ( cuMemsetD8 ( gpuVar(&m_FluidTemp, FEPIGEN), 0,	m_FParams.szPnts *sizeof(uint[NUM_GENES]) ), "ComputeGenesCL", "cuMemsetD8", "gpuVar(&m_FluidTemp, FEPIGEN)", mbDebug );
    status = clEnqueueFillBuffer(m_queue, gpuVar(&m_Fluid, FEPIGEN), 0, sizeof(uint[NUM_GENES]), 0, m_GridTotal * sizeof(uint[NUM_GENES]), 0, NULL, NULL);

    for (int gene=0;gene<NUM_GENES;gene++) {
        uint list_length = bufI(&m_Fluid, FDENSE_LIST_LENGTHS)[gene];

        void* args[3] = { &m_FParams.pnumActive, &gene, &list_length };
        status  = clSetKernelArg(m_Kern[FUNC_COMPUTE_GENE_ACTION], 0, sizeof(int), &m_FParams.pnumActive);
        status  = clSetKernelArg(m_Kern[FUNC_COMPUTE_GENE_ACTION], 1, sizeof(int), &gene);
        status  = clSetKernelArg(m_Kern[FUNC_COMPUTE_GENE_ACTION], 2, sizeof(uint), &list_length);

        size_t numGroups, numItems;
        computeNumBlocks ( list_length , local_work_size, numGroups, numItems);


        size_t global_work_size = numGroups * numItems;
        size_t local_work_size = numItems;

        if (verbosity>1) std::cout<<"\nComputeGenesCL (): gene ="<<gene<<", list_length="<<list_length<<", m_FParams.itemsPerGroup="<<m_FParams.itemsPerGroup<<", numGroups="<<numGroups<<",  numItems="<<numItems<<". args={mNumPoints="<<mNumPoints<<", list_length="<<list_length<<", gene ="<<gene<<"}"<<std::flush;

        if( numGroups>0 && numItems>0){
            //std::cout<<"\nCalling m_Kern[FUNC_COMPUTE_GENE_ACTION], list_length="<<list_length<<", numGroups="<<numGroups<<", numItems="<<numItems<<"\n"<<std::flush;
            //clCheck ( cuLaunchKernel ( m_Kern[FUNC_COMPUTE_GENE_ACTION],  numGroups, 1, 1, numItems, 1, 1, 0, NULL, args, NULL), "ComputeGenesCL", "cuLaunch", "FUNC_COMPUTE_GENE_ACTION", mbDebug);
            status = clEnqueueNDRangeKernel(m_queue, m_Kern[FUNC_COMPUTE_GENE_ACTION], 1, NULL, &global_work_size, &local_work_size, 0, NULL, NULL);
        }
    }
    clCheck(clFinish(m_queue), "ComputeGenesCL", "clFinish", "After FUNC_COMPUTE_GENE_ACTION & before FUNC_TALLY_GENE_ACTION", mbDebug);
    for (int gene=0;gene<NUM_GENES;gene++) {
        uint list_length = bufI(&m_Fluid, FDENSE_LIST_LENGTHS)[gene];
        void* args[3] = { &m_FParams.pnumActive, &gene, &list_length };
        status  = clSetKernelArg(m_Kern[FDENSE_LIST_LENGTHS], 0, sizeof(int), &m_FParams.pnumActive);
        status  = clSetKernelArg(m_Kern[FDENSE_LIST_LENGTHS], 1, sizeof(int), &gene);
        status  = clSetKernelArg(m_Kern[FDENSE_LIST_LENGTHS], 2, sizeof(uint), &list_length);

        size_t numGroups, numItems;
        computeNumBlocks ( list_length , local_work_size, numGroups, numItems);

        size_t global_work_size = numGroups * numItems;
        size_t local_work_size = numItems;

        if( numGroups>0 && numItems>0){
            if (verbosity>1) std::cout<<"\nCalling m_Kern[FUNC_TALLY_GENE_ACTION], gene="<<gene<<", list_length="<<list_length<<", numGroups="<<numGroups<<", numItems="<<numItems<<"\n"<<std::flush;
            //clCheck ( cuLaunchKernel ( m_Kern[FUNC_TALLY_GENE_ACTION],  numGroups, 1, 1, numItems, 1, 1, 0, NULL, args, NULL), "ComputeGenesCL", "cuLaunch", "FUNC_TALLY_GENE_ACTION", mbDebug);
            status = clEnqueueNDRangeKernel(m_queue, m_Kern[FUNC_TALLY_GENE_ACTION], 1, NULL, &global_work_size, &local_work_size, 0, NULL, NULL);

        }
    }
}

void FluidSystem::AssembleFibresCL (){  //kernel: void assembleMuscleFibres ( int pnum, uint list, uint list_length )
    if (verbosity>1)cout << "\n\nAssembleFibresCL ()\n"<<std::flush;
    cl_int status;
    uint gene = 7; // muscle
    uint list_length = bufI(&m_Fluid, FDENSE_LIST_LENGTHS)[gene];
    void* args[3] = { &m_FParams.pnumActive, &gene, &list_length };
    status  = clSetKernelArg(m_Kern[FDENSE_LIST_LENGTHS], 0, sizeof(int), &m_FParams.pnumActive);
    status  = clSetKernelArg(m_Kern[FDENSE_LIST_LENGTHS], 1, sizeof(int), &gene);
    status  = clSetKernelArg(m_Kern[FDENSE_LIST_LENGTHS], 2, sizeof(uint), &list_length);
    size_t numGroups, numItems;
    computeNumBlocks ( list_length , local_work_size, numGroups, numItems);

    /*CUCLCUCL Kein KERNEL?!
    if( numGroups>0 && numItems>0){
        clCheck ( cuLaunchKernel ( m_Kern[FUNC_ASSEMBLE_MUSCLE_FIBRES_OUTGOING],  numGroups, 1, 1, numItems, 1, 1, 0, NULL, args, NULL), "ComputeGenesCL", "cuLaunch", "FUNC_COMPUTE_GENE_ACTION", mbDebug);
    }
    */

    clCheck(clFinish(m_queue), "Run", "clFinish", "In AssembleFibresCL, after OUTGOING", mbDebug);

    /*
    if( numGroups>0 && numItems>0){
        clCheck ( cuLaunchKernel ( m_Kern[FUNC_ASSEMBLE_MUSCLE_FIBRES_INCOMING],  numGroups, 1, 1, numItems, 1, 1, 0, NULL, args, NULL), "ComputeGenesCL", "cuLaunch", "FUNC_COMPUTE_GENE_ACTION", mbDebug);
    }
    clCheck(clFinish(), "Run", "clFinish", "In AssembleFibresCL, after OUTGOING", mbDebug);
    */




    // Kernels:  call by tissue type using dense lists by gene.
    //assembleMuscleFibres()
    //assembleFasciaFibres ()
    if (verbosity>1) cout << "\nFinished AssembleFibresCL ()\n\n"<<std::flush;
}

void FluidSystem::ComputeBondChangesCL (uint steps_per_InnerPhysicalLoop){// Given the action of the genes, compute the changes to particle properties & splitting/combining  NB also "inserts changes"
//  if (verbosity>1)printf("\n gpuVar(&m_Fluid, FBIN_OFFSET_CHANGES)=%llu   ,\t gpuVar(&m_Fluid, FBIN_COUNT_CHANGES)=%llu   \n",gpuVar(&m_Fluid, FBIN_OFFSET_CHANGES) , gpuVar(&m_Fluid, FBIN_COUNT_CHANGES)   );
    cl_int status;
    //clCheck ( cuMemsetD8 ( gpuVar(&m_Fluid, FBIN_OFFSET_CHANGES), 0,	m_GridTotal *sizeof(uint[NUM_CHANGES]) ), "ComputeBondChangesCL", "cuMemsetD8", "FBIN_OFFSET", mbDebug );
    status = clEnqueueFillBuffer(m_queue, gpuVar(&m_Fluid, FBIN_OFFSET_CHANGES), 0, sizeof(uint[NUM_CHANGES]), 0, m_GridTotal * sizeof(uint[NUM_CHANGES]), 0, NULL, NULL);
                                            //NB list for all living cells. (non senescent) = FEPIGEN[2]
    //clCheck ( cuMemsetD8 ( gpuVar(&m_Fluid, FBIN_COUNT_CHANGES), 0,	m_GridTotal *sizeof(uint[NUM_CHANGES]) ), "ComputeBondChangesCL", "cuMemsetD8", "FBIN_COUNT", mbDebug );
    status = clEnqueueFillBuffer(m_queue, gpuVar(&m_Fluid, FBIN_COUNT_CHANGES), 0, sizeof(uint[NUM_CHANGES]), 0, m_GridTotal * sizeof(uint[NUM_CHANGES]), 0, NULL, NULL);

    uint list_length = bufI(&m_Fluid, FDENSE_LIST_LENGTHS)[2];    // call for dense list of living cells (gene'2'living/telomere (has genes))
    void* args[3] = { &mActivePoints, &list_length, &steps_per_InnerPhysicalLoop};
    status  = clSetKernelArg(m_Kern[FUNC_COMPUTE_BOND_CHANGES], 0, sizeof(int), &mActivePoints);
    status  = clSetKernelArg(m_Kern[FUNC_COMPUTE_BOND_CHANGES], 1, sizeof(int), &list_length);
    status  = clSetKernelArg(m_Kern[FUNC_COMPUTE_BOND_CHANGES], 2, sizeof(uint), &steps_per_InnerPhysicalLoop);

    size_t numGroups, numItems;
    computeNumBlocks (list_length, local_work_size, numGroups, numItems);

    size_t global_work_size = numGroups * numItems;
    size_t local_work_size = numItems;

    //std::cout<<"\n\nComputeBondChangesCL (): verbosity = "<<verbosity<<", (verbosity>1)="<<(verbosity>1)<<"\n"<<std::flush;

    if (verbosity>1) std::cout<<"\n\nComputeBondChangesCL (): list_length="<<list_length<<", m_FParams.itemsPerGroup="<<m_FParams.itemsPerGroup<<", numGroups="<<numGroups<<",  numItems="<<numItems<<". \t\t args={mActivePoints="<<mActivePoints<<", list_length="<<list_length<<"}\n\n"<<std::flush;

    //clCheck ( cuLaunchKernel ( m_Kern[FUNC_COMPUTE_BOND_CHANGES],  m_FParams.numGroups, 1, 1, m_FParams.numItems, 1, 1, 0, NULL, args, NULL), "computeBondChanges", "cuLaunch", "FUNC_COMPUTE_BOND_CHANGES", mbDebug);

    status = clEnqueueNDRangeKernel(m_queue, m_Kern[FUNC_COMPUTE_BOND_CHANGES], 1, NULL, &global_work_size, &local_work_size, 0, NULL, NULL);

}

void FluidSystem::ComputeParticleChangesCL (){// Call each for dense list to execute particle changes. NB Must run concurrently without interfering => no clFinish()
    cl_int status;
    uint startNewPoints = mActivePoints + 1;
    if (verbosity>2)printf("\n");
    for (int change_list = 0; change_list<NUM_CHANGES;change_list++){
    //int change_list = 0; // TODO debug, chk one kernel at a time
        uint list_length = bufI(&m_Fluid, FDENSE_LIST_LENGTHS_CHANGES)[change_list];  // num blocks and threads by list length
        //uint list_length = bufI(&m_Fluid, FDENSE_BUF_LENGTHS_CHANGES)[change_list];
        //if (change_list!=0 && change_list!=1)continue; // only test heal() and lengthenTissue() for now.

        if (verbosity>2)printf("\n\nComputeParticleChangesCL(): startNewPoints=%u, change_list=%u, list_length=%u, mMaxPoints=%u \t",
            startNewPoints, change_list, list_length, mMaxPoints);

        if ((change_list >0)&&(startNewPoints + list_length > mMaxPoints)){         // NB heal() does not create new bonds.
            printf("\n\n### Run out of spare particles. startNewPoints=%u, change_list=%u, list_length=%u, mMaxPoints=%u ###\n",
            startNewPoints, change_list, list_length, mMaxPoints);
            list_length = mMaxPoints - startNewPoints;
            exit_(status);
        }//

        void* args[5] = {&mActivePoints, &list_length, &change_list, &startNewPoints, &mMaxPoints};

        status  = clSetKernelArg(m_Kern[FUNC_HEAL+change_list], 0, sizeof(int), &mActivePoints);
        status  = clSetKernelArg(m_Kern[FUNC_HEAL+change_list], 1, sizeof(uint), &list_length);
        status  = clSetKernelArg(m_Kern[FUNC_HEAL+change_list], 2, sizeof(int), &change_list);
        status  = clSetKernelArg(m_Kern[FUNC_HEAL+change_list], 3, sizeof(uint), &startNewPoints);
        status  = clSetKernelArg(m_Kern[FUNC_HEAL+change_list], 4, sizeof(int), &mMaxPoints);

        size_t numItems, numGroups;

        //int numItems = 1;//m_FParams.itemsPerGroup;
        //int numGroups  = 1;//iDivUp ( list_length, numItems );

        computeNumBlocks (list_length, local_work_size, numGroups, numItems);

        size_t global_work_size = m_FParams.numGroups * m_FParams.numItems;
        size_t local_work_size = m_FParams.numItems;

        if (verbosity>2) std::cout
            <<"\nComputeParticleChangesCL ():"
            <<" frame ="                    <<m_FParams.frame
            <<", mActivePoints="            <<mActivePoints
            <<", change_list ="             <<change_list
            <<", list_length="              <<list_length
            <<", m_FParams.itemsPerGroup="<<m_FParams.itemsPerGroup
            <<", numGroups="                <<numGroups
            <<", numItems="               <<numItems
            <<". args={mActivePoints="      <<mActivePoints
            <<", list_length="              <<list_length
            <<", change_list="              <<change_list
            <<", startNewPoints="           <<startNewPoints
            <<"\t"<<std::flush;

        if( (list_length>0) && (numGroups>0) && (numItems>0)){
            if (verbosity>0) std::cout
                <<"\nComputeParticleChangesCL ():"
                <<"\tCalling m_Kern[FUNC_HEAL+"             <<change_list
                <<"], list_length="                         <<list_length
                <<", numGroups="                            <<numGroups
                <<", numItems="                           <<numItems
                <<",\t m_FParams.itemsPerGroup="          <<m_FParams.itemsPerGroup
                <<", numGroups*m_FParams.itemsPerGroup="  <<numGroups*m_FParams.itemsPerGroup
                <<"\t"<<std::flush;

            //clCheck ( cuLaunchKernel ( m_Kern[FUNC_HEAL+change_list], numGroups, 1, 1, numItems, 1, 1, 0, NULL, args, NULL), "ComputeParticleChangesCL", "cuLaunch", "FUNC_HEAL+change_list", mbDebug);
            clCheck(clEnqueueNDRangeKernel(m_queue, m_Kern[FUNC_HEAL + change_list], 1, NULL, &global_work_size, &local_work_size, 0, NULL, NULL), "ComputeParticleChangesCL", "cuLaunch", "FUNC_HEAL+change_list", mbDebug);
        }
        clCheck(clFinish(m_queue), "ComputeParticleChangesCL", "clFinish", "In ComputeParticleChangesCL", mbDebug);
                                                                                // Each thread will pick different new particles from surplus particles.
        if (change_list==2 || change_list==6) startNewPoints+=  list_length;    // Increment by num new particles used by previous kernels.
        //if (change_list==1 || change_list==5) startNewPoints+=  list_length*3;  // Increment by 3 particles for muscle.
        /*
    0   #define FUNC_HEAL                       23 //heal
    1   #define FUNC_LENGTHEN_MUSCLE            24 //lengthen_muscle
    2   #define FUNC_LENGTHEN_TISSUE            25 //lengthen_tissue
    3   #define FUNC_SHORTEN_MUSCLE             26 //shorten_muscle
    4   #define FUNC_SHORTEN_TISSUE             27 //shorten_tissue

    5   #define FUNC_STRENGTHEN_MUSCLE          28 //strengthen_muscle
    6   #define FUNC_STRENGTHEN_TISSUE          29 //strengthen_tissue
    7   #define FUNC_WEAKEN_MUSCLE              30 //weaken_muscle
    8   #define FUNC_WEAKEN_TISSUE              31 //weaken_tissue
         */
    }
    if (verbosity>1) std::cout<<"\nFinished ComputeParticleChangesCL ()\n"<<std::flush;
}

void FluidSystem::ZeroVelCL (){                                       // Used to remove velocity, kinetic energy and momentum during initialization.
    clCheck(clFinish(m_queue), "Run", "clFinish", "After freeze Run2PhysicalSort ", mbDebug);
    clCheck ( clMemsetD32 ( gpuVar(&m_Fluid, FVEL),   0.0,  mMaxPoints ),  "ZeroVelCL", "clMemsetD32", "FVEL",        mbDebug);
    clCheck ( clMemsetD32 ( gpuVar(&m_Fluid, FVEVAL), 0.0,  mMaxPoints ),  "ZeroVelCL", "clMemsetD32", "FVEVAL",      mbDebug);
    clCheck(clFinish(m_queue), "Run", "clFinish", "After freeze ZeroVelCL ", mbDebug);
}

void FluidSystem::AdvanceCL ( float tm, float dt, float ss ){

    cl_int status;
    cout<< "-------AdvanceCL() started... -------\n" << flush;

    //void* args[4] = { &tm, &dt, &ss, &m_FParams.pnumActive };
    clCheck(clSetKernelArg(m_Kern[FUNC_ADVANCE], 0, sizeof(cl_mem),     &m_FParamsDevice),      "AdvanceCL", "clSetKernelArg", "FUNC_ADVANCE 0", mbDebug);
    clCheck(clSetKernelArg(m_Kern[FUNC_ADVANCE], 1, sizeof(cl_mem),     &m_FluidDevice),        "AdvanceCL", "clSetKernelArg", "FUNC_ADVANCE 1", mbDebug);
    clCheck(clSetKernelArg(m_Kern[FUNC_ADVANCE], 2, sizeof(cl_mem),     &m_FluidTempDevice),    "AdvanceCL", "clSetKernelArg", "FUNC_ADVANCE 2", mbDebug);
    clCheck(clSetKernelArg(m_Kern[FUNC_ADVANCE], 3, sizeof(float),      &tm),                   "AdvanceCL", "clSetKernelArg", "FUNC_ADVANCE 3", mbDebug);
    clCheck(clSetKernelArg(m_Kern[FUNC_ADVANCE], 4, sizeof(float),      &dt),                   "AdvanceCL", "clSetKernelArg", "FUNC_ADVANCE 4", mbDebug);
    clCheck(clSetKernelArg(m_Kern[FUNC_ADVANCE], 5, sizeof(float),      &ss),                   "AdvanceCL", "clSetKernelArg", "FUNC_ADVANCE 5", mbDebug);
    clCheck(clSetKernelArg(m_Kern[FUNC_ADVANCE], 6, sizeof(int),        &m_FParams.pnumActive), "AdvanceCL", "clSetKernelArg", "FUNC_ADVANCE 6", mbDebug);
    //cout<<"\nAdvanceCL: m_FParams.pnumActive="<<m_FParams.pnumActive<<std::flush;

    computeNumBlocks ( m_FParams.pnumActive, local_work_size, m_FParams.numGroups, m_FParams.numItems); // particles


    //clCheck ( cuLaunchKernel ( m_Kern[FUNC_ADVANCE],  m_FParams.numGroups, 1, 1, m_FParams.numItems, 1, 1, 0, NULL, args, NULL), "AdvanceCL", "cuLaunch", "FUNC_ADVANCE", mbDebug);
    clCheck(clEnqueueNDRangeKernel(

        m_queue,
        m_Kern[FUNC_ADVANCE],
        1,
        NULL,
        &m_FParams.numItems,
        &m_FParams.numGroups,
        0,
        NULL,
        NULL
        ), "AdvanceCL", "clEnqueueNDRangeKernel", "FUNC_ADVANCE", mbDebug);

    cout<< "-------AdvanceCL() finished. -------\n" << flush;

}

void FluidSystem::SpecialParticlesCL (float tm, float dt, float ss){   // For interaction.Using dense lists for gene 1 & 0.

    cl_int status;
    int gene = 12;                                                           // 'externally actuated' particles
    uint list_length = bufI(&m_Fluid, FDENSE_LIST_LENGTHS)[gene];
    //void* args[5] = {&list_length, &tm, &dt, &ss, &m_FParams.pnumActive};         // void externalActuation (uint list_len,  float time, float dt, float ss, int numPnts )
    status  = clSetKernelArg(m_Kern[FUNC_EXTERNAL_ACTUATION], 0, sizeof(uint),     &list_length);
    status  = clSetKernelArg(m_Kern[FUNC_EXTERNAL_ACTUATION], 1, sizeof(float),    &tm);
    status  = clSetKernelArg(m_Kern[FUNC_EXTERNAL_ACTUATION], 2, sizeof(float),    &dt);
    status  = clSetKernelArg(m_Kern[FUNC_EXTERNAL_ACTUATION], 3, sizeof(float),    &ss);
    status  = clSetKernelArg(m_Kern[FUNC_EXTERNAL_ACTUATION], 4, sizeof(int),      &m_FParams.pnumActive);

    size_t numGroups, numItems;
    computeNumBlocks ( list_length , local_work_size, numGroups, numItems);

    size_t global_work_size = numGroups * numItems;
    size_t local_work_size = numItems;

    if (verbosity>1) std::cout<<"\nSpecialParticlesCL:EXTERNAL_ACTUATION: list_length="<<list_length<<" , m_FParams.itemsPerGroup="<< m_FParams.itemsPerGroup <<", numGroups="<< numGroups <<", numItems="<< numItems <<", args{m_FParams.pnum="<< m_FParams.pnum <<",  gene="<< gene <<", list_length="<< list_length <<"  }  \n"<<std::flush;

    if( numGroups>0 && numItems>0){
        if (verbosity>1) std::cout<<"\nCalling m_Kern[FUNC_EXTERNAL_ACTUATION], list_length="<<list_length<<", numGroups="<<numGroups<<", numItems="<<numItems<<"\n"<<std::flush;
        //clCheck ( cuLaunchKernel ( m_Kern[FUNC_EXTERNAL_ACTUATION],  numGroups, 1, 1, numItems, 1, 1, 0, NULL, args, NULL), "SpecialParticlesCL", "cuLaunch", "FUNC_EXTERNAL_ACTUATION", mbDebug);

        status = clEnqueueNDRangeKernel(m_queue, m_Kern[FUNC_EXTERNAL_ACTUATION], 1, NULL, &global_work_size, &local_work_size, 0, NULL, NULL);

    }



    gene = 11;                                                                // 'fixed' particles
    list_length = bufI(&m_Fluid, FDENSE_LIST_LENGTHS)[gene];
    //args[0] = &list_length;                                                 // void fixedParticles (uint list_len, int numPnts )
    //args[1] = &m_FParams.pnum;
    status  = clSetKernelArg(m_Kern[FUNC_FIXED], 0, sizeof(uint),  &list_length);
    status  = clSetKernelArg(m_Kern[FUNC_FIXED], 1, sizeof(int),   &m_FParams.pnum);

    computeNumBlocks ( list_length , local_work_size, numGroups, numItems);

    if (verbosity>1) std::cout<<"\nSpecialParticlesCL:FIXED: list_length="<<list_length<<" , m_FParams.itemsPerGroup="<< m_FParams.itemsPerGroup <<", numGroups="<< numGroups <<", numItems="<< numItems <<", args{m_FParams.pnum="<< m_FParams.pnum <<",  gene="<< gene <<", list_length="<< list_length <<"  }  \n"<<std::flush;

    if( numGroups>0 && numItems>0){
        if (verbosity>1) std::cout<<"\nCalling m_Kern[FUNC_FIXED], list_length="<<list_length<<", numGroups="<<numGroups<<", numItems="<<numItems<<"\n"<<std::flush;
        //clCheck ( cuLaunchKernel ( m_Kern[FUNC_FIXED],  numGroups, 1, 1, numItems, 1, 1, 0, NULL, args, NULL), "SpecialParticlesCL", "cuLaunch", "FUNC_FIXED", mbDebug);
        status = clEnqueueNDRangeKernel(m_queue, m_Kern[FUNC_FIXED], 1, NULL, &global_work_size, &local_work_size, 0, NULL, NULL);
    }
}

void FluidSystem::EmitParticlesCL ( float tm, int cnt ){
    cl_int status;

    //void* args[3] = { &tm, &cnt, &m_FParams.pnum };

    status  = clSetKernelArg(m_Kern[FUNC_EMIT], 0, sizeof(float),  &tm);
    status  = clSetKernelArg(m_Kern[FUNC_EMIT], 1, sizeof(int),    &cnt);
    status  = clSetKernelArg(m_Kern[FUNC_EMIT], 2, sizeof(int),    &m_FParams.pnum);


    size_t global_work_size = m_FParams.numGroups * m_FParams.numItems;
    size_t local_work_size = m_FParams.numItems;

    //clCheck ( cuLaunchKernel ( m_Kern[FUNC_EMIT],  m_FParams.numGroups, 1, 1, m_FParams.numItems, 1, 1, 0, NULL, args, NULL), "EmitParticlesCL", "cuLaunch", "FUNC_EMIT", mbDebug);

    status = clEnqueueNDRangeKernel(m_queue, m_Kern[FUNC_EMIT], 1, NULL, &global_work_size, &local_work_size, 0, NULL, NULL);

}

