//----------------------------------------------------------------------------------
//
// FLUIDS v.3 - SPH Fluid Simulator for CPU and GPU
// Copyright (C) 2012-2013. Rama Hoetzlein, http://fluids3.com
//
// BSD 3-clause:
// Redistribution and use in source and binary forms, with or without modification,
// are permitted provided that the following conditions are met:
// 1. Redistributions of source code must retain the above copyright notice, this
//    list of conditions and the following disclaimer.
// 2. Redistributions in binary form must reproduce the above copyright notice, this
//    list of conditions and the following disclaimer in the documentation and/or
//    other materials provided with the distribution.
// 3. Neither the name of the copyright holder nor the names of its contributors may
//    be used to endorse or promote products derived from this software without specific
//   prior written permission.
//
// THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS" AND ANY
// EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE IMPLIED WARRANTIES
// OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE DISCLAIMED. IN NO EVENT
// SHALL THE COPYRIGHT HOLDER OR CONTRIBUTORS BE LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL,
// SPECIAL, EXEMPLARY, OR CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT
// OF SUBSTITUTE GOODS OR SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION)
// HOWEVER CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY, OR
// TORT (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE OF THIS SOFTWARE,
// EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
//----------------------------------------------------------------------------------

#define CL_KERNEL
#define SCAN_BLOCKSIZE		128
#define GRID_UNDEF			4294967295
//#define FLT_MIN  0.000000001              // set here as 2^(-30) //REDEFINED
//#define UINT_MAX 65535    //REDEFINED
#define A 48271
#define M 2147483647
#define Q (M / A)
#define R (M % A)

#pragma OPENCL EXTENSION cl_khr_fp64: enable
#pragma OPENCL EXTENSION cl_khr_int64_base_atomics: enable
#pragma OPENCL EXTENSION cl_khr_global_int32_base_atomics: enable
#pragma OPENCL EXTENSIONcl_khr_global_int32_extended_atomics: enable
#pragma OPENCL EXTENSION cl_khr_local_int32_base_atomics: enable
#pragma OPENCL EXTENSION cl_khr_local_int32_extended_atomics: enable
#pragma OPENCL_VERSION 120


#include "../fluid_system_opencl.h"
#include "../fluid.h"
#include "../randCL_well512.cl"


// __constant struct FParams	fparam  = {
//         .debug = 2,
// 		.numItems = 0, .numGroups = 0, .itemsPerGroup = 0,
// 		.gridThreads = 0, .gridBlocks = 0,
// 		.szPnts = 0,
// 		.szGrid = 0,
// 		.stride = 0, .pnum = 0, .pnumActive = 0, .maxPoints = 0,
//         .freeze = 0,
//         .freezeBoolToInt = 0,
//         .frame = 0,
// 		.chk = 0,
// 		.pdist = 0, .pmass = 0, .prest_dens = 0,
// 		.pextstiff = 0, .pintstiff = 0,
// 		.pradius = 0, .psmoothradius = 0, .r2 = 0, .psimscale = 0, .pvisc = 0, .psurface_t = 0,
// 		.pforce_min = 0, .pforce_max = 0, .pforce_freq = 0, .pground_slope = 0,
// 		.pvel_limit = 0, .paccel_limit = 0, .pdamp = 0,
// 		.pboundmin = 0, .pboundmax = 0, .pgravity = 0,
// 		.AL = 0, .AL2 = 0, .VL = 0, .VL2 = 0,
// 		.H = 0, .d2 = 0, .rd2 = 0, .vterm = 0,		// used in force calculation
// 		.poly6kern = 0, .spikykern = 0, .lapkern = 0, .gausskern = 0, .wendlandC2kern = 0,
// 		.gridSize = 0, .gridDelta = 0, .gridMin = 0, .gridMax = 0,
// 		.gridRes = 0, .gridScanMax = 0,
// 		.gridSrch = 0, .gridTotal = 0, .gridAdjCnt = 0, .gridActive = 0,
//         .actuation_factor = 0,
//         .actuation_period = 0,
// 		.gridAdj = {0, 0, 0, 0, 0, 0, 0, 0,
//                     0, 0, 0, 0, 0, 0, 0, 0,
//                     0, 0, 0, 0, 0, 0, 0, 0,
//                     0, 0, 0, 0, 0, 0, 0, 0,
//                     0, 0, 0, 0, 0, 0, 0, 0,
//                     0, 0, 0, 0, 0, 0, 0, 0,
//                     0, 0, 0, 0, 0, 0, 0, 0,
//                     0, 0, 0, 0, 0, 0, 0, 0}
// 	};			// CPU Fluid params
__constant struct FBufs			fbuf = {};
			// GPU Particle buffers (unsorted). An FBufs struct holds an array of pointers.
__constant struct FBufs			ftemp = {};			// GPU Particle buffers (sorted)

__constant struct FGenome		fgenome = {};		// GPU Genome for particle automata behaviour. Also holds morphogen diffusability.

__constant struct FPrefix       fprefix = {};
//__constant_FBondParams    fbondparams;    // GPU copy of remodelling parameters.
//__constant uint			gridActive;


///////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
///////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
///////////////////////////////////////////////////////////////// Kernels /////////////////////////////////////////////////////////////////////////////
///////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
///////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////


__kernel void memset32d_kernel(

    volatile __global int* buffer,
    const int value
    )
{
    int gid = get_global_id(0);
    buffer[gid] = value;
}
/*
__kernel void insertParticlesCL(
    __global struct FParams* m_FParamsDevice,
    int pnum,
    volatile __global float4* fpos,
    volatile __global uint* fgcell,
    volatile __global uint* fgndx,
    volatile __global uint* fepigen,
    volatile __global int* fgridcnt,
    volatile __global int* fgridcnt_active_genes
    )
{
    uint i = get_global_id(0);
    if ( i >= pnum ) return;

    float4  gridMin     =  m_FParamsDevice->gridMin;
    float4  gridDelta   =  m_FParamsDevice->gridDelta;
    int4    gridRes     =  m_FParamsDevice->gridRes;
    int4    gridScan    =  m_FParamsDevice->gridScanMax;
    int     gridTot     =  m_FParamsDevice->gridTotal;

    float4 gcf = (fpos[i] - gridMin) * gridDelta;

    printf("GCF after Calculation: Thread ID: %u, gcf: (%f, %f, %f), gridDelta: (%f, %f, %f)\n",
        i, gcf.x, gcf.y, gcf.z, gridDelta.x, gridDelta.y, gridDelta.z);
    int4 gc  = (int4)(gcf.x, gcf.y, gcf.z, gcf.w);
    int gs  = (gc.y * gridRes.z + gc.z) * gridRes.x + gc.x;

//     printf("Thread ID: %u, m_FParamsDevice->gridDelta: (%f, %f, %f)\n",
//         i, m_FParamsDevice->gridDelta.x, m_FParamsDevice->gridDelta.y, m_FParamsDevice->gridDelta.z);

    printf("Thread ID: %u, float3 fpos[%u]: (%v4hlf)\n",
        i, i, fpos[i]);

// printf("Thread ID: %u, m_FParamsDevice->gridTotal: (%u)\n",
//        i, m_FParamsDevice->gridTotal);

    if ( gc.x >= 1 && gc.x <= gridScan.x && gc.y >= 1 && gc.y <= gridScan.y && gc.z >= 1 && gc.z <= gridScan.z ) {

		fgcell[i] = gs;                                    // Grid cell insert.
		printf("InsideLoop: Thread ID: %u, fgcell: (%u)\n",             // all zero (0)
        i, fgcell[i]);
        fgndx[i] = atomic_add(&fgridcnt[gs], 1);          // Grid counts.
                                                                                                  //  ## add counters for dense lists. ##############
        // for each gene, if active, then atomicAdd bin count for gene
        for(int gene=0; gene<NUM_GENES; gene++){ // NB data ordered FEPIGEN[gene][particle] AND +ve int values -> active genes.
            //if(m_FParamsDevice->debug>2 && i==0)printf("\n");
            if (fepigen[i + gene * m_FParamsDevice->maxPoints] > 0) {  // "if((int)bufI(m_FluidDevice, FEPIGEN)" may clash with INT_MAX
                atomic_add( &fgridcnt_active_genes[gene*gridTot +gs], 1 );
            }
        }
    } else {
        fgcell[i] = GRID_UNDEF;
        printf("Thread ID: %u, GRID_UNDEF\n",i);

    }

    printf("Thread ID: %u, gc: (%d, %d, %d), Grid Cell: %d\n",
        i, gc.x, gc.y, gc.z, gs);

//     printf("Thread ID: %u, fgcell: (%u)\n",
//         i, fgcell[i]);
}*/


__kernel void insertParticlesCL(
    __global struct FParams* m_FParamsDevice,
    int pnum,
    volatile __global float4* fpos,
    volatile __global uint* fgcell,
    volatile __global uint* fgndx,
    volatile __global uint* fepigen,
    volatile __global int* fgridcnt,
    volatile __global int* fgridcnt_active_genes
    )
{
uint i = get_global_id(0);
    if ( i >= pnum ) return;

    float4  gridMin     =  m_FParamsDevice->gridMin;
    float4  gridDelta   =  m_FParamsDevice->gridDelta;
    int4    gridRes     =  m_FParamsDevice->gridRes;
    int4    gridScan    =  m_FParamsDevice->gridScanMax;
    int     gridTot     =  m_FParamsDevice->gridTotal;

    float4 gcf = (fpos[i] - gridMin) * gridDelta;

    printf("GCF after Calculation: Thread ID: %u, gcf: (%f, %f, %f), gridDelta: (%f, %f, %f)\n",
        i, gcf.x, gcf.y, gcf.z, gridDelta.x, gridDelta.y, gridDelta.z);
    int4 gc  = (int4)(gcf.x, gcf.y, gcf.z, gcf.w);
    int gs  = (gc.y * gridRes.z + gc.z) * gridRes.x + gc.x;

    printf("Thread ID: %u, m_FParamsDevice->gridDelta: (%f, %f, %f)\n",
        i, m_FParamsDevice->gridDelta.x, m_FParamsDevice->gridDelta.y, m_FParamsDevice->gridDelta.z);

    printf("Thread ID: %u, float4 fpos[%u]: (%v4hlf)\n",
        i, i, fpos[i]);

    if(i==pnum-1) printf("\n\ninsertParticles()1: gridTot=%i,  i=%u: gc.x=%i, gc.y=%i, gc.z=%i, gs=%i \t gridScan.x=%i, gridScan.y=%i, gridScan.z=%i, gridTot=%u,\t gridDelta=(%f,%f,%f) gridMin=(%f,%f,%f) gridRes=(%i,%i,%i)",
    gridTot, i, gc.x, gc.y, gc.z, gs,  gridScan.x, gridScan.y, gridScan.z, gridTot, gridDelta.x, gridDelta.y, gridDelta.z,  gridMin.x, gridMin.y, gridMin.z, gridRes.x, gridRes.y, gridRes.z );

    if ( gc.x >= 1 && gc.x <= gridScan.x && gc.y >= 1 && gc.y <= gridScan.y && gc.z >= 1 && gc.z <= gridScan.z ) {

		fgcell[i] = gs;                                    // Grid cell insert.

        printf("Thread ID: %u, fgridcnt[%d] before atomic_add: %d\n", i, gs, fgridcnt[gs]);
        fgndx[i] = atomic_add(&fgridcnt[gs], 1);          // Grid counts.
        printf("Thread ID: %u, fgridcnt[%d] after atomic_add: %d\n", i, gs, fgridcnt[gs]);

        // Index of each particle within its bin.
        printf("Thread ID: %u, fgndx: %u\n", i, fgndx[i]);

        for(int gene=0; gene<NUM_GENES; gene++){ // NB data ordered FEPIGEN[gene][particle] AND +ve int values -> active genes.
            //if(m_FParamsDevice->debug>2 && i==0)printf("\n");
            if (fepigen[i + gene * m_FParamsDevice->maxPoints] > 0) {  // "if((int)bufI(m_FluidDevice, FEPIGEN)" may clash with INT_MAX
                atomic_add( &fgridcnt_active_genes[gene*gridTot +gs], 1 );
            }

        }
    } else {
        fgcell[i] = GRID_UNDEF;
        printf("Thread ID: %u, GRID_UNDEF\n",i);

    }

    // Synchronize to ensure all work items have finished
    barrier(CLK_GLOBAL_MEM_FENCE);
}

__kernel void prefixUp(

    __global uint *input,
    __global uint *aux,
    int len
    )
{

    unsigned int t = get_local_id(0);
    unsigned int start = t + 2 * get_group_id(0) * SCAN_BLOCKSIZE;
    if (start < len)					input[start] += aux[get_group_id(0)];
    if (start + SCAN_BLOCKSIZE < len)   input[start + SCAN_BLOCKSIZE] += aux[get_group_id(0)];
}

__kernel void prefixSumChanges(

    __global uint *input,
    __global uint *output,
    __global uint *aux,
    __global uint *addInputOffset,
    __global uint *addOutputOffset,
              int  len,
              int  zeroff
        ) //, __constant size_t *offset_scan0)
{
    // Declares a locally shared, temporary array, with dimemsions 2 x BLOCKSIZE (1024 as of rn)
    __local uint scan_array[SCAN_BLOCKSIZE << 1];

    //-------------------------------------------------------------------------------------
    //      Assigning t1 and t2
    //
    // t1 represents a position for each work item, so does t2, but with an offset of SCAN_BLOCKSIZE
    //
    // Visualisation:
    //  t2[
    //      ITEM_1[0*2*BLOCKSIZE+256],
    //      ITEM_2[1*2*BLOCKSIZE+256],
    //      ITEM_3[2*2*BLOCKSIZE+256]
    //      [...]
    //      [...]...
    //  ]
    unsigned int t1 = get_global_id(0) * 2 * SCAN_BLOCKSIZE;
    unsigned int t2 = t1 + SCAN_BLOCKSIZE;

    //-------------------------------------------------------------------------------------
    //      Pre-load into shared memory
    // This also checks, if t1 or t2 exceed len = m_GridTotal (= 10000 as of rn).
    // If not, it asigns the corresponding element from the input[] array.
    //
    scan_array[get_local_id(0)] = (t1 < len) ? input[t1] : 0;
    scan_array[get_local_id(0) + SCAN_BLOCKSIZE] = (t2 < len) ? input[t2] : 0;
    barrier(CLK_LOCAL_MEM_FENCE);

    //-------------------------------------------------------------------------------------
    //      Reduction
    // Each iteration, the first half of the previously calculated threads will be added to the second half
    //
    int stride;
    for (stride = 1; stride <= SCAN_BLOCKSIZE; stride <<= 1) {
        int index = (get_local_id(0) + 1) * stride * 2 - 1;
        if (index < 2 * SCAN_BLOCKSIZE)
            scan_array[index] += scan_array[index - stride];
        barrier(CLK_LOCAL_MEM_FENCE);
    }

    // Post reduction
    for (stride = SCAN_BLOCKSIZE >> 1; stride > 0; stride >>= 1) {
        int index = (get_local_id(0) + 1) * stride * 2 - 1;
        if (index + stride < 2 * SCAN_BLOCKSIZE)
            scan_array[index + stride] += scan_array[index];
        barrier(CLK_LOCAL_MEM_FENCE);
    }
    barrier(CLK_LOCAL_MEM_FENCE);

    // Output values & aux
    if (t1 < len)    output[t1] = scan_array[get_local_id(0)];
    if (t2 < len)    output[t2] = (get_local_id(0) == SCAN_BLOCKSIZE - 1 && zeroff) ? 0 : scan_array[get_local_id(0) + SCAN_BLOCKSIZE];
    if (get_local_id(0) == 0) {
        if (zeroff) output[0] = 0;
        if (aux) aux[get_group_id(0)] = scan_array[2 * SCAN_BLOCKSIZE - 1];
    }

}

__kernel void prefixSum(

    __global uint *input,
    __global uint *output,
    __global uint *aux,
              int  len,
              int  zeroff
        )

{
    uint i = get_global_id(0);
    uint l = get_local_id(0);
    uint g = get_group_id(0);

    // Declares a locally shared, temporary array, with dimemsions 2 x BLOCKSIZE (256 as of rn)
    __local uint scan_array[SCAN_BLOCKSIZE << 1];
/*
                                                                                //-------------------------------------------------------------------------------------
                                                                                //      Assigning t1 and t2
                                                                                //
                                                                                // t1 represents a position for each work item, so does t2, but with an offset of SCAN_BLOCKSIZE
                                                                                //
                                                                                // Visualisation:
                                                                                //  t2[
                                                                                //      ITEM_1[0 + 0*2*BLOCKSIZE],
                                                                                //      ITEM_2[1 + 1*2*BLOCKSIZE],
                                                                                //      ITEM_3[2 + 2*2*BLOCKSIZE]
                                                                                //      [...]
                                                                                //      [...]...
                                                                                //  ]
*/
    uint t1 = get_local_id(0) + get_group_id(0) * SCAN_BLOCKSIZE * 2;

    //printf("XXXXXXXXXXXXXXXXXXXXX THREAD: %u, LOCAL ID: %u, GROUP ID: %u, t1: %u\n", i, l, g, t1);

    uint t2 = t1 + SCAN_BLOCKSIZE;
/*
                                                                                //-------------------------------------------------------------------------------------
                                                                                //      Pre-load into shared memory
                                                                                // This also checks, if t1 or t2 exceed len = m_GridTotal (= 10000 as of rn).
                                                                                // If not, it asigns the corresponding element from the input[] array.

//     if (input[t1] != 0) printf("INPUT ARRAY Thread [%u]: %u. \t t1 = %u\n", i, input[t1], t1);            //Tells you which bins contain particles and how many.
//     if (input[t2] != 0) printf("INPUT ARRAY Thread [%u]: %u. \t t2 = %u\n", i, input[t2],  t2);            //Tells you which bins contain particles and how many.
//
*/
    scan_array[get_local_id(0)] = (t1 < len) ? input[t1] : 0;
    barrier(CLK_LOCAL_MEM_FENCE);
/*
//     if (l == 8) printf("scan_array[get_local_id(0)] = %u\n", scan_array[get_local_id(0)]);
    // Print value of scan_array[1543]
//         if (t1 == 1544) {
//             printf("Value of scan_array[2044] after assignment: %u, Local ID: %u\n", scan_array[l], l);
//         }

                                                                                    // Print out non-zero elements of scan_array
                                                                                //     for (int j = 0; j < 2 * SCAN_BLOCKSIZE; ++j) {
                                                                                //         if (scan_array[j] != 0) {
                                                                                //                                                                         printf("--------- NON ZERO Thread %u: SCAN_ARRAY[%d] = %u\n", i, j, scan_array[j]);
                                                                                //         }
                                                                                //     }

*/
    scan_array[get_local_id(0) + SCAN_BLOCKSIZE] = (t2 < len) ? input[t2] : 0;
    barrier(CLK_LOCAL_MEM_FENCE);
/*
                                                                                //-------------------------------------------------------------------------------------
                                                                                //      Reduction
                                                                                // Each iteration, the first half of the previously calculated threads will be added to the second half
                                                                                //
*/
    int stride;
    for (stride = 1; stride <= SCAN_BLOCKSIZE; stride <<= 1) {

        int index = (get_local_id(0) + 1) * stride * 2 - 1;
        //printf("INDEX = %d\n", i);
        if (index < 2 * SCAN_BLOCKSIZE)
            scan_array[index] += scan_array[index - stride];
            //if (scan_array[index] != 0) printf("scan_array[%u] = %u\n", index, scan_array[index]);

        barrier(CLK_LOCAL_MEM_FENCE);
    }

    // Post reduction
    for (stride = SCAN_BLOCKSIZE >> 1; stride > 0; stride >>= 1) {
        int index = (get_local_id(0) + 1) * stride * 2 - 1;
        if (index + stride < 2 * SCAN_BLOCKSIZE)
            scan_array[index + stride] += scan_array[index];

        // Buffer out (if thread < 16, print thread num and the data)
        barrier(CLK_LOCAL_MEM_FENCE);
    }
    //barrier(CLK_LOCAL_MEM_FENCE);

    // Output values & aux
    if (t1 + zeroff < len)    output[t1 + zeroff] = scan_array[get_local_id(0)];
/*
    //if (t1 + zeroff < len && scan_array[get_local_id(0)] != 0)    printf("!!! scan_array[%lu] = %u\n", get_local_id(0), scan_array[get_local_id(0)]);
    //     if (t1 + zeroff < len && output[t1] != 0)    output[t1 + zeroff] = printf("OUTPUT: %u\n",  output[t1]);
*/
    if (t2 + zeroff < len)    output[t2 + zeroff] = (get_local_id(0) == SCAN_BLOCKSIZE - 1 && zeroff) ? 0 : scan_array[get_local_id(0) + SCAN_BLOCKSIZE];
/*
//     if (t2 + zeroff < len && output[t2] != 0)    output[t2 + zeroff] = printf("OUTPUT: %u\n",  output[t2]);
*/
    if (get_local_id(0) == 0) {
        if (zeroff) output[0] = 0;
        if (aux) aux[get_group_id(0)] = scan_array[2 * SCAN_BLOCKSIZE - 1];
    }

}



__kernel void tally_denselist_lengths(
    __global struct FParams* m_FParamsDevice,
    __global struct FBufs* m_Fluid,
                int num_lists,
                int fdense_list_lengths,
                int fgridcnt,
                int fgridoff
    )
{
    uint list = get_group_id(0) * get_local_size(0) + get_local_id(0);
    if ( list >= num_lists ) return;
    int gridTot = m_FParamsDevice->gridTotal;
    bufI(m_Fluid, fdense_list_lengths)[list] = bufI(m_Fluid, fgridcnt)[(list+1)*gridTot -1] + bufI(m_Fluid, fgridoff)[(list+1)*gridTot -1];
}
/*
__kernel void countingSortFull(
    __global struct FParams*    m_FParamsDevice,
                  int pnum,
    volatile __global float4* fposTemp

    )
{
    uint i = get_global_id(0);

    // Print the value along with the thread index
    printf("\nThread Index: %d, Value: (%f, %f, %f, %f)\n", i, fposTemp[i].x, fposTemp[i].y, fposTemp[i].z, fposTemp[i].w);
}*/


__kernel void countingSortFull(
    __global struct FParams*    m_FParamsDevice,
                  int pnum,
    volatile __global float4* fbin,
    volatile __global uint*   fbin_offset,
    volatile __global float4* fpos,
    volatile __global float4* fvel,
    volatile __global float4* fveval,
    volatile __global float4* fforce,
    volatile __global float*  fpress,
    volatile __global float*  fdensity,
    volatile __global uint*   fage,
    volatile __global uint*   fcolor,
    volatile __global uint*   fgcell,
    volatile __global uint*   fgndx,
    volatile __global uint*   felastidx,
    volatile __global uint*   fparticleidx,
    volatile __global uint*   fparticle_id,
    volatile __global uint*   fmass_radius,
    volatile __global uint*   fnerveidx,
    volatile __global uint*   fepigen,
    volatile __global uint*   fconc,
    volatile __global float4* fposTemp,
    volatile __global float4* fvelTemp,
    volatile __global float4* fvevalTemp,
    volatile __global float4* fforceTemp,
    volatile __global float*  fpressTemp,
    volatile __global float*  fdensityTemp,
    volatile __global uint*   fageTemp,
    volatile __global uint*   fcolorTemp,
    volatile __global uint*   fgcellTemp,
    volatile __global uint*   fgndxTemp,
    volatile __global uint*   felastidxTemp,
    volatile __global uint*   fparticleidxTemp,
    volatile __global uint*   fparticle_idTemp,
    volatile __global uint*   fmass_radiusTemp,
    volatile __global uint*   fnerveidxTemp,
    volatile __global uint*   fepigenTemp,
    volatile __global uint*   fconcTemp
    )
{
    uint i = get_global_id(0);

    // Print the value along with the thread index
    //printf("Thread Index: %d, Value: (%f, %f, %f, %f)\n", i, fposTemp[i].x, fposTemp[i].y, fposTemp[i].z, fposTemp[i].w);

if(get_global_id(0) == 1) {
        printf("Thread Index: %d\n", i);
        printf("fbin: (%f, %f, %f, %f)\n", fbin[i].x, fbin[i].y, fbin[i].z, fbin[i].w);
        printf("fbin_offset: %u\n", fbin_offset[i]);
        printf("fpos: (%f, %f, %f, %f)\n", fpos[i].x, fpos[i].y, fpos[i].z, fpos[i].w);
        printf("fvel: (%f, %f, %f, %f)\n", fvel[i].x, fvel[i].y, fvel[i].z, fvel[i].w);
        printf("fveval: (%f, %f, %f, %f)\n", fveval[i].x, fveval[i].y, fveval[i].z, fveval[i].w);
        printf("fforce: (%f, %f, %f, %f)\n", fforce[i].x, fforce[i].y, fforce[i].z, fforce[i].w);
        printf("fpress: %f\n", fpress[i]);
        printf("fdensity: %f\n", fdensity[i]);
        printf("fage: %u\n", fage[i]);
        printf("fcolor: %u\n", fcolor[i]);
        printf("fgcell: %u\n", fgcell[i]);
        printf("fgndx: %u\n", fgndx[i]);
        printf("felastidx: %u\n", felastidx[i]);
        printf("fparticleidx: %u\n", fparticleidx[i]);
        printf("fparticle_id: %u\n", fparticle_id[i]);
        printf("fmass_radius: %u\n", fmass_radius[i]);
        printf("fnerveidx: %u\n", fnerveidx[i]);
        printf("fepigen: %u\n", fepigen[i]);
        printf("fconc: %u\n", fconc[i]);
        printf("fposTemp: (%f, %f, %f, %f)\n", fposTemp[i].x, fposTemp[i].y, fposTemp[i].z, fposTemp[i].w);
        printf("fvelTemp: (%f, %f, %f, %f)\n", fvelTemp[i].x, fvelTemp[i].y, fvelTemp[i].z, fvelTemp[i].w);
        printf("fvevalTemp: (%f, %f, %f, %f)\n", fvevalTemp[i].x, fvevalTemp[i].y, fvevalTemp[i].z, fvevalTemp[i].w);
        printf("fforceTemp: (%f, %f, %f, %f)\n", fforceTemp[i].x, fforceTemp[i].y, fforceTemp[i].z, fforceTemp[i].w);
        printf("fpressTemp: %f\n", fpressTemp[i]);
        printf("fdensityTemp: %f\n", fdensityTemp[i]);
        printf("fageTemp: %u\n", fageTemp[i]);
        printf("fcolorTemp: %u\n", fcolorTemp[i]);
        printf("fgcellTemp: %u\n", fgcellTemp[i]);
        printf("fgndxTemp: %u\n", fgndxTemp[i]);
        printf("felastidxTemp: %u\n", felastidxTemp[i]);
        printf("fparticleidxTemp: %u\n", fparticleidxTemp[i]);
        printf("fparticle_idTemp: %u\n", fparticle_idTemp[i]);
        printf("fmass_radiusTemp: %u\n", fmass_radiusTemp[i]);
        printf("fnerveidxTemp: %u\n", fnerveidxTemp[i]);
        printf("fepigenTemp: %u\n", fepigenTemp[i]);
        printf("fconcTemp: %u\n", fconcTemp[i]);

}

    if (i >= pnum) return;
    if (m_FParamsDevice->debug > 1 && i == 0) printf("\ncountingSortFull(): pnum=%u\n", pnum);
    uint icell = fgcell[i];
    if (icell != GRID_UNDEF) {
        uint indx = fgcell[i];
        int sort_ndx = fbin_offset[icell] + indx;
        //printf("\nIndices: indx = %u, sort_ndx = %d\n", indx, sort_ndx);

        float4 zero = (float4)(0, 0, 0, 0);

        fbin      [sort_ndx] = sort_ndx;
        fpos      [sort_ndx] = fposTemp[i];
//         printf("\nPosition assigned to m_FluidDevice at Thread %u, index %u: (%f, %f, %f, %f)\n", i, sort_ndx,
//         bufF4(m_FluidTempDevice, FPOS)[i].x, bufF4(m_FluidTempDevice, FPOS)[i].y,
//         bufF4(m_FluidTempDevice, FPOS)[i].z, bufF4(m_FluidTempDevice, FPOS)[i].w);

        fvel      [sort_ndx] = fvelTemp[i];
        fveval    [sort_ndx] = fvevalTemp[i];
        fforce    [sort_ndx] = zero;
        fpress    [sort_ndx] = fpressTemp[i];
        fdensity  [sort_ndx] = fdensityTemp[i];
        fage      [sort_ndx] = fageTemp[i];
        fcolor    [sort_ndx] = fcolorTemp[i];
        fgcell    [sort_ndx] = icell;
        fgndx     [sort_ndx] = indx;
        float4 pos = fpos[i];
        for (int a = 0; a < BONDS_PER_PARTICLE; a++) {
            uint j = felastidx[i * BOND_DATA + a * DATA_PER_BOND];
            uint j_sort_ndx = UINT_MAX;
            uint jcell = GRID_UNDEF;
            if (j < pnum) {
                jcell = fgcell[j];
                uint jndx = UINT_MAX;
                if (jcell != GRID_UNDEF) {
                    jndx = fgndx[j];
                    if ((fbin_offset[jcell] + jndx) < pnum) {
                        j_sort_ndx = fbin_offset[jcell] + jndx;
                    }
                }
            }
            fparticleidx[sort_ndx * BOND_DATA + a * DATA_PER_BOND] = j_sort_ndx;
            for (int b = 1; b < 5 / DATA_PER_BOND; b++) {
                fparticleidx[sort_ndx * BOND_DATA + a * DATA_PER_BOND + b] =
                fparticleidxTemp[i * BOND_DATA + a * DATA_PER_BOND + b];
            }
            felastidx[sort_ndx * BOND_DATA + a * DATA_PER_BOND + 5] = fparticleidxTemp[i * BOND_DATA + a * DATA_PER_BOND + 5];
            felastidx[sort_ndx * BOND_DATA + a * DATA_PER_BOND + 6] = fparticleidxTemp[i * BOND_DATA + a * DATA_PER_BOND + 6];
            felastidx[sort_ndx * BOND_DATA + a * DATA_PER_BOND + 7] = fparticleidxTemp[i * BOND_DATA + a * DATA_PER_BOND + 7];
            felastidx[sort_ndx * BOND_DATA + a * DATA_PER_BOND + 8] = fparticleidxTemp[i * BOND_DATA + a * DATA_PER_BOND + 8];
            felastidx[sort_ndx * BOND_DATA + a * DATA_PER_BOND + 8] = fparticleidxTemp[i * BOND_DATA + a * DATA_PER_BOND + 8];

        }
            for (int a = 0; a < BONDS_PER_PARTICLE; a++) {
                uint k = fparticleidxTemp[i * BONDS_PER_PARTICLE * 2 + a * 2];
                uint b = fparticleidxTemp[i * BONDS_PER_PARTICLE * 2 + a * 2 + 1];
                uint kndx, kcell, ksort_ndx = UINT_MAX;
                if (k < pnum) {
                    kcell = fgcellTemp[k];
                    if (kcell != GRID_UNDEF) {
                        kndx = fgndxTemp[k];
                        ksort_ndx = fbin_offset[kcell] + kndx;
                    }
                }
                fparticleidx[sort_ndx * BONDS_PER_PARTICLE * 2 + a * 2] = ksort_ndx;
                fparticleidx[sort_ndx * BONDS_PER_PARTICLE * 2 + a * 2 + 1] = b;
                fparticleidxTemp[i * BONDS_PER_PARTICLE * 2 + a * 2] = UINT_MAX;
            }

            fparticle_id[sort_ndx] = fparticle_idTemp[i];
            fmass_radius[sort_ndx] = fmass_radiusTemp[i];
            fnerveidx   [sort_ndx] = fnerveidxTemp   [i];



            if (i < NUM_GENES) {
                // Access memory directly and perform element-wise assignment for fbuf_epigen
                fepigen[sort_ndx + pnum * i] = fepigenTemp[i + pnum * i];
            }

            if (i < NUM_TF) {
                // Access memory directly and perform element-wise assignment for fbuf_conc
                fconc[sort_ndx * NUM_TF + i] = fconcTemp[i * NUM_TF + i];
            }

            if(get_global_id(0) == 1) {
                printf("Thread Index: %d\n", i);
                printf("sort_ndx: %d\n", sort_ndx);
                printf("fbin: (%f, %f, %f, %f)\n", fbin[sort_ndx].x, fbin[sort_ndx].y, fbin[sort_ndx].z, fbin[sort_ndx].w);
                printf("fbin_offset: %u\n", fbin_offset[sort_ndx]);
                printf("fpos: (%f, %f, %f, %f)\n", fpos[sort_ndx].x, fpos[sort_ndx].y, fpos[sort_ndx].z, fpos[sort_ndx].w);
                printf("fvel: (%f, %f, %f, %f)\n", fvel[sort_ndx].x, fvel[sort_ndx].y, fvel[sort_ndx].z, fvel[sort_ndx].w);
                printf("fveval: (%f, %f, %f, %f)\n", fveval[sort_ndx].x, fveval[sort_ndx].y, fveval[sort_ndx].z, fveval[sort_ndx].w);
                printf("fforce: (%f, %f, %f, %f)\n", fforce[sort_ndx].x, fforce[sort_ndx].y, fforce[sort_ndx].z, fforce[sort_ndx].w);
                printf("fpress: %f\n", fpress[sort_ndx]);
                printf("fdensity: %f\n", fdensity[sort_ndx]);
                printf("fage: %u\n", fage[sort_ndx]);
                printf("fcolor: %u\n", fcolor[sort_ndx]);
                printf("fgcell: %u\n", fgcell[sort_ndx]);
                printf("fgndx: %u\n", fgndx[sort_ndx]);
                printf("felastidx: %u\n", felastidx[sort_ndx]);
                printf("fparticleidx: %u\n", fparticleidx[sort_ndx]);
                printf("fparticle_id: %u\n", fparticle_id[sort_ndx]);
                printf("fmass_radius: %u\n", fmass_radius[sort_ndx]);
                printf("fnerveidx: %u\n", fnerveidx[sort_ndx]);
                printf("fepigen: %u\n", fepigen[sort_ndx]);
                printf("fconc: %u\n", fconc[sort_ndx]);

                printf("fposTemp: (%f, %f, %f, %f)\n", fposTemp[i].x, fposTemp[i].y, fposTemp[i].z, fposTemp[i].w);
                printf("fvelTemp: (%f, %f, %f, %f)\n", fvelTemp[i].x, fvelTemp[i].y, fvelTemp[i].z, fvelTemp[i].w);
                printf("fvevalTemp: (%f, %f, %f, %f)\n", fvevalTemp[i].x, fvevalTemp[i].y, fvevalTemp[i].z, fvevalTemp[i].w);
                printf("fforceTemp: (%f, %f, %f, %f)\n", fforceTemp[i].x, fforceTemp[i].y, fforceTemp[i].z, fforceTemp[i].w);
                printf("fpressTemp: %f\n", fpressTemp[i]);
                printf("fdensityTemp: %f\n", fdensityTemp[i]);
                printf("fageTemp: %u\n", fageTemp[i]);
                printf("fcolorTemp: %u\n", fcolorTemp[i]);
                printf("fgcellTemp: %u\n", fgcellTemp[i]);
                printf("fgndxTemp: %u\n", fgndxTemp[i]);
                printf("felastidxTemp: %u\n", felastidxTemp[i]);
                printf("fparticleidxTemp: %u\n", fparticleidxTemp[i]);
                printf("fparticle_idTemp: %u\n", fparticle_idTemp[i]);
                printf("fmass_radiusTemp: %u\n", fmass_radiusTemp[i]);
                printf("fnerveidxTemp: %u\n", fnerveidxTemp[i]);
                printf("fepigenTemp: %u\n", fepigenTemp[i]);
                printf("fconcTemp: %u\n", fconcTemp[i]);
            }
        }
}

__kernel void countingSortDenseLists (
    __global struct FParams*    m_FParamsDevice,
    volatile __global uint* fbin_offset,
    volatile __global uint* fbin_count,
    volatile __global uint* fdense_lists,
    volatile __global uint* fbin_count_active_genes,
    volatile __global uint* fbin_offset_active_genes,
    volatile __global uint* fepigen,
    volatile __global uint* fparticle_id,
                       int  pnum
    )
{
    uint i = get_global_id(0);
    unsigned int bin = i * SCAN_BLOCKSIZE / 2 + get_group_id(0) * SCAN_BLOCKSIZE / 2;
    int gridTot = m_FParamsDevice->gridTotal;

    if (m_FParamsDevice->debug > 2 && bin == 0) {
        printf("\n\n######countingSortDenseLists###### bin==0  gridTot=%u, fbuf.bufI (FBIN_OFFSET)[bin]=%u \n", gridTot, fbin_offset[0]);
    }

    if (bin >= gridTot)
        return;

    uint count = fbin_count[bin];

    if (count == 0)
        return;

    uint binoffset = fbin_offset[bin];
    uint gene_counter[NUM_GENES] = {0};

    if (binoffset + count > pnum) {
        printf("\n\n!!Overflow: (binoffset+count > pnum), bin=%u \n", bin);
        return;
    }

    for (uint particle = binoffset; particle < binoffset + count; particle++) {
        if (m_FParamsDevice->debug > 2 && particle >= 22000 && particle < 20030) {
            printf("\nparticle==%u, ", particle);
        }
        for (int gene = 0; gene < NUM_GENES; gene++) {
            if (fepigen[particle + pnum * gene] > 0) {
                fdense_lists[gene * NUM_GENES + particle] = particle;
                gene_counter[gene]++;
                if (m_FParamsDevice->debug > 2 && gene_counter[gene] > fbin_count_active_genes[gene * gridTot + bin]) {
                    printf("\n Overflow: particle=,%u, ID=,%u, gene=,%u, bin=,%u, gene_counter[gene]=,%u, fbin_count_active_genes[gene*gridTot +bin]=,%u \t\t",
                        particle, fparticle_id[particle], gene, bin, gene_counter[gene], fbin_count_active_genes[gene * gridTot + bin]);
                }
            } else if (m_FParamsDevice->debug > 2 && gene == 2 && particle % 1000 == 0) {
                printf("*");
            }
        }
    }

}


float contributePressure (
    __global struct FParams* m_FParamsDevice,
    __global struct FBufs* m_FluidDevice,
    int i,
    float3 p,
    int cell,
    float *sum_p6k
    )
{

    if ( bufI(m_FluidDevice, FBIN_COUNT)[cell] == 0 ) return 0.0;                       // If the cell is empty, skip it.

    float3 dist;
    float dsq, r, q, b, c, sum = 0.0;
    float d2 = m_FParamsDevice->psimscale * m_FParamsDevice->psimscale;
    float r2 = m_FParamsDevice->r2;
    float sr = m_FParamsDevice->psmoothradius;

    int clast = bufI(m_FluidDevice, FBIN_OFFSET)[cell] + bufI(m_FluidDevice, FBIN_COUNT)[cell];

    for (int cndx = bufI(m_FluidDevice, FBIN_OFFSET)[cell]; cndx < clast; cndx++) {
        int pndx = bufI(m_FluidDevice, FBIN)[cndx];
        dist = p - bufF3(m_FluidDevice, FPOS)[pndx];
        dsq = (dist.x*dist.x + dist.y*dist.y + dist.z*dist.z);

        if (dsq < r2 && dsq > 0.0) {
            r = sqrt(dsq);
            q = r / sr;
            b = (1 - q);
            b *= b;
            b *= b;
            sum += b * (4 * q + 1);

            c = (r2 - dsq) * d2;
            sum_p6k[i] += c * c * c;
        }
    }
    return sum;;
}

__kernel void computePressure (
    __global struct FParams* m_FParamsDevice,
    __global struct FBufs* m_FluidDevice,
    int pnum
    )
{
    int i = get_global_id(0); // particle index
    if (i >= pnum) return;

    // Get search cell
    int nadj = (1 * m_FParamsDevice->gridRes.z + 1) * m_FParamsDevice->gridRes.x + 1;
    uint gc = bufI(m_FluidDevice, FGCELL)[i]; // get grid cell of the current particle.
    if (gc == GRID_UNDEF) return;   // IF particle not in the simulation
    gc -= nadj;

    // Sum Pressures
    float3 pos = bufF3(m_FluidDevice, FPOS)[i];
    float sum = 0.0;
    float sum_p6k = 0.0;
    for (int c = 0; c < m_FParamsDevice->gridAdjCnt; c++) {
        sum += contributePressure(m_FParamsDevice, m_FluidDevice, i, pos, gc + m_FParamsDevice->gridAdj[c], &sum_p6k);
    }
    barrier(CLK_LOCAL_MEM_FENCE);

    // Compute Density & Pressure
    sum = sum * m_FParamsDevice->pmass * m_FParamsDevice->wendlandC2kern;

    if (sum == 0.0) sum = 1.0;
    bufF(m_FluidDevice, FPRESS)[i] = (sum - m_FParamsDevice->prest_dens) * m_FParamsDevice->pintstiff; // pressure = (diff from rest density) * stiffness
    bufF(m_FluidDevice, FDENSITY)[i] = 1.0f / sum;
}

float3 contributeForce (
    __global struct FParams* m_FParamsDevice,
    __global struct FBufs* m_FluidDevice,
    int i,
    float3 ipos,
    float3 iveleval,
    float ipress,
    float idens,
    int cell
    )
{
	if ( bufI(m_FluidDevice, FBIN_COUNT)[cell] == 0 ) return (float3)(0,0,0);                                        // If the cell is empty, skip it.
	float  dsq, sdist, c, r, sr=m_FParamsDevice->psmoothradius;//1.0;//
    float3 pterm= (float3)(0,0,0), sterm= (float3)(0,0,0), vterm= (float3)(0,0,0), forcej= (float3)(0,0,0), delta_v= (float3)(0,0,0);                                                              // pressure, surface tension and viscosity terms.
	float3 dist     = (float3)(0,0,0),      eterm = (float3)(0,0,0),    force = (float3)(0,0,0);
	uint   j;
	int    clast    = bufI(m_FluidDevice, FBIN_OFFSET)[cell] + bufI(m_FluidDevice, FBIN_COUNT)[cell];                                // index of last particle in this cell
    uint k =0 ;
    for (int cndx = bufI(m_FluidDevice, FBIN_OFFSET)[cell]; cndx < clast; cndx++ ) {                                     // For particles in this cell.
        k++;
		j           = bufI(m_FluidDevice, FBIN)[ cndx ];
		dist        = ( ipos - bufF3(m_FluidDevice, FPOS)[ j ] );                                                     // dist in cm (Rama's comment)
		dsq         = (dist.x*dist.x + dist.y*dist.y + dist.z*dist.z);                                      // scalar distance squared
		r           = sqrt(dsq);


        if ( dsq < 1 /*m_FParamsDevice->rd2*/ && dsq > 0) {                                                           // IF in-range && not the same particle
            float kern = pow((sr - r),3);                                                                   // used as a component of surface tension kernel AND directly in viscosity
            sdist   = sqrt(dsq * m_FParamsDevice->d2);                                                                // smoothing distance
            float press = 100*(ipress+bufF(m_FluidDevice, FPRESS)[j]);///sdist

            pterm = idens * bufF(m_FluidDevice, FDENSITY)[j] *  100.0f* (dist/r) *(press*kern - (m_FParamsDevice->psurface_t/*0.4*/)*pow((sr - r),2));       // 1000 = hydroststic stiffness
            delta_v = bufF3(m_FluidDevice, FVEVAL)[j] - iveleval;
            vterm =  100000.0f* delta_v * kern;// (1/2)*pow((sr - r),3) ; // 10000.0 gives fluid, 100000.0 gives visco-elastic behaviour.
            //if (i==1) printf("\n contributeForce : m_FParamsDevice->psurface_t=%f,  m_FParamsDevice->sterm=%f, m_FParamsDevice->pvisc=%f, m_FParamsDevice->vterm=%f ", m_FParamsDevice->psurface_t, m_FParamsDevice->sterm, m_FParamsDevice->pvisc, m_FParamsDevice->vterm );
            /*
             sdist   = sqrt(dsq * m_FParamsDevice->d2);                                                                // smoothing distance = sqrt(dist^2 * sim_scale^2))
             c       = ( m_FParamsDevice->psmoothradius - sdist );
             pterm   = (dist/sdist) * pow((m_FParamsDevice->psmoothradius - sqrt(dsq)), 3) * (m_FParamsDevice->psmoothradius - dsq) ;
             * m_FParamsDevice->psimscale * -0.5f * c * m_FParamsDevice->spikykern   * ( ipress + bufF(m_FluidDevice, FPRESS)[ j ] )/ sdist )  ;       // pressure term
            //sterm   = (dist/dsq) * m_FParamsDevice->sterm * cos(3*CUDART_PI_F*r/(2*m_FParamsDevice->psmoothradius));  // can we use sdist in placeof r ?  or in place od dsq? What about pressure?
			//vterm   =  m_FParamsDevice->vterm * ( bufF3(m_FluidDevice, FVEVAL)[ j ] - iveleval );  // make_float3(0,0,0);//
			forcej  += ( pterm + sterm + vterm) * c * idens * (bufF(m_FluidDevice, FDENSITY)[ j ] );  // fluid force
            */
            force   +=  pterm + vterm  ;
            /*
            if(m_FParamsDevice->debug>0 && i<5 && k<2)  printf("\ncontribForce : debug=%u. i=%u, r=,%f, sr=,%f, (sr-r)^3=,%f, delta_v=,(%f,%f,%f), vterm=(%f,%f,%f), pterm(%f,%f,%f), \t\t press=,%f, sdist=,%f, dsq=,%f, m_FParamsDevice->d2=,%f  kern=,%f, \t\t idens=%f,, bufF(m_FluidDevice, FDENSITY)[j]=,%f, ",m_FParamsDevice->debug, i, r, sr, kern, delta_v.x,delta_v.y,delta_v.z, vterm.x,vterm.y,vterm.z, pterm.x,pterm.y,pterm.z, press, sdist, dsq, m_FParamsDevice->d2, kern, idens, bufF(m_FluidDevice, FDENSITY)[j]);
            */
            /*
            if(i<10) printf("\ncontribForce() : i=,%u, ,cell=,%u,  ,cndx=,%u, ,r=,%f, ,sqrt(m_FParamsDevice->rd2)=r_basis=,%f, ,m_FParamsDevice->psmoothradius=,%f,,sdist=,%f, ,(m_FParamsDevice->psmoothradius-sdist)= c =,%f, \t,ipress=,%f, ,jpress=,%f, ,idens=,%f, ,jdens=,%f,  press=,%f,     \t ,pterm=(,%f,%f,%f,),  ,sterm=(,%f,%f,%f,), ,vterm=(,%f,%f,%f,), ,forcej=(,%f,%f,%f,) ,  ,m_FParamsDevice->vterm=,%f, ,bufF3(m_FluidDevice, FVEVAL)[ j ]=(,%f,%f,%f,), ,iveleval=(,%f,%f,%f,) ",
                i, cell, cndx, r, sqrt(m_FParamsDevice->rd2), m_FParamsDevice->psmoothradius, sdist, c,  ipress, bufF(m_FluidDevice, FPRESS)[j], idens, bufF(m_FluidDevice, FDENSITY)[j], press,   pterm.x,pterm.y,pterm.z, sterm.x,sterm.y,sterm.z, vterm.x,vterm.y,vterm.z, forcej.x,forcej.y,forcej.z,
                m_FParamsDevice->vterm, bufF3(m_FluidDevice, FVEVAL)[j].x, bufF3(m_FluidDevice, FVEVAL)[j].y, bufF3(m_FluidDevice, FVEVAL)[j].z, iveleval.x, iveleval.y, iveleval.z
            );
            */
        }                                                                                                   // end of: IF in-range && not the same particle
    }                                                                                                       // end of loop round particles in this cell
    //if(i<10)  printf("\ncontribForce : i=%u, force=(%f,%f,%f)  ",i, force.x,force.y,force.z  );
    return force;                                                                                           // return fluid force && list of potential bonds fron this cell
}

__kernel void computeForce (
    __global struct FParams* m_FParamsDevice,
    __global struct FBufs* m_FluidDevice,
    int pnum,
    int freezeBoolToInt,
    uint frame
    )
{
    uint i = get_global_id(0);
    if (i >= pnum) return;
    uint gc = bufI(m_FluidDevice, FGCELL)[i];
    if (gc == GRID_UNDEF) return;

    gc -= (1 * m_FParamsDevice->gridRes.z + 1) * m_FParamsDevice->gridRes.x + 1;
    float3 force = (float3)(0, 0, 0);
    float3 eterm = (float3)(0, 0, 0);
    float3 dist = (float3)(0, 0, 0);
    float dsq, abs_dist;
    uint bondsToFill = 0;
    uint bonds[BONDS_PER_PARTICLE][2];
    float bond_dsq[BONDS_PER_PARTICLE];
    for (int a = 0; a < BONDS_PER_PARTICLE; a++) {
        bonds[a][0] = UINT_MAX;
        bonds[a][1] = UINT_MAX;
        bond_dsq[a] = m_FParamsDevice->rd2;
    }
    uint i_ID = bufI(m_FluidDevice, FPARTICLE_ID)[i];

    float3 pvel = bufF3(m_FluidDevice, FVEVAL)[i];
    bool hide;
    bool long_bonds = false;
//     for (int a = 0; a < BONDS_PER_PARTICLE; a++) {
//         uint bond = i * BOND_DATA + a * DATA_PER_BOND;
//         uint j = bufI(m_FluidDevice, FELASTIDX)[bond];
//         float restlength = bufF(m_FluidDevice, FELASTIDX)[bond + 2];
//         if (j >= pnum || restlength < 0.000000001) {
//             hide = true;
//             continue;
//         } else hide = false;
//
//         float elastic_limit = bufF(m_FluidDevice, FELASTIDX)[bond + 1];
//         float modulus = bufF(m_FluidDevice, FELASTIDX)[bond + 3];
//         float damping_coeff = bufF(m_FluidDevice, FELASTIDX)[bond + 4];
//         uint other_particle_ID = bufI(m_FluidDevice, FELASTIDX)[bond + 5];
//         uint bondIndex = bufI(m_FluidDevice, FELASTIDX)[bond + 6];
//
//         float3 j_pos = bufF3(m_FluidDevice, FPOS)[j];
//
//         dist = (bufF3(m_FluidDevice, FPOS)[i] - j_pos);
//         dsq = (dist.x * dist.x + dist.y * dist.y + dist.z * dist.z);
//         abs_dist = sqrt(dsq) + FLT_MIN;
//         float3 rel_vel = bufF3(m_FluidDevice, FVEVAL)[j] - pvel;
//
//         float spring_strain = fmax(0.0f, (abs_dist - restlength) / restlength);
//         bufF(m_FluidDevice, FELASTIDX)[bond + /*6*/strain_sq_integrator] = (bufF(m_FluidDevice, FELASTIDX)[bond + /*6*/strain_sq_integrator] + spring_strain * spring_strain);
//         bufF(m_FluidDevice, FELASTIDX)[bond + /*7*/strain_integrator] = (bufF(m_FluidDevice, FELASTIDX)[bond + /*7*/strain_integrator] + spring_strain);
//
//         eterm = ((float)(abs_dist < elastic_limit)) * (((dist / abs_dist) * spring_strain * modulus) - damping_coeff * rel_vel) / (m_FParamsDevice->pmass);
//
//         if (m_FParamsDevice->debug > 0 && abs_dist > 1.5) {
//             long_bonds = true;
//             printf("\ncomputeForce() 1: frame=%u, i=%u, i_ID=%u, j=%u, j_ID=%u, other_particle_ID=%u, bond=%u, eterm=(%f,%f,%f) restlength=%f, modulus=%f , abs_dist=%f , spring_strain=%f , strain_integrator=%f, damping_coeff*rel_vel.z/m_FParamsDevice->pmass=%f, ((dist/abs_dist) * spring_strain * modulus) / m_FParamsDevice->pmass=%f ",
//                 frame, i, i_ID, j, bufI(m_FluidDevice, FPARTICLE_ID)[j], other_particle_ID, a, eterm.x, eterm.y, eterm.z, restlength, modulus, abs_dist, spring_strain, bufF(m_FluidDevice, FELASTIDX)[bond + 7],
//                 damping_coeff * rel_vel.z / m_FParamsDevice->pmass, (((dist.z / abs_dist) * spring_strain * modulus) / m_FParamsDevice->pmass)
//                 );
//         }
//
//         if (isnan(eterm.x) || isnan(eterm.y) || isnan(eterm.z)) {
//             if (!hide) {
//                 printf("\n#### i=%i, j=%i, bond=%i, eterm.x=%f, eterm.y=%f, eterm.z=%f \t####", i,j,a, eterm.x,eterm.y,eterm.z);
//                 printf("\ncomputeForce() chk3: ParticleID=%u, bond=%u, restlength=%f, modulus=%f , abs_dist=%f , spring_strain=%f , strain_integrator=%f  ", bufI(m_FluidDevice, FPARTICLE_ID)[i], a, restlength, modulus, abs_dist, spring_strain, bufF(m_FluidDevice, FELASTIDX)[bond + 7]);
//             }
//         }else {
//             force -= eterm;
//             atomic_fetch_add(bufF3(m_FluidDevice, FFORCE)[j].x += 1.0f);
//             atomic_fetch_add(bufF3(m_FluidDevice, FFORCE)[j].y += 1.0f);
//             atomic_fetch_add(bufF3(m_FluidDevice, FFORCE)[j].z += 1.0f);
//         }
//         if (abs_dist >= elastic_limit && freezeBoolToInt == 0) {
//             bufF(m_FluidDevice, FELASTIDX)[i * BOND_DATA + a * DATA_PER_BOND + 2] = 0.0;
//
//             uint bondIndex_ = bufI(m_FluidDevice, FELASTIDX)[i * BOND_DATA + a * DATA_PER_BOND + 6];
//             if (m_FParamsDevice->debug > 2) printf("\n#### Set to broken, i=%i, j=%i, b=%i, bufI(m_FluidDevice, FPARTICLEIDX)[j*BONDS_PER_PARTICLE*2 + b]=UINT_MAX\t####", i,j,bondIndex_);
//             bondsToFill++;
//         }
//
//         barrier(CLK_LOCAL_MEM_FENCE);
//
//     }


    bondsToFill = BONDS_PER_PARTICLE;
    float3 fluid_force_sum = (float3)(0, 0, 0);
    for (int c = 0; c < m_FParamsDevice->gridAdjCnt; c++) {
        float3 fluid_force = (float3)(0, 0, 0);
        fluid_force = contributeForce(m_FParamsDevice, m_FluidDevice, i, bufF3(m_FluidDevice, FPOS)[i], bufF3(m_FluidDevice, FVEVAL)[i], bufF(m_FluidDevice, FPRESS)[i], bufF(m_FluidDevice, FDENSITY)[i], gc + m_FParamsDevice->gridAdj[c]);
        fluid_force_sum += fluid_force;
    }
    force += fluid_force_sum * m_FParamsDevice->pmass;
    if (m_FParamsDevice->debug > 0 && long_bonds == true) {
        fluid_force_sum *= m_FParamsDevice->pmass;
        printf("\nComputeForce 2: i=%u, fluid_force_sum=(%f,%f,%f) force=(%f,%f,%f)",
            i, fluid_force_sum.x, fluid_force_sum.y, fluid_force_sum.z, force.x, force.y, force.z);
    }

    bufF3(m_FluidDevice, FFORCE)[i].x += force.x;                                 // atomicAdd req due to other particles contributing forces via incomming bonds.
    bufF3(m_FluidDevice, FFORCE)[i].y += force.y;                                 // NB need to reset FFORCE to zero in  CountingSortFull(..)
    bufF3(m_FluidDevice, FFORCE)[i].z += force.z;                                 // temporary hack, ? better to write a float3 atomicAdd using atomicCAS ?  ########

}

__kernel void init_RandCL (
     uint num,
     global long* seed,
     global uint* res
     )
{
uint gid = get_global_id(0);
uint gsize = get_global_size(0);
well512_state state;
well512_seed(&state, seed [gid]);
    for (uint i = gid; i < num; i += gsize) {
        res [i] = well512_uint (state);
    }
}

// __kernel void initialize_bonds (
//     __global struct FParams* m_FParamsDevice,
//     __global struct FBufs* m_FluidDevice,
//     int ActivePoints,
//     uint list_length,
//     int gene
//     )
// {
//     uint particle_index = get_global_id(0);
//     if ( particle_index >= list_length ) return;
//
//     uint i = bufII(m_FluidDevice,FDENSE_LISTS)[gene][particle_index];
//     if ( i >= ActivePoints ) return;
//
//     uint gc = bufI(m_FluidDevice, FGCELL)[ i ];
//     uint bondToIdx[BONDS_PER_PARTICLE];
//     for(int bond=0; bond<BONDS_PER_PARTICLE; bond++)
//         bondToIdx[bond] = UINT_MAXSIZE;
//
//     float3 tpos = bufF3(m_FluidDevice, FPOS)[ i ];
//     uint *uintptr = &bufI(m_FluidDevice, FELASTIDX)[i*BOND_DATA];
//     float *floatptr = &bufF(m_FluidDevice, FELASTIDX)[i*BOND_DATA];
//
//     uint *fbufFEPIGEN = &bufI(m_FluidDevice, FEPIGEN)[i];
//     uint bond_type[BONDS_PER_PARTICLE] = {fgenome.elastin, fgenome.elastin, fgenome.elastin, fgenome.elastin };
//
//     unsigned int tissueType;
//     if (fbufFEPIGEN[9*m_FParamsDevice->maxPoints]>0) {
//         tissueType = 9;
//         bond_type[0]=2; bond_type[1]=2; bond_type[2]=2; bond_type[3]=2;
//     }
//     else if (fbufFEPIGEN[6*m_FParamsDevice->maxPoints]>0) {
//         tissueType = 6;
//         bond_type[0]=1; bond_type[1]=0; bond_type[2]=0; bond_type[3]=0;
//     }
//     else if (fbufFEPIGEN[7*m_FParamsDevice->maxPoints]>0) {
//         tissueType = 7;
//         bond_type[0]=1; bond_type[1]=0; bond_type[2]=0; bond_type[3]=0;
//     }
//     else if (fbufFEPIGEN[10*m_FParamsDevice->maxPoints]>0) {
//         tissueType = 10;
//         bond_type[0]=1; bond_type[1]=0; bond_type[2]=0; bond_type[3]=0;
//     }
//     else if (fbufFEPIGEN[6*m_FParamsDevice->maxPoints]>0) {
//         tissueType = 8;
//         bond_type[0]=1; bond_type[1]=1; bond_type[2]=1; bond_type[3]=1;
//     }
//     else {
//         tissueType = 0;
//         bond_type[0]=0; bond_type[1]=0; bond_type[2]=0; bond_type[3]=0;
//     }
//
//     for (int bond=0; bond<BONDS_PER_PARTICLE; bond++) {
//         float best_theta = FLT_MAX, bond_dsq = m_FParamsDevice->rd2;
//
//         for (int c=0; c < m_FParamsDevice->gridAdjCnt; c++)
//             contribFindBonds ( i, tpos, gc + m_FParamsDevice->gridAdj[c], bond, bondToIdx, &bond_dsq, &best_theta, m_FParamsDevice->maxPoints);
//
//         if(bondToIdx[bond]<ActivePoints) {
//             uintptr [bond*DATA_PER_BOND +0] = bondToIdx[bond];
//             floatptr[bond*DATA_PER_BOND +1] = fgenome.param[bond_type[bond]][fgenome.elastLim];
//             floatptr[bond*DATA_PER_BOND +2] = fgenome.param[bond_type[bond]][fgenome.default_rest_length];
//             floatptr[bond*DATA_PER_BOND +3] = fgenome.param[bond_type[bond]][fgenome.default_modulus];
//             floatptr[bond*DATA_PER_BOND +4] = fgenome.param[bond_type[bond]][fgenome.default_damping];
//             uintptr [bond*DATA_PER_BOND +5] = bufI(&fbuf, FPARTICLE_ID)[bondToIdx[bond]];
//             uintptr [bond*DATA_PER_BOND +6] = 0;
//             uintptr [bond*DATA_PER_BOND +7] = 0;
//         }
//     }
// }

__kernel void advanceParticles (

    __global struct FParams*    m_FParamsDevice,
    __global struct FBufs*      m_FluidDevice,
    __global struct FBufs*      m_FluidTempDevice,
    float time,
    float dt,
    float ss,
    int numPnts
    )
{
    uint i = get_global_id(0); // particle index
    if (i >= numPnts) return;

    if (bufI(m_FluidDevice, FGCELL)[i] == GRID_UNDEF) {
        bufF4(m_FluidDevice, FPOS)[i] = (float4)(m_FParamsDevice->pboundmin.x, m_FParamsDevice->pboundmin.y, m_FParamsDevice->pboundmin.z - 2 * m_FParamsDevice->gridRes.z, 0);
        bufF4(m_FluidDevice, FVEL)[i] = (float4)(0, 0, 0, 0);
        return;
    }

    float4 accel, norm;
    float diff, adj, speed;
    float4 pos = bufF4(m_FluidDevice, FPOS)[i];
    float4 veval = bufF4(m_FluidDevice, FVEVAL)[i];

    // Leapfrog integration
    accel = bufF4(m_FluidDevice, FFORCE)[i];

    // Boundaries
    // Y-axis
    diff = m_FParamsDevice->pradius - (pos.y - (m_FParamsDevice->pboundmin.y + (pos.x - m_FParamsDevice->pboundmin.x) * m_FParamsDevice->pground_slope));
    if (diff > EPSILON) {
        norm = (float4)(-m_FParamsDevice->pground_slope, 1.0 - m_FParamsDevice->pground_slope, 0, 0);
        adj = m_FParamsDevice->pextstiff * diff - m_FParamsDevice->pdamp * dot(norm, veval);
        norm *= adj;
        accel += norm;
    }

    diff = m_FParamsDevice->pradius - (m_FParamsDevice->pboundmax.y - pos.y);
    if (diff > EPSILON) {
        norm = (float4)(0, -1, 0, 0);
        adj = m_FParamsDevice->pextstiff * diff - m_FParamsDevice->pdamp * dot(norm, veval);
        norm *= adj;
        accel += norm;
    }

    // X-axis
    diff = m_FParamsDevice->pradius - (pos.x - m_FParamsDevice->pboundmin.x);
    if (diff > EPSILON) {
        norm = (float4)(1, 0, 0, 0);
        adj = (m_FParamsDevice->pforce_min + 1) * m_FParamsDevice->pextstiff * diff - m_FParamsDevice->pdamp * dot(norm, veval);
        norm *= adj;
        accel += norm;
    }

    diff = m_FParamsDevice->pradius - (m_FParamsDevice->pboundmax.x - pos.x);
    if (diff > EPSILON) {
        norm = (float4)(-1, 0, 0, 0);
        adj = (m_FParamsDevice->pforce_max + 1) * m_FParamsDevice->pextstiff * diff - m_FParamsDevice->pdamp * dot(norm, veval);
        norm *= adj;
        accel += norm;
    }

    // Z-axis
    diff = m_FParamsDevice->pradius - (pos.z - m_FParamsDevice->pboundmin.z);
    if (diff > EPSILON) {
        norm = (float4)(0, 0, 1, 0);
        adj = m_FParamsDevice->pextstiff * diff - m_FParamsDevice->pdamp * dot(norm, veval);
        norm *= adj;
        accel += norm;
    }

    diff = m_FParamsDevice->pradius - (m_FParamsDevice->pboundmax.z - pos.z);
    if (diff > EPSILON) {
        norm = (float4)(0, 0, -1, 0);
        adj = m_FParamsDevice->pextstiff * diff - m_FParamsDevice->pdamp * dot(norm, veval);
        norm *= adj;
        accel += norm;
    }

    // Shield for particle store at fparam.pboundmax
    float4 dist = m_FParamsDevice->pboundmax - pos;
    diff = 2 * m_FParamsDevice->pradius - (dist.x + dist.y + dist.z);
    if (diff > EPSILON) {
        norm = (float4)(1, 1, 1, 0);
        adj = m_FParamsDevice->pextstiff * diff - m_FParamsDevice->pdamp * dot(norm, veval);
        norm *= adj;
        accel += norm;
    }

    // Gravity
    accel += m_FParamsDevice->pgravity;

    // Accel Limit
    speed = accel.x * accel.x + accel.y * accel.y + accel.z * accel.z;
    if (speed > m_FParamsDevice->AL2) {
        accel *= m_FParamsDevice->AL / sqrt(speed);
        if (m_FParamsDevice->debug > 1) printf("\nadvanceParticles() Accel Limit: i=%u, mass=%f, accel=(%f,%f,%f), speed=%f, m_FParamsDevice->AL2=%f, m_FParamsDevice->pgravity=(%f,%f,%f)",
            i, m_FParamsDevice->pmass, accel.x, accel.y, accel.z, speed, m_FParamsDevice->AL2, m_FParamsDevice->pgravity.x, m_FParamsDevice->pgravity.y, m_FParamsDevice->pgravity.z);
    }

    // Velocity Limit
    float4 vel = bufF4(m_FluidDevice, FVEL)[i];
    speed = vel.x * vel.x + vel.y * vel.y + vel.z * vel.z;
    if (speed > m_FParamsDevice->VL2) {
        speed = m_FParamsDevice->VL2;
        vel *= m_FParamsDevice->VL / sqrt(speed);
        if (m_FParamsDevice->debug > 1) printf("\nadvanceParticles() Velocity Limit: i=%u, mass=%f, accel=(%f,%f,%f), speed=%f, m_FParamsDevice->VL2=%f, m_FParamsDevice->pgravity=(%f,%f,%f)",
            i, m_FParamsDevice->pmass, accel.x, accel.y, accel.z, speed, m_FParamsDevice->VL2, m_FParamsDevice->pgravity.x, m_FParamsDevice->pgravity.y, m_FParamsDevice->pgravity.z);
    }

    // Leap-frog Integration
    float4 vnext = accel * dt + vel;
    bufF4(m_FluidTempDevice, FVEVAL)[i] = (float4)(vel.x, vel.y, vel.z, 0.0f) + (float4)(vnext.x, vnext.y, vnext.z, 0.0f) * 0.5f;
    bufF4(m_FluidTempDevice,FVEL)[i] = vnext;
    bufF4(m_FluidTempDevice,FPOS)[i] += (vnext * dt); /*(dt/ss)) + bmotion*/

    if (m_FParamsDevice->debug > 2 && i < 10) {
        printf("\nadvanceParticles()3: i=%u, accel=(%f,%f,%f), vel=(%f,%f,%f), dt=%f, vnext=(%f,%f,%f), ss=%f",
            i, accel.x, accel.y, accel.z, vel.x, vel.y, vel.z, dt, vnext.x, vnext.y, vnext.z, ss);
    }

}
