#include <assert.h>
//#include <cuda.h>
#include <CL/cl.h>
#include "cutil_math.h"
#include <unistd.h>
#include <curand_kernel.h> // ../cuda-11.2/targets/x86_64-linux/include/
#include "fluid_system.h"
#include "fluid_system.cpp"



int iDivUp (int a, int b) {
    return (a % b != 0) ? (a / b + 1) : (a / b);
}

void computeNumBlocks (int numPnts, int minThreads, int &numGroups, int &numItems){
    numItems = min( minThreads, numPnts );
    numGroups = (numItems==0) ? 1 : iDivUp ( numPnts, numItems );
}

// void FluidSystem::TransferToTempCL ( int buf_id, int sz ){
//     clCheck ( cuMemcpyDtoD ( gpuVar(&m_FluidTemp, buf_id), gpuVar(&m_Fluid, buf_id), sz ), "TransferToTempCL", "cuMemcpyDtoD", "m_FluidTemp", mbDebug);
// }

void FluidSystem::TransferFromTempCL ( int buf_id, int sz ){
    clCheck ( cuMemcpyDtoD ( gpuVar(&m_Fluid, buf_id), gpuVar(&m_FluidTemp, buf_id), sz ), "TransferFromTempCL", "cuMemcpyDtoD", "m_Fluid", mbDebug);
}

void FluidSystem::FluidSetupCL ( int num, int gsrch, int3 res, cl_float3 size, cl_float3 delta, cl_float3 gmin, cl_float3 gmax, int total, int chk ){
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
    int3 temp = make_int3(m_FParams.gridSrch, m_FParams.gridSrch, m_FParams.gridSrch);
    m_FParams.gridScanMax.x -= temp.x;
    m_FParams.gridScanMax.y -= temp.y;
    m_FParams.gridScanMax.z -= temp.z;
    m_FParams.chk = chk;

    // Build Adjacency Lookup
    int cell = 0;
    for (int y=0; y < gsrch; y++ )
        for (int z=0; z < gsrch; z++ )
            for (int x=0; x < gsrch; x++ )
                m_FParams.gridAdj [ cell++]  = ( y * m_FParams.gridRes.z+ z )*m_FParams.gridRes.x +  x ;

    // Compute number of blocks and threads
    m_FParams.itemsPerGroup = 512;                    //TODO probe hardware to set m_FParams.itemsPerGroup

    computeNumBlocks ( m_FParams.pnum, m_FParams.itemsPerGroup, m_FParams.numGroups, m_FParams.numItems);				// particles
    computeNumBlocks ( m_FParams.gridTotal, m_FParams.itemsPerGroup, m_FParams.gridBlocks, m_FParams.gridThreads);		// grid cell

    // Compute particle buffer & grid dimensions
    m_FParams.szPnts = (m_FParams.numGroups  * m_FParams.numItems);

    std::cout << "\nhost_CUDA.cpp  FluidSystem::FluidSetupCL ()  :   m_FParams.numItems = " << m_FParams.numItems << flush;
}

// void FluidSystem::FluidParamCL ( float ss, float sr, float pr, float mass, float rest, cl_float3 bmin, cl_float3 bmax, float estiff, float istiff, float visc, float surface_tension, float damp, float fmin, float fmax, float ffreq, float gslope, float gx, float gy, float gz, float al, float vl, float a_f, float a_p ){
//     m_FParams.psimscale = ss;
//     m_FParams.psmoothradius = sr;
//     m_FParams.pradius = pr;
//     m_FParams.r2 = sr * sr;
//     m_FParams.pmass = mass;
//     m_FParams.prest_dens = rest;
//     m_FParams.pboundmin = bmin;
//     m_FParams.pboundmax = bmax;
//     m_FParams.pextstiff = estiff;
//     m_FParams.pintstiff = istiff;
//     m_FParams.pvisc = visc;
//     m_FParams.psurface_t = surface_tension;
//     m_FParams.pdamp = damp;
//     m_FParams.pforce_min = fmin;
//     m_FParams.pforce_max = fmax;
//     m_FParams.pforce_freq = ffreq;
//     m_FParams.pground_slope = gslope;
//     m_FParams.pgravity = make_cl_float3( gx, gy, gz );
//     m_FParams.AL = al;
//     m_FParams.AL2 = al * al;
//     m_FParams.VL = vl;
//     m_FParams.VL2 = vl * vl;
//     //m_FParams.pemit = emit;
//
//     m_FParams.pdist = pow ( m_FParams.pmass / m_FParams.prest_dens, 1/3.0f );
//                                                                                 // Normalization constants.
//     m_FParams.poly6kern = 315.0f / (64.0f * 3.141592f * pow( sr, 9.0f) );
//     m_FParams.wendlandC2kern = 21 / (2 * 3.141592f );   // This is the value calculated in SymPy as per Wendland C2 as per (Dehnen & Aly 2012)
//     // 16   // The  WC2 kernel in DualSPHysics assumes  values of 0<=q<=2 , hence the divisor 16pi in the normalisation constant for 3D.
//     /* My notes from Sympy my notebook.
//     Where Wendland C2 kernel:
//
//         wc2 = (1-r*ss/2*sr)**4  * ((2*q) +1)
//
//     Normalisation constant = 1/integrate( (wc2*(4*pi*r**2)), (r,0, 2*sr/ss)),  NB *(4*pi*r**2) area of a sphere, & 2=basis of wc2.
//
//         =  1/ (288pi - 15552.0πss^2/sr^2 + 77760.0πss^3/sr^3 - 149965.714285714πss^4/sr^4 + 104976.0πss^5/sr^5  )
//
//     */
//     /* Notes from DualSPHysics Wiki
//     // Normalization const = reciprocal of radial integral of (kernel * area of sphere), found using Sympy.
//     // NB using W(r,h)=alpha_D (1-q/2)**4 *(2*q +1), 0<=q<=2, as per DualSPHysics Wiki. Where alpha_D is the normaliation constant.
//     // * m_FParams.pmass * m_FParams.psimscale
//     */
//     m_FParams.spikykern = -45.0f / (3.141592f * pow( sr, 6.0f) );            // spikykern used for force due to pressure.
//     m_FParams.lapkern = 45.0f / (3.141592f * pow( sr, 6.0f) );
//     // NB Viscosity uses a different kernel, this is the constant portion of its Laplacian.
//     // NB Laplacian is a scalar 2nd order differential, "The divergence of the gradient"
//     // This Laplacian comes from Muller et al 2003, NB The kernel is defined by the properties of  its Laplacian, gradient and value at the basis (outer limit) of the kernel. The Laplacian is the form used in the code. The equation of the kernel in Muller et al seems to be wrong, but this does not matter.
//
// /*
//     // -32*(1 - r)**3 + 12*(1 - r)**2*(4*r + 1)  // the Laplacian of  WC2 = (1-r)**4 *(1+4*r)
// //(15*r**2*(h/r**3 + 2/h**2 - 3*r/h**3)/(2*pi*h**3) + 15*r*(-h/(2*r**2) + 2*r/h**2 - 3*r**2/(2*h**3))/(pi*h**3))/r**2
// //(45/pi*h^6)((h^2/12r^3)+(2h/3)-(3r/4))
//
// //(r**2*(h/r**3 + 2/h**2 - 3*r/h**3) + 2*r*(-h/(2*r**2) + 2*r/h**2 - 3*r**2/(2*h**3) ) )/r**2
// */
//
//     m_FParams.gausskern = 1.0f / pow(3.141592f * 2.0f*sr*sr, 3.0f/2.0f);     // Gaussian not currently used.
//
//     m_FParams.H = m_FParams.psmoothradius / m_FParams.psimscale;
//     m_FParams.d2 = m_FParams.psimscale * m_FParams.psimscale;
//     m_FParams.rd2 = m_FParams.r2 / m_FParams.d2;
//     m_FParams.vterm = m_FParams.lapkern * m_FParams.pvisc;
//
//     m_FParams.actuation_factor = a_f;
//     m_FParams.actuation_period = a_p;
//
//
//     // Transfer sim params to device
//     clCheck ( cuMemcpyHtoD ( clFParams,	&m_FParams,		sizeof(FParams) ), "FluidParamCL", "cuMemcpyHtoD", "clFParams", mbDebug);
// }

void FluidSystem::TransferToCL (){
if (m_FParams.debug>1) std::cout<<"\nTransferToCL ()\n"<<std::flush;
    // Send particle buffers
    clCheck( cuMemcpyHtoD ( gpuVar(&m_Fluid, FPOS),          bufC(&m_Fluid, FPOS),         mMaxPoints *sizeof(float) * 3),                     "TransferToCL", "cuMemcpyHtoD", "FPOS",           mbDebug);
    clCheck( cuMemcpyHtoD ( gpuVar(&m_Fluid, FVEL),          bufC(&m_Fluid, FVEL),         mMaxPoints *sizeof(float)*3 ),                      "TransferToCL", "cuMemcpyHtoD", "FVEL",           mbDebug);
    clCheck( cuMemcpyHtoD ( gpuVar(&m_Fluid, FVEVAL),        bufC(&m_Fluid, FVEVAL),       mMaxPoints *sizeof(float)*3 ),                      "TransferToCL", "cuMemcpyHtoD", "FVELAL",         mbDebug);
    clCheck( cuMemcpyHtoD ( gpuVar(&m_Fluid, FFORCE),        bufC(&m_Fluid, FFORCE),       mMaxPoints *sizeof(float)*3 ),                      "TransferToCL", "cuMemcpyHtoD", "FFORCE",         mbDebug);
    clCheck( cuMemcpyHtoD ( gpuVar(&m_Fluid, FPRESS),        bufC(&m_Fluid, FPRESS),       mMaxPoints *sizeof(float) ),                        "TransferToCL", "cuMemcpyHtoD", "FPRESS",         mbDebug);
    clCheck( cuMemcpyHtoD ( gpuVar(&m_Fluid, FDENSITY),      bufC(&m_Fluid, FDENSITY),     mMaxPoints *sizeof(float) ),                        "TransferToCL", "cuMemcpyHtoD", "FDENSITY",       mbDebug);
    clCheck( cuMemcpyHtoD ( gpuVar(&m_Fluid, FCOLOR),          bufC(&m_Fluid, FCOLOR),         mMaxPoints *sizeof(uint) ),                         "TransferToCL", "cuMemcpyHtoD", "FCLR",           mbDebug);
    // add extra data for morphogenesis
    clCheck( cuMemcpyHtoD ( gpuVar(&m_Fluid, FELASTIDX),     bufC(&m_Fluid, FELASTIDX),    mMaxPoints *sizeof(uint[BOND_DATA]) ),              "TransferToCL", "cuMemcpyHtoD", "FELASTIDX",      mbDebug);
    clCheck( cuMemcpyHtoD ( gpuVar(&m_Fluid, FPARTICLEIDX),  bufC(&m_Fluid, FPARTICLEIDX), mMaxPoints *sizeof(uint[BONDS_PER_PARTICLE *2]) ),  "TransferToCL", "cuMemcpyHtoD", "FPARTICLEIDX",   mbDebug);
    clCheck( cuMemcpyHtoD ( gpuVar(&m_Fluid, FPARTICLE_ID),  bufC(&m_Fluid, FPARTICLE_ID), mMaxPoints *sizeof(uint) ),                         "TransferToCL", "cuMemcpyHtoD", "FPARTICLE_ID",   mbDebug);
    clCheck( cuMemcpyHtoD ( gpuVar(&m_Fluid, FMASS_RADIUS),  bufC(&m_Fluid, FMASS_RADIUS), mMaxPoints *sizeof(uint) ),                         "TransferToCL", "cuMemcpyHtoD", "FMASS_RADIUS",   mbDebug);
    clCheck( cuMemcpyHtoD ( gpuVar(&m_Fluid, FNERVEIDX),     bufC(&m_Fluid, FNERVEIDX),    mMaxPoints *sizeof(uint) ),                         "TransferToCL", "cuMemcpyHtoD", "FNERVEIDX",      mbDebug);
    clCheck( cuMemcpyHtoD ( gpuVar(&m_Fluid, FCONC),         bufC(&m_Fluid, FCONC),        mMaxPoints *sizeof(float[NUM_TF]) ),                "TransferToCL", "cuMemcpyHtoD", "FCONC",          mbDebug);
    clCheck( cuMemcpyHtoD ( gpuVar(&m_Fluid, FEPIGEN),       bufC(&m_Fluid, FEPIGEN),      mMaxPoints *sizeof(uint[NUM_GENES]) ),              "TransferToCL", "cuMemcpyHtoD", "FEPIGEN",        mbDebug);
if (m_FParams.debug>1) std::cout<<"TransferToCL ()  finished\n"<<std::flush;

}
  
void FluidSystem::TransferFromCL (){
//std::cout<<"\nTransferFromCL () \n"<<std::flush;    
    // Return particle buffers
    clCheck( cuMemcpyDtoH ( bufC(&m_Fluid, FPOS),         gpuVar(&m_Fluid, FPOS),          mMaxPoints *sizeof(float)*3 ),                         "TransferFromCL", "cuMemcpyDtoH", "FPOS",         mbDebug);
    clCheck( cuMemcpyDtoH ( bufC(&m_Fluid, FVEL),         gpuVar(&m_Fluid, FVEL),          mMaxPoints *sizeof(float)*3 ),                         "TransferFromCL", "cuMemcpyDtoH", "FVEL",         mbDebug);
    
    clCheck( cuMemcpyDtoH ( bufC(&m_Fluid, FVEVAL),       gpuVar(&m_Fluid, FVEVAL),        mMaxPoints *sizeof(float)*3 ),                         "TransferFromCL", "cuMemcpyDtoH", "FVELAL",       mbDebug);
    clCheck( cuMemcpyDtoH ( bufC(&m_Fluid, FFORCE),       gpuVar(&m_FluidTemp, FFORCE),    mMaxPoints *sizeof(float)*3 ),                         "TransferFromCL", "cuMemcpyDtoH", "FFORCE",       mbDebug);
    //NB PhysicalSort zeros gpuVar(&m_Fluid, FFORCE)
    clCheck( cuMemcpyDtoH ( bufC(&m_Fluid, FPRESS),       gpuVar(&m_Fluid, FPRESS),        mMaxPoints *sizeof(float) ),                           "TransferFromCL", "cuMemcpyDtoH", "FPRESS",       mbDebug);
    clCheck( cuMemcpyDtoH ( bufC(&m_Fluid, FDENSITY),     gpuVar(&m_Fluid, FDENSITY),      mMaxPoints *sizeof(float) ),                           "TransferFromCL", "cuMemcpyDtoH", "FDENSITY",     mbDebug);
    
    clCheck( cuMemcpyDtoH ( bufI(&m_Fluid, FAGE),         gpuVar(&m_Fluid, FAGE),          mMaxPoints *sizeof(uint) ),                            "TransferFromCL", "cuMemcpyDtoH", "FAGE",         mbDebug);
    clCheck( cuMemcpyDtoH ( bufI(&m_Fluid, FCOLOR),         gpuVar(&m_Fluid, FCOLOR),          mMaxPoints *sizeof(uint) ),                            "TransferFromCL", "cuMemcpyDtoH", "FCLR",         mbDebug);
    
    // add extra data for morphogenesis
    clCheck( cuMemcpyDtoH ( bufC(&m_Fluid, FELASTIDX),	gpuVar(&m_Fluid, FELASTIDX),	    mMaxPoints *sizeof(uint[BOND_DATA]) ),                 "TransferFromCL", "cuMemcpyDtoH", "FELASTIDX",    mbDebug);
    clCheck( cuMemcpyDtoH ( bufI(&m_Fluid, FPARTICLEIDX),	gpuVar(&m_Fluid, FPARTICLEIDX),	mMaxPoints *sizeof(uint[BONDS_PER_PARTICLE *2]) ),     "TransferFromCL", "cuMemcpyDtoH", "FPARTICLEIDX", mbDebug);
    clCheck( cuMemcpyDtoH ( bufI(&m_Fluid, FPARTICLE_ID),	gpuVar(&m_Fluid, FPARTICLE_ID),	mMaxPoints *sizeof(uint) ),                            "TransferFromCL", "cuMemcpyDtoH", "FPARTICLE_ID", mbDebug);
    clCheck( cuMemcpyDtoH ( bufI(&m_Fluid, FMASS_RADIUS),	gpuVar(&m_Fluid, FMASS_RADIUS),	mMaxPoints *sizeof(uint) ),                            "TransferFromCL", "cuMemcpyDtoH", "FMASS_RADIUS", mbDebug);
    clCheck( cuMemcpyDtoH ( bufI(&m_Fluid, FNERVEIDX),	gpuVar(&m_Fluid, FNERVEIDX),	    mMaxPoints *sizeof(uint) ),                            "TransferFromCL", "cuMemcpyDtoH", "FNERVEIDX",    mbDebug);
    clCheck( cuMemcpyDtoH ( bufF(&m_Fluid, FCONC),	    gpuVar(&m_Fluid, FCONC),	        mMaxPoints *sizeof(float[NUM_TF]) ),                   "TransferFromCL", "cuMemcpyDtoH", "FCONC",        mbDebug);
    clCheck( cuMemcpyDtoH ( bufI(&m_Fluid, FEPIGEN),	    gpuVar(&m_Fluid, FEPIGEN),	    mMaxPoints *sizeof(uint[NUM_GENES]) ),                 "TransferFromCL", "cuMemcpyDtoH", "FEPIGEN",      mbDebug);
}   // NB found FEPIGEN needed bufI and mMaxPoints, otherwise produced garbled files.



//CUDAs random number generator
// void FluidSystem::Init_FCURAND_STATE_CL (){ // designed to use to bootstrap itself. Set j=0 from host, call kernel repeatedly for 256^n threads, n=0-> to pnum threads.
//     unsigned long long  seed=0; // sequence=0, offset=1,
//     srand (time(NULL));
//     for (int i=0;i<mNumPoints;i++){ // generate seeds
//         seed = rand();
//         //seed = seed << 32;
//         //seed += rand();
//         //seed = clock();
//         bufI(&m_Fluid, FCURAND_SEED)[i] = seed;
//         //curand_init(seed, sequence, offset, &m_Fluid.bufCuRNDST(FCURAND_STATE)[i]);
//         //if (m_FParams.debug>1)printf("\n(2:seed=%llu,(FCURAND_SEED)[i]=%llu, rand()=%u), ",seed, m_Fluid.bufULL(FCURAND_SEED)[i], rand() );
//         //if (m_FParams.debug>1) cout<<"\t(seed="<<seed<<",(FCURAND_SEED)[i]="<<bufI(&m_Fluid, FCURAND_SEED)[i]<<"), "<<std::flush;
//
//     }
//     // transfer to cuda
//     //clCheck( cuMemcpyDtoH ( gcell,	gpuVar(&m_Fluid, FGCELL),	mNumPoints *sizeof(uint) ), "InsertParticlesCL", "cuMemcpyDtoH", "FGCELL", mbDebug );
//     //clCheck( cuMemcpyDtoH ( m_Fluid.bufCuRNDST(FCURAND_STATE),	gpuVar(&m_Fluid, FCURAND_STATE),	mNumPoints *sizeof(curandState_t) ),
//     //         "Init_FCURAND_STATE_CL", "cuMemcpyDtoH", "FCURAND_STATE", mbDebug );
//
//     clCheck( cuMemcpyHtoD (gpuVar(&m_Fluid, FCURAND_SEED), bufC(&m_Fluid, FCURAND_SEED), mNumPoints *sizeof(uint) ),
//              "Init_FCURAND_STATE_CL", "cuMemcpyDtoH", "FCURAND_SEED", mbDebug );
//
//     if (m_FParams.debug>1) std::cout <<"\nInit_FCURAND_STATE_CL_2.0\n\n"<<std::flush;
//     /*
//     int n=0;
//     void* args[1] = {&n};
//     int numGroups_=1, numItems_=1;
//     if (m_FParams.debug>1) std::cout <<"\nInit_FCURAND_STATE_CL_1.0\t n="<<n<<",  pow(256,n)="<<pow(256,n)<<",  mNumPoints/256="<<mNumPoints/256<<",\t mNumPoints="<<mNumPoints<<", mMaxPoints="<<mMaxPoints<<"  \n"<<std::flush;
//
//     do {
//         computeNumBlocks ( pow(256,n), m_FParams.itemsPerGroup, numGroups_, numItems_);
//
//         if (m_FParams.debug>1) std::cout <<"\nInit_FCURAND_STATE_CL_2.0\t n="<<n<<",  pow(256,n)="<<pow(256,n)<<",  mNumPoints/256="<<mNumPoints/256<<
//         "\t numGroups_="<<numGroups_<<", numItems_="<<numItems_<<"  \n"<<std::flush;
//
//         clCheck(clFinish(), "Init_FCURAND_STATE_CL", "clFinish", "Before m_Kern[FUNC_INIT_RANDOMCL], in do-while loop", 1);
//
//         clCheck ( cuLaunchKernel ( m_Kern[FUNC_INIT_RANDOMCL],  numGroups_, 1, 1, numItems_, 1, 1, 0, NULL, args, NULL), "Init_FCURAND_STATE_CL", "cuLaunch", "FUNC_INIT_RANDOMCL", mbDebug);
//
//         n++;
//     } while (pow(256,n) < mNumPoints/256) ;
//
//     if (m_FParams.debug>1) std::cout <<"\nInit_FCURAND_STATE_CL_3.0\t n="<<n<<",  pow(256,n)="<<pow(256,n)<<",  mNumPoints/256="<<mNumPoints/256<<
//     "\t m_FParams.numGroups="<<m_FParams.numGroups<<",  m_FParams.numItems="<<m_FParams.numItems<<".  \n"<<std::flush;
//
//     */
//     void* args[1] = {&mNumPoints};
//
//     clCheck(clFinish(), "Init_FCURAND_STATE_CL", "clFinish", "Before m_Kern[FUNC_INIT_RANDOMCL], after do-while loop", 1);
//
//     clCheck ( cuLaunchKernel ( m_Kern[FUNC_INIT_RANDOMCL],  m_FParams.numGroups, 1, 1, m_FParams.numItems, 1, 1, 0, NULL, args, NULL), "Init_FCURAND_STATE_CL", "cuLaunch", "FUNC_INIT_RANDOMCL", mbDebug);
//
//     clCheck(clFinish(), "Init_FCURAND_STATE_CL", "clFinish", "After cuMemcpyDtoH FCURAND_STATE, before 1st timestep", 1);
//     if (m_FParams.debug>1) std::cout <<"\nInit_FCURAND_STATE_CL_4.0\n"<<std::flush;
//
// }
///////////////////////////////// above here is all about setting up CUDA
/*
void FluidSystem::InsertParticlesCL ( uint* gcell, uint* gndx, uint* gcnt ){   // first zero the counters

    clCheck ( cuMemsetD8 ( gpuVar(&m_Fluid, FBIN_COUNT), 0,	m_GridTotal*sizeof(int) ), "InsertParticlesCL", "cuMemsetD8", "FBIN_COUNT", mbDebug );
    clCheck ( cuMemsetD8 ( gpuVar(&m_Fluid, FBIN_OFFSET), 0,	m_GridTotal*sizeof(int) ), "InsertParticlesCL", "cuMemsetD8", "FBIN_OFFSET", mbDebug );
    
    clCheck ( cuMemsetD8 ( gpuVar(&m_Fluid, FBIN_COUNT_ACTIVE_GENES), 0,	m_GridTotal *sizeof(uint[NUM_GENES]) ), "InsertParticlesCL", "cuMemsetD8", "FBIN_COUNT", mbDebug );
    clCheck ( cuMemsetD8 ( gpuVar(&m_Fluid, FBIN_OFFSET_ACTIVE_GENES), 0,	m_GridTotal *sizeof(uint[NUM_GENES]) ), "InsertParticlesCL", "cuMemsetD8", "FBIN_OFFSET", mbDebug );
    
    // Set long list to sort all particles.
    computeNumBlocks ( m_FParams.pnum, m_FParams.itemsPerGroup, m_FParams.numGroups, m_FParams.numItems);				// particles
    // launch kernel "InsertParticles"
    void* args[1] = { &mMaxPoints };  //&mNumPoints
    clCheck(cuLaunchKernel(m_Kern[FUNC_INSERT], m_FParams.numGroups, 1, 1, m_FParams.numItems, 1, 1, 0, NULL, args, NULL),
            "InsertParticlesCL", "cuLaunch", "FUNC_INSERT", mbDebug);
    if (m_FParams.debug>1) cout<<"\n########\nCalling InsertParticles kernel: args[1] = {"<<mNumPoints<<"}, mMaxPoints="<<mMaxPoints
        <<"\t m_FParams.numGroups="<<m_FParams.numGroups<<", m_FParams.numItems="<<m_FParams.numItems<<" \t"<<std::flush;

    // Transfer data back if requested (for validation)
    if (gcell != 0x0) {
        clCheck( cuMemcpyDtoH ( gcell,	gpuVar(&m_Fluid, FGCELL),	mNumPoints *sizeof(uint) ), "InsertParticlesCL", "cuMemcpyDtoH", "FGCELL", mbDebug );
        clCheck( cuMemcpyDtoH ( gndx,	gpuVar(&m_Fluid, FGNDX),		mNumPoints *sizeof(uint) ), "InsertParticlesCL", "cuMemcpyDtoH", "FGNDX", mbDebug);
        clCheck( cuMemcpyDtoH ( gcnt,	gpuVar(&m_Fluid, FBIN_COUNT),	m_GridTotal*sizeof(uint) ), "InsertParticlesCL", "cuMemcpyDtoH", "FBIN_COUNT", mbDebug);

        clFinish ();
    }
    if(m_debug>4){
        if (m_FParams.debug>1) cout<<"\nSaving (FGCELL) InsertParticlesCL: (particleIdx, cell) , mMaxPoints="<<mMaxPoints<<"\t"<<std::flush;
        clCheck( cuMemcpyDtoH ( bufI(&m_Fluid, FGCELL), gpuVar(&m_Fluid, FGCELL),	sizeof(uint[mMaxPoints]) ), "PrefixSumChangesCL", "cuMemcpyDtoH", "FGCELL", mbDebug);
        SaveUintArray( bufI(&m_Fluid, FGCELL), mMaxPoints, "InsertParticlesCL__bufI(&m_Fluid, FGCELL).csv" );
    }
}*/
/*
void FluidSystem::InsertParticlesCL ( uint* gcell, uint* gndx, uint* gcnt ){
    // first zero the counters
    clEnqueueFillBuffer(queue, gpuVar(&m_Fluid, FBIN_COUNT), 0, sizeof(int), 0, m_GridTotal * sizeof(int), 0, NULL, NULL);
    clEnqueueFillBuffer(queue, gpuVar(&m_Fluid, FBIN_OFFSET), 0, sizeof(int), 0, m_GridTotal * sizeof(int), 0, NULL, NULL);

    clEnqueueFillBuffer(queue, gpuVar(&m_Fluid, FBIN_COUNT_ACTIVE_GENES), 0, sizeof(uint[NUM_GENES]), 0, m_GridTotal * sizeof(uint[NUM_GENES]), 0, NULL, NULL);
    clEnqueueFillBuffer(queue, gpuVar(&m_Fluid, FBIN_OFFSET_ACTIVE_GENES), 0, sizeof(uint[NUM_GENES]), 0, m_GridTotal * sizeof(uint[NUM_GENES]), 0, NULL, NULL);

    // Set long list to sort all particles.
    computeNumBlocks (m_FParams.pnum, m_FParams.itemsPerGroup, m_FParams.numGroups, m_FParams.numItems);
    // launch kernel "InsertParticles"
    size_t global_work_size = ...;

    clSetKernelArg(m_Kern[FUNC_INSERT], 0, sizeof(cl_uint), &pnum);
    clEnqueueNDRangeKernel(queue,m_Kern[FUNC_INSERT],1,NULL,&global_work_size,NULL ,0,NULL,NULL);

    if (m_FParams.debug>1) cout<<"\n########\nCalling InsertParticles kernel: args[1] = {"<<mNumPoints<<"}, mMaxPoints="<<mMaxPoints
        <<"\t m_FParams.numGroups="<<m_FParams.numGroups<<", m_FParams.numItems="<<m_FParams.numItems<<" \t"<<std::flush;

    // Transfer data back if requested (for validation)
    if (gcell != 0x0) {
        clEnqueueReadBuffer(queue,gpuVar(&m_Fluid, FGCELL),CL_TRUE ,0,mNumPoints *sizeof(uint) ,gcell ,0,NULL,NULL);
        clEnqueueReadBuffer(queue,gpuVar(&m_Fluid, FGNDX),CL_TRUE ,0,mNumPoints *sizeof(uint) ,gndx ,0,NULL,NULL);
        clEnqueueReadBuffer(queue,gpuVar(&m_Fluid, FBIN_COUNT),CL_TRUE ,0,m_GridTotal*sizeof(uint) ,gcnt ,0,NULL,NULL);
        clFinish (queue);
    }
}*/

void FluidSystem::PrefixSumCellsCL ( int zero_offsets ){
    // Prefix Sum - determine grid offsets
    int blockSize = SCAN_BLOCKSIZE << 1;                // NB 1024 = 512 << 1.  NB SCAN_BLOCKSIZE is the number of threads per block
    int numElem1 = m_GridTotal;                         // tot num bins, computed in SetupGrid() 
    int numElem2 = int ( numElem1 / blockSize ) + 1;    // num sheets of bins? NB not spatial, but just dividing the linear array of bins, by a factor of 512*2
    int numElem3 = int ( numElem2 / blockSize ) + 1;    // num rows of bins?
    int threads = SCAN_BLOCKSIZE;
    int zon=1;

    cl_device_idptr array1  = gpuVar(&m_Fluid, FBIN_COUNT);		// input
    cl_device_idptr scan1   = gpuVar(&m_Fluid, FBIN_OFFSET);		// output
    cl_device_idptr array2  = gpuVar(&m_Fluid, FAUXARRAY1);
    cl_device_idptr scan2   = gpuVar(&m_Fluid, FAUXSCAN1);
    cl_device_idptr array3  = gpuVar(&m_Fluid, FAUXARRAY2);
    cl_device_idptr scan3   = gpuVar(&m_Fluid, FAUXSCAN2);

#ifndef xlong
    typedef unsigned long long	xlong;		// 64-bit integer
#endif
    if ( numElem1 > SCAN_BLOCKSIZE*xlong(SCAN_BLOCKSIZE)*SCAN_BLOCKSIZE) { if (m_FParams.debug>1)printf ( "\nERROR: Number of elements exceeds prefix sum max. Adjust SCAN_BLOCKSIZE.\n" );  }

    void* argsA[5] = {&array1, &scan1, &array2, &numElem1, &zero_offsets };     // sum array1. output -> scan1, array2.         i.e. FBIN_COUNT -> FBIN_OFFSET, FAUXARRAY1
    clCheck ( cuLaunchKernel ( m_Kern[FUNC_FPREFIXSUM], numElem2, 1, 1, threads, 1, 1, 0, NULL, argsA, NULL ), "PrefixSumCellsCL", "cuLaunch", "FUNC_PREFIXSUM", mbDebug);

    void* argsB[5] = { &array2, &scan2, &array3, &numElem2, &zon };             // sum array2. output -> scan2, array3.         i.e. FAUXARRAY1 -> FAUXSCAN1, FAUXARRAY2
    clCheck ( cuLaunchKernel ( m_Kern[FUNC_FPREFIXSUM], numElem3, 1, 1, threads, 1, 1, 0, NULL, argsB, NULL ), "PrefixSumCellsCL", "cuLaunch", "FUNC_PREFIXSUM", mbDebug);

    if ( numElem3 > 1 ) {
        cl_device_idptr nptr = {0};
        void* argsC[5] = { &array3, &scan3, &nptr, &numElem3, &zon };	        // sum array3. output -> scan3                  i.e. FAUXARRAY2 -> FAUXSCAN2, &nptr
        clCheck ( cuLaunchKernel ( m_Kern[FUNC_FPREFIXSUM], 1, 1, 1, threads, 1, 1, 0, NULL, argsC, NULL ), "PrefixSumCellsCL", "cuLaunch", "FUNC_PREFIXFIXUP", mbDebug);

        void* argsD[3] = { &scan2, &scan3, &numElem2 };	                        // merge scan3 into scan2. output -> scan2      i.e. FAUXSCAN2, FAUXSCAN1 -> FAUXSCAN1
        clCheck ( cuLaunchKernel ( m_Kern[FUNC_FPREFIXUP], numElem3, 1, 1, threads, 1, 1, 0, NULL, argsD, NULL ), "PrefixSumCellsCL", "cuLaunch", "FUNC_PREFIXFIXUP", mbDebug);
    }
    void* argsE[3] = { &scan1, &scan2, &numElem1 };		                        // merge scan2 into scan1. output -> scan1      i.e. FAUXSCAN1, FBIN_OFFSET -> FBIN_OFFSET
    clCheck ( cuLaunchKernel ( m_Kern[FUNC_FPREFIXUP], numElem2, 1, 1, threads, 1, 1, 0, NULL, argsE, NULL ), "PrefixSumCellsCL", "cuLaunch", "FUNC_PREFIXFIXUP", mbDebug);
    
    // Loop to PrefixSum the Dense Lists - NB by doing one gene at a time, we reuse the FAUX* arrays & scans.
    // For each gene, input FBIN_COUNT_ACTIVE_GENES[gene*m_GridTotal], output FBIN_OFFSET_ACTIVE_GENES[gene*m_GridTotal]
    cl_device_idptr array0  = gpuVar(&m_Fluid, FBIN_COUNT_ACTIVE_GENES);
    cl_device_idptr scan0   = gpuVar(&m_Fluid, FBIN_OFFSET_ACTIVE_GENES);

    for(int gene=0;gene<NUM_GENES;gene++){
      //if (m_FParams.debug>1) cout<<"\nPrefixSumCellsCL()1:gene="<<gene<<"\t"<<std::flush;
        array1  = array0 + gene*numElem1*sizeof(int); //gpuVar(&m_Fluid, FBIN_COUNT_ACTIVE_GENES);//[gene*numElem1]   ;///
        scan1   = scan0 + gene*numElem1*sizeof(int);

        //clCheck ( cuMemsetD8 ( array1, 0,	numElem1*sizeof(int) ), "PrefixSumCellsCL", "cuMemsetD8", "FBIN_COUNT", mbDebug );
        clCheck ( cuMemsetD8 ( scan1,  0,	numElem1*sizeof(int) ), "PrefixSumCellsCL", "cuMemsetD8", "FBIN_COUNT", mbDebug );
        
        clCheck ( cuMemsetD8 ( array2, 0,	numElem2*sizeof(int) ), "PrefixSumCellsCL", "cuMemsetD8", "FBIN_COUNT", mbDebug );
        clCheck ( cuMemsetD8 ( scan2,  0,	numElem2*sizeof(int) ), "PrefixSumCellsCL", "cuMemsetD8", "FBIN_COUNT", mbDebug );
        
        clCheck ( cuMemsetD8 ( array3, 0,	numElem3*sizeof(int) ), "PrefixSumCellsCL", "cuMemsetD8", "FBIN_COUNT", mbDebug );
        clCheck ( cuMemsetD8 ( scan3,  0,	numElem3*sizeof(int) ), "PrefixSumCellsCL", "cuMemsetD8", "FBIN_COUNT", mbDebug );
        
        void* argsA[5] = {&array1, &scan1, &array2, &numElem1, &zero_offsets };     // sum array1. output -> scan1, array2.         i.e. FBIN_COUNT -> FBIN_OFFSET, FAUXARRAY1
        clCheck ( cuLaunchKernel ( m_Kern[FUNC_FPREFIXSUM], numElem2, 1, 1, threads, 1, 1, 0, NULL, argsA, NULL ),
                  "PrefixSumCellsCL", "cuLaunch", "FUNC_PREFIXSUM", mbDebug);

        void* argsB[5] = { &array2, &scan2, &array3, &numElem2, &zon };             // sum array2. output -> scan2, array3.         i.e. FAUXARRAY1 -> FAUXSCAN1, FAUXARRAY2
        clCheck ( cuLaunchKernel ( m_Kern[FUNC_FPREFIXSUM], numElem3, 1, 1, threads, 1, 1, 0, NULL, argsB, NULL ), "PrefixSumCellsCL", "cuLaunch", "FUNC_PREFIXSUM", mbDebug);

        if ( numElem3 > 1 ) {
            cl_device_idptr nptr = {0};
            void* argsC[5] = { &array3, &scan3, &nptr, &numElem3, &zon };	        // sum array3. output -> scan3                  i.e. FAUXARRAY2 -> FAUXSCAN2, &nptr
            clCheck ( cuLaunchKernel ( m_Kern[FUNC_FPREFIXSUM], 1, 1, 1, threads, 1, 1, 0, NULL, argsC, NULL ), "PrefixSumCellsCL", "cuLaunch", "FUNC_PREFIXFIXUP", mbDebug);

            void* argsD[3] = { &scan2, &scan3, &numElem2 };	                        // merge scan3 into scan2. output -> scan2      i.e. FAUXSCAN2, FAUXSCAN1 -> FAUXSCAN1
            clCheck ( cuLaunchKernel ( m_Kern[FUNC_FPREFIXUP], numElem3, 1, 1, threads, 1, 1, 0, NULL, argsD, NULL ), "PrefixSumCellsCL", "cuLaunch", "FUNC_PREFIXFIXUP", mbDebug);
        }
        void* argsE[3] = { &scan1, &scan2, &numElem1 };		                        // merge scan2 into scan1. output -> scan1      i.e. FAUXSCAN1, FBIN_OFFSET -> FBIN_OFFSET
        clCheck ( cuLaunchKernel ( m_Kern[FUNC_FPREFIXUP], numElem2, 1, 1, threads, 1, 1, 0, NULL, argsE, NULL ), "PrefixSumCellsCL", "cuLaunch", "FUNC_PREFIXFIXUP", mbDebug);
    }
    
    int num_lists = NUM_GENES, length = FDENSE_LIST_LENGTHS, fgridcnt = FBIN_COUNT_ACTIVE_GENES, fgridoff = FBIN_OFFSET_ACTIVE_GENES;
    void* argsF[4] = {&num_lists, &length,&fgridcnt,&fgridoff};
    clCheck ( cuLaunchKernel ( m_Kern[FUNC_TALLYLISTS], NUM_GENES, 1, 1, NUM_GENES, 1, 1, 0, NULL, argsF, NULL ), "PrefixSumCellsCL", "cuLaunch", "FUNC_TALLYLISTS", mbDebug); //256 threads launched
    clCheck( cuMemcpyDtoH ( bufI(&m_Fluid, FDENSE_LIST_LENGTHS), gpuVar(&m_Fluid, FDENSE_LIST_LENGTHS),	sizeof(uint[NUM_GENES]) ), "PrefixSumCellsCL", "cuMemcpyDtoH", "FDENSE_LIST_LENGTHS", mbDebug);
                                                                                    //if active particles for gene > existing buff, then enlarge buff.
    uint * densebuff_len = bufI(&m_Fluid, FDENSE_BUF_LENGTHS);                    // and only bufI(&m_Fluid, FDENSE_LIST_LENGTHS); copied to host.
    uint * denselist_len = bufI(&m_Fluid, FDENSE_LIST_LENGTHS);                   // For each gene allocate intial buffer,
    
    for(int gene=0;gene<NUM_GENES;gene++){                                          // Note this calculation could be done by a kernel, 
      //if (m_FParams.debug>1) cout<<"\nPrefixSumCellsCL()2:gene="<<gene<<", densebuff_len["<<gene<<"]="<<densebuff_len[gene]<<", denselist_len["<<gene<<"]="<<denselist_len[gene]<<" \t"<<std::flush;
        if (denselist_len[gene] > densebuff_len[gene]) {                            // write pointer and size to FDENSE_LISTS and FDENSE_LIST_LENGTHS 
            if (m_FParams.debug>1)printf("\n\nPrefixSumCellsCL: enlarging densebuff_len[%u],  gpuptr(&m_Fluid, FDENSE_LIST_LENGTHS)[gene]=%llu .\t",gene, gpuptr(&m_Fluid, FDENSE_LIST_LENGTHS)[gene] );
            while(denselist_len[gene] >  densebuff_len[gene]) densebuff_len[gene] *=4;                  // bufI(&m_Fluid, FDENSE_BUF_LENGTHS)[i]
            AllocateBufferDenseLists( gene, sizeof(uint), bufI(&m_Fluid, FDENSE_LIST_LENGTHS)[gene], FDENSE_LISTS );   // NB frees previous buffer &=> clears data
        }
    }
    cuMemcpyHtoD(gpuVar(&m_Fluid, FDENSE_LISTS),         bufC(&m_Fluid, FDENSE_LISTS),         NUM_GENES * sizeof(cl_device_idptr)  );  // update pointers to lists on device
    cuMemcpyHtoD(gpuVar(&m_Fluid, FDENSE_BUF_LENGTHS),   bufC(&m_Fluid, FDENSE_BUF_LENGTHS),   NUM_GENES * sizeof(cl_device_idptr)  );

    if (m_FParams.debug>1){ 
        std::cout << "\nChk: PrefixSumCellsCL 4"<<std::flush;
        for(int gene=0;gene<NUM_GENES;gene++){    std::cout<<"\ngene list_length["<<gene<<"]="<<bufI(&m_Fluid, FDENSE_LIST_LENGTHS)[gene]<<"\t"<<std::flush;}
        }
//#endif
}

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
    cl_device_idptr scan0   = gpuVar(&m_Fluid, FBIN_OFFSET);
    
    if(m_debug>3){
        // debug chk
        cout<<"\nSaving (FBIN_COUNT_CHANGES): (bin,#particles) , numElem1="<<numElem1<<"\t"<<std::flush;
        clCheck( cuMemcpyDtoH ( bufI(&m_Fluid, FBIN_COUNT_CHANGES), gpuVar(&m_Fluid, FBIN_COUNT_CHANGES),	sizeof(uint[numElem1]) ), "PrefixSumChangesCL", "cuMemcpyDtoH", "FBIN_COUNT_CHANGES", mbDebug); // NUM_CHANGES*
        //### print to a csv file   AND do the same afterwards for FBIN_OFFSET ###
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
    
    int num_lists = NUM_CHANGES, length = FDENSE_LIST_LENGTHS_CHANGES, fgridcnt = FBIN_COUNT_CHANGES, fgridoff = FBIN_OFFSET;
    void* argsF[4] = {&num_lists, &length,&fgridcnt,&fgridoff};
    clCheck ( cuLaunchKernel ( m_Kern[FUNC_TALLYLISTS], NUM_CHANGES, 1, 1, NUM_CHANGES, 1, 1, 0, NULL, argsF, NULL ), "PrefixSumChangesCL", "cuLaunch", "FUNC_TALLYLISTS", mbDebug); //256 threads launched
    clCheck( cuMemcpyDtoH ( bufI(&m_Fluid, FDENSE_LIST_LENGTHS_CHANGES), gpuVar(&m_Fluid, FDENSE_LIST_LENGTHS_CHANGES),	sizeof(uint[NUM_CHANGES]) ), "PrefixSumChangesCL", "cuMemcpyDtoH", "FDENSE_LIST_LENGTHS_CHANGES", mbDebug);

                                                                                                                    // If active particles for change_list > existing buff, then enlarge buff.
    for(int change_list=0;change_list<NUM_CHANGES;change_list++){                                                   // Note this calculation could be done by a kernel, 
        uint * densebuff_len = bufI(&m_Fluid, FDENSE_BUF_LENGTHS_CHANGES);                                            // and only bufI(&m_Fluid, FDENSE_LIST_LENGTHS); copied to host.
        uint * denselist_len = bufI(&m_Fluid, FDENSE_LIST_LENGTHS_CHANGES);                                           // For each change_list allocate intial buffer,
        //if (m_FParams.debug>1)printf("\nPrefixSumChangesCL: change_list=%u,  densebuff_len[change_list]=%u, denselist_len[change_list]=%u ,",change_list, densebuff_len[change_list], denselist_len[change_list] );
        if (denselist_len[change_list] > densebuff_len[change_list]) {                                              // write pointer and size to FDENSE_LISTS and FDENSE_LIST_LENGTHS 
            while(denselist_len[change_list] >  densebuff_len[change_list])   densebuff_len[change_list] *=4;       // bufI(&m_Fluid, FDENSE_BUF_LENGTHS)[i].
                                                                                                                    // NB Need 2*densebuff_len[change_list] for particle & bond
            if (m_FParams.debug>1)printf("\nPrefixSumChangesCL: ## enlarging buffer## change_list=%u,  densebuff_len[change_list]=%u, denselist_len[change_list]=%u ,",change_list, densebuff_len[change_list], denselist_len[change_list] );
            AllocateBufferDenseLists( change_list, sizeof(uint), 2*densebuff_len[change_list], FDENSE_LISTS_CHANGES );// NB frees previous buffer &=> clears data
        }                                                                                                           // NB buf[2][list_length] holding : particleIdx, bondIdx
    }
    clCheck( cuMemcpyHtoD ( gpuVar(&m_Fluid, FDENSE_BUF_LENGTHS_CHANGES), bufC(&m_Fluid, FDENSE_BUF_LENGTHS_CHANGES),	sizeof(uint[NUM_CHANGES]) ), "PrefixSumChangesCL", "cuMemcpyHtoD", "FDENSE_BUF_LENGTHS_CHANGES", mbDebug);
    cuMemcpyHtoD(gpuVar(&m_Fluid, FDENSE_LISTS_CHANGES), bufC(&m_Fluid, FDENSE_LISTS_CHANGES),  NUM_CHANGES * sizeof(cl_device_idptr)  );                      // update pointers to lists on device
    
    if (m_FParams.debug>1) {
        std::cout << "\nChk: PrefixSumChangesCL 4"<<std::flush;
        for(int change_list=0;change_list<NUM_CHANGES;change_list++){
            std::cout<<"\nPrefixSumChangesCL: change list_length["<<change_list<<"]="<<bufI(&m_Fluid, FDENSE_LIST_LENGTHS_CHANGES)[change_list]<<"\t"<<std::flush;
        }
    }
}

void FluidSystem::CountingSortFullCL ( Vector3DF* ppos ){
    if (m_FParams.debug>1) std::cout << "\nCountingSortFullCL()1: mMaxPoints="<<mMaxPoints<<", mNumPoints="<<mNumPoints<<",\tmActivePoints="<<mActivePoints<<".\n"<<std::flush;
    // get number of active particles & set short lists for later kernels
    int grid_ScanMax = (m_FParams.gridScanMax.y * m_FParams.gridRes.z + m_FParams.gridScanMax.z) * m_FParams.gridRes.x + m_FParams.gridScanMax.x;
    
    clCheck( cuMemcpyDtoH ( &mNumPoints,  gpuVar(&m_Fluid, FBIN_OFFSET)+(m_GridTotal-1/*grid_ScanMax+1*/)*sizeof(int), sizeof(int) ), "CountingSortFullCL1", "cuMemcpyDtoH", "FBIN_OFFSET", mbDebug);
    
    clCheck( cuMemcpyDtoH ( &mActivePoints,  gpuVar(&m_Fluid, FBIN_OFFSET)+(grid_ScanMax/*-1*/)*sizeof(int), sizeof(int) ), "CountingSortFullCL2", "cuMemcpyDtoH", "FBIN_OFFSET", mbDebug);
    /*
    int totalPoints = 0;
    clCheck( cuMemcpyDtoH ( &totalPoints,  gpuVar(&m_Fluid, FBIN_OFFSET)+(m_GridTotal)*sizeof(int), sizeof(int) ), "CountingSortFullCL3", "cuMemcpyDtoH", "FBIN_OFFSET", mbDebug);
    std::cout<<"\nCountingSortFullCL(): totalPoints="<<totalPoints<<std::flush;
    */
    m_FParams.pnumActive = mActivePoints;                                     // TODO eliminate duplication of information & variables between fluid.h and fluid_system.h                               
    clCheck ( cuMemcpyHtoD ( clFParams,	&m_FParams, sizeof(FParams) ), "CountingSortFullCL3", "cuMemcpyHtoD", "clFParams", mbDebug); // seems the safest way to update fparam.pnumActive on device.
    
    if (m_FParams.debug>1) std::cout<<"\nCountingSortFullCL()2: mMaxPoints="<<mMaxPoints<<" mNumPoints="<<mNumPoints<<",\tmActivePoints="<<mActivePoints<<",  m_GridTotal="<<m_GridTotal<<", grid_ScanMax="<<grid_ScanMax<<"\n"<<std::flush;

    // Transfer particle data to temp buffers
    //  (gpu-to-gpu copy, no sync needed)
    //TransferToTempCL ( FPOS,		mMaxPoints *sizeof(Vector3DF) );    // NB if some points have been removed, then the existing list is no longer dense,  
    //TransferToTempCL ( FVEL,		mMaxPoints *sizeof(Vector3DF) );    // hence must use mMaxPoints, not mNumPoints
    //TransferToTempCL ( FVEVAL,	mMaxPoints *sizeof(Vector3DF) );    // { Could potentially use (old_mNumPoints + mNewPoints) instead of mMaxPoints}
    TransferToTempCL ( FFORCE,	mMaxPoints *sizeof(Vector3DF) );    // NB buffers are declared and initialized on mMaxPoints.
    TransferToTempCL ( FPRESS,	mMaxPoints *sizeof(float) );
    TransferToTempCL ( FDENSITY,	mMaxPoints *sizeof(float) );
    TransferToTempCL ( FCOLOR,		mMaxPoints *sizeof(uint) );
    TransferToTempCL ( FAGE,		mMaxPoints *sizeof(uint) );
    TransferToTempCL ( FGCELL,	mMaxPoints *sizeof(uint) );
    TransferToTempCL ( FGNDX,		mMaxPoints *sizeof(uint) );
    
    // extra data for morphogenesis
    TransferToTempCL ( FELASTIDX,		mMaxPoints *sizeof(uint[BOND_DATA]) );
    TransferToTempCL ( FPARTICLEIDX,	mMaxPoints *sizeof(uint[BONDS_PER_PARTICLE *2]) );
    TransferToTempCL ( FPARTICLE_ID,	mMaxPoints *sizeof(uint) );
    TransferToTempCL ( FMASS_RADIUS,	mMaxPoints *sizeof(uint) );
    TransferToTempCL ( FNERVEIDX,		mMaxPoints *sizeof(uint) );
    TransferToTempCL ( FCONC,		    mMaxPoints *sizeof(float[NUM_TF]) );
    TransferToTempCL ( FEPIGEN,	    mMaxPoints *sizeof(uint[NUM_GENES]) );

    // debug chk
    //clCheck( cuMemcpyDtoH ( bufI(&m_Fluid, FEPIGEN), gpuVar(&m_FluidTemp, FEPIGEN),	mMaxPoints *sizeof(uint[NUM_GENES]) ), "CountingSortFullCL4", "cuMemcpyDtoH", "FBIN_COUNT", mbDebug);
    //SaveUintArray_2D( bufI(&m_Fluid, FEPIGEN), mMaxPoints, NUM_GENES, "CountingSortFullCL__m_FluidTemp.bufI(FEPIGEN)2.csv" );
    
    // reset bonds and forces in fbuf FELASTIDX, FPARTICLEIDX and FFORCE, required to prevent interference between time steps, 
    // because these are not necessarily overwritten by the FUNC_COUNTING_SORT_FULL kernel.
    clFinish ();    // needed to prevent colision with previous operations
    
    float max_pos = max(max(m_Vec[PVOLMAX].x, m_Vec[PVOLMAX].y), m_Vec[PVOLMAX].z);
    uint * uint_max_pos = (uint*)&max_pos;
    clCheck ( clMemsetD32 ( gpuVar(&m_Fluid, FPOS), *uint_max_pos, mMaxPoints * 3 ),  "CountingSortFullCL", "clMemsetD32", "FELASTIDX",   mbDebug);
    
    //cout<<"\nCountingSortFullCL: m_Vec[PVOLMAX]=("<<m_Vec[PVOLMAX].x<<", "<<m_Vec[PVOLMAX].y<<", "<<m_Vec[PVOLMAX].z<<"),  max_pos = "<< max_pos <<std::flush;
    // NB resetting  gpuVar(&m_Fluid, FPOS)  ensures no zombie particles. ?hopefully?
    
    clCheck ( clMemsetD32 ( gpuVar(&m_Fluid, FELASTIDX),    UINT_MAXSIZE,  mMaxPoints * BOND_DATA              ),  "CountingSortFullCL", "clMemsetD32", "FELASTIDX",    mbDebug);
    clCheck ( clMemsetD32 ( gpuVar(&m_Fluid, FPARTICLEIDX), UINT_MAXSIZE,  mMaxPoints * BONDS_PER_PARTICLE *2  ),  "CountingSortFullCL", "clMemsetD32", "FPARTICLEIDX", mbDebug);
    clCheck ( clMemsetD32 ( gpuVar(&m_Fluid, FPARTICLE_ID), UINT_MAXSIZE,  mMaxPoints                          ),  "CountingSortFullCL", "clMemsetD32", "FPARTICLEIDX", mbDebug);
    clCheck ( clMemsetD32 ( gpuVar(&m_Fluid, FFORCE),      (uint)0.0,  mMaxPoints * 3 /* ie num elements */),  "CountingSortFullCL", "clMemsetD32", "FFORCE",       mbDebug);
    clCheck ( clMemsetD32 ( gpuVar(&m_Fluid, FCONC),             0.0,  mMaxPoints * NUM_TF                 ),  "CountingSortFullCL", "clMemsetD32", "FCONC",        mbDebug);
    clCheck ( clMemsetD32 ( gpuVar(&m_Fluid, FEPIGEN),     (uint)0.0,  mMaxPoints * NUM_GENES              ),  "CountingSortFullCL", "clMemsetD32", "FEPIGEN",      mbDebug);
    clFinish ();    // needed to prevent colision with previous operations

    // Reset grid cell IDs
    // clCheck(clMemsetD32(gpuVar(&m_Fluid, FGCELL), GRID_UNDEF, numPoints ), "clMemsetD32(Sort)");
    void* args[1] = { &mMaxPoints };
    clCheck ( cuLaunchKernel ( m_Kern[FUNC_COUNTING_SORT_FULL], m_FParams.numGroups, 1, 1, m_FParams.numItems, 1, 1, 0, NULL, args, NULL),
              "CountingSortFullCL5", "cuLaunch", "FUNC_COUNTING_SORT", mbDebug );

    // Having sorted the particle data, we can start using a shortened list of particles.
    // NB have to reset to long list at start of time step. 
    computeNumBlocks ( m_FParams.pnumActive, m_FParams.itemsPerGroup, m_FParams.numGroups, m_FParams.numItems);				// particles
    
    if (m_FParams.debug>1) std::cout<<"\n CountingSortFullCL : FUNC_COUNT_SORT_LISTS\n"<<std::flush;
    // countingSortDenseLists ( int pnum ) // NB launch on bins not particles.
    int blockSize = SCAN_BLOCKSIZE/2 << 1; 
    int numElem1 = m_GridTotal;  
    int numElem2 = 2*  int( numElem1 / blockSize ) + 1;  
    int threads = SCAN_BLOCKSIZE/2;
    clFinish ();
    clCheck ( cuLaunchKernel ( m_Kern[FUNC_COUNT_SORT_DENSE_LISTS], /*m_FParams.numGroups*/ numElem2, 1, 1, /*m_FParams.numItems/2*/ threads , 1, 1, 0, NULL, args, NULL),
              "CountingSortFullCL7", "cuLaunch", "FUNC_COUNT_SORT_LISTS", mbDebug );                                   // NB threads/2 required on GTX970m
    clFinish ();
    
    if(m_FParams.debug>3){//debug chk
        std::cout<<"\n### Saving UintArray .csv files."<<std::flush;
        
        clCheck( cuMemcpyDtoH ( bufI(&m_Fluid, FEPIGEN), gpuVar(&m_FluidTemp, FEPIGEN),	mMaxPoints *sizeof(uint[NUM_GENES]) ), "CountingSortFullCL8", "cuMemcpyDtoH", "FBIN_COUNT", mbDebug);
        SaveUintArray_2D( bufI(&m_Fluid, FEPIGEN), mMaxPoints, NUM_GENES, "CountingSortFullCL__m_FluidTemp.bufI(FEPIGEN)3.csv" );
        
        clCheck( cuMemcpyDtoH ( bufI(&m_Fluid, FEPIGEN), gpuVar(&m_Fluid, FEPIGEN),	/*mMaxPoints*/mNumPoints *sizeof(uint[NUM_GENES]) ), "PrefixSumChangesCL", "cuMemcpyDtoH", "FBIN_COUNT", mbDebug);
        SaveUintArray_2D( bufI(&m_Fluid, FEPIGEN), mMaxPoints, NUM_GENES, "CountingSortFullCL__bufI(&m_Fluid, FEPIGEN)3.csv" );
        
        clCheck( cuMemcpyDtoH ( bufI(&m_Fluid, FBIN_COUNT), gpuVar(&m_Fluid, FBIN_COUNT),	sizeof(uint[m_GridTotal]) ), "CountingSortFullCL9", "cuMemcpyDtoH", "FBIN_COUNT", mbDebug);
        SaveUintArray( bufI(&m_Fluid, FBIN_COUNT), m_GridTotal, "CountingSortFullCL__bufI(&m_Fluid, FBIN_COUNT).csv" );
        
        clCheck( cuMemcpyDtoH ( bufI(&m_Fluid, FBIN_OFFSET), gpuVar(&m_Fluid, FBIN_OFFSET),	sizeof(uint[m_GridTotal]) ), "CountingSortFullCL10", "cuMemcpyDtoH", "FBIN_OFFSET", mbDebug);
        SaveUintArray( bufI(&m_Fluid, FBIN_OFFSET), m_GridTotal, "CountingSortFullCL__bufI(&m_Fluid, FBIN_OFFSET).csv" );
    
       // uint fDenseList2[100000];
       // cl_device_idptr*  _list2pointer = (cl_device_idptr*) &bufC(&m_Fluid, FDENSE_LISTS)[2 * sizeof(cl_device_idptr)];
       // clCheck( cuMemcpyDtoH ( fDenseList2, *_list2pointer,	sizeof(uint[ bufI(&m_Fluid, FDENSE_LIST_LENGTHS)[2] ])/*sizeof(uint[2000])*/ ), "CountingSortFullCL11", "cuMemcpyDtoH", "FBIN_COUNT", mbDebug);
       // SaveUintArray( fDenseList2, bufI(&m_Fluid, FDENSE_LIST_LENGTHS)[2], "CountingSortFullCL__m_Fluid.bufII(FDENSE_LISTS)[2].csv" );
    }
}

void FluidSystem::CountingSortChangesCL ( ){
    //std::cout<<"\n\n#### CountingSortChangesCL ( ):   m_FParams.debug = "<< m_FParams.debug <<"\n";
    if (m_FParams.debug>1) {std::cout<<"\n\n#### CountingSortChangesCL ( )"<<std::flush;}
    /* ////////
    clCheck( cuMemcpyDtoH ( bufI(&m_Fluid, FDENSE_LIST_LENGTHS_CHANGES), gpuVar(&m_Fluid, FDENSE_LIST_LENGTHS_CHANGES),	sizeof(uint[NUM_CHANGES]) ), "PrefixSumCellsCL", "cuMemcpyDtoH", "FDENSE_LIST_LENGTHS_CHANGES", mbDebug);
    
                                                                                                                    // If active particles for change_list > existing buff, then enlarge buff.
    for(int change_list=0;change_list<NUM_CHANGES;change_list++){                                                   // Note this calculation could be done by a kernel, 
        uint * densebuff_len = bufI(&m_Fluid, FDENSE_BUF_LENGTHS_CHANGES);                                            // and only bufI(&m_Fluid, FDENSE_LIST_LENGTHS); copied to host.
        uint * denselist_len = bufI(&m_Fluid, FDENSE_LIST_LENGTHS_CHANGES);                                           // For each change_list allocate intial buffer,
        if (m_FParams.debug>1)printf("\nCountingSortChangesCL1: change_list=%u,  densebuff_len[change_list]=%u, denselist_len[change_list]=%u ,",change_list, densebuff_len[change_list], denselist_len[change_list] );
    }
    *//////////
    int blockSize = SCAN_BLOCKSIZE/2 << 1; 
    int numElem1 = m_GridTotal;  
    int numElem2 = 2* int( numElem1 / blockSize ) + 1;  
    int threads = SCAN_BLOCKSIZE/2;
    void* args[1] = { &mActivePoints };
    clCheck ( cuLaunchKernel ( m_Kern[FUNC_COUNTING_SORT_CHANGES], numElem2, 1, 1, threads , 1, 1, 0, NULL, args, NULL),
              "CountingSortChangesCL", "cuLaunch", "FUNC_COUNTING_SORT_CHANGES", mbDebug );   
     /////////
    clCheck(clFinish(), "CountingSortChangesCL()", "clFinish", "After FUNC_COUNTING_SORT_CHANGES", mbDebug);
    
    clCheck( cuMemcpyDtoH ( bufI(&m_Fluid, FDENSE_LIST_LENGTHS_CHANGES), gpuVar(&m_Fluid, FDENSE_LIST_LENGTHS_CHANGES),	sizeof(uint[NUM_CHANGES]) ), "PrefixSumCellsCL", "cuMemcpyDtoH", "FDENSE_LIST_LENGTHS_CHANGES", mbDebug);
                                                                                                                    // If active particles for change_list > existing buff, then enlarge buff.
    for(int change_list=0;change_list<NUM_CHANGES;change_list++){                                                   // Note this calculation could be done by a kernel, 
        uint * densebuff_len = bufI(&m_Fluid, FDENSE_BUF_LENGTHS_CHANGES);                                            // and only bufI(&m_Fluid, FDENSE_LIST_LENGTHS); copied to host.
        uint * denselist_len = bufI(&m_Fluid, FDENSE_LIST_LENGTHS_CHANGES);                                           // For each change_list allocate intial buffer,
        if (m_FParams.debug>1)printf("\nCountingSortChangesCL2: change_list=%u,  densebuff_len[change_list]=%u, denselist_len[change_list]=%u ,\t\t threads=%u, numElem2=%u,  m_GridTotal=%u \t",
               change_list, densebuff_len[change_list], denselist_len[change_list], threads, numElem2,  m_GridTotal );
        clFinish ();
        if(m_FParams.debug>0){
            uint fDenseList2[1000000] = {UINT_MAXSIZE};//TODO make this array size safe!  NB 10* num particles.
            cl_device_idptr*  _list2pointer = (cl_device_idptr*) &bufC(&m_Fluid, FDENSE_LISTS_CHANGES)[change_list*sizeof(cl_device_idptr)];
                                                                                                                // Get device pointer to FDENSE_LISTS_CHANGES[change_list].
            clCheck( cuMemcpyDtoH ( fDenseList2, *_list2pointer,	2*sizeof(uint[densebuff_len[change_list]]) ), "PrefixSumChangesCL", "cuMemcpyDtoH", "FBIN_COUNT", mbDebug);
            char filename[256];
            sprintf(filename, "CountingSortChangesCL__m_Fluid.bufII(FDENSE_LISTS_CHANGES)[%u].csv", change_list);
            SaveUintArray_2Columns( fDenseList2, denselist_len[change_list], densebuff_len[change_list], filename );
            ///
            printf("\n\n*_list2pointer=%llu",*_list2pointer);
            
        }
    }
}

void FluidSystem::InitializeBondsCL (){
    if (m_FParams.debug>1)cout << "\n\nInitializeBondsCL ()\n"<<std::flush;
    uint gene           = 1;                                                            // solid  (has springs)
    uint list_length    = bufI(&m_Fluid, FDENSE_LIST_LENGTHS)[gene];
    void* args[3]       = { &m_FParams.pnumActive, &list_length, &gene};                //initialize_bonds (int ActivePoints, uint list_length, uint gene)
    int numGroups, numItems;
    computeNumBlocks (list_length, m_FParams.itemsPerGroup, numGroups, numItems);

    if (m_FParams.debug>1)cout << "\nInitializeBondsCL (): list_length="<<list_length<<", m_FParams.itemsPerGroup="<<m_FParams.itemsPerGroup<<", numGroups="<<numGroups<<", numItems="<<numItems<<" \t args{m_FParams.pnumActive="<<m_FParams.pnumActive<<", list_length="<<list_length<<", gene="<<gene<<"}"<<std::flush;

    clCheck ( cuLaunchKernel ( m_Kern[FUNC_INITIALIZE_BONDS],  m_FParams.numGroups, 1, 1, m_FParams.numItems, 1, 1, 0, NULL, args, NULL), "ComputePressureCL", "cuLaunch", "FUNC_COMPUTE_PRESS", mbDebug);
}

void FluidSystem::ComputePressureCL (){
    void* args[1] = { &mActivePoints };
    //cout<<"\nComputePressureCL: mActivePoints="<<mActivePoints<<std::flush;
    clCheck ( cuLaunchKernel ( m_Kern[FUNC_COMPUTE_PRESS],  m_FParams.numGroups, 1, 1, m_FParams.numItems, 1, 1, 0, NULL, args, NULL), "ComputePressureCL", "cuLaunch", "FUNC_COMPUTE_PRESS", mbDebug);
}

void FluidSystem::ComputeDiffusionCL(){
    //if (m_FParams.debug>1) std::cout << "\n\nRunning ComputeDiffusionCL()" << std::endl;
    void* args[1] = { &mActivePoints };
    clCheck ( cuLaunchKernel ( m_Kern[FUNC_COMPUTE_DIFFUSION],  m_FParams.numGroups, 1, 1, m_FParams.numItems, 1, 1, 0, NULL, args, NULL), "ComputeDiffusionCL", "cuLaunch", "FUNC_COMPUTE_DIFFUSION", mbDebug);
}

void FluidSystem::ComputeForceCL (){
    //if (m_FParams.debug>1)printf("\n\nFluidSystem::ComputeForceCL (),  m_FParams.freeze=%s",(m_FParams.freeze==true) ? "true" : "false");
    void* args[3] = { &m_FParams.pnumActive ,  &m_FParams.freeze, &m_FParams.frame};
    clCheck ( cuLaunchKernel ( m_Kern[FUNC_COMPUTE_FORCE],  m_FParams.numGroups, 1, 1, m_FParams.numItems, 1, 1, 0, NULL, args, NULL), "ComputeForceCL", "cuLaunch", "FUNC_COMPUTE_FORCE", mbDebug);
}

void FluidSystem::ComputeGenesCL (){  // for each gene, call a kernel wih the dese list for that gene
    // NB Must zero ftemp.bufI(FEPIGEN) and ftemp.bufI(FCONC) before calling kernel. ftemp is used to collect changes before FUNC_TALLY_GENE_ACTION.
    clCheck ( cuMemsetD8 ( gpuVar(&m_FluidTemp, FCONC),   0,	m_FParams.szPnts *sizeof(float[NUM_TF])   ), "ComputeGenesCL", "cuMemsetD8", "gpuVar(&m_FluidTemp, FCONC)",   mbDebug );
    clCheck ( cuMemsetD8 ( gpuVar(&m_FluidTemp, FEPIGEN), 0,	m_FParams.szPnts *sizeof(uint[NUM_GENES]) ), "ComputeGenesCL", "cuMemsetD8", "gpuVar(&m_FluidTemp, FEPIGEN)", mbDebug );
    for (int gene=0;gene<NUM_GENES;gene++) {
        uint list_length = bufI(&m_Fluid, FDENSE_LIST_LENGTHS)[gene];
        void* args[3] = { &m_FParams.pnumActive, &gene, &list_length };
        int numGroups, numItems;
        computeNumBlocks ( list_length , m_FParams.itemsPerGroup, numGroups, numItems);
        
        if (m_FParams.debug>1) std::cout<<"\nComputeGenesCL (): gene ="<<gene<<", list_length="<<list_length<<", m_FParams.itemsPerGroup="<<m_FParams.itemsPerGroup<<", numGroups="<<numGroups<<",  numItems="<<numItems<<". args={mNumPoints="<<mNumPoints<<", list_length="<<list_length<<", gene ="<<gene<<"}"<<std::flush;
        
        if( numGroups>0 && numItems>0){
            //std::cout<<"\nCalling m_Kern[FUNC_COMPUTE_GENE_ACTION], list_length="<<list_length<<", numGroups="<<numGroups<<", numItems="<<numItems<<"\n"<<std::flush;
            clCheck ( cuLaunchKernel ( m_Kern[FUNC_COMPUTE_GENE_ACTION],  numGroups, 1, 1, numItems, 1, 1, 0, NULL, args, NULL), "ComputeGenesCL", "cuLaunch", "FUNC_COMPUTE_GENE_ACTION", mbDebug);
        }
    }
    clCheck(clFinish(), "ComputeGenesCL", "clFinish", "After FUNC_COMPUTE_GENE_ACTION & before FUNC_TALLY_GENE_ACTION", mbDebug);
    for (int gene=0;gene<NUM_GENES;gene++) {
        uint list_length = bufI(&m_Fluid, FDENSE_LIST_LENGTHS)[gene];
        void* args[3] = { &m_FParams.pnumActive, &gene, &list_length };
        int numGroups, numItems;
        computeNumBlocks ( list_length , m_FParams.itemsPerGroup, numGroups, numItems);
        
        if( numGroups>0 && numItems>0){
            if (m_FParams.debug>1) std::cout<<"\nCalling m_Kern[FUNC_TALLY_GENE_ACTION], gene="<<gene<<", list_length="<<list_length<<", numGroups="<<numGroups<<", numItems="<<numItems<<"\n"<<std::flush;
            clCheck ( cuLaunchKernel ( m_Kern[FUNC_TALLY_GENE_ACTION],  numGroups, 1, 1, numItems, 1, 1, 0, NULL, args, NULL), "ComputeGenesCL", "cuLaunch", "FUNC_TALLY_GENE_ACTION", mbDebug);
        }
    }
}

void FluidSystem::AssembleFibresCL (){  //kernel: void assembleMuscleFibres ( int pnum, uint list, uint list_length )
    if (m_FParams.debug>1)cout << "\n\nAssembleFibresCL ()\n"<<std::flush;
    uint gene = 7; // muscle
    uint list_length = bufI(&m_Fluid, FDENSE_LIST_LENGTHS)[gene];
    void* args[3] = { &m_FParams.pnumActive, &gene, &list_length };
    int numGroups, numItems;
    computeNumBlocks ( list_length , m_FParams.itemsPerGroup, numGroups, numItems);
    
    /*
    if( numGroups>0 && numItems>0){
        clCheck ( cuLaunchKernel ( m_Kern[FUNC_ASSEMBLE_MUSCLE_FIBRES_OUTGOING],  numGroups, 1, 1, numItems, 1, 1, 0, NULL, args, NULL), "ComputeGenesCL", "cuLaunch", "FUNC_COMPUTE_GENE_ACTION", mbDebug);
    }
    */
    
    clCheck(clFinish(), "Run", "clFinish", "In AssembleFibresCL, after OUTGOING", mbDebug); 
    
    /*
    if( numGroups>0 && numItems>0){
        clCheck ( cuLaunchKernel ( m_Kern[FUNC_ASSEMBLE_MUSCLE_FIBRES_INCOMING],  numGroups, 1, 1, numItems, 1, 1, 0, NULL, args, NULL), "ComputeGenesCL", "cuLaunch", "FUNC_COMPUTE_GENE_ACTION", mbDebug);
    }
    clCheck(clFinish(), "Run", "clFinish", "In AssembleFibresCL, after OUTGOING", mbDebug); 
    */
    

    
    
    // Kernels:  call by tissue type using dense lists by gene.
    //assembleMuscleFibres()
    //assembleFasciaFibres ()
    if (m_FParams.debug>1) cout << "\nFinished AssembleFibresCL ()\n\n"<<std::flush;
}

void FluidSystem::ComputeBondChangesCL (uint steps_per_InnerPhysicalLoop){// Given the action of the genes, compute the changes to particle properties & splitting/combining  NB also "inserts changes" 
//  if (m_FParams.debug>1)printf("\n gpuVar(&m_Fluid, FBIN_OFFSET)=%llu   ,\t gpuVar(&m_Fluid, FBIN_COUNT_CHANGES)=%llu   \n",gpuVar(&m_Fluid, FBIN_OFFSET) , gpuVar(&m_Fluid, FBIN_COUNT_CHANGES)   );
  
    clCheck ( cuMemsetD8 ( gpuVar(&m_Fluid, FBIN_OFFSET), 0,	m_GridTotal *sizeof(uint[NUM_CHANGES]) ), "ComputeBondChangesCL", "cuMemsetD8", "FBIN_OFFSET", mbDebug );
                                            //NB list for all living cells. (non senescent) = FEPIGEN[2]
    clCheck ( cuMemsetD8 ( gpuVar(&m_Fluid, FBIN_COUNT_CHANGES), 0,	m_GridTotal *sizeof(uint[NUM_CHANGES]) ), "ComputeBondChangesCL", "cuMemsetD8", "FBIN_COUNT", mbDebug );
    
    uint list_length = bufI(&m_Fluid, FDENSE_LIST_LENGTHS)[2];    // call for dense list of living cells (gene'2'living/telomere (has genes))
    void* args[3] = { &mActivePoints, &list_length, &steps_per_InnerPhysicalLoop};
    int numGroups, numItems;
    computeNumBlocks (list_length, m_FParams.itemsPerGroup, numGroups, numItems);
    
    //std::cout<<"\n\nComputeBondChangesCL (): m_FParams.debug = "<<m_FParams.debug<<", (m_FParams.debug>1)="<<(m_FParams.debug>1)<<"\n"<<std::flush;
    
    if (m_FParams.debug>1) std::cout<<"\n\nComputeBondChangesCL (): list_length="<<list_length<<", m_FParams.itemsPerGroup="<<m_FParams.itemsPerGroup<<", numGroups="<<numGroups<<",  numItems="<<numItems<<". \t\t args={mActivePoints="<<mActivePoints<<", list_length="<<list_length<<"}\n\n"<<std::flush;
    
    clCheck ( cuLaunchKernel ( m_Kern[FUNC_COMPUTE_BOND_CHANGES],  m_FParams.numGroups, 1, 1, m_FParams.numItems, 1, 1, 0, NULL, args, NULL), "computeBondChanges", "cuLaunch", "FUNC_COMPUTE_BOND_CHANGES", mbDebug);
}

void FluidSystem::ComputeParticleChangesCL (){// Call each for dense list to execute particle changes. NB Must run concurrently without interfering => no clFinish()
    uint startNewPoints = mActivePoints + 1;
    if (m_FParams.debug>2)printf("\n");
    for (int change_list = 0; change_list<NUM_CHANGES;change_list++){
    //int change_list = 0; // TODO debug, chk one kernel at a time
        uint list_length = bufI(&m_Fluid, FDENSE_LIST_LENGTHS_CHANGES)[change_list];  // num blocks and threads by list length
        //uint list_length = bufI(&m_Fluid, FDENSE_BUF_LENGTHS_CHANGES)[change_list];
        //if (change_list!=0 && change_list!=1)continue; // only test heal() and lengthenTissue() for now.
        
        if (m_FParams.debug>2)printf("\n\nComputeParticleChangesCL(): startNewPoints=%u, change_list=%u, list_length=%u, mMaxPoints=%u \t", 
            startNewPoints, change_list, list_length, mMaxPoints); 
    
        if ((change_list >0)&&(startNewPoints + list_length > mMaxPoints)){         // NB heal() does not create new bonds.
            printf("\n\n### Run out of spare particles. startNewPoints=%u, change_list=%u, list_length=%u, mMaxPoints=%u ###\n", 
            startNewPoints, change_list, list_length, mMaxPoints); 
            list_length = mMaxPoints - startNewPoints;
            Exit();
        }//
    
        void* args[5] = {&mActivePoints, &list_length, &change_list, &startNewPoints, &mMaxPoints};
        int numItems, numGroups;
        
        //int numItems = 1;//m_FParams.itemsPerGroup;
        //int numGroups  = 1;//iDivUp ( list_length, numItems );
        
        computeNumBlocks (list_length, m_FParams.itemsPerGroup, numGroups, numItems);
        
        if (m_FParams.debug>2) std::cout
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
            if (m_FParams.debug>0) std::cout
                <<"\nComputeParticleChangesCL ():"
                <<"\tCalling m_Kern[FUNC_HEAL+"             <<change_list
                <<"], list_length="                         <<list_length
                <<", numGroups="                            <<numGroups
                <<", numItems="                           <<numItems
                <<",\t m_FParams.itemsPerGroup="          <<m_FParams.itemsPerGroup
                <<", numGroups*m_FParams.itemsPerGroup="  <<numGroups*m_FParams.itemsPerGroup
                <<"\t"<<std::flush;
            
            clCheck ( cuLaunchKernel ( m_Kern[FUNC_HEAL+change_list], numGroups, 1, 1, numItems, 1, 1, 0, NULL, args, NULL),
                  "ComputeParticleChangesCL", "cuLaunch", "FUNC_HEAL+change_list", mbDebug);
        }
        clCheck(clFinish(), "ComputeParticleChangesCL", "clFinish", "In ComputeParticleChangesCL", mbDebug);
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
    if (m_FParams.debug>1) std::cout<<"\nFinished ComputeParticleChangesCL ()\n"<<std::flush;
}

void FluidSystem::CleanBondsCL (){
    void* args[3] = { &m_FParams.pnumActive};
    clCheck ( cuLaunchKernel ( m_Kern[FUNC_CLEAN_BONDS],  m_FParams.numGroups, 1, 1, m_FParams.numItems, 1, 1, 0, NULL, args, NULL), "CleanBondsCL", "cuLaunch", "FUNC_CLEAN_BONDS", mbDebug);
}

// void FluidSystem::TransferPosVelVeval (){
//     TransferToTempCL ( FPOS,		mMaxPoints *sizeof(Vector3DF) );    // NB if some points have been removed, then the existing list is no longer dense,
//     TransferToTempCL ( FVEL,		mMaxPoints *sizeof(Vector3DF) );    // hence must use mMaxPoints, not mNumPoints
//     TransferToTempCL ( FVEVAL,	mMaxPoints *sizeof(Vector3DF) );
// }

void FluidSystem::TransferPosVelVevalFromTemp (){
    TransferFromTempCL ( FPOS,	mMaxPoints *sizeof(Vector3DF) );    // NB if some points have been removed, then the existing list is no longer dense,  
    TransferFromTempCL ( FVEL,	mMaxPoints *sizeof(Vector3DF) );    // hence must use mMaxPoints, not mNumPoints
    TransferFromTempCL ( FVEVAL,	mMaxPoints *sizeof(Vector3DF) );
}

void FluidSystem::ZeroVelCL (){                                       // Used to remove velocity, kinetic energy and momentum during initialization.
    clCheck(clFinish(), "Run", "clFinish", "After freeze Run2PhysicalSort ", mbDebug);
    clCheck ( clMemsetD32 ( gpuVar(&m_Fluid, FVEL),   0.0,  mMaxPoints ),  "ZeroVelCL", "clMemsetD32", "FVEL",        mbDebug);
    clCheck ( clMemsetD32 ( gpuVar(&m_Fluid, FVEVAL), 0.0,  mMaxPoints ),  "ZeroVelCL", "clMemsetD32", "FVEVAL",      mbDebug);
    clCheck(clFinish(), "Run", "clFinish", "After freeze ZeroVelCL ", mbDebug);
}

void FluidSystem::AdvanceCL ( float tm, float dt, float ss ){
    void* args[4] = { &tm, &dt, &ss, &m_FParams.pnumActive };
    clCheck ( cuLaunchKernel ( m_Kern[FUNC_ADVANCE],  m_FParams.numGroups, 1, 1, m_FParams.numItems, 1, 1, 0, NULL, args, NULL), "AdvanceCL", "cuLaunch", "FUNC_ADVANCE", mbDebug);
    //cout<<"\nAdvanceCL: m_FParams.pnumActive="<<m_FParams.pnumActive<<std::flush;
}

void FluidSystem::SpecialParticlesCL (float tm, float dt, float ss){   // For interaction.Using dense lists for gene 1 & 0.
    int gene = 12;                                                           // 'externally actuated' particles
    uint list_length = bufI(&m_Fluid, FDENSE_LIST_LENGTHS)[gene];
    void* args[5] = {&list_length, &tm, &dt, &ss, &m_FParams.pnumActive};         // void externalActuation (uint list_len,  float time, float dt, float ss, int numPnts )
    int numGroups, numItems;
    computeNumBlocks ( list_length , m_FParams.itemsPerGroup, numGroups, numItems);
    
    if (m_FParams.debug>1) std::cout<<"\nSpecialParticlesCL:EXTERNAL_ACTUATION: list_length="<<list_length<<" , m_FParams.itemsPerGroup="<< m_FParams.itemsPerGroup <<", numGroups="<< numGroups <<", numItems="<< numItems <<", args{m_FParams.pnum="<< m_FParams.pnum <<",  gene="<< gene <<", list_length="<< list_length <<"  }  \n"<<std::flush;
    
    if( numGroups>0 && numItems>0){
        if (m_FParams.debug>1) std::cout<<"\nCalling m_Kern[FUNC_EXTERNAL_ACTUATION], list_length="<<list_length<<", numGroups="<<numGroups<<", numItems="<<numItems<<"\n"<<std::flush;
        clCheck ( cuLaunchKernel ( m_Kern[FUNC_EXTERNAL_ACTUATION],  numGroups, 1, 1, numItems, 1, 1, 0, NULL, args, NULL), "SpecialParticlesCL", "cuLaunch", "FUNC_EXTERNAL_ACTUATION", mbDebug);
    }
    gene =11;                                                                // 'fixed' particles
    list_length = bufI(&m_Fluid, FDENSE_LIST_LENGTHS)[gene];
    args[0] = &list_length;                                                 // void fixedParticles (uint list_len, int numPnts )
    args[1] = &m_FParams.pnum;
    computeNumBlocks ( list_length , m_FParams.itemsPerGroup, numGroups, numItems);
    
    if (m_FParams.debug>1) std::cout<<"\nSpecialParticlesCL:FIXED: list_length="<<list_length<<" , m_FParams.itemsPerGroup="<< m_FParams.itemsPerGroup <<", numGroups="<< numGroups <<", numItems="<< numItems <<", args{m_FParams.pnum="<< m_FParams.pnum <<",  gene="<< gene <<", list_length="<< list_length <<"  }  \n"<<std::flush;
    
    if( numGroups>0 && numItems>0){
        if (m_FParams.debug>1) std::cout<<"\nCalling m_Kern[FUNC_FIXED], list_length="<<list_length<<", numGroups="<<numGroups<<", numItems="<<numItems<<"\n"<<std::flush;
        clCheck ( cuLaunchKernel ( m_Kern[FUNC_FIXED],  numGroups, 1, 1, numItems, 1, 1, 0, NULL, args, NULL), "SpecialParticlesCL", "cuLaunch", "FUNC_FIXED", mbDebug);
    }
}

void FluidSystem::EmitParticlesCL ( float tm, int cnt ){
    void* args[3] = { &tm, &cnt, &m_FParams.pnum };
    clCheck ( cuLaunchKernel ( m_Kern[FUNC_EMIT],  m_FParams.numGroups, 1, 1, m_FParams.numItems, 1, 1, 0, NULL, args, NULL), "EmitParticlesCL", "cuLaunch", "FUNC_EMIT", mbDebug);
}

