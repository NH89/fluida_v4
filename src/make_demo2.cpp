#include <stdio.h>
#include <inttypes.h>
#include <errno.h>
#include <string.h>
#include <chrono>
#include <filesystem>
#include <fstream>
#include <iostream>
//#include <boost/filesystem.hpp>

#include "fluid_system.h"

typedef	unsigned int		uint;	
using namespace std;

int main ( int argc, const char** argv ) 
{
    char input_folder[256];
    char output_folder[256];
    if ((argc != 3) && (argc !=2)) {
        printf ( "usage: make_demo2 input_folder output_folder.\
        \nNB input_folder must contain \"SpecificationFile.txt\", output will be wrtitten to \"output_folder/out_data_time/\".\
        \nIf output_folder is not given the value from SpecificationFile.txt will be used.\n" );
        return 0;
    } else {
        auto now = std::chrono::system_clock::now();
        auto in_time_t = std::chrono::system_clock::to_time_t(now);
        std::stringstream datetime;
        datetime << std::put_time(std::localtime(&in_time_t), "%Y-%m-%d_%X");

        sprintf ( input_folder, "%s", argv[1] );
        sprintf ( output_folder, "%s_%s", argv[2], datetime.str().c_str() );                // Add timestamp to output folder name.
        printf ( "input_folder = %s , output_folder = %s\n", input_folder, output_folder );
    }
    if(mkdir(output_folder, 0755) == -1) cerr << "\nError :  failed to create output_folder.\n" << strerror(errno) << endl;
    else cout << "output_folder created\n";                                                 // NB Folder access setting: 0755 = rwx owner, rx for others.
    stringstream outfile;
    outfile << "./" << output_folder <<  "/make_demo2_output.txt";
    cout << "\ncout outfile = " << outfile.str();
    fflush (stdout);
    freopen (outfile.str().c_str(), "w", stdout);                                           // Redirecting cout to write to "output.txt"

    cuInit ( 0 );                                                                           // Initialize
    int deviceCount = 0;
    cuDeviceGetCount ( &deviceCount );
    if ( deviceCount == 0 ) {
        printf ( "There is no device supporting CUDA.\n" );
        exit ( 0 );
    }
    CUdevice cuDevice;
    cuDeviceGet ( &cuDevice, 0 );
    CUcontext cuContext;
    cuCtxCreate ( &cuContext, 0, cuDevice );
    FluidSystem fluid;

    fluid.launchParams.debug = 3;                                                           // High debug setting, for fluid.InitializeCuda ().
    fluid.InitializeCuda ();
    fluid.ReadSpecificationFile ( input_folder );                                           // Debug reset to value in SpecificationFile.

    cout <<"\nfluid.launchParams.paramsPath = "<< fluid.launchParams.paramsPath ;
    cout <<"\nfluid.launchParams.pointsPath = "<< fluid.launchParams.pointsPath ;
    cout <<"\nfluid.launchParams.genomePath = "<< fluid.launchParams.genomePath ;

    if (fluid.launchParams.debug>0) std::cout<<"\n\nmake_demo2 chk6, fluid.launchParams.debug="<<fluid.launchParams.debug<<", fluid.launchParams.genomePath=" <<fluid.launchParams.genomePath  << ",  fluid.launchParams.spacing="<<fluid.launchParams.spacing<<std::flush;

    for(int i=0; i<256; i++){fluid.launchParams.outPath[i]    = output_folder[i];}

    if (fluid.launchParams.debug>0) std::cout<<"\n\nmake_demo2 chk7, fluid.launchParams.debug="<<fluid.launchParams.debug<<", fluid.launchParams.genomePath=" <<fluid.launchParams.genomePath  << ",  fluid.launchParams.spacing="<<fluid.launchParams.spacing<<std::flush;

    if(fluid.launchParams.create_demo=='y'){
        fluid.WriteDemoSimParams(                                                           // Generates the simulation from data previously loaded from SpecificationFile.txt .
            fluid.launchParams.outPath/*paramsPath*/, GPU_DUAL, CPU_YES, fluid.launchParams.num_particles, fluid.launchParams.spacing, fluid.launchParams.x_dim, fluid.launchParams.y_dim, fluid.launchParams.z_dim, fluid.launchParams.demoType, fluid.launchParams.simSpace, fluid.launchParams.debug
        );
    }else{
        fluid.ReadSimParams (   fluid.launchParams.paramsPath);
        fluid.ReadGenome(       fluid.launchParams.genomePath);
        fluid.ReadPointsCSV2(   fluid.launchParams.pointsPath, GPU_DUAL, CPU_YES );         // NB Also transfers params, genome and points to gpu.
        fluid.TransferFromCUDA();
        fluid.SavePointsCSV2(fluid.launchParams.outPath, -1 );// to check points from GPU.
    }
    uint num_particles_start=fluid.ActivePoints();
    
    fluid.TransferToCUDA (); 
    fluid.Run2Simulation ();
    cudaDeviceSynchronize();

    fluid.WriteResultsCSV(input_folder, output_folder, num_particles_start);// NB post-slurm script to (i) cat results.csv files, (ii)tar-gzip and ftp folders to recipient.
    
    size_t   free1, free2, total;
    cudaMemGetInfo(&free1, &total);
    printf("\n\nmake_demo2: Cuda Memory, before cuCtxDestroy(cuContext): free=%lu, total=%lu.\t",free1,total);
    
    CUresult cuResult = cuCtxDestroy ( cuContext ) ;
    if ( cuResult!=0 ) {printf ( "error closing, cuResult = %i \n",cuResult );}
    
    cudaMemGetInfo(&free2, &total);
    printf("\nmake_demo2: After cuCtxDestroy(cuContext): free=%lu, total=%lu, released=%lu.\n",free2,total,(free2-free1) );
    printf("\nClosed make_demo2.\n" );

    fflush (stdout);
    fclose (stdout);
    return 0;
}
