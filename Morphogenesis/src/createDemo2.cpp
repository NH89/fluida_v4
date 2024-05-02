#define __CL_ENABLE_EXCEPTIONS
#define CL_HPP_ENABLE_EXCEPTIONS
#define CL_TARGET_OPENCL_VERSION 300
#define SDK_SUCCESS 0
#define SDK_FAILURE 1



#include <stdio.h>
#include <iostream>
#include <fstream>
#include <string>
#include <vector>
#include <inttypes.h>
#include <errno.h>
#include <math.h>
#include <regex>
#include <jsoncpp/json/json.h>
#define CHECK_ERROR(status) if (status != CL_SUCCESS) { printf("Error: %d\n", status); exit(1); }
// #include "fluid.h"
#include "fluid_system.h"
#include "host_CL.cpp"
#include <chrono>


int main ( int argc, const char** argv )
{
    //Initialize Variables--------------------
    char input_folder[256];
    char specfile_folder[256];
    char output_folder[256];
    cout << "argc: " << argc << "\n" << flush;
    //----------------------------------------


    //Find working directory-----------------------------
    fs::path Path = fs::current_path().parent_path();
    const char* directory = Path.c_str();
    std::cout << "Directory: " << directory << std::endl;
    //---------------------------------------------------

    //Initialize JSON----------------------------------------------------------------------------------------------------------------------------
    char jsonPath[512]; sprintf(jsonPath, "%s/%s", directory, argv[1]); cout << "Concatenated path: " << jsonPath << std::endl;
    ifstream ifs(jsonPath);
    Json::Reader reader;
    Json::Value obj_;
    Json::Value obj;
    bool b = reader.parse(ifs, obj);
    if (!b) cout << "Error: " << reader.getFormattedErrorMessages();  else cout << "NB lists .json file entries alphabetically: \n" << obj ;
    cout << "\n\n\n" << endl;
    //if (!obj.isMember("verbosity")) std::cerr<< "\nError: 'verbosity' key not found in JSON file." << std::endl;return 1;
    uint verbosity = obj["verbosity"].asUInt();
    obj["opencl_platform"] = 0;
    obj["opencl_device"] = 0;
    std::string in_path_str = obj.isMember("demo_path") && obj["demo_path"].isString() ? std::string(obj["demo_path"].asCString()) + "/check" : (std::cerr << "Error: 'demo_path' is missing or invalid." << std::endl, nullptr);
    const char* in_path = in_path_str.c_str();
    std::string out_path_str = obj.isMember("demo_path") && obj["demo_path"].isString() ? std::string(obj["demo_path"].asCString()) + "/out" : (std::cerr << "Error: 'demo_path' is missing or invalid." << std::endl, nullptr);
    const char* out_path = out_path_str.c_str();
    const char* specfile_path = obj.isMember("demo_path") && obj["demo_path"].isString() ? obj["demo_path"].asCString() : (std::cerr << "Error: 'demo_path' in JSON file) is missing or invalid." << std::endl, nullptr);
    //-------------------------------------------------------------------------------------------------------------------------------------------


    if ((argc != 2) && (argc !=1)) {
        printf ( "usage: create_demo2 config.json .\
        \nNB input_folder must contain \"SpecificationFile.txt\", output will be wrtitten to \"output_folder/out_data_time/\".\
        \nIf output_folder is not given the value from SpecificationFile.txt will be used.\n" );
        return 0;
    } else {

        sprintf ( input_folder, "%s/%s", directory, in_path);
        sprintf ( specfile_folder, "%s/%s", directory, specfile_path);
        sprintf ( output_folder, "%s/%s", directory, out_path);

        printf ( "input_folder = %s , output_folder = %s \n", input_folder, output_folder );

    }
    FluidSystem fluid(obj);
    fluid.Initialize();
    fluid.InitializeOpenCL();

    //Setup Simulation------------------------------------------------------------
    fluid.ReadSpecificationFile ( specfile_folder );
    fluid.launchParams.verbosity = verbosity;

    std::cout<<"\n\nmake_demo2 chk1, fluid.launchParams.verbosity="<<fluid.launchParams.verbosity<<", fluid.launchParams.genomePath=" <<fluid.launchParams.genomePath  << ",  fluid.launchParams.spacing="<<fluid.launchParams.spacing<< ", fluid.launchParams.paramsPath="<<fluid.launchParams.paramsPath<< ",  fluid.launchParams.num_particles="<<fluid.launchParams.num_particles<< ",  fluid.launchParams.demoType="<<fluid.launchParams.demoType<<std::flush;

    for(int i=0; i<256; i++){fluid.launchParams.paramsPath[i] = input_folder[i];}
    for(int i=0; i<256; i++){fluid.launchParams.pointsPath[i] = input_folder[i];}
    //for(int i=0; i<256; i++){fluid.launchParams.genomePath[i] = input_folder[i];} // obtained from SpecificationFile.txt above.
    if(argc==2 || argc == 1){
        for(int i=0; i<256; i++) fluid.launchParams.outPath[i] = output_folder[i];
        cout << "\nfluid.launchParams.outPath: " << fluid.launchParams.outPath << flush;

        for(int i=0; i<256; i++) fluid.launchParams.paramsPath[i] = output_folder[i];
        cout << "\nfluid.launchParams.paramsPath: " << fluid.launchParams.paramsPath << flush;
    }
    if (!fs::exists(output_folder)) {
            if (!fs::create_directories(output_folder)) {std::cerr << "Error: Failed to create output folder: " << output_folder << std::endl;return 1;}
            else{cout << "output_folder created\n";}}

    cout << "\nSTARTING WriteDemoSimParams() WITH fluid.launchParams.paramsPath = " << fluid.launchParams.paramsPath << "\n" << flush;
    fluid.WriteDemoSimParams(           // Generates the simulation from data previously loaded from SpecificationFile.txt .
        fluid.launchParams.paramsPath, GPU_SINGLE, CPU_YES, fluid.launchParams.num_particles, fluid.launchParams.spacing, fluid.launchParams.x_dim, fluid.launchParams.y_dim, fluid.launchParams.z_dim, fluid.launchParams.demoType, fluid.launchParams.simSpace, fluid.launchParams.verbosity
    ); /*const char * relativePath*/
    //std::cout<<"\n\nmake_demo2 chk2 "<<std::flush;
    uint num_particles_start=fluid.ActivePoints();

    fluid.CreateFluidBuffers();
    fluid.TransferToCL ();
    //---------------------------------------------------

    //Run Simulation---------
    fluid.Run2Simulation ();
    //-----------------------

    //std::cout<<"\n\nmake_demo2 chk3 "<<std::flush;
    fluid.WriteResultsCSV(input_folder, output_folder, num_particles_start);// NB post-slurm script to (i) cat results.csv files, (ii)tar-gzip and ftp folders to recipient.

    printf ( "\nClosed createDemo2.\n" );
    return 0;
}
