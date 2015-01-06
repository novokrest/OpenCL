#define __CL_ENABLE_EXCEPTIONS
#include <CL/cl.h>
#include "cl.hpp"

#include <cassert>
#include <iostream>
#include <iomanip>
#include <fstream>
#include <iterator>
#include <vector>
#include <string>

using std::cout;
using std::endl;
using std::setprecision;
using std::vector;
using std::ifstream;
using std::ofstream;
using std::string;

int read_input(string const& file_name, vector<float>& input);
void write_input(string const& file_name, vector<float> const& buffer);
void write_output(string const& file_name, vector<float> const& buffer);

void generate_io(int N);
int nearest2power(int n);
void print_buffer(vector<float> const& buffer);

void generate_io(int N)
{
    vector<float> input(N);
    for (int i = 0; i < N; ++i) {
        input[i] = i % 10;
    }
    
    vector<float> output(N);
    output[0] = input[0];
    for (int i = 1; i < N; ++i) {
        output[i] = input[i] + output[i - 1];
    }
    
    write_input("input.txt", input);
    write_output("output.txt", output);
}

int nearest2power(int n)
{
    int length = 0;
    int tmp = n;
    while (tmp >>= 1) {
        ++length;
    }

    int floor = (1 << length);

    return n > floor ? floor << 1 : n;
}

int read_input(string const& file_name, vector<float>& input)
{
    ifstream fin(file_name.c_str());
    int N;
    fin >> N;
    
    input.resize(N);
    
    for (int i = 0; i < N; ++i) {
        float val;
        fin >> val;
        input[i] = val;
    }
    
    return N;
}

void write_input(string const& file_name, vector<float> const& buffer)
{
    ofstream fout(file_name);
    fout << buffer.size() << endl;
    
    for (vector<float>::const_iterator it = buffer.begin(); it != buffer.end(); ++it) {
        fout << setprecision(3) << *it << " ";
    }
    
    fout << endl;
}

void write_output(string const& file_name, vector<float> const& buffer)
{
    ofstream fout(file_name);
    
    for (vector<float>::const_iterator it = buffer.begin(); it != buffer.end(); ++it) {
        fout << setprecision(3) << *it << " ";
    }
    
    fout << endl;
}

void print_buffer(vector<float> const& buffer)
{
    for (vector<float>::const_iterator it = buffer.begin(); it != buffer.end(); ++it) {
        cout << setprecision(3) << *it << " ";
    }
    cout << ", size: " << buffer.size() << endl;
}

struct GPU_Thread
{
    int gid_;
    int gsize_;
    int block_size_;
    int lid_;
    int bid_;
    int blocks_count_;
    
    GPU_Thread(int gid, int gsize, int block_size)
        : gid_(gid), gsize_(gsize), block_size_(block_size), lid_(gid_ % block_size_), bid_(gid_ / block_size_), blocks_count_(gsize_ / block_size_)
    {}
    
    void print()
    {
        cout << "GPU_Thread: "
             << "gid: " << gid_
             << ", gsize: " << gsize_
             << ", block_size: " << block_size_
             << ", lid: " << lid_
             << ", bid: " << bid_
             << ", blocks_count: " << blocks_count_
             << endl;
    }
    
    void inclusive_scan(vector<float>& data, int offset)
    {}
};

int compute_total_array_size(int array_size, int block_size)
{
    int total_array_size = 0;
    int s = array_size;
    while(s) {
        total_array_size += s;
        s /= block_size;
    }
    
    return total_array_size;
}

void inclusive_scan(vector<float> data, int gsize, int block_size)
{
    int offset = 0;
    for (int gid = 0; gid < gsize; ++gid) {
        GPU_Thread gpu_thread(gid, gsize, block_size);
        gpu_thread.inclusive_scan(data, offset);
    }
}

int main_cpu()
{
    int const block_size_max = 2;
    
    vector<float> input;
    int const N = read_input("input.txt", input);
    int const array_size = nearest2power(N);
    int const block_size = array_size > block_size_max ? block_size_max : array_size;
    int const total_array_size = compute_total_array_size(array_size, block_size);
    
    input.resize(total_array_size);
    
    cout << "N: " << N << endl;
    cout << "array_size: " << array_size << endl;
    cout << "total_array_size: " << total_array_size << endl;
    print_buffer(input);
    
    inclusive_scan(input, array_size, block_size);
    
    return 0;
}

int main(int argc, char* argv[])
{
    generate_io(7);
//    main_cpu();
//    return 0;

    std::vector<cl::Platform> platforms;
    std::vector<cl::Device> devices;
    std::vector<cl::Kernel> kernels;

    try {

        cl::Platform::get(&platforms);
        cout << "Platforms size: " << platforms.size() << endl;
        for (size_t p = 0; p < platforms.size(); ++p) {
            try {
                platforms[p].getDevices(CL_DEVICE_TYPE_GPU, &devices);
            }
            catch (cl::Error) {}
        }
        
        cout << "Devices size: " << devices.size() << endl;

        cl::Context context(devices);
        cl::CommandQueue queue(context, devices[0], CL_QUEUE_PROFILING_ENABLE);

        std::ifstream cl_file("scan.cl");
        std::string cl_string_source(std::istreambuf_iterator<char>(cl_file), (std::istreambuf_iterator<char>()));
        cl::Program::Sources source(1, std::make_pair(cl_string_source.c_str(), cl_string_source.length() + 1));

        cl::Program program(context, source);
        cout << "Prebuild" << endl;
        program.build(devices);
        cout << "After build" << endl;

        size_t const block_size_max = 2;

        //read input data
        std::ifstream fin("input.txt");
        int N;
        fin >> N;

        
        size_t const array_size = nearest2power(N);
        size_t total_array_size = 0;
        size_t const block_size = array_size > block_size_max ? block_size_max : array_size;
        for (size_t s = array_size; s; s /= block_size) {
            total_array_size += s;
        }
        
        std::vector<float> input(total_array_size, 0);
        std::vector<float> output(total_array_size, 0);

        for (int i = 0; i < N; ++i) {
            float number;
            fin >> number;
            input[i] = number;
        }

        cl::Buffer dev_input(context, CL_MEM_READ_ONLY, sizeof(float) * total_array_size);
        cl::Buffer dev_lasts(context, CL_MEM_READ_WRITE, sizeof(float) * total_array_size);
        cl::Buffer dev_output(context, CL_MEM_READ_WRITE, sizeof(float) * total_array_size);

        queue.enqueueWriteBuffer(dev_lasts, CL_TRUE, 0, sizeof(float) * total_array_size, &input[0]);
        queue.finish();
        
        cout << "array size: " << array_size << endl;
        cout << "block size: " << block_size << endl;
        cout << "total array size: " << total_array_size << endl;

        int offset = 0;
        cl::Kernel kernel_hs(program, "inclusive_scan_hillis_steele");

        cout << "after kernel_hs" << endl;
        for (size_t s = array_size; s > 1; s /= block_size) {
            size_t cur_block_size = s > block_size ? block_size : s;
        cl::KernelFunctor scan_hs(kernel_hs, queue, cl::NullRange, cl::NDRange(s), cl::NDRange(cur_block_size));
            cout << "after scan_hs" << endl;
            cl::Event cl_event = scan_hs(dev_input, dev_lasts, dev_output, cl::__local(sizeof(float) * cur_block_size), cl::__local(sizeof(float) * cur_block_size), offset, 0);
            cout << "after cl_event" << endl;
            cout << "offset: " << offset << endl;
            cout << "cur_block_size: " << cur_block_size << endl;
            cout << "total s: " << s << endl;
            
            cl_event.wait();
            offset += s;
        }
        
        cout << "Prefinal" << endl;
        cl::KernelFunctor scan_hs(kernel_hs, queue, cl::NullRange, cl::NDRange(array_size), cl::NDRange(block_size));
        cl::Event cl_event = scan_hs(dev_input, dev_lasts, dev_output, cl::__local(sizeof(float) * block_size), cl::__local(sizeof(float) * block_size), offset, 1);

        cout << "After wait" << endl;
//        cl_ulong start_time = cl_event.getProfilingInfo<CL_PROFILING_COMMAND_START>();
//        cl_ulong end_time = cl_event.getProfilingInfo<CL_PROFILING_COMMAND_END>();
//        cl_ulong elapsed_time = end_time - start_time;

        queue.enqueueReadBuffer(dev_lasts, CL_TRUE, 0, sizeof(float) * total_array_size, &output[0]);
        queue.enqueueReadBuffer(dev_output, CL_TRUE, 0, sizeof(float) * total_array_size, &output[0]);

        for (size_t i = 0; i < total_array_size; ++i) {
            cout << output[i] << " ";
        }
        cout << endl;

//        cout << std::setprecision(3) << "Total time: " << elapsed_time / 1000000.0 << " ms" << endl;

    }
    catch (cl::Error e) {
        cout << endl << e.what() << " : " << e.err() << endl;
        return 1;
    }

    return 0;
}


