#define __CL_ENABLE_EXCEPTIONS
#include <OpenCL/cl.h>
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
void write_output(string const& file_name, vector<float> const& buffer, int count);

void generate_io(int N);
int nearest2power(int n);
void print_buffer(vector<float> const& buffer, int count);

void generate_io(int N)
{
    vector<float> input(N);
    for (int i = 0; i < N; ++i) {
        input[i] = 1;
    }
    
    vector<float> output(N);
    output[0] = input[0];
    for (int i = 1; i < N; ++i) {
        output[i] = input[i] + output[i - 1];
    }
    
    write_input("input.txt", input);
    write_output("output_cpu.txt", output, output.size());
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

int read_output(string const& file_name, vector<float>& output)
{
    ifstream fin(file_name.c_str());
    int k = 0;
    while (!fin.eof()) {
        float val;
        fin >> val;
        output.push_back(val);
        ++k;
    }

    return k;
}

void write_input(string const& file_name, vector<float> const& buffer)
{
    ofstream fout(file_name);
    fout << buffer.size() << endl;
    
    for (vector<float>::const_iterator it = buffer.begin(); it != buffer.end(); ++it) {
        fout << std::fixed << setprecision(3) << *it << " ";
    }
    
    fout << endl;
}

void write_output(string const& file_name, vector<float> const& buffer, int count)
{
    ofstream fout(file_name);
    
    int k = 0;
    for (vector<float>::const_iterator it = buffer.begin(); it != buffer.end(); ++it) {
        if (k == count) {
            break;
        }
        fout << std::fixed << setprecision(3) << *it << " ";
        ++k;
    }
    
    fout << endl;
}

void print_buffer(vector<float> const& buffer, int count)
{
    int k = 0;
    for (vector<float>::const_iterator it = buffer.begin(); it != buffer.end(); ++it) {
        if (k == count) {
            break;
        }
        cout << setprecision(3) << *it << " ";
        ++k;
    }
    cout << endl;
}

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

int main(int argc, char* argv[])
{
    //generate_io(1048576);

    int const BLOCK_SIZE_MAX = 256;

    std::vector<cl::Platform> platforms;
    std::vector<cl::Device> devices;
    std::vector<cl::Kernel> kernels;

    try {

        cl::Platform::get(&platforms);
        for (size_t p = 0; p < platforms.size(); ++p) {
            try {
                platforms[p].getDevices(CL_DEVICE_TYPE_GPU, &devices);
            }
            catch (cl::Error) {}
        }

        cl::Context context(devices);
        cl::CommandQueue queue(context, devices[0], CL_QUEUE_PROFILING_ENABLE);

        std::ifstream cl_file("scan.cl");
        std::string cl_string_source(std::istreambuf_iterator<char>(cl_file), (std::istreambuf_iterator<char>()));
        cl::Program::Sources source(1, std::make_pair(cl_string_source.c_str(), cl_string_source.length() + 1));

        cl::Program program(context, source);
        program.build(devices);

        size_t const block_size_max = 256;

        //read input data
        vector<float> input, output;
        int const N = read_input("input.txt", input);

        int const array_size = nearest2power(N);
        int const block_size = array_size > block_size_max ? block_size_max : array_size;
        int const total_array_size = compute_total_array_size(array_size, block_size);

        input.resize(total_array_size);
        output.resize(array_size);

        cl::Buffer dev_input(context, CL_MEM_READ_ONLY, sizeof(float) * total_array_size);
        cl::Buffer dev_output(context, CL_MEM_READ_ONLY, sizeof(float) * array_size);
        cl::Buffer dev_lasts(context, CL_MEM_READ_WRITE, sizeof(float) * total_array_size);

        queue.enqueueWriteBuffer(dev_lasts, CL_TRUE, 0, sizeof(float) * total_array_size, &input[0]);
        queue.finish();
        
        int offset = 0;
        cl::Kernel kernel_hs(program, "inclusive_scan_hillis_steele");
        for (int s = array_size; s > 1; s /= block_size) {
            int cur_block_size = s > block_size ? block_size : s;

            cl::KernelFunctor scan_hs(kernel_hs, queue, cl::NullRange, cl::NDRange(s), cl::NDRange(cur_block_size));
            cl::Event cl_event = scan_hs(dev_input, dev_lasts, dev_output, cl::__local(sizeof(float) * cur_block_size), cl::__local(sizeof(float) * cur_block_size), offset, 0);
            cl_event.wait();

            offset += s;
        }
        
        cl::KernelFunctor scan_hs(kernel_hs, queue, cl::NullRange, cl::NDRange(array_size), cl::NDRange(block_size));
        cl::Event cl_event = scan_hs(dev_input, dev_lasts, dev_output, cl::__local(sizeof(float) * block_size), cl::__local(sizeof(float) * block_size), -1, 1);

        queue.enqueueReadBuffer(dev_output, CL_TRUE, 0, sizeof(float) * N, &output[0]);
        print_buffer(output, N);
        write_output("output.txt", output, N);
    }

    catch (cl::Error e) {
        cout << endl << e.what() << " : " << e.err() << endl;
        return 1;
    }

    return 0;
}


