#define __CL_ENABLE_EXCEPTIONS
#include <CL/cl.h>
#include "cl.hpp"

#include <vector>
#include <fstream>
#include <iostream>
#include <iterator>
#include <iomanip>

#define DEVICE_MAX_BLOCK_SIZE 1024

typedef std::vector<float> vfloat;

void read_data(std::ifstream & input, vfloat & a)
{
	for (vfloat::iterator it = a.begin(); it != a.end(); ++it)
	{
		input >> *it;
	}
}

size_t get_thread_count(size_t tasks_count)
{
	int power_of_2 = 0;
	while (tasks_count)
	{
		tasks_count >>= 1;
		++power_of_2;
	}

	return 1 << power_of_2;
}

void write_data(std::ofstream & output, vfloat & a, int n)
{
	int i = 0;
	for (vfloat::iterator it = a.begin(); it != a.end(); ++it)
	{
		output << std::setprecision(3) << std::fixed << *it << ' ';
		++i;
		if (i % n == 0)
		{
			i = 0;
			output << std::endl;
		}
	}
	output << std::endl;
}

int main()
{
   std::vector<cl::Platform> platforms;
   std::vector<cl::Device> devices;
   std::vector<cl::Kernel> kernels;

   try {

      // create platform
      cl::Platform::get(&platforms);
	  for (size_t i = 0; i < platforms.size(); ++i)
	  {
		  try
		  {
			  platforms[i].getDevices(CL_DEVICE_TYPE_GPU, &devices);
		  }
		  catch (cl::Error e)
		  {
			  std::cout << std::endl << e.what() << " : " << e.err() << std::endl;
			  if (i + 1 == platforms.size())
			  {
				  return 1;
			  }
		  }
	  }

      // create context
      cl::Context context(devices);

      // create command queue
      cl::CommandQueue queue(context, devices[0], CL_QUEUE_PROFILING_ENABLE);

      // load opencl source
      std::ifstream cl_file("convolution.cl");
      std::string cl_string(std::istreambuf_iterator<char>(cl_file), (std::istreambuf_iterator<char>()));
      cl::Program::Sources source(1, std::make_pair(cl_string.c_str(),
         cl_string.length() + 1));

      // create program
      cl::Program program(context, source);

      // compile opencl source
      program.build(devices);

      // create a message to send to kernel
	  std::ifstream input("input.txt");
	  int n, m;
	  input >> n >> m;

	  const int data_size = n * n;
	  const int mask_data_size = m * m;
	  vfloat input_matrix(data_size);
	  vfloat output_matrix(data_size);
	  vfloat mask(mask_data_size);
	  read_data(input, input_matrix);
	  read_data(input, mask);

      size_t const global_wi_count = get_thread_count(data_size);
	  size_t const block_size = global_wi_count < DEVICE_MAX_BLOCK_SIZE ? global_wi_count : DEVICE_MAX_BLOCK_SIZE;

      // allocate device buffer to hold message
	  cl::Buffer dev_input(context, CL_MEM_READ_ONLY, sizeof(float) * data_size);
	  cl::Buffer dev_output(context, CL_MEM_WRITE_ONLY, sizeof(float) * data_size);
	  cl::Buffer dev_mask(context, CL_MEM_READ_ONLY, sizeof(float) * mask_data_size);

      // copy from cpu to gpu
	  queue.enqueueWriteBuffer(dev_input, CL_TRUE, 0, sizeof(float) * data_size, &input_matrix[0]);
	  queue.enqueueWriteBuffer(dev_mask, CL_TRUE, 0, sizeof(float)* mask_data_size, &mask[0]);

      // load named kernel from opencl source
      queue.finish();
      cl::Kernel kernel_gmem(program, "gpu_convolution_gmem");
      cl::KernelFunctor convolution_gmem(kernel_gmem, queue, cl::NullRange, cl::NDRange(global_wi_count), cl::NDRange(block_size));
      cl::Event event = convolution_gmem(dev_input, dev_mask, dev_output, n, m);
     
      event.wait();
      cl_ulong start_time = event.getProfilingInfo<CL_PROFILING_COMMAND_START>();
      cl_ulong end_time   = event.getProfilingInfo<CL_PROFILING_COMMAND_END>();
      cl_ulong elapsed_time = end_time - start_time;

      queue.enqueueReadBuffer(dev_output, CL_TRUE, 0, sizeof(float) * data_size, &output_matrix[0]);
	  std::ofstream output("output.txt");
	  write_data(output, output_matrix, n);
   }
   catch (cl::Error e)
   {
      std::cout << std::endl << e.what() << " : " << e.err() << std::endl;
   }

   return 0;
}