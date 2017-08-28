/* RUN: %{execute}%s

   Test that we increment the reference count of a \c cl_kernel and
   \c cl_context when building a \c cl::sycl::kernel or
   \c cl::sycl::context from it in OpenCL Compatibility mode.
 */

#include <boost/compute.hpp>
#include <boost/test/minimal.hpp>
#include <CL/sycl.hpp>

#define MAX_OPENCL_PLATFORMS 3

int test_main(int argc, char *argv[]) {

  const char* sources = "__kernel void empty() { }";

  cl_uint platform_num;
  cl_uint device_num;
  cl_uint ref_count_1;
  cl_uint ref_count_2;

  cl_platform_id cl_platforms[MAX_OPENCL_PLATFORMS];
  cl_device_id cl_device;

  clGetPlatformIDs(MAX_OPENCL_PLATFORMS, cl_platforms, &platform_num);

  for(cl_uint i = 0; i < platform_num; i++){
    clGetDeviceIDs(cl_platforms[i], CL_DEVICE_TYPE_CPU, 1, &cl_device, &device_num);
    if(device_num > 0) break;
  }

  cl_context cl_context = clCreateContext(0, 1, &cl_device, NULL, NULL, NULL);
  clGetContextInfo(cl_context, CL_CONTEXT_REFERENCE_COUNT, sizeof(ref_count_1), &ref_count_1, NULL);
  {
    cl::sycl::context sycl_context { cl_context };

    clGetContextInfo(cl_context, CL_CONTEXT_REFERENCE_COUNT, sizeof(ref_count_2), &ref_count_2, NULL);
    BOOST_CHECK(ref_count_2 == ref_count_1+1);
  }
  clGetContextInfo(cl_context, CL_CONTEXT_REFERENCE_COUNT, sizeof(ref_count_1), &ref_count_1, NULL);
  BOOST_CHECK(ref_count_1 == ref_count_2-1);



  cl_program cl_program = clCreateProgramWithSource(cl_context, 1, &sources, NULL, NULL);
  clBuildProgram(cl_program, 0, NULL, NULL, NULL, NULL);

  cl_kernel cl_kernel = clCreateKernel(cl_program, "empty", NULL);
  clGetKernelInfo(cl_kernel, CL_KERNEL_REFERENCE_COUNT, sizeof(ref_count_1), &ref_count_1, NULL);
  {
    cl::sycl::kernel sycl_kernel { cl_kernel };

    clGetKernelInfo(cl_kernel, CL_KERNEL_REFERENCE_COUNT, sizeof(ref_count_2), &ref_count_2, NULL);
    BOOST_CHECK(ref_count_2 == ref_count_1+1);
  }
  clGetKernelInfo(cl_kernel, CL_KERNEL_REFERENCE_COUNT, sizeof(ref_count_1), &ref_count_1, NULL);
  BOOST_CHECK(ref_count_1 == ref_count_2-1);

  return 0;
}
