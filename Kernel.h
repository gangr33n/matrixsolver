/**
 * @file
 * @author Wayne Madden <s3197676@student.rmit.edu.au>
 * @version 0.3
 *
 * @section LICENSE
 * Free to re-use and reference from within code as long as the original owner
 * is referenced as per GNU standards
 *
 * @section DESCRIPTION
 * Kernel functions used by the matrix solver
 */

#include <cuda_runtime_api.h>
//#include <cuda.h>
//#include "cuda_runtime.h"
#include "device_launch_parameters.h"

/*kernels to be executed on gpu*/
__global__ void kernel_forElim(float*, float*, int);
__global__ void kernel_backSub(float*, float*, float*, int);
