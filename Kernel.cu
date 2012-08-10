/**
 * @file
 * @author Wayne Madden <s3197676@student.rmit.edu.au>
 * @version 0.2
 *
 * @section LICENSE
 * Free to re-use and reference from within code as long as the original owner
 * is referenced as per GNU standards
 *
 * @section DESCRIPTION
 * Kernel functions used by the matrix solver
 */

#include "Kernel.h"

/**
 *
 */
__global__ void kernel_forSub(float* A, float* B)
{
   /*int index = blockIdx.x * blockDim.x + threadIdx.x;*/
   int i = blockIdx.x;
   int j, k;
   float n;

   /*for (j = 0; j < i; j++)
   {
      while (i > 0 && A[(i-1) * blockDim.x + j] != 0);
      n = A[i * blockDim.x + j] / A[j * blockDim.x + j];
      for (k = 0; k < blockDim.x; k++)
         A[i * blockDim.x + k] -= n * A[j * blockDim.x + k];
      B[i] -= n * B[j];
   }*/

   for (i = 1; i < blockDim.x; i++)
   {
      for (j = 0; j < i; j++)
      {
         n = A[i * blockDim.x + j] / A[j * blockDim.x + j];
         for (k = 0; k < blockDim.x; k++)
            A[i * blockDim.x + k] -= n * A[j * blockDim.x + k];
         B[i] -= n * B[j];
      }
   }

   /*B[0] = blockDim.x;
   B[1] = gridDim.x;
   B[2] = threadIdx.x; //blockIdx*/
}

/**
 *
 */
__global__ void kernel_backSub(float* A, float* X, float* B)
{
   /*int index = blockIdx.x * blockDim.x + threadIdx.x;*/
   int i, j;
   float t;

   /*a.x + b.y + c.z = n => (n - c.z - b.y) / a*/
   for (i = (blockDim.x-1); i >= 0; i--)
   {
      t = B[i];
      for (j = (blockDim.x-1); j > i; j--)
         t -= A[i * blockDim.x + j] * X[j];
      X[i] = t / A[i * blockDim.x + i];
   }
}
