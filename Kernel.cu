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
 *
 * note: int index = blockIdx.x * blockDim.x + threadIdx.x;
 */

#include "Kernel.h"

/**
 * Device Kernel function which is called by multiple threads from the main
 * program. The function performs forward elimination on a set of linear
 * equations in a parallel fashion - cannot be called in a non concurrent
 * fashion. Future addition: check for any thread index value outside the
 * matrix size boundary, e.g. if 32 threads are scheduled for 10x10 matrix
 * then discount index 11 through 31.
 *
 * @param A The square matrix containing the operands of the unknown matrix X
 * @param B The single column matrix containing the right hand side of the
 * linear equations
 * @param matrixSide The number of elements expressing height and width of A
 * and height of B
 */
__global__ void kernel_forElim(float* A, float* B, int matrixSide)
{
   int i, j, k;
   float n;

   i = threadIdx.x;

   /*don't process empty threads when matrixSide is not divisible by WARP_SIZE*/
   if (i < matrixSide)
   {
      for (j = 0; j < threadIdx.x; j++)
      {
         /*index 0 is first row processed, last is index matrixSide-1*/
         n = A[i * matrixSide + j] / A[j * matrixSide + j];
         for (k = 0; k < matrixSide; k++)
            A[i * matrixSide + k] -= n * A[j * matrixSide + k];
         B[i] -= n * B[j];
      }
   }
}

/**
 * Device Kernel function which is called by multiple threads from the main
 * program. The function performs backward substitution on a set of linear
 * equations in a parallel fashion - cannot be called in a non concurrent
 * fashion.
 *
 * @param A The square matrix containing the operands of the unknown matrix X
 * @param X The single column matrix containing the unknown values expressed
 * in each linear equation
 * @param B The single column matrix containing the right hand side of the
 * linear equations
 * @param matrixSide The number of elements expressing height and width of A
 * and height of X and B
 */
__global__ void kernel_backSub(float* A, float* X, float* B, int matrixSide)
{
   int i, j;
   float t;

   i = threadIdx.x;
   /*e.g. a.x + b.y + c.z = n => (n - c.z - b.y) / a*/
   /*don't process empty threads when matrixSide is not divisible by WARP_SIZE*/
   if (i < matrixSide)
   {
      t = B[i];
      for (j = matrixSide; j > i; j--)
      {
         if (j < matrixSide)
            t -= A[i * matrixSide + j] * X[j];
         /*this calculation will always occur first due to the thread
		 scheduler*/
         if (i == (j-1))
            X[i] = t / A[i * matrixSide + i];
      }
   }
}
