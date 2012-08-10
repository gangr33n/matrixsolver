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
 * Basic matrix solver. Accepts two input matrices A and B in the form of
 * equation 'A . X = B'. Uses gaussian elimination.
 */

#include "SolverMain.h"

/**
 * Main method of matrix solver program
 *
 * @param argc Number of items to be stored
 * @param argsv[] Size to be used for each item by the Memory Manager
 */
int main(int argc, char* argv[])
{
   float *hA, *dA;
   float *hX, *dX;
   float *hB, *dB;
   float n, t;
   int i, j, k;
   int matrixSide;
   int status;
   FILE* fp;
   char line[BUFFER_SIZE]; /*EXISTING ISSUE - SPACE IS ALLOCATED ON DEVICE, CAN'T BE TOO LARGE*/
   char* token;

   /*validate arguments*/
   if (argc != 4)
   {
      cout << "Invalid arguments! Press enter to continue...";
      cin.ignore(1);
   }

   /*set device*/
   status = cudaSetDevice(0);
   if (status != cudaSuccess)
   {
      cout << "No valid device found! Press enter to continue...";
      cin.ignore(1);
   }

   /*read command line input*/
   matrixSide = atoi(argv[1]);

   /*allocate host memory*/
   hA = new float[matrixSide * matrixSide];
   hX = new float[matrixSide];
   hB = new float[matrixSide];

   /*check host memory*/
   if (hA == NULL || hX == NULL || hB == NULL)
   {
      cout << "Unable to allocate host memory! Press enter to continue...";
      cin.ignore(1);
	  exit(EXIT_FAILURE);
   }

   /*allocate device memory*/
   status = cudaSuccess;
   status += cudaMalloc((void**) &dA, sizeof(float) * matrixSide * matrixSide);
   status += cudaMalloc((void**) &dX, sizeof(float) * matrixSide);
   status += cudaMalloc((void**) &dB, sizeof(float) * matrixSide);

   /*check device memory*/
   if (status != cudaSuccess)
   {
      cout << "Unable to allocate host memory! Press enter to continue...";
      cin.ignore(1);
	  exit(EXIT_FAILURE);
   }

   /*read matrix a into memory*/
   fp = fopen(argv[2], "r");
   i = 0;
   while (i < matrixSide && fgets(line, 1000000, fp) != NULL)
   {
      j = 0;
      token = strtok(line, " ,");
      do
      {
         hA[i * matrixSide + j] = atof(token);
         j++;
      } while (j < matrixSide && (token = strtok(NULL, " ,")) != NULL);
      i++;
   }
   fclose(fp);

   /*read matrix b into memory*/
   fp = fopen(argv[3], "r");
   i = 0;
   while (i < matrixSide && fgets(line, 1000, fp) != NULL)
   {
      hB[i] = atof(line);
      i++;
   }
   fclose(fp);

   /*copy host memory to device*/
   cudaMemcpy(dA, hA, sizeof(float) * matrixSide * matrixSide, cudaMemcpyHostToDevice);
   cudaMemcpy(dB, hB, sizeof(float) * matrixSide, cudaMemcpyHostToDevice);

   /*first display*/
   displayEquation(hA, hX, hB, matrixSide);

   /*forward substitution*/
   kernel_forSub<<<1, matrixSide>>>(dA, dB);
   status = cudaDeviceSynchronize();
   if (status != cudaSuccess)
   {
      cout << "Unable to complete forward substitution! Press enter to continue...";
      cin.ignore(1);
	  exit(EXIT_FAILURE);
   }

   /*backwards substitution*/
   kernel_backSub<<<1, matrixSide>>>(dA, dX, dB); /*UP YOURS VISUAL STUDIO THE EXTENSION SHOULD NOT MATTER*/
   status = cudaDeviceSynchronize();
   if (status != cudaSuccess)
   {
      cout << "Unable to complete backward substitution! Press enter to continue...";
      cin.ignore(1);
	  exit(EXIT_FAILURE);
   }

   /*copy host memory to device*/
   cudaMemcpy(hX, dX, sizeof(float) * matrixSide, cudaMemcpyDeviceToHost);

   /*display the solved values of matrix X*/
   displayEquation(hA, hX, hB, matrixSide);
   
   /*free host memory*/
   delete[] hA;
   delete[] hX;
   delete[] hB;

   /*free device memory*/
   cudaFree(dA);
   cudaFree(dX);
   cudaFree(dB);

   /*reset device for profiling tool traces*/
   cudaDeviceReset();

   /*prompt to continue - to allow the user to read output before exiting*/
   cout << "Press enter to continue...";
   cin.ignore(1);

   exit(EXIT_SUCCESS);
}
