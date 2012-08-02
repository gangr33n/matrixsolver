/**
 * @file
 * @author Wayne Madden <s3197676@student.rmit.edu.au>
 * @version 0.1
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
   float** A;
   float* X;
   float* B;
   float n, t;
   int i, j, k;
   int matrixSide;
   FILE* fp;
   char line[BUFFER_SIZE];
   char* token;

   /*read command line input*/
   matrixSide = atoi(argv[1]);

   /*allocate memory*/
   A = new float*[matrixSide];
   for (i = 0; i < matrixSide; i++)
      A[i] = new float[matrixSide];
   X = new float[matrixSide];
   B = new float[matrixSide];

   /*read matrix a into memory*/
   fp = fopen(argv[2], "r");
   i = 0;
   while (i < matrixSide && fgets(line, 1000000, fp) != NULL)
   {
      j = 0;
      token = strtok(line, " ,");
      do
      {
         A[i][j] = atof(token);
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
      B[i] = atof(line);
      i++;
   }
   fclose(fp);

   displayEquation(A, X, B, matrixSide);
   std::cout << endl;

   /*forward substitution*/
   for (i = 1; i < matrixSide; i++)
   {
      for (j = 0; j < i; j++)
      {
         n = A[i][j] / A[j][j];
         for (k = 0; k < matrixSide; k++)
            A[i][k] -= n * A[j][k];
         B[i] -= n * B[j];
         std::cerr << "....." << endl;
         displayEquation(A, X, B, matrixSide);
      }
   }
   
   /*backwards substitution*/
   /*a.x + b.y + c.z = n => (n - c.z - b.y) / a*/
   for (i = (matrixSide-1); i >= 0; i--)
   {
      t = B[i];
      for (j = (matrixSide-1); j > i; j--)
         t -= A[i][j] * X[j];
      X[i] = t / A[i][i];
   }

   /*display the solved values of matrix X*/
   std::cerr << endl;
   displayEquation(A, X, B, matrixSide);
   
   /*clean memory*/
   for (i = 0; i < matrixSide; i++)
      delete[] A[i];
   delete[] A;
   delete[] X;
   delete[] B;

   exit(EXIT_SUCCESS);
}
