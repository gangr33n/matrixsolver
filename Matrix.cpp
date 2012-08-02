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
 * Library functions for the basic matrix solver
 */

#include "Matrix.h"

/**
 * Display the matrix data in a human readable format
 *
 * @param A 
 * @param X The matrix to be solved
 * @param B 
 * @param matrixSide The size of eeach dimension of matrix A and the larger
 * dimension of both X and B
 */
void displayEquation(float** A, float* X, float* B, int matrixSide)
{
   int i, j;

   for (i = 0; i < matrixSide; i++)
   {
      /*output matrix a, one line at a time*/
      cerr << "[ ";
      for (j = 0; j < matrixSide; j++)
         cerr << setw(10) << setprecision(2) << A[i][j];
      cerr << " ]";
      
      /*output multiplication sign, centred vertically*/
      (i == (matrixSide/2)) ? cerr << " x " : cerr << "   ";
      
      /*output matrix x, one line at a time*/
      cerr << "[ " << setw(10) << setprecision(2) << X[i] << " ]";
      
      /*output equals sign, centred vertically*/
      (i == (matrixSide/2)) ? printf(" = ") : printf("   ");
      
      /*output matrix b, one line at a time*/
      cerr << "[ " << setw(10) << setprecision(2) << B[i] << " ]" << endl;
   }
}