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
 * Library functions for the basic matrix solver
 */

#include "Matrix.h"

/**
 * Display the matrix data in a human readable format
 *
 * @param A The square matrix containing linear system operands
 * @param X The matrix to be solved
 * @param B The right ahdn side of the linear equations
 * @param matrixSide The size of eeach dimension of matrix A and the larger
 * dimension of both X and B
 */

#include "Matrix.h"

void displayEquation(float* A, float* X, float* B, int matrixSide)
{
   int i, j;

   for (i = 0; i < matrixSide; i++)
   {
      /*output matrix a, one line at a time*/
      cerr << "[ ";
      for (j = 0; j < matrixSide; j++)
         cerr << setw(10) << setprecision(2) << A[i * matrixSide + j];
      cerr << " ]";
      
      /*output multiplication sign, centred vertically*/
      (i == (matrixSide/2)) ? cerr << " x " : cerr << "   ";
      
      /*output matrix x, one line at a time*/
	  if (X != NULL)
         cerr << "[ " << setw(10) << setprecision(2) << X[i] << " ]";
	  else
	     cerr << "[ " << setw(10) << setprecision(2) << "?" << " ]";
      
      /*output equals sign, centred vertically*/
      (i == (matrixSide/2)) ? printf(" = ") : printf("   ");
      
      /*output matrix b, one line at a time*/
      cerr << "[ " << setw(10) << setprecision(2) << B[i] << " ]" << endl;
   }

   cerr << endl;
}
