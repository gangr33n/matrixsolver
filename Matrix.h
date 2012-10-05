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
 * Header file for the basic matrix solver libraries
 */

#include <cstdio>
#include <iostream>
#include <iomanip>

#include "Kernel.h"

using namespace std;

/*function prototypes*/
void displayEquation(float*, float*, float*, int);

void gaussianElimination(float*, float*, float*, int, int, int);
void galerkin(float*, float*, float*, int, int, int);
void rungekutta(float*, float*, float*, int, int, int);
