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
 * Header file for the basic matrix solver. Contains definitions and libraries
 */

#include <cstdio>
#include <iostream>
#include <cstdlib>
#include <cstring>
//#include <Windows.h>

#include "Matrix.h"
#include "Kernel.h"

/*used to initialise the input buffer*/
/*EXISTING ISSUE - SPACE IS ALLOCATED ON DEVICE*/
#define BUFFER_SIZE 100000

/*used to organise the threafd scheduling*/
#define WARP_SIZE 32
#define GRID_MAX 8
