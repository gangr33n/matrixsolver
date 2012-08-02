#
# Author: Wayne Madden <s3197676@student.rmit.edu.au>
# Version: 0.1
#
# Free to re-use and reference from within code as long as the original owner
# is referenced as per GNU standards
#
# Makefile to compile the basic matrix solver program modules that is under
# developing for my final year Design project at RMIT University
#

MAIN = SolverMain.o
UTILS = Matrix.o

BINARY = solver
CC = g++
CC_FLAGS = -ansi -Wall -pedantic -gstabs

#####

all: $(MAIN) $(UTILS)
	$(CC) $(MAIN) $(UTILS) -o $(BINARY)

#####

SolverMain.o: SolverMain.cpp SolverMain.h
	$(CC) -c $(CC_FLAGS) SolverMain.cpp

#####

Matrix.o: Matrix.cpp Matrix.h
	$(CC) -c $(CC_FLAGS) Matrix.cpp

#####

clean:
	rm -f *.o $(BINARY)
