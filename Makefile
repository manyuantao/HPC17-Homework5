CC = gcc
FLAGS = -lm -lrt
EXECS = jacobi2D-multigrid

all: ${EXECS}

jacobi2D-multigrid: jacobi2D-multigrid.c
	${CC} ${FLAGS} jacobi2D-multigrid.c -o jacobi2D-multigrid

clean:
	rm -f ${EXECS}