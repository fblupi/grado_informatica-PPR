
CC = g++
CFLAGS = -O3 -fopenmp
NAME_SEC = floydSecuencial
NAME_1D = floyd1D
NAME_2D = floyd2D
MODULES_GRAPH = Graph.o
MODULES_SEC = floydSecuencial.o
MODULES_1D = floyd1D.o
MODULES_2D = floyd2D.o

all : floydSecuencial floyd1D #floyd2D
Graph.o	: Graph.cpp Graph.h
	$(CC) $(CFLAGS) -c Graph.cpp
floydSecuencial.o :	floydSecuencial.cpp
	$(CC) $(CFLAGS) -c floydSecuencial.cpp
floydSecuencial : $(MODULES_GRAPH) $(MODULES_SEC)
	$(CC) $(CFLAGS) $(MODULES_GRAPH) $(MODULES_SEC) -o $(NAME_SEC) $(EXTRALIBS)
floyd1D.o :	floyd1D.cpp
	$(CC) $(CFLAGS) -c floyd1D.cpp
floyd1D : $(MODULES_GRAPH) $(MODULES_1D)
	$(CC) $(CFLAGS) $(MODULES_GRAPH) $(MODULES_1D) -o $(NAME_1D) $(EXTRALIBS)
#floyd2D.o :	floyd2D.cpp
#	$(CC) $(CFLAGS) -c floyd2D.cpp
#floyd2D : $(MODULES_GRAPH) $(MODULES_2D)
#	$(CC) $(CFLAGS) $(MODULES_GRAPH) $(MODULES_2D) -o $(NAME_2D) $(EXTRALIBS)
clean :
	rm -f $(MODULES_GRAPH) $(MODULES_SEC) $(NAME_SEC) $(MODULES_1D) $(NAME_1D) #$(MODULES_2D) $(NAME_2D)
