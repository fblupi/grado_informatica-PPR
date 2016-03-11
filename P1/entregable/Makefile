
CC = mpicxx 
CFLAGS = -O3
NAME_SEC = floydSecuencial
NAME_1D = floyd1D
MODULES_GRAPH = Graph.o
MODULES_SEC = floydSecuencial.o
MODULES_1D = floyd1D.o

all : floydSecuencial floyd1D
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
clean :
	rm -f $(MODULES_GRAPH) $(MODULES_SEC) $(MODULES_1D) $(NAME_SEC) $(NAME_1D)