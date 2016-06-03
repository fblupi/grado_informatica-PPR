
CC = g++
CFLAGS = -O3 -fopenmp

all : main
main.o :	main.cpp
	$(CC) $(CFLAGS) -c main.cpp
main : main.o
	$(CC) $(CFLAGS) main.o -o prueba
clean :
	rm -f main.o prueba
