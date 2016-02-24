#include "mpi.h"
#include <math.h>
#include <cstdlib> // Incluido para el uso de atoi
#include <iostream>
using namespace std;

int main(int argc, char *argv[]) 
{
    int n;
    double h = 1.0 / (double) n;
    double sum = 0.0;
    double sumLocal = 0.0;
    int rank, size;
    MPI_Status estado;
 
    MPI_Init(&argc, &argv); // Inicializamos la comunicacion de los procesos
    MPI_Comm_size(MPI_COMM_WORLD, &size); // Obtenemos el número total de procesos
    MPI_Comm_rank(MPI_COMM_WORLD, &rank); // Obtenemos el valor de nuestro identificador

    if (rank == 0) {
        cout << "introduce la precision del calculo (n > 0): ";
        cin >> n;
    }

    MPI_Bcast(&n // referencia al vector de elementos a enviar
             ,1 // tamaño del vector a enviar
             ,MPI_INT // tipo de dato que envia
             ,0 // rango del proceso raiz
             ,MPI_COMM_WORLD); // Comunicador por el que se recibe

    for (int i = rank; i < n; i += size) {
        double x = h * ((double)(i + 1) - 0.5);
        sumLocal += (4.0 / (1.0 + x * x));
    }

    MPI_Reduce(&sumLocal
              ,&sum
              ,1
              ,MPI_DOUBLE
              ,MPI_SUM
              ,0
              ,MPI_COMM_WORLD);

    if (rank == 0) {
        cout << sum * h << endl;
    }
 
    MPI_Finalize();
    return 0;
}
