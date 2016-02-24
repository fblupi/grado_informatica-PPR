#include "mpi.h"
#include <math.h>
#include <cstdlib> // Incluido para el uso de atoi
#include <iostream>
using namespace std;

int main(int argc, char *argv[]) 
{
    int n, rank, size, sizeBloque;
    const double PI25DT = 3.141592653589793238462643;
    MPI_Status estado;
 
    MPI_Init(&argc, &argv); // Inicializamos la comunicacion de los procesos
    MPI_Comm_size(MPI_COMM_WORLD, &size); // Obtenemos el número total de procesos
    MPI_Comm_rank(MPI_COMM_WORLD, &rank); // Obtenemos el valor de nuestro identificador

    // El proceso 0 captura el número de partes utilizadas para el cálculo
    if (rank == 0) {
        do {
            cout << "introduce la precision del calculo (n > 0): ";
            cin >> n;
        } while (n <= 0); // El número de partes debe ser mayor que cero
    }

    // El proceso 0 envía el número de partes al resto de procesos
    MPI_Bcast(&n // referencia al vector de elementos a enviar
             ,1 // tamaño del vector a enviar
             ,MPI_INT // tipo de dato que envia
             ,0 // rango del proceso raiz
             ,MPI_COMM_WORLD); // Comunicador por el que se recibe

    // Cálculo de PI
    double h = 1.0 / (double)n, 
           sum = 0.0, 
           sumLocal = 0.0;
    sizeBloque = ceil((double)n / size); // tamaño del bloque

    for (int i = rank * sizeBloque; i < (rank + 1) * sizeBloque && i < n; i++) { // reparto por bloques
        double x = h * ((double)i + 1.0 - 0.5);
        sumLocal += (4.0 / (1.0 + x * x));
    }
    sumLocal *= h; // resultado local

    // El proceso 0 recolecta y suma todas las sumas locales
    MPI_Reduce(&sumLocal // referencia al vector de elementos a enviar
              ,&sum // referencia del al vector donde se almacena el resultado
              ,1 // tamaño del vector a enviar
              ,MPI_DOUBLE // tipo de dato que envia
              ,MPI_SUM // operación de reducción
              ,0 // rango del proceso raiz
              ,MPI_COMM_WORLD); // Comunicador por el que se recibe

    // El proceso 0 envía el resultado al resto de procesos
    MPI_Bcast(&sum // referencia al vector de elementos a enviar
             ,1 // tamaño del vector a enviar
             ,MPI_DOUBLE // tipo de dato que envia
             ,0 // rango del proceso raiz
             ,MPI_COMM_WORLD); // Comunicador por el que se recibe

    cout << "Proceso: " << rank << " --> El valor aproximado de PI es: " << sum << ", con un error de " << fabs(sum - PI25DT) << endl;
 
    MPI_Finalize();
    return 0;
}
