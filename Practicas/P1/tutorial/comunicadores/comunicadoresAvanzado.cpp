#include "mpi.h"
#include <vector>
#include <cstdlib>
#include <iostream>
using namespace std;
 
int main(int argc, char *argv[]) {
    int rank, size;
 
    MPI_Init(&argc, &argv); // Inicializamos la comunicacion de los procesos
    MPI_Comm_size(MPI_COMM_WORLD, &size); // Obtenemos el número total de procesos
    MPI_Comm_rank(MPI_COMM_WORLD, &rank); // Obtenemos el valor de nuestro identificador

    MPI_Comm comm_par_impar, comm_inverso; // nuevos comunicadores

    int rank_par_impar, rank_inverso, size_par_impar, size_inverso; // nuevo tamaño y rango en los nuevos comunicadores
    int dato = 0;
    vector<int> V;

    int color = rank % 2; // color para saber si es de pares o de impares

    // Se crean los comunicadores de pares e impares
    MPI_Comm_split(MPI_COMM_WORLD // Comunicador padre
                  ,color // Identifica si par o impar
                  ,rank // Prioridad para orden del rango en el nuevo comunicador
                  ,&comm_par_impar); // Referencia al nuevo comunicador

    // Se crea el comunicador de inversos
    MPI_Comm_split(MPI_COMM_WORLD // Comunicador padre
                  ,0 // Identifica si par o impar
                  ,-rank // Prioridad para orden del rango en el nuevo comunicador
                  ,&comm_inverso); // Referencia al nuevo comunicador

    MPI_Comm_size(comm_par_impar, &size_par_impar); // Obtenemos el número total de procesos en el comunicador par/impar
    MPI_Comm_rank(comm_par_impar, &rank_par_impar); // Obtenemos el valor de nuestro identificador en el comunicador par/impar

    MPI_Comm_size(comm_inverso, &size_inverso); // Obtenemos el número total de procesos en el comunicador inverso
    MPI_Comm_rank(comm_inverso, &rank_inverso); // Obtenemos el valor de nuestro identificador en el comunicador inverso

    V.resize(size_par_impar, 0);

    // El proceso 1 (0 en el impar) inicializa el vector
    if (rank == 1) {
        for (int i = 0; i < size_par_impar; i++) {
            V[i] = i * 14;
        }
    }

    // Repartir vector
    MPI_Scatter(&V[0] // referencia al vector de elementos a enviar
               ,1 // tamaño del vector a enviar
               ,MPI_INT // tipo de dato que envia
               ,&dato // referencia al vector donde se almacenarán los datos recibidos
               ,1 // tamaño del vector a recibir
               ,MPI_INT // tipo de dato que recibe
               ,0 // rango del proceso raiz
               ,comm_par_impar); // Comunicador por el que se realiza la acción

    // Comprobar valores
    if (color == 1) {
        cout << "Global: " << rank << " --> Dato: " << dato << endl;
    }

    MPI_Finalize();
    return 0;
}