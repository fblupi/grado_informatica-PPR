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
    int a, b;

    if (rank == 0) { // El proceso 0 del comm global inicializa a 2000, 1
        a = 2000;
        b = 1;
    } else { // El resto de procesos a 0
        a = 0;
        b = 0;
    }

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

    // Comprobar comunicadores
    if (rank == 0) cout << "COMUNICADORES" << endl;
    cout << endl
         << "Global: " << rank << endl 
         << "Par/Impar: " << rank_par_impar << endl 
         << "Inverso: " << rank_inverso << endl;
    // Difundir a solo a los pares
    MPI_Bcast(&a // referencia al vector de elementos a enviar
        ,1 // tamaño del vector a enviar
        ,MPI_INT // tipo de dato que envia
        ,0 // rango del proceso raiz
        ,comm_par_impar); // Comunicador por el que se recibe

    // Difundir b a todos
    MPI_Bcast(&b // referencia al vector de elementos a enviar
        ,1 // tamaño del vector a enviar
        ,MPI_INT // tipo de dato que envia
        ,size - 1 // rango del proceso raiz
        ,comm_inverso); // Comunicador por el que se recibe

    // Comprobar valores
    if (rank == 0) cout << "VALORES" << endl;
    cout << endl
         << "Global: " << rank << endl
         << "\tA: " << a << endl
         << "\tB: " << b << endl;

    MPI_Finalize();
    return 0;
}