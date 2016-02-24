#include "mpi.h"
#include <iostream>
using namespace std;
 
int main(int argc, char *argv[])
{
    int rank, size, contador;
    MPI_Status estado;
 
    MPI_Init(&argc, &argv); // Inicializamos la comunicacion de los procesos
    MPI_Comm_size(MPI_COMM_WORLD, &size); // Obtenemos el número total de procesos
    MPI_Comm_rank(MPI_COMM_WORLD, &rank); // Obtenemos el valor de nuestro identificador

    if (rank == 0) { // El primer proceso solo envía
        // Envia mensaje
        MPI_Send(&rank // referencia al vector de elementos a enviar
                ,1 // tamaño del vector a enviar
                ,MPI_INT // Tipo de dato que envias
                ,rank + 1 // pid del proceso destino
                ,0 // etiqueta
                ,MPI_COMM_WORLD); //Comunicador por el que se manda
    } else if (rank == size - 1) { // El último proceso solo recibe
        // Recibe mensaje
        MPI_Recv(&contador // Referencia al vector donde se almacenara lo recibido
                ,1 // tamaño del vector a recibir
                ,MPI_INT // Tipo de dato que recibe
                ,rank - 1 // pid del proceso origen de la que se recibe
                ,0 // etiqueta
                ,MPI_COMM_WORLD // Comunicador por el que se recibe
                ,&estado); // estructura informativa del estado
        cout << "Soy el proceso " << rank << " y he recibido " << contador << endl;
    } else { // El resto de procesos reciben y envían
        // Recibe mensaje
        MPI_Recv(&contador // Referencia al vector donde se almacenara lo recibido
                ,1 // tamaño del vector a recibir
                ,MPI_INT // Tipo de dato que recibe
                ,rank - 1 // pid del proceso origen de la que se recibe
                ,0 // etiqueta
                ,MPI_COMM_WORLD // Comunicador por el que se recibe
                ,&estado); // estructura informativa del estado
        cout << "Soy el proceso " << rank << " y he recibido " << contador << endl;
        contador++;
        // Envia mensaje
        MPI_Send(&rank // referencia al vector de elementos a enviar
                ,1 // tamaño del vector a enviar
                ,MPI_INT // Tipo de dato que envias
                ,rank + 1 // pid del proceso destino
                ,0 // etiqueta
                ,MPI_COMM_WORLD); //Comunicador por el que se manda
    }
 
    MPI_Finalize();
    return 0;
}