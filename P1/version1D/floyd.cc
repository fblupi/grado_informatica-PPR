#include <iostream>
#include <fstream>
#include <string.h>
#include "Graph.h"
#include "mpi.h"

using namespace std;

int pasarFilaALocal(int localSize, int globalIndex) {
  return globalIndex % localSize;
}

int pasarFilaAGlobal(int rank, int localSize, int localIndex) {
  return rank * localSize + localIndex;
}

int main (int argc, char *argv[]) 
{
  int rank, size, tama;

  MPI_Init(&argc, &argv); // Inicializamos la comunicacion de los procesos
  MPI_Comm_size(MPI_COMM_WORLD, &size); // Obtenemos el número total de procesos
  MPI_Comm_rank(MPI_COMM_WORLD, &rank); // Obtenemos el valor de nuestro identificador

  if (argc != 2) {
    if (rank == 0) {
      cerr << "Sintaxis: " << argv[0] << " <archivo de grafo>" << endl;
    }
    MPI_Finalize();
    return -1;
  }

  Graph G;
  int nverts;
  if (rank == 0) {
    G.lee(argv[1]);
    //cout << "El Grafo de entrada es:" << endl;
    //G.imprime();
    nverts = G.vertices;
  }

  // Enviar el número de vértices a todos los procesos
  MPI_Bcast(&nverts, 1, MPI_INT, 0, MPI_COMM_WORLD);
  cout << "Proceso " << rank << " recibe tamaño " << nverts << endl;

  double t = MPI_Wtime();
  /*
  // BUCLE PPAL DEL ALGORITMO
  int i,j,k,vikj;
  for (k = 0; k < nverts; k++) {
    for (i = 0; i < nverts; i++) {
      for (j = 0; j < nverts; j++) {
        if (i != j && i != k && j != k) {
          vikj = G.arista(i, k) + G.arista(k, j);
          vikj = min(vikj, G.arista(i, j));
          G.inserta_arista(i, j, vikj);   
        }
      }
    }
  }
  */
  t = MPI_Wtime() - t;
  MPI_Finalize();
 
  //cout << endl << "EL Grafo con las distancias de los caminos más cortos es:" << endl << endl;
  //G.imprime();
  cout << "Tiempo gastado= " << t<< endl << endl;
}



