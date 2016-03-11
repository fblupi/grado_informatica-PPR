#include <iostream>
#include <fstream>
#include <string.h>
#include "Graph.h"
#include "mpi.h"

using namespace std;
/*
int pasarFilaALocal(int tamaBloque, int indiceGlobal) {
  return indiceGlobal % tamaBloque;
}

int pasarFilaAGlobal(int rank, int tamaBloque, int indiceLocal) {
  return rank * tamaBloque + indiceLocal;
}

bool filaMePertenece(int rank, int, tamaBloque, int fila) {
  if (fila >= pasarFilaAGlobal(0) && fila < pasarFilaAGlobal(tamaBloque))
    return true;
  else 
    return false;
}
*/
int main (int argc, char *argv[]) 
{
  int rank, size, tama;

  /**
    * Paso 1: Iniciar MPI y obtener tamaño e id para cada proceso
    */
  MPI_Init(&argc, &argv); // Inicializamos la comunicacion de los procesos
  MPI_Comm_size(MPI_COMM_WORLD, &size); // Obtenemos el número total de procesos
  MPI_Comm_rank(MPI_COMM_WORLD, &rank); // Obtenemos el valor de nuestro identificador

  /**
    * Paso 2: Comprobar entradas
    */
  if (argc != 2) { // Debe haber dos argumentos
    if (rank == 0) { // El proceso 0 imprime el error
      cerr << "Sintaxis: " << argv[0] << " <archivo de grafo>" << endl;
    }
    MPI_Finalize();
    return -1;
  }
  
  /**
    * Paso 3: Crear grafo y obtener número de vértices
    */
  Graph G;
  int nverts, tamaLocal, tamaBloque;
  if (rank == 0) { // Solo lo hace un proceso
    G.lee(argv[1]);
    cout << "El Grafo de entrada es:" << endl;
    G.imprime();
    nverts = G.vertices;
  }

  /**
    * Paso 4: Hacer broadcast del número de vértices a todos los procesos
    */
  MPI_Bcast(&nverts, 1, MPI_INT, 0, MPI_COMM_WORLD);

  /**
    * Paso 5: Reservar espacio para matriz y fila k
    */
  tamaLocal = nverts * nverts / size;
  int * M = new int [tamaLocal], 
      * filaK = new int [nverts];

  /**
    * Paso 6: Repartir matriz entre los procesos
    */
  MPI_Scatter(G.ptrMatriz(), tamaLocal, MPI_INT, M, tamaLocal, MPI_INT, 0, MPI_COMM_WORLD);

/*
  double t = MPI_Wtime();

  // BUCLE PPAL DEL ALGORITMO
  int i, j, k, vikj;
  for (k = 0; k < nverts; k++) {
    if (filaMePertenece, tamaBloque, k) {
      filaK = M[k];
      MPI_Bcast(&filaK, nverts, MPI_INT, 0, MPI_COMM_WORLD);
    }
    MPI_Bcast(&M[pasarFilaALocal(k)], nverts, MPI_INT, 0, MPI_COMM_WORLD);
    for (i = pasarFilaAGlobal(0); i < pasarFilaAGlobal(tamaBloque); i++) {
      for (j = 0; j < nverts; j++) {
        if (i != j && i != k && j != k) {
          vikj = G.arista(i, k) + G.arista(k, j);
          vikj = min(vikj, G.arista(i, j));
          G.inserta_arista(i, j, vikj);   
        }
      }
    }
  }

  t = MPI_Wtime() - t;
  */

  /**
    * Paso N - 1: Recoger resultados en la matriz
    */
  MPI_Gather(&M[0], tamaLocal, MPI_INT, G.ptrMatriz(), tamaLocal, MPI_INT, 0, MPI_COMM_WORLD);

  /**
    * Paso N: Finalizar e imprimir resultados
    */
  MPI_Finalize();
 
  //cout << endl << "EL Grafo con las distancias de los caminos más cortos es:" << endl << endl;
  //G.imprime();
  //cout << "Tiempo gastado= " << t<< endl << endl;

}



