#include <iostream>
#include <fstream>
#include <string.h>
#include "Graph.h"
#include "mpi.h"

using namespace std;

int main (int argc, char *argv[]) 
{
  /**
    * Paso 1: Iniciar MPI y obtener tamaño e id para cada proceso
    */
  int rank, size, tama;

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
    //cout << "El Grafo de entrada es:" << endl;
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
  tamaBloque = nverts / size;
  int M[tamaBloque][nverts], 
      K[nverts];

  /**
    * Paso 6: Repartir matriz entre los procesos
    */
  MPI_Scatter(G.ptrMatriz(), tamaLocal, MPI_INT, &M[0][0], tamaLocal, MPI_INT, 0, MPI_COMM_WORLD);

  /**
    * Paso 7: Bucle principal del algoritmo
    */
  int i, j, k, vikj,
      iniLocal = rank * tamaBloque, // fila inicial en global
      finLocal = (rank + 1) * tamaBloque; // fila final en global

  double t = MPI_Wtime();

  for (k = 0; k < nverts; k++) {
    if (k >= iniLocal && k < finLocal) { // La fila K pertenece al proceso
      copy(M[k % tamaBloque], M[k % tamaBloque] + nverts, K);
    }
    MPI_Bcast(&K, nverts, MPI_INT, k / tamaBloque, MPI_COMM_WORLD);
    for (i = 0; i < tamaBloque; i++) { // valores locales
      for (j = 0; j < nverts; j++) {
        if (i != j && i != k && j != k) { // No iterar sobre celdas de valor 0
          vikj = M[i][k] + K[j];
          vikj = min(vikj, M[i][j]);
          M[i][j] = vikj;
        }
      }
    }
  }

  t = MPI_Wtime() - t;

  for (int i = 0; i < tamaBloque; i++) {
    for (int j = 0; j < nverts; j++) {
      cout << "[P" << rank << "] --> M[" << i << "][" << j << "] = " << M[i][j] << endl;
    }
  }

  /**
    * Paso 8: Recoger resultados en la matriz
    */
  MPI_Gather(&M[0][0], tamaLocal, MPI_INT, G.ptrMatriz(), tamaLocal, MPI_INT, 0, MPI_COMM_WORLD);

  /**
    * Paso 9: Finalizar e imprimir resultados
    */
  MPI_Finalize();
 
  if (rank == 0) { // Solo lo hace un proceso
    cout << endl << "El Grafo con las distancias de los caminos más cortos es:" << endl << endl;
    G.imprime();
    cout << "Tiempo gastado = " << t << endl << endl;
  }

}