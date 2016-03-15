#include <iostream>
#include <fstream>
#include <string.h>
#include <math.h>
#include "Graph.h"
#include "mpi.h"

#define PRINT_ALL

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
  int nverts, tamaLocal, tamaBloque, raizP;
  if (rank == 0) { // Solo lo hace un proceso
    G.lee(argv[1]);
    #ifdef PRINT_ALL
      cout << "El grafo de entrada es:" << endl;
      G.imprime();
    #endif
    nverts = G.vertices;
  }

  /**
    * Paso 4: Hacer broadcast del número de vértices a todos los procesos
    */
  MPI_Bcast(&nverts, 1, MPI_INT, 0, MPI_COMM_WORLD);
  raizP = sqrt(size);
  tamaBloque = nverts / raizP;

  /**
    * Paso 5: Empaquetar
    */
  MPI_Datatype MPI_BLOQUE;
  int buffEnvio[nverts][nverts]; // Buffer de envío para almacenar los datos empaquetados
  int filaP, columnaP, comienzo;

  if (rank == 0) {
    MPI_Type_vector(tamaBloque, tamaBloque, nverts, MPI_INT, &MPI_BLOQUE); // Se define el tipo de bloque cuadrado
    MPI_Type_commit(&MPI_BLOQUE); // Se crea el nuevo tipo
    for (int i = 0, posicion = 0; i < size; i++) {
      // Cálculo de la posición de comienzo de cada submatriz
      filaP = i / raizP;
      columnaP = i % raizP;
      comienzo = columnaP * tamaBloque + filaP * tamaBloque * tamaBloque * raizP;
      MPI_Pack(G.ptrMatriz() + comienzo, 1, MPI_BLOQUE, buffEnvio, sizeof(int) * nverts * nverts, &posicion, MPI_COMM_WORLD);
    }
    MPI_Type_free(&MPI_BLOQUE); // Se libera el tipo bloque
  }

  /**
    * Paso 6: Distribuir la matriz entre los procesos
    */
  int M[tamaBloque][tamaBloque]; // Matriz local
  MPI_Scatter(buffEnvio, sizeof(int) * tamaBloque * tamaBloque, MPI_PACKED, M, tamaBloque * tamaBloque, MPI_INT, 0, MPI_COMM_WORLD);

//  for (int i = 0; i < tamaBloque; i++)
//    for (int j = 0; j < tamaBloque; j++)
//      cout << "[P" << rank << "] --> M[" << i << "][" << j << "] = " << M[i][j] << endl;

  /**
    * Paso 7: Bucle principal del algoritmo
    */

  double t = MPI_Wtime();

  // Se ejecutaría....
//  M[0][0] = 14;

  t = MPI_Wtime() - t;

  /**
    * Paso 8: Recoger resultados en la matriz
    */
  MPI_Gather(M, tamaBloque * tamaBloque, MPI_INT, buffEnvio, sizeof(int) * tamaBloque * tamaBloque, MPI_PACKED, 0, MPI_COMM_WORLD);

  /**
    * Paso 9: Desempaquetar
    */
  if (rank == 0) {
    MPI_Type_vector(tamaBloque, tamaBloque, nverts, MPI_INT, &MPI_BLOQUE); // Se define el tipo de bloque cuadrado
    MPI_Type_commit(&MPI_BLOQUE); // Se crea el nuevo tipo
    for (int i = 0, posicion = 0; i < size; i++) {
      // Cálculo de la posición de comienzo de cada submatriz
      filaP = i / raizP;
      columnaP = i % raizP;
      comienzo = columnaP * tamaBloque + filaP * tamaBloque * tamaBloque * raizP;
      MPI_Unpack(buffEnvio, sizeof(int) * nverts * nverts, &posicion, G.ptrMatriz() + comienzo, 1, MPI_BLOQUE, MPI_COMM_WORLD);
    }
    MPI_Type_free(&MPI_BLOQUE); // Se libera el tipo bloque
  }

  /**
    * Paso 10: Finalizar e imprimir resultados
    */
  MPI_Finalize();

  if (rank == 0) { // Solo lo hace un proceso
    #ifdef PRINT_ALL
      cout << endl << "El grafo con las distancias de los caminos más cortos es:" << endl;
      G.imprime();
      cout << "Tiempo gastado = " << t << endl << endl;
    #else
      cout << t << endl;
    #endif
  }
}