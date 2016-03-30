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
  int nverts;

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
  int raizP, tamaBloque;
  MPI_Bcast(&nverts, 1, MPI_INT, 0, MPI_COMM_WORLD);

  raizP = sqrt(size);
  tamaBloque = nverts / raizP;

  /**
    * Paso 5: Crear comunicadores
    */
  int colorHorizontal, colorVertical, rankHorizontal, rankVertical;
  MPI_Comm commHorizontal, commVertical;

  colorHorizontal = rank / raizP;
  colorVertical = rank % raizP;

  MPI_Comm_split(MPI_COMM_WORLD, colorHorizontal, rank, &commHorizontal);
  MPI_Comm_split(MPI_COMM_WORLD, colorVertical, rank, &commVertical);

  MPI_Comm_rank(commHorizontal, &rankHorizontal);
  MPI_Comm_rank(commVertical, &rankVertical);

  /**
    * Paso 6: Empaquetar
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
    * Paso 7: Distribuir la matriz entre los procesos
    */
  int M[tamaBloque][tamaBloque], FilK[tamaBloque], ColK[tamaBloque]; // Matriz local

  MPI_Scatter(buffEnvio, sizeof(int) * tamaBloque * tamaBloque, MPI_PACKED, M, tamaBloque * tamaBloque, MPI_INT, 0, MPI_COMM_WORLD);

  /**
    * Paso 8: Bucle principal del algoritmo
    */
  int i, j, k, a, vikj, iGlobal, jGlobal, iIniLocal, iFinLocal, jIniLocal, jFinLocal, kEntreTama, kModuloTama; 

  iIniLocal = colorHorizontal * tamaBloque; // Fila inicial del proceso (valor global)
  iFinLocal = (colorHorizontal + 1) * tamaBloque; // Fila final del proceso (valor global)
  jIniLocal = colorVertical * tamaBloque; // Columna inicial del proceso (valor global)
  jFinLocal = (colorVertical + 1) * tamaBloque; // Columna final del proceso (valor global)

  double t = MPI_Wtime();

  for (k = 0; k < nverts; k++) {
    kEntreTama = k / tamaBloque;
    kModuloTama = k % tamaBloque;
    if (k >= iIniLocal && k < iFinLocal) { // La fila K pertenece al proceso
      copy(M[kModuloTama], M[kModuloTama] + tamaBloque, FilK); // Copia la fila en el vector FilK
    }
    if (k >= jIniLocal && k < jFinLocal) { // La columna K pertenece al proceso
      for (a = 0; a < tamaBloque; a++) {
        ColK[a] = M[a][kModuloTama]; // Copia la columna en el vector ColK
      }
    }
    MPI_Bcast(FilK, tamaBloque, MPI_INT, kEntreTama, commVertical);
    MPI_Bcast(ColK, tamaBloque, MPI_INT, kEntreTama, commHorizontal);
    for (i = 0; i < tamaBloque; i++) { // Recorrer las filas (valores locales)
      iGlobal = iIniLocal + i; // Convertir la fila a global
      for (j = 0; j < tamaBloque; j++) {  // Recorrer las columnas (valores locales)
        jGlobal = jIniLocal + j;
        if (iGlobal != jGlobal && iGlobal != k && jGlobal != k) { // No iterar sobre celdas de valor 0
          vikj = ColK[i] + FilK[j];
          vikj = min(vikj, M[i][j]);
          M[i][j] = vikj;
        }
      }
    }
  }

  t = MPI_Wtime() - t;

  /**
    * Paso 9: Recoger resultados en la matriz
    */
  MPI_Gather(M, tamaBloque * tamaBloque, MPI_INT, buffEnvio, sizeof(int) * tamaBloque * tamaBloque, MPI_PACKED, 0, MPI_COMM_WORLD);

  /**
    * Paso 10: Desempaquetar
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
    * Paso 11: Finalizar e imprimir resultados
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