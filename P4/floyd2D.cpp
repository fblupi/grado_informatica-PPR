#include <iostream>
#include <cstring>
#include <cstdlib>
#include <fstream>
#include <string.h>
#include <omp.h>
#include <math.h>
#include <unistd.h>
#include "Graph.h"

#define PRINT_ALL

using namespace std;

int main(int argc, char *argv[]) {
  int procs, sqrtP, tamaBloque, nverts, i, j, k, ik, kj, ij, *M, *colK, *filK, id,
      iIni, iFin, jIni, jFin, iGlobal, jGlobal;
  double t;

  switch(argc) {
    case 3: // se especifica el número de procesadores
      procs = atoi(argv[2]);
      break;
    case 2: // no se especifica el número de procesadores => se usan todos los que tenga el equipo
      procs = omp_get_num_procs();
      break;
    default: // número incorrecto de parámetros => se termina la ejecución
      cerr << "Sintaxis: " << argv[0] << "<archivo de grafo> <num procs>" << endl;
      return(-1);
  }
  omp_set_num_threads(procs);

  Graph G;
  G.lee(argv[1]);	// Lee el grafo
  #ifdef PRINT_ALL
    cout << "El grafo de entrada es:" << endl;
    G.imprime();
  #endif

  sqrtP = sqrt(procs);
  nverts = G.vertices;          // Número de vértices
  tamaBloque = nverts / sqrtP;  // Tamaño de bloque

  #ifdef PRINT_ALL
    cout << endl;
    cout << "El tamaño del problema es: " << nverts << endl;
    cout << "El número de procesos es: " << procs << endl;
    cout << "El tamaño de bloque (tama submatriz) es: " << tamaBloque << "x" << tamaBloque << endl;
  #endif

  M = (int *) malloc(nverts * nverts * sizeof(int));    // Se reserva espacio en memoria para M
  G.copia_matriz(M);                                    // Se copia la matriz del grafo
  colK = (int *) malloc(nverts * sizeof(int));
  filK = (int *) malloc(nverts * sizeof(int));

  t = omp_get_wtime();

  #pragma omp parallel private(id, i, j, ik, ij, kj, iIni, iFin, jIni, jFin, iGlobal, jGlobal) // inicio de la región paralela
  {
    id = omp_get_thread_num();
    iIni = id / sqrtP * tamaBloque;
    iFin = (id / sqrtP + 1) * tamaBloque;
    jIni = id % sqrtP * tamaBloque;
    jFin = (id % sqrtP + 1) * tamaBloque;
    //printf("%d --> i = %d ~ %d, j == %d ~ %d\n", id, iIni, iFin, jIni, jFin);
    for (k = 0; k < nverts; k++) {
      #pragma omp master
      for (i = 0; i < nverts; i++) {
        colK[i] = M[i * nverts + k];
        filK[i] = M[k * nverts + i];
      }
      #pragma omp barrier
      #pragma omp flush(colK)
      #pragma omp flush(filK)
      for (i = 0; i < tamaBloque; i++) {
        iGlobal = iIni + i;
        for (j = 0; j < tamaBloque; j++) {
          jGlobal = jIni + j;
          if (iGlobal != jGlobal && iGlobal != k && jGlobal != k) {
            ij = iGlobal * nverts + jGlobal;
            M[ij] = min(colK[iGlobal] + filK[jGlobal], M[ij]);
          }
        }
      }
    }
  }
  t = omp_get_wtime() - t;

  G.lee_matriz(M);  // Se copia en el grafo el resultado calculado en la matriz

  #ifdef PRINT_ALL
    cout << endl << "El grafo con las distancias de los caminos más cortos es:" << endl;
    G.imprime();
    cout << "Tiempo gastado = " << t << endl << endl;
  #else
    cout << t << endl;
  #endif

  delete[] M;
  delete[] filK;
  delete[] colK;

  return(0);
}
