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
  int procs, sqrtP, tamaBloque, nverts, i, j, k, ij, *M, *colK, *filK, id,
      iIni, iFin, jIni, jFin, iGlobal, jGlobal, nvertsPorK, nvertsPorI, idEntreSqrtP, idModuloSqrtP;
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

  #pragma omp parallel private(id, i, j, k, ij, iIni, iFin, jIni, jFin, iGlobal, jGlobal, idEntreSqrtP, idModuloSqrtP, nvertsPorK, nvertsPorI) // inicio de la región paralela
  {
    id = omp_get_thread_num();
    idEntreSqrtP = id / sqrtP;
    idModuloSqrtP = id % sqrtP;
    iIni = idEntreSqrtP * tamaBloque;
    iFin = (idEntreSqrtP + 1) * tamaBloque;
    jIni = idModuloSqrtP * tamaBloque;
    jFin = (idModuloSqrtP + 1) * tamaBloque;
    for (k = 0; k < nverts; k++) {
      nvertsPorK = k * nverts;
      #pragma omp barrier
      for (i = 0; i < nverts; i++) {
        colK[i] = M[nverts * i + k];
        filK[i] = M[nvertsPorK + i];
      }
      for (i = iIni; i < iFin; i++) {
        nvertsPorI = i * nverts;
        for (j = jIni; j < jFin; j++) {
          if (i != j && i != k && j != k) {
            ij = nvertsPorI + j;
            M[ij] = min(colK[i] + filK[j], M[ij]);
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

  //free(M);
  //free(filK);
  //free(colK);

  return(0);
}
