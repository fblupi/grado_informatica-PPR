#include <iostream>
#include <cstring>
#include <cstdlib>
#include <fstream>
#include <string.h>
#include <omp.h>
#include "Graph.h"

#define PRINT_ALL

using namespace std;

int main(int argc, char *argv[]) {
  int procs, chunk, nverts, i, j, k, ik, kj, ij, *M;
  double t;

  switch(argc) {
    case 3:
      procs = atoi(argv[2]);
    break;
    case 2:
      procs = omp_get_num_procs();
      break;
    default:
      cerr << "Sintaxis: " << argv[0] << "<archivo de grafo> <num procs>" << endl;
      return(-1);
  }

  Graph G;
  G.lee(argv[1]);	// Read the Graph
  #ifdef PRINT_ALL
    cout << "El grafo de entrada es:" << endl;
    G.imprime();
  #endif

  nverts = G.vertices;
  chunk = nverts / procs;

  M = (int *) malloc(nverts * nverts * sizeof(int));
  G.copia_matriz(M);

  t = omp_get_wtime();
  for (k = 0; k < nverts; k++) {
    #pragma omp parallel for private(i, j, ik, ij, kj) schedule(static, chunk)
    for (i = 0; i < nverts; i++) {
      ik = i * nverts + k;
      for (j = 0; j < nverts; j++) {
        if (i != j && i != k && j != k) {
          kj = k * nverts + j;
          ij = i * nverts + j;
          M[ij] = min(M[ik] + M[kj], M[ij]);
        }
      }
    }
  }
  t = omp_get_wtime() - t;

  G.lee_matriz(M);

  #ifdef PRINT_ALL
    cout << endl << "El grafo con las distancias de los caminos más cortos es:" << endl;
    G.imprime();
    cout << "Tiempo gastado = " << t << endl << endl;
  #else
    cout << t << endl;
  #endif

  delete[] M;
}
