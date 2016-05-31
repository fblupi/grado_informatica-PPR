#include <iostream>
#include <cstring>
#include <cstdlib>
#include <fstream>
#include <string.h>
#include <omp.h>
#include "Graph.h"

//#define PRINT_ALL

using namespace std;

int main(int argc, char *argv[]) {
  int procs, chunk, nverts, i, j, k, ik, kj, ij, *M;
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

  nverts = G.vertices;                                  // Número de vértices
  procs > nverts ? chunk = 1 : chunk = nverts / procs;  // Tamaño de bloque

  #ifdef PRINT_ALL
    cout << endl;
    cout << "El tamaño del problema es: " << nverts << endl;
    cout << "El número de procesos es: " << procs << endl;
    cout << "El tamaño de bloque es: " << chunk << endl;
  #endif

  M = (int *) malloc(nverts * nverts * sizeof(int));    // Se reserva espacio en memoria para M
  G.copia_matriz(M);                                    // Se copia la matriz del grafo

  t = omp_get_wtime();
  for (k = 0; k < nverts; k++) {
    #pragma omp parallel for schedule(static, chunk) private(i, j, ik, ij, kj)  // inicio de la región paralela, reparto estático por bloques
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

  G.lee_matriz(M);  // Se copia en el grafo el resultado calculado en la matriz

  #ifdef PRINT_ALL
    cout << endl << "El grafo con las distancias de los caminos más cortos es:" << endl;
    G.imprime();
    cout << "Tiempo gastado = " << t << endl << endl;
  #else
    cout << t << endl;
  #endif

  delete[] M;

  return(0);
}
