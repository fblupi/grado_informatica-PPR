#include <iostream>
#include <fstream>
#include <string.h>
#include <time.h>
#include "Graph.h"

#define PRINT_ALL

using namespace std;

int main (int argc, char *argv[])
{

  if (argc != 2) {
    cerr << "Sintaxis: " << argv[0] << " <archivo de grafo>" << endl;
    return(-1);
  }

  Graph G;
  G.lee(argv[1]);	// Read the Graph
  #ifdef PRINT_ALL
    cout << "El grafo de entrada es:" << endl;
    G.imprime();
  #endif

  int nverts = G.vertices;

  double t = clock();
  // BUCLE PPAL DEL ALGORITMO
  int i, j, k, vikj;
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
  t = (clock() - t) / CLOCKS_PER_SEC;

  #ifdef PRINT_ALL
    cout << endl << "El grafo con las distancias de los caminos más cortos es:" << endl;
    G.imprime();
    cout << "Tiempo gastado = " << t << endl << endl;
  #else
    cout << t << endl;
  #endif
}
