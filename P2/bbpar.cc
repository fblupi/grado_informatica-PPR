/* ******************************************************************** */
/*               Algoritmo Branch-And-Bound Paralelo                    */
/* ******************************************************************** */
#include <cstdlib>
#include <cstdio>
#include <iostream>
#include <mpi.h>
#include "libbb.h"

using namespace std;

unsigned int NCIUDADES;
int rank, size;

/* ******************************************************************** */

int main (int argc, char **argv) {
  MPI_Init(&argc, &argv);
  MPI_Comm_size(MPI_COMM_WORLD, &size);
  MPI_Comm_rank(MPI_COMM_WORLD, &rank);

  switch (argc) {
    case 3:
      NCIUDADES = atoi(argv[1]);
      break;
    default:
      if (rank == 0)
        cerr << "La sintaxis es: bbseq <tamaño> <archivo>" << endl;
      MPI_Finalize();
      exit(-1);
    break;
  }

  int** tsp0 = reservarMatrizCuadrada(NCIUDADES);
  tNodo   nodo,                   // nodo a explorar
          nodo_izq,               // hijo izquierdo
          nodo_dch,               // hijo derecho
          solucion;               // mejor solucion
  bool    fin = false,            // condicion de fin
          nueva_U;                // hay nuevo valor de c.s.
  int     U;                      // valor de c.s.
  int     iteraciones = 0;
  tPila   pila;                   // pila de nodos a explorar

  U = INFINITO;                   // inicializa cota superior

  extern MPI_Comm comunicadorCarga;	// Para la distribución de la carga
  extern MPI_Comm comunicadorCota;	// Para la difusión de una nueva cota superior detectada

  MPI_Comm_dup(MPI_COMM_WORLD, &comunicadorCarga);
  MPI_Comm_dup(MPI_COMM_WORLD, &comunicadorCota);

  if (rank == 0) {
    LeerMatriz(argv[2], tsp0);
    MPI_Bcast(&tsp0[0][0], NCIUDADES * NCIUDADES, MPI_INT, 0, MPI_COMM_WORLD);
    InicNodo(&nodo);                // inicializa estructura nodo
  } else {
    MPI_Bcast(&tsp0[0][0], NCIUDADES * NCIUDADES, MPI_INT, 0, MPI_COMM_WORLD);
    Equilibrado_Carga(&pila, &fin);
    if (!fin)
      pila.pop(nodo);
  }

  double t = MPI_Wtime();
  while (!fin) { // ciclo de Branch&Bound
    cout << "[" << rank << "]: " << "it " << iteraciones << endl;
    Ramifica(&nodo, &nodo_izq, &nodo_dch, tsp0);
    nueva_U = false;

    if (Solucion(&nodo_dch)) {
      //cout << "[" << rank << "]: " << "Hijo dcha. es solución" << endl;
      if (nodo_dch.ci() < U) {
        //cout << "[" << rank << "]: " << "Hijo dcha. es nueva CS" << endl;
        U = nodo_dch.ci(); // actualiza c.s.
        nueva_U = true;
        CopiaNodo(&nodo_dch, &solucion);
      }
    } else { // no es nodo hoja
      //cout << "[" << rank << "]: " << "Hijo dcha. no es solución" << endl;
      if (nodo_dch.ci() < U) {
        //cout << "[" << rank << "]: " << "Hijo dcha. a la pila" << endl;
        pila.push(nodo_dch);
      }
    }

    if (Solucion(&nodo_izq)) {
      //cout << "[" << rank << "]: " << "Hijo izda. es solución" << endl;
      if (nodo_izq.ci() < U) {
        //cout << "[" << rank << "]: " << "Hijo izda. es nueva CS" << endl;
        U = nodo_izq.ci(); // actualiza c.s.
        nueva_U = true;
        CopiaNodo(&nodo_izq, &solucion);
      }
    } else { // no es nodo hoja
      //cout << "[" << rank << "]: " << "Hijo izda. no es solución" << endl;
      if (nodo_izq.ci() < U) {
        //cout << "[" << rank << "]: " << "Hijo izda. a la pila" << endl;
        pila.push(nodo_izq);
      }
    }

    Difusion_Cota_Superior(&U);
    if (nueva_U)
      pila.acotar(U);

    Equilibrado_Carga(&pila, &fin);
    if (!fin)
      pila.pop(nodo);

    iteraciones++;
  }
  t = MPI_Wtime() - t;

  cout << "[" << rank << "]: Solución = " << endl;
  EscribeNodo(&solucion);
  cout << "[" << rank << "]: " << iteraciones << endl;

  MPI_Finalize();
  liberarMatriz(tsp0);
}
