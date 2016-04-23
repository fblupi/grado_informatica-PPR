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

  int**   tsp0 = reservarMatrizCuadrada(NCIUDADES);
  tNodo   *nodo = new tNodo(),      // nodo a explorar
          *nodo_izq = new tNodo(),  // hijo izquierdo
          *nodo_dch = new tNodo(),  // hijo derecho
          *solucion = new tNodo();  // mejor solucion
  bool    fin = false,              // condicion de fin
          nueva_U;                  // hay nuevo valor de c.s.
  int     U,                        // valor de c.s.
          iteraciones = 0,
          acotaciones = 0;
  tPila   *pila = new tPila();      // pila de nodos a explorar

  extern bool     token_presente;   // poseedor del token
  extern int      anterior,         // proceso anterior
                  siguiente;        // proceso siguiente
  extern MPI_Comm comunicadorCarga,	// Para la distribución de la carga
                  comunicadorCota;	// Para la difusión de una nueva cota superior detectada

  U = INFINITO;                         // inicializa cota superior
  anterior = (rank - 1 + size) % size;  // inicializa proceso anterior
  siguiente = (rank + 1) % size;        // inicializa proceso siguiente

  MPI_Comm_dup(MPI_COMM_WORLD, &comunicadorCarga);
  MPI_Comm_dup(MPI_COMM_WORLD, &comunicadorCota);

  if (rank == 0) {
    token_presente = true;
    LeerMatriz(argv[2], tsp0);
    MPI_Bcast(&tsp0[0][0], NCIUDADES * NCIUDADES, MPI_INT, 0, MPI_COMM_WORLD);
    InicNodo(nodo);                // inicializa estructura nodo
  } else {
    token_presente = false;
    MPI_Bcast(&tsp0[0][0], NCIUDADES * NCIUDADES, MPI_INT, 0, MPI_COMM_WORLD);
    Equilibrado_Carga(pila, &fin, solucion);
    if (!fin)
      pila->pop(*nodo);
  }

  double t = MPI_Wtime();
  while (!fin) { // ciclo de Branch&Bound
    Ramifica(nodo, nodo_izq, nodo_dch, tsp0);
    nueva_U = false;

    if (Solucion(nodo_dch)) {
      if (nodo_dch->ci() < U) {
        U = nodo_dch->ci(); // actualiza c.s.
        nueva_U = true;
        CopiaNodo(nodo_dch, solucion);
      }
    } else { // no es nodo hoja
      if (nodo_dch->ci() < U) {
        pila->push(*nodo_dch);
      }
    }

    if (Solucion(nodo_izq)) {
      if (nodo_izq->ci() < U) {
        U = nodo_izq->ci(); // actualiza c.s.
        nueva_U = true;
        CopiaNodo(nodo_izq, solucion);
      }
    } else { // no es nodo hoja
      if (nodo_izq->ci() < U) {
        pila->push(*nodo_izq);
      }
    }

    //Difusion_Cota_Superior(&U, &nueva_U);
    if (nueva_U) {
      acotaciones++;
      pila->acotar(U);
    }

    Equilibrado_Carga(pila, &fin, solucion);
    if (!fin) {
      pila->pop(*nodo);
    }

    iteraciones++;
  }
  t = MPI_Wtime() - t;

  if (rank == 0) {
     cout << "Solución = " << endl;
     EscribeNodo(solucion);
     cout << "Tiempo gastado= " << t << endl;
  }
  cout << "[" << rank << "] Numero de iteraciones: " << iteraciones << endl;
  cout << "[" << rank << "] Numero de acotaciones: " << acotaciones << endl;

  liberarMatriz(tsp0);
  delete pila;
  delete nodo;
  delete nodo_izq;
  delete nodo_dch;
  delete solucion;

  MPI_Finalize();

  exit(0);
}
