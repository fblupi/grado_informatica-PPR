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

void Equilibrado_Carga(tPila *pila, bool *fin) {
  //cout << "[" << rank << "]: " << "Equilibrando carga" << endl;
  int solicitante, flag, tamanio;
  MPI_Status estado;
  tNodo nodo;
  tPila pilaNueva;
  if (pila->vacia()) { // el proceso no tiene trabajo: pide a otros procesos
    //cout << "[" << rank << "]: " << "Pila vacía" << endl;
    /* Enviar petición de trabajo al proceso (rank + 1) % size */
    //cout << "[" << rank << "]: " << "Voy a enviar petición de nodo" << endl;
    MPI_Send(&rank, 1, MPI_INT, (rank + 1) % size, PETICION, comunicadorCarga);
    //cout << "[" << rank << "]: " << "Petición de nodo enviada" << endl;
    while (pila->vacia() && !*fin) {
      /* Esperar mensaje de otro proceso */
      //cout << "[" << rank << "]: " << "Esperando mensaje" << endl;
      MPI_Probe(MPI_ANY_SOURCE, MPI_ANY_TAG, comunicadorCarga, &estado);
      //cout << "[" << rank << "]: " << "Mensaje recibido" << endl;
      switch (estado.MPI_TAG) {
        case PETICION: // peticion de trabajo
          //cout << "[" << rank << "]: " << "Recibir mensaje de tipo petición" << endl;
          /* Recibir mensaje de petición de trabajo */
          MPI_Recv(&solicitante, 1, MPI_INT, MPI_ANY_SOURCE, PETICION, comunicadorCarga, &estado);
          if (solicitante == rank) { // peticion devuelta
            /* Reenviar petición de trabajo al proceso (rank + 1) % size */
            //cout << "[" << rank << "]: " << "Voy a reenviar petición de nodo" << endl;
            MPI_Send(&solicitante, 1, MPI_INT, (rank + 1) % size, PETICION, comunicadorCarga);
            //cout << "[" << rank << "]: " << "Petición de nodo reenviada" << endl;
            /* Iniciar detección de posible situación de fin */
            //cout << "[" << rank << "]: " << "Posible fin" << endl;
          } else { // petición de otro proceso: la retransmite al siguiente
            /* Pasar petición de trabajo al proceso (rank + 1) % size */
            //cout << "[" << rank << "]: " << "Voy a pasar petición" << endl;
            MPI_Send(&solicitante, 1, MPI_INT, (rank + 1) % size, PETICION, comunicadorCarga);
            //cout << "[" << rank << "]: " << "Petición pasada de nodo " << solicitante << endl;
          }
          break;
        case NODOS: // resultado de una petición de trabajo
          MPI_Get_count(&estado, MPI_INT, &tamanio);
          //cout << "[" << rank << "]: " << "Recibir mensaje de tipo nodos" << endl;
          /* Recibir nodos del proceso donante */
          MPI_Recv(&pila->nodos[0], tamanio, MPI_INT, MPI_ANY_SOURCE, NODOS, comunicadorCarga, &estado);
          /* Almacenar nodos recibidos en la pila */
          pila->tope = tamanio;
          //cout << "[" << rank << "]: " << "Nodo almacenado" << endl;
      }
    }
  }
  if (!*fin) { // el proceso tiene nodos para trabajar
    /* Sondear si hay mensajes pendientes de otros procesos */
    //cout << "[" << rank << "]: " << "Sondea si hay mensajes pendientes de otros procesos" << endl;
    MPI_Iprobe(MPI_ANY_SOURCE, MPI_ANY_TAG, comunicadorCarga, &flag, &estado);
    while (flag) { // atiende peticiones mientras haya mensajes
      //cout << "[" << rank << "]: " << "Recibe mensaje de petición de trabajo. Tamaño pila: " << pila->tamanio() << endl;
      /* Recibir mensaje de petición de trabajo */
      MPI_Recv(&solicitante, 1, MPI_INT, MPI_ANY_SOURCE, PETICION, comunicadorCarga, &estado);
      if (pila->tamanio() > 1) {
        /* Enviar nodos al proceso solicitante */
        pila->divide(pilaNueva);
        //cout << "[" << rank << "]: " << "Voy a enviar nodo" << endl;
        MPI_Send(&pilaNueva.nodos[0], pilaNueva.tope, MPI_INT, solicitante, NODOS, comunicadorCarga);
        //cout << "[" << rank << "]: " << "Nodo enviado" << endl;
      } else {
        /* Pasar petición de trabajo al proceso (rank + 1) % size */
        //cout << "[" << rank << "]: " << "Voy a reenviar petición de nodo" << endl;
        MPI_Send(&solicitante, 1, MPI_INT, (rank + 1) % size, PETICION, comunicadorCarga);
        //cout << "[" << rank << "]: " << "Petición de nodo reenviada" << endl;
      }
      /* Sondear si hay mensajes pendientes de otros procesos */
      //cout << "[" << rank << "]: " << "Sondea si hay mensajes pendientes de otros procesos" << endl;
      MPI_Iprobe(MPI_ANY_SOURCE, MPI_ANY_TAG, comunicadorCarga, &flag, &estado);
    }
  }
}

// void Difusion_Cota_Superior() {
//   if (difundir_cs_local && !pendiente_retorno_cs) {
//     /* Enviar valor local de cs al proceos (rank + 1) % size */
//     pendiente_retorno_cs = true;
//     difundir_cs_local = false;
//   }
//   /* Sondear si hay mensajes de cota superior pendientes */
//   while (hay_mensajes) {
//     /* Recibir mensajes con valor de cota superior desde el proceso (rank - 1 + size) % size */
//     /* Actualizar valor local de cota superior */
//
//     if (origen_mensaje == rank && difundir_cs_local) {
//       /* Enviar valor local de cs al proceso (rank + 1) % size */
//       pendiente_retorno_cs = true;
//       difundir_cs_local = false;
//     } else if (origen_mensaje == rank && !difundir_cs_local)
//       pendiente_retorno_cs = false;
//     else // origen mensaje == otro proceso
//       /* Reenviar mensaje al proceso (rank + 1) % size */
//     /* Sondear si hay mensajes de cota superior pendientes */
//   }
// }

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
  int     U = INFINITO;           // valor de c.s.
  int     iteraciones = 0;
  tPila   pila;                   // pila de nodos a explorar

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

    // Difusion_Cota_Superior(&U);
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
