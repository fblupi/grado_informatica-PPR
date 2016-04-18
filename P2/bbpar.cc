/* ******************************************************************** */
/*               Algoritmo Branch-And-Bound Paralelo                    */
/* ******************************************************************** */
#include <cstdlib>
#include <cstdio>
#include <iostream>
#include <mpi.h>
#include "libbb.h"

using namespace std;

#define PETIC 0
#define NODOS 1

unsigned int NCIUDADES;
int id, P;

void Equilibrado_Carga(tPila *pila, bool *fin) {
  //cout << "[" << id << "]: " << "Equilibrando carga. FIN = " << *fin << endl;
  int solicitante, flag;
  MPI_Status estado;
  tNodo nodo;
  if (pila->vacia()) { // el proceso no tiene trabajo: pide a otros procesos
    //cout << "[" << id << "]: " << "Pila vacía" << endl;
    /* Enviar petición de trabajo al proceso (id + 1) % P */
    //cout << "[" << id << "]: " << "Voy a enviar petición de nodo" << endl;
    MPI_Send(&id, 1, MPI_INT, (id + 1) % P, PETIC, MPI_COMM_WORLD);
    //cout << "[" << id << "]: " << "Petición de nodo enviada" << endl;
    while (pila->vacia() && !*fin) {
      /* Esperar mensaje de otro proceso */
      //cout << "[" << id << "]: " << "Esperando mensaje" << endl;
      MPI_Probe((id - 1) % P, MPI_ANY_TAG, MPI_COMM_WORLD, &estado);
      //cout << "[" << id << "]: " << "Mensaje recibido" << endl;
      switch (estado.MPI_TAG) {
        case PETIC: // peticion de trabajo
          //cout << "[" << id << "]: " << "Recibir mensaje de tipo petición" << endl;
          /* Recibir mensaje de petición de trabajo */
          MPI_Recv(&solicitante, 1, MPI_INT, MPI_ANY_SOURCE, PETIC, MPI_COMM_WORLD, &estado);
          if (solicitante == id) { // peticion devuelta
            /* Reenviar petición de trabajo al proceso (id + 1) % P */
            //cout << "[" << id << "]: " << "Voy a reenviar petición de nodo" << endl;
            MPI_Send(&solicitante, 1, MPI_INT, (id + 1) % P, PETIC, MPI_COMM_WORLD);
            //cout << "[" << id << "]: " << "Petición de nodo reenviada" << endl;
            /* Iniciar detección de posible situación de fin */
            //cout << "[" << id << "]: " << "Posible fin" << endl;
          } else { // petición de otro proceso: la retransmite al siguiente
            /* Pasar petición de trabajo al proceso (id + 1) % P */
            //cout << "[" << id << "]: " << "Voy a pasar petición" << endl;
            MPI_Send(&solicitante, 1, MPI_INT, (id + 1) % P, PETIC, MPI_COMM_WORLD);
            //cout << "[" << id << "]: " << "Petición pasada" << endl;
          }
          break;
        case NODOS: // resultado de una petición de trabajo
          //cout << "[" << id << "]: " << "Recibir mensaje de tipo nodos" << endl;
          /* Recibir nodos del proceso donante */
          MPI_Recv(&nodo.datos[0], 2 * NCIUDADES, MPI_INT, MPI_ANY_SOURCE, NODOS, MPI_COMM_WORLD, &estado);
          /* Almacenar nodos recibidos en la pila */
          pila->push(nodo);
          //cout << "[" << id << "]: " << "Nodo almacenado" << endl;
      }
    }
  }
  if (!*fin) { // el proceso tiene nodos para trabajar
    /* Sondear si hay mensajes pendientes de otros procesos */
    //cout << "[" << id << "]: " << "Sondea si hay mensajes pendientes de otros procesos" << endl;
    MPI_Iprobe((id - 1) % P, MPI_ANY_TAG, MPI_COMM_WORLD, &flag, &estado);
    while (flag) { // atiende peticiones mientras haya mensajes
      //cout << "[" << id << "]: " << "Recibe mensaje de petición de trabajo" << endl;
      /* Recibir mensaje de petición de trabajo */
      MPI_Recv(&solicitante, 1, MPI_INT, MPI_ANY_SOURCE, PETIC, MPI_COMM_WORLD, &estado);
      if (pila->tamanio() > 1) {
        /* Enviar nodos al proceso solicitante */
        pila->pop(nodo);
        //cout << "[" << id << "]: " << "Voy a enviar nodo" << endl;
        MPI_Send(&nodo.datos[0], 2 * NCIUDADES, MPI_INT, solicitante, NODOS, MPI_COMM_WORLD);
        //cout << "[" << id << "]: " << "Nodo enviado" << endl;
      } else {
        /* Pasar petición de trabajo al proceso (id + 1) % P */
        //cout << "[" << id << "]: " << "Voy a reenviar petición de nodo" << endl;
        MPI_Send(&solicitante, 1, MPI_INT, (id + 1) % P, PETIC, MPI_COMM_WORLD);
        //cout << "[" << id << "]: " << "Petición de nodo reenviada" << endl;
      }
      /* Sondear si hay mensajes pendientes de otros procesos */
      //cout << "[" << id << "]: " << "Sondea si hay mensajes pendientes de otros procesos" << endl;
      MPI_Iprobe((id - 1) % P, MPI_ANY_TAG, MPI_COMM_WORLD, &flag, &estado);
    }
  }
}

// void Difusion_Cota_Superior() {
//   if (difundir_cs_local && !pendiente_retorno_cs) {
//     /* Enviar valor local de cs al proceos (id + 1) % P */
//     pendiente_retorno_cs = true;
//     difundir_cs_local = false;
//   }
//   /* Sondear si hay mensajes de cota superior pendientes */
//   while (hay_mensajes) {
//     /* Recibir mensajes con valor de cota superior desde el proceso (id - 1 + P) % P */
//     /* Actualizar valor local de cota superior */
//
//     if (origen_mensaje == id && difundir_cs_local) {
//       /* Enviar valor local de cs al proceso (id + 1) % P */
//       pendiente_retorno_cs = true;
//       difundir_cs_local = false;
//     } else if (origen_mensaje == id && !difundir_cs_local)
//       pendiente_retorno_cs = false;
//     else // origen mensaje == otro proceso
//       /* Reenviar mensaje al proceso (id + 1) % P */
//     /* Sondear si hay mensajes de cota superior pendientes */
//   }
// }

/* ******************************************************************** */

int main (int argc, char **argv) {
  MPI_Init(&argc, &argv);
  MPI_Comm_size(MPI_COMM_WORLD, &P);
  MPI_Comm_rank(MPI_COMM_WORLD, &id);

  switch (argc) {
    case 3:
      NCIUDADES = atoi(argv[1]);
      break;
    default:
      if (id == 0)
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

  if (id == 0) {
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
    //cout << "[" << id << "]: " << "it " << iteraciones << endl;
    Ramifica(&nodo, &nodo_izq, &nodo_dch, tsp0);
    nueva_U = false;

    if (Solucion(&nodo_dch)) {
      //cout << "[" << id << "]: " << "Hijo dcha. es solución" << endl;
      if (nodo_dch.ci() < U) {
        //cout << "[" << id << "]: " << "Hijo dcha. es nueva CS" << endl;
        U = nodo_dch.ci(); // actualiza c.s.
        nueva_U = true;
        CopiaNodo(&nodo_dch, &solucion);
      }
    } else { // no es nodo hoja
      //cout << "[" << id << "]: " << "Hijo dcha. no es solución" << endl;
      if (nodo_dch.ci() < U) {
        //cout << "[" << id << "]: " << "Hijo dcha. a la pila" << endl;
        pila.push(nodo_dch);
      }
    }

    if (Solucion(&nodo_izq)) {
      //cout << "[" << id << "]: " << "Hijo izda. es solución" << endl;
      if (nodo_izq.ci() < U) {
        //cout << "[" << id << "]: " << "Hijo izda. es nueva CS" << endl;
        U = nodo_izq.ci(); // actualiza c.s.
        nueva_U = true;
        CopiaNodo(&nodo_izq, &solucion);
      }
    } else { // no es nodo hoja
      //cout << "[" << id << "]: " << "Hijo izda. no es solución" << endl;
      if (nodo_izq.ci() < U) {
        //cout << "[" << id << "]: " << "Hijo izda. a la pila" << endl;
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

  if (id == 0) { EscribeNodo(&solucion); if (id != 0) cout << iteraciones << endl; }

  MPI_Finalize();
  liberarMatriz(tsp0);
}
