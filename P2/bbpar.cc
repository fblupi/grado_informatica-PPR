/* ******************************************************************** */
/*               Algoritmo Branch-And-Bound Paralelo                    */
/* ******************************************************************** */
#include <cstdlib>
#include <cstdio>
#include <iostream>
#include <mpi.h>
#include "libbb.h"

using namespace std;

#define PETICION 0
#define TRABAJO 1
#define TESTIGO_FIN 2
#define FIN 3

unsigned int NCIUDADES;
int id, P;
MPI_Comm COMM_EQUILIBRADO_CARGA, COMM_DIFUSION_COTA;

void Equilibrado_Carga(tPila *pila, bool *fin) {
  //cout << "[" << id << "]: " << "Equilibrando carga" << endl;
  int solicitante, flag, tamanio;
  MPI_Status estado;
  tNodo nodo;
  tPila pilaNueva;
  if (pila->vacia()) { // el proceso no tiene trabajo: pide a otros procesos
    //cout << "[" << id << "]: " << "Pila vacía" << endl;
    /* Enviar petición de trabajo al proceso (id + 1) % P */
    //cout << "[" << id << "]: " << "Voy a enviar petición de nodo" << endl;
    MPI_Send(&id, 1, MPI_INT, (id + 1) % P, PETICION, COMM_EQUILIBRADO_CARGA);
    //cout << "[" << id << "]: " << "Petición de nodo enviada" << endl;
    while (pila->vacia() && !*fin) {
      /* Esperar mensaje de otro proceso */
      //cout << "[" << id << "]: " << "Esperando mensaje" << endl;
      MPI_Probe(MPI_ANY_SOURCE, MPI_ANY_TAG, COMM_EQUILIBRADO_CARGA, &estado);
      //cout << "[" << id << "]: " << "Mensaje recibido" << endl;
      switch (estado.MPI_TAG) {
        case PETICION: // peticion de trabajo
          //cout << "[" << id << "]: " << "Recibir mensaje de tipo petición" << endl;
          /* Recibir mensaje de petición de trabajo */
          MPI_Recv(&solicitante, 1, MPI_INT, MPI_ANY_SOURCE, PETICION, COMM_EQUILIBRADO_CARGA, &estado);
          if (solicitante == id) { // peticion devuelta
            /* Reenviar petición de trabajo al proceso (id + 1) % P */
            //cout << "[" << id << "]: " << "Voy a reenviar petición de nodo" << endl;
            MPI_Send(&solicitante, 1, MPI_INT, (id + 1) % P, PETICION, COMM_EQUILIBRADO_CARGA);
            //cout << "[" << id << "]: " << "Petición de nodo reenviada" << endl;
            /* Iniciar detección de posible situación de fin */
            //cout << "[" << id << "]: " << "Posible fin" << endl;
          } else { // petición de otro proceso: la retransmite al siguiente
            /* Pasar petición de trabajo al proceso (id + 1) % P */
            //cout << "[" << id << "]: " << "Voy a pasar petición" << endl;
            MPI_Send(&solicitante, 1, MPI_INT, (id + 1) % P, PETICION, COMM_EQUILIBRADO_CARGA);
            //cout << "[" << id << "]: " << "Petición pasada de nodo " << solicitante << endl;
          }
          break;
        case TRABAJO: // resultado de una petición de trabajo
          MPI_Get_count(&estado, MPI_INT, &tamanio);
          //cout << "[" << id << "]: " << "Recibir mensaje de tipo nodos" << endl;
          /* Recibir nodos del proceso donante */
          MPI_Recv(&pila->nodos[pila->tope], tamanio, MPI_INT, MPI_ANY_SOURCE, TRABAJO, COMM_EQUILIBRADO_CARGA, &estado);
          /* Almacenar nodos recibidos en la pila */
          pila->tope = tamanio;
          //cout << "[" << id << "]: " << "Nodo almacenado" << endl;
      }
    }
  }
  if (!*fin) { // el proceso tiene nodos para trabajar
    /* Sondear si hay mensajes pendientes de otros procesos */
    //cout << "[" << id << "]: " << "Sondea si hay mensajes pendientes de otros procesos" << endl;
    MPI_Iprobe(MPI_ANY_SOURCE, MPI_ANY_TAG, COMM_EQUILIBRADO_CARGA, &flag, &estado);
    while (flag) { // atiende peticiones mientras haya mensajes
      //cout << "[" << id << "]: " << "Recibe mensaje de petición de trabajo. Tamaño pila: " << pila->tamanio() << endl;
      /* Recibir mensaje de petición de trabajo */
      MPI_Recv(&solicitante, 1, MPI_INT, MPI_ANY_SOURCE, PETICION, COMM_EQUILIBRADO_CARGA, &estado);
      if (pila->tamanio() > 1) {
        /* Enviar nodos al proceso solicitante */
        pila->divide(pilaNueva);
        //cout << "[" << id << "]: " << "Voy a enviar nodo" << endl;
        MPI_Send(&pilaNueva.nodos[0], pilaNueva.tope, MPI_INT, solicitante, TRABAJO, COMM_EQUILIBRADO_CARGA);
        //cout << "[" << id << "]: " << "Nodo enviado" << endl;
      } else {
        /* Pasar petición de trabajo al proceso (id + 1) % P */
        //cout << "[" << id << "]: " << "Voy a reenviar petición de nodo" << endl;
        MPI_Send(&solicitante, 1, MPI_INT, (id + 1) % P, PETICION, COMM_EQUILIBRADO_CARGA);
        //cout << "[" << id << "]: " << "Petición de nodo reenviada" << endl;
      }
      /* Sondear si hay mensajes pendientes de otros procesos */
      //cout << "[" << id << "]: " << "Sondea si hay mensajes pendientes de otros procesos" << endl;
      MPI_Iprobe(MPI_ANY_SOURCE, MPI_ANY_TAG, COMM_EQUILIBRADO_CARGA, &flag, &estado);
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

  MPI_Comm_split(MPI_COMM_WORLD, 0 ,id, &COMM_EQUILIBRADO_CARGA);
  MPI_Comm_split(MPI_COMM_WORLD, 0 ,id, &COMM_DIFUSION_COTA);

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
    cout << "[" << id << "]: " << "it " << iteraciones << endl;
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

  cout << "[" << id << "]: Solución = " << endl;
  EscribeNodo(&solucion);
  cout << "[" << id << "]: " << iteraciones << endl;

  MPI_Finalize();
  liberarMatriz(tsp0);
}
