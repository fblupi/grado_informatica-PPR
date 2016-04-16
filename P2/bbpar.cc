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
int id, P;

// void Equilibrado_Carga(tPila *pila, bool *fin) {
//   if (pila.vacia()) { // el proceso no tiene trabajo: pide a otros procesos
//     /* Enviar petición de trabajo al proceso (id + 1) % P */
//     while (pila.vacia() && !fin) {
//       /* Esperar mensaje de otro proceso */
//       switch (tipo_de_mensaje) {
//         case PETIC: // peticion de trabajo
//           /* Recibir mensaje de petición de trabajo */
//           if (solicitante == id) { // peticion devuelta
//             /* Reenviar petición de trabajo al proceso (id + 1) % P */
//             /* Iniciar detección de posible situación de fin */
//           } else // petición de otro proceso: la retransmite al siguiente
//             /* Pasar petición de trabajo al proceso (id + 1) % P */
//           break;
//         case NODOS: // resultado de una petición de trabajo
//           /* Recibir nodos del proceso donante */
//           /* Almacenar nodos recibidos en la pila */
//       }
//     }
//   }
//   if (!fin) { // el proceso tiene nodos para trabajar
//     /* Sondear si hay mensajes pendientes de otros procesos */
//     while (hay_mensajes) { // atiende peticiones mientras haya mensajes
//       /* Recibir mensaje de petición de trabajo */
//       if (hay_suficientes_nodos_en_la_pila_para_ceder)
//         /* Enviar nodos al proceso solicitante */
//       else
//         /* Pasar petición de trabajo al proceso (id + 1) % P */
//       /* Sondear si hay mensajes pendientes de otros procesos */
//     }
//   }
// }

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

  InicNodo(&nodo);                // inicializa estructura nodo

  if (id == 0) {
    LeerMatriz(argv[2], tsp0);
    MPI_Bcast(&tsp0[0][0], NCIUDADES * NCIUDADES, MPI_INT, 0, MPI_COMM_WORLD);
  } else {
    MPI_Bcast(&tsp0[0][0], NCIUDADES * NCIUDADES, MPI_INT, 0, MPI_COMM_WORLD);
    // Equilibrado_Carga(&pila, &fin);
    // if (!fin)
    //   pila.pop(&nodo);
  }

  // double t = MPI_Wtime();
  // while (!fin) { // ciclo de Branch&Bound
  //   Ramifica(&nodo, &nodo_izq, &nodo_dch, tsp0);
  //   nueva_U = false;
  //
  //   if (Solucion(&nodo_dch)) {
  //     if (nodo_dch.ci() < U)
  //       U = nodo_dch.ci(); // actualiza c.s.
  //       nueva_U = true;
  //       CopiaNodo(&nodo_dch, &solucion);
  //   } else { // no es nodo hoja
  //     if (nodo_dch.ci() < U)
  //       pila.push(nodo_dch);
  //   }
  //
  //   if (Solucion(&nodo_izq)) {
  //     if (nodo_izq.ci() < U)
  //       U = nodo_izq.ci(); // actualiza c.s.
  //       nueva_U = true;
  //       CopiaNodo(&nodo_izq, &solucion);
  //   } else {
  //     if (nodo_izq.ci() < U)
  //       pila.push(nodo_izq);
  //   }
  //
  //   // Difusion_Cota_Superior(&U);
  //   // if (nueva_U)
  //   //   pila.acotar(U);
  //
  //   // Equilibrado_Carga(&pila, &fin);
  //   if (!fin)
  //     pila.pop(nodo);
  //   iteraciones++;
  // }
  // t = MPI_Wtime() - t;
  MPI_Finalize();
  liberarMatriz(tsp0);
}
