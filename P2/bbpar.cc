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

Equilibrado_Carga(tPila *pila, bool *fin) {
  if (Vacia (pila)) { // el proceso no tiene trabajo: pide a otros procesos
    /* Enviar petición de trabajo al proceso (id + 1) % P */
    while (Vacia(pila) && !fin) {
      /* Esperar mensaje de otro proceso */
      switch (tipo_de_mensaje) {
        case PETIC: // peticion de trabajo
          /* Recibir mensaje de petición de trabajo */
          if (solicitante == id) { // peticion devuelta
            /* Reenviar petición de trabajo al proceso (id + 1) % P */
            /* Iniciar detección de posible situación de fin */
          } else // petición de otro proceso: la retransmite al siguiente
            /* Pasar petición de trabajo al proceso (id + 1) % P */
          break;
        case NODOS: // resultado de una petición de trabajo
          /* Recibir nodos del proceso donante */
          /* Almacenar nodos recibidos en la pila */
      }
    }
  }
  if (!fin) { // el proceso tiene nodos para trabajar
    /* Sondear si hay mensajes pendientes de otros procesos */
    while (hay_mensajes) { // atiende peticiones mientras haya mensajes
      /* Recibir mensaje de petición de trabajo */
      if (hay_suficientes_nodos_en_la_pila_para_ceder)
        /* Enviar nodos al proceso solicitante */
      else
        /* Pasar petición de trabajo al proceso (id + 1) % P */
      /* Sondear si hay mensajes pendientes de otros procesos */
    }
  }
}

Difusion_Cota_Superior() {
  if (difundir_cs_local && !pendiente_retorno_cs) {
    /* Enviar valor local de cs al proceos (id + 1) % P */
    pendiente_retorno_cs = true;
    difundir_cs_local = false;
  }
  /* Sondear si hay mensajes de cota superior pendientes */
  while (hay_mensajes) {
    /* Recibir mensajes con valor de cota superior desde el proceso (id - 1 + P) % P */
    /* Actualizar valor local de cota superior */

    if (origen_mensaje == id && difundir_cs_local) {
      /* Enviar valor local de cs al proceso (id + 1) % P */
      pendiente_retorno_cs = true;
      difundir_cs_local = false;
    } else if (origen_mensaje == id && !difundir_cs_local)
      pendiente_retorno_cs = false;
    else // origen mensaje == otro proceso
      /* Reenviar mensaje al proceso (id + 1) % P */
    /* Sondear si hay mensajes de cota superior pendientes */
  }
}

/* ******************************************************************** */

main (int argc, char **argv) {
    MPI_Init(argc, argv);
    MPI_Comm_size(MPI_COMM_WORLD, &P);
    MPI_Comm_rank(MPI_COMM_WORLD, &id);

    U = INFINITO; // inicializa cota superior

    if (id == 0)
      Leer_Problema_Inicial(&nodo);
    else {
      Equilibrar_Carga(&pila, &fin);
      if (!fin)
        Pop(&fila, &nodo);
    }

    while (!fin) { // ciclo de Branch&Bound
      Ramifica(&nodo, &nodo_izq, &nodo_dch);

      if (Solucion(&nodo_dch)) {
        if (ci(nodo_dch) < U)
          U = ci(nodo_dch); // actualiza c.s.
      } else { // no es nodo hoja
        if (ci(nodo_dch) < U)
          Push(&pila, &nodo_dch)
      }
    }

    if (Solucion(&nodo_izq)) {
      if (ci(nodo_izq) < U)
        U = ci(nodo_izq); // actualiza c.s.
    } else {
      if (ci(nodo_izq) < U)
        Push(&pila, &nodo_izq);
    }

    Difusion_Cota_Superior(&U);
    if (hay_nueva_cota_superior)
      Acotar (&pila, U);

    Equilibrado_Carga(&pila, &fin);
    if (!fin)
      Pop(&pila, &nodo);

}
