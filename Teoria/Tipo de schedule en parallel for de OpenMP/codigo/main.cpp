#include <iostream>
#include <stdlib.h>
#include <math.h>
#include <omp.h>

using namespace std;

/**
  * Funcion que en n iteraciones realiza cuatro operaciones:
  * - asignacion
  * - suma
  * - potencia
  * - modulo
  */
double f(int n) {
  double result = 0;
  for (int i = 0; i < n; i++) {
    result += ((int) pow(i, 2) % 37);
  }
  return result;
}

/***********************************************************************/

int main(int argc, char *argv[]) {
  int P, N, i;
  double t;

  switch(argc) {
    case 3: // se especifica el numero de procesadores
      P = atoi(argv[2]);
      break;
    case 2: // no se especifica el numero de procesadores => se usan todos los que tenga el equipo
      P = omp_get_num_procs();
      break;
    default: // numero incorrecto de parametros => se termina la ejecucion
      cerr << "Sintaxis: " << argv[0] << " <num iters> <num procs>" << endl;
      return(-1);
  }

  omp_set_num_threads(P);
  N = atoi(argv[1]);

  /*********************************************************************/
  // Static por bloques

  t = omp_get_wtime();
  #pragma omp parallel for schedule(static, N / P)
  for (i = 0; i < N; i++) {
    f(i);
  }
  t = omp_get_wtime() - t;

  cout << "Tiempo gastado (static por bloques) = " << t << endl;

  /*********************************************************************/
  // Static ciclico

  t = omp_get_wtime();
  #pragma omp parallel for schedule(static, 1)
  for (i = 0; i < N; i++) {
    f(i);
  }
  t = omp_get_wtime() - t;

  cout << "Tiempo gastado (static ciclico) = " << t << endl;

  /*********************************************************************/
  // Static dynamic

  t = omp_get_wtime();
  #pragma omp parallel for schedule(dynamic, 1)
  for (i = 0; i < N; i++) {
    f(i);
  }
  t = omp_get_wtime() - t;

  cout << "Tiempo gastado (dynamic) = " << t << endl;

  /*********************************************************************/
  // Static guided

  t = omp_get_wtime();
  #pragma omp parallel for schedule(guided, 1)
  for (i = 0; i < N; i++) {
    f(i);
  }
  t = omp_get_wtime() - t;

  cout << "Tiempo gastado (guided) = " << t << endl;

  return(0);
}
