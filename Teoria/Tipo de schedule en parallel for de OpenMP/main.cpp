#include <iostream>
#include <stdlib.h>
#include <math.h>
#include <omp.h>

using namespace std;

/**
  * Función que en n iteraciones realiza cuatro operaciones:
  * - asignación
  * - suma
  * - potencia
  * - módulo
  */
double f(int n) {
  double result = 0;
  for (int i = 0; i < n; i++) {
    result += ((int) pow(i, 2) % 37);
  }
  return result;
}

int main(int argc, char *argv[]) {
  int P, N, i;
  double t;

  switch(argc) {
    case 3: // se especifica el número de procesadores
      P = atoi(argv[2]);
      break;
    case 2: // no se especifica el número de procesadores => se usan todos los que tenga el equipo
      P = omp_get_num_procs();
      break;
    default: // número incorrecto de parámetros => se termina la ejecución
      cerr << "Sintaxis: " << argv[0] << "<num iters> <num procs>" << endl;
      return(-1);
  }
  omp_set_num_threads(P);

  N = atoi(argv[1]);

  t = omp_get_wtime();
  for (i = 0; i < N; i++) {
    f(i);
  }
  t = omp_get_wtime() - t;

  cout << "Tiempo gastado = " << t << endl << endl;

  return(0);
}
