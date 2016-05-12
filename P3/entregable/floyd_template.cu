#include <iostream>
#include <fstream>
#include <string.h>
#include <time.h>
#include "Graph.h"

#define blocksize 256

using namespace std;

//**************************************************************************
// Kernel to update the Matrix at k-th iteration
__global__ void floyd_kernel1D(int * M, const int nverts, const int k) {
  int i = blockIdx.y * blockDim.y + threadIdx.y,
      j = blockIdx.x * blockDim.x + threadIdx.x;
  if (i < nverts && j < nverts && i != j && i != k && j != k) {
    int ij = i * nverts + j, ik = i * nverts + k, kj = k * nverts + j;
    M[ij] = min(M[ik] + M[kj], M[ij]);
  }
}

int main(int argc, char *argv[]) {
  if (argc != 2) {
    cerr << "Sintaxis: " << argv[0] << " <archivo de grafo>" << endl;
    return(-1);
	}

  // Get GPU information
  int devID;
  cudaDeviceProp props;
  cudaError_t err;
  err = cudaGetDevice(&devID);
  if (err != cudaSuccess) {
    cout << "ERRORRR" << endl;
  }
  cudaGetDeviceProperties(&props, devID);
  printf("Device %d: \"%s\" with Compute %d.%d capability\n",
        devID, props.name, props.major, props.minor);

  //****************************************************************************

  Graph G;
  G.lee(argv[1]);		// Read the Graph
  //cout << "El Grafo de entrada es:" << endl;
  //G.imprime();

  const int nverts = G.vertices;        // Vertices
  const int niters = nverts;            // Iteraciones
  const int nverts2 = nverts * nverts;  // Elementos de la matriz
  const int nblocks = nverts / blocksize + (nverts % blocksize == 0 ? 0 : 1);
  int * c_Out_M = new int[nverts2];     // Matriz en el HOST
  int size = nverts2 * sizeof(int);     // Tama en bytes de la matriz de salida
  int * d_In_M = NULL;                  // Matriz en DEVICE

  // GPU phase
  // Reservar espacio en memoria para la matriz en DEVICE
  err = cudaMalloc((void **) &d_In_M, size);
  if (err != cudaSuccess) {
    cout << "ERROR: Bad Allocation in Device Memory" << endl;
  }

  double  T = clock();

  // Copiar los datos de la matriz en HOST en la matriz en DEVICE
  err = cudaMemcpy(d_In_M, G.Get_Matrix(), size, cudaMemcpyHostToDevice);
  if (err != cudaSuccess) {
    cout << "ERROR: COPY MATRIX TO DEVICE" << endl;
  }

  for (int k = 0; k < niters; k++) {
    // Kernel Launch
    floyd_kernel1D <<< nblocks, blocksize >>> (d_In_M, nverts, k);
    err = cudaGetLastError();
    if (err != cudaSuccess) {
      fprintf(stderr, "Failed to launch kernel!\n");
      exit(EXIT_FAILURE);
    }
  }

  // Copiar los datos de la matriz en DEVICE en la matriz en HOST
  cudaMemcpy(c_Out_M, d_In_M, size, cudaMemcpyDeviceToHost);

  double Tgpu = clock();
  Tgpu = (Tgpu - T) / CLOCKS_PER_SEC;
  cout << "Tiempo gastado GPU = " << Tgpu << endl << endl;

  //****************************************************************************

  // CPU phase
  T = clock();
  // Bucle ppal del algoritmo
  for (int k = 0; k < niters; k++)
    for (int i = 0; i < nverts; i++)
      for (int j = 0; j < nverts; j++)
        if (i != j && i != k && j != k) {
          int vikj = min(G.arista(i, k) + G.arista(k, j), G.arista(i, j));
          G.inserta_arista(i, j, vikj);
        }

  double Tcpu = clock();
  Tcpu = (Tcpu - T) / CLOCKS_PER_SEC;
  //cout << endl << "El Grafo con las distancias de los caminos más cortos es:"
  //     << endl << endl;
  //G.imprime();
  cout << "Tiempo gastado CPU = " << Tcpu << endl << endl;
  cout << "Ganancia = " << Tcpu / Tgpu << endl;

  //****************************************************************************

  // Comprobar que los resultados en CPU y GPU son los mismos
  for (int i = 0; i < nverts; i++)
    for (int j = 0; j < nverts; j++)
       if (abs(c_Out_M[i * nverts + j] - G.arista(i, j)) > 0)
         cout << "Error (" << i << "," << j << ")   " << c_Out_M[i * nverts + j]
              << "..." << G.arista(i, j) << endl;
}
