#include <iostream>
#include <fstream>
#include <string.h>
#include <time.h>
#include "Graph.h"

//#define TIEMPOS // Comentar para obtener resultados de la CPU y comparar con estos los de la GPU

#define BLOCK_SIZE_1D 256
#define BLOCK_SIZE_2D 16

using namespace std;

//**************************************************************************************************
// Kernels to update the Matrix at k-th iteration

// Kernel 1D
__global__ void floyd_kernel1D(int * M, const int nverts, const int k) {
  int ij = blockIdx.x * blockDim.x + threadIdx.x,
      i = ij / nverts,
      j = ij - i * nverts;
  if (i < nverts && j < nverts) {
    if (i != j && i != k && j != k) {
      M[ij] = min(M[i * nverts + k]  + M[k * nverts + j], M[ij]);
    }
  }
}

// Kernel 2D
__global__ void floyd_kernel2D(int * M, const int nverts, const int k) {
  int ii = blockIdx.y * blockDim.y + threadIdx.y,
      jj = blockIdx.x * blockDim.x + threadIdx.x,
      ij = ii * nverts + jj,
      i = ij / nverts,
      j = ij - i * nverts;
  if (i < nverts && j < nverts) {
    if (i != j && i != k && j != k) {
      M[ij] = min(M[i * nverts + k] + M[k * nverts + j], M[ij]);
    }
  }
}

// Kernel Shared 1D
__global__ void floyd_kernel1DShared(int * d_M, const int nverts, const int k) {
  int gi = blockIdx.x * blockDim.x + threadIdx.x, // índice global de memoria == ij
      li = threadIdx.x,                           // índice local en el vector de memoria compartida
      i = gi / nverts,
      j = gi - i * nverts;

  __shared__ int s_M[2 * BLOCK_SIZE_1D + 2];
  s_M[li] = d_M[gi];
  s_M[li + BLOCK_SIZE_1D] = d_M[k * nverts + j];
  if (li == 0) {
    s_M[BLOCK_SIZE_1D - 2] = d_M[i * nverts + k];
  }
  if (li == BLOCK_SIZE_1D - 1) {
    s_M[BLOCK_SIZE_1D - 1] = d_M[i * nverts + k];
  }

  __syncthreads();

  if (i < nverts && j < nverts) {
    if (i != j && i != k && j != k) {
      //printf("(i=%u, j=%u, k=%u)\n\t[ij] => d_M=%u...s_M=%u\n\t[kj] => d_M=%u...s_M=%u\n\t[ik] => d_M=%u...s_M=%u\n\n", i, j, k, d_M[gi], s_M[li], d_M[k * nverts + j], s_M[li + BLOCK_SIZE_1D], d_M[i * nverts + k], s_M[BLOCK_SIZE_1D - 1]);
      d_M[gi] = min(s_M[BLOCK_SIZE_1D - 1] + s_M[li + BLOCK_SIZE_1D], s_M[li]);
    }
  }
}


//**************************************************************************************************
// Main

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
    return(-1);
  }
  cudaGetDeviceProperties(&props, devID);
  printf("Device %d: \"%s\" with Compute %d.%d capability\n",
        devID, props.name, props.major, props.minor);

//**************************************************************************************************

  Graph G;
  G.lee(argv[1]); // Read the Graph
  //cout << "El Grafo de entrada es:" << endl;
  //G.imprime();

  const int nverts = G.vertices;        // Vertices
  const int niters = nverts;            // Iteraciones
  const int nverts2 = nverts * nverts;  // Elementos

  const dim3 blocksize1D (BLOCK_SIZE_1D);                                 // Tama Bloque 1D
  const dim3 blocksize2D (BLOCK_SIZE_2D, BLOCK_SIZE_2D);                  // Tama Bloque 2D
  const dim3 nblocks1D (ceil((float) (nverts * nverts) / blocksize1D.x)); // Num Bloques 1D
  const dim3 nblocks2D (ceil((float) nverts / blocksize2D.x),             // Num Bloques 2D
                        ceil((float) nverts / blocksize2D.y));

  int * c_out_M_1D = new int[nverts2];        // Matriz en el HOST 1D
  int * c_out_M_2D = new int[nverts2];        // Matriz en el HOST 2D
  int * c_out_M_1DShared = new int[nverts2];  // Matriz en el HOST 1D Shared
  int size = nverts2 * sizeof(int);           // Tama en bytes de la matriz de salida
  int * d_In_M_1D = NULL;                     // Matriz en DEVICE para 1D
  int * d_In_M_2D = NULL;                     // Matriz en DEVICE para 2D
  int * d_In_M_1DShared = NULL;               // Matriz en DEVICE para 2D

  int i, j, k;
  double T, Tgpu1D, Tgpu2D, Tgpu1DShared, Tcpu;

  //************************************************************************************************
  // GPU phase (1D)

  // Reservar espacio en memoria para la matriz en DEVICE
  err = cudaMalloc((void **) &d_In_M_1D, size);
  if (err != cudaSuccess) {
    cout << "ERROR: Bad Allocation in Device Memory" << endl;
  }

  T = clock();

  // Copiar los datos de la matriz en HOST en la matriz en DEVICE
  err = cudaMemcpy(d_In_M_1D, G.Get_Matrix(), size, cudaMemcpyHostToDevice);
  if (err != cudaSuccess) {
    cout << "ERROR: COPY MATRIX TO DEVICE" << endl;
  }

  for (k = 0; k < niters; k++) {
    // Kernel Launch
    floyd_kernel1D <<< nblocks1D, blocksize1D >>> (d_In_M_1D, nverts, k);
    err = cudaGetLastError();
    if (err != cudaSuccess) {
      fprintf(stderr, "Failed to launch kernel!\n");
      exit(EXIT_FAILURE);
    }
  }

  // Copiar los datos de la matriz en DEVICE en la matriz en HOST
  cudaMemcpy(c_out_M_1D, d_In_M_1D, size, cudaMemcpyDeviceToHost);

  Tgpu1D = clock();
  Tgpu1D = (Tgpu1D - T) / CLOCKS_PER_SEC;
  cout << "Tiempo gastado GPU (1D) = " << Tgpu1D << endl;

  //************************************************************************************************
  // GPU phase (2D)

  // Reservar espacio en memoria para la matriz en DEVICE
  err = cudaMalloc((void **) &d_In_M_2D, size);
  if (err != cudaSuccess) {
    cout << "ERROR: Bad Allocation in Device Memory" << endl;
  }

  T = clock();

  // Copiar los datos de la matriz en HOST en la matriz en DEVICE
  err = cudaMemcpy(d_In_M_2D, G.Get_Matrix(), size, cudaMemcpyHostToDevice);
  if (err != cudaSuccess) {
    cout << "ERROR: COPY MATRIX TO DEVICE" << endl;
  }

  for (k = 0; k < niters; k++) {
    // Kernel Launch
    floyd_kernel2D <<< nblocks2D, blocksize2D >>> (d_In_M_2D, nverts, k);
    err = cudaGetLastError();
    if (err != cudaSuccess) {
      fprintf(stderr, "Failed to launch kernel!\n");
      exit(EXIT_FAILURE);
    }
  }

  // Copiar los datos de la matriz en DEVICE en la matriz en HOST
  cudaMemcpy(c_out_M_2D, d_In_M_2D, size, cudaMemcpyDeviceToHost);

  Tgpu2D = clock();
  Tgpu2D = (Tgpu2D - T) / CLOCKS_PER_SEC;
  cout << "Tiempo gastado GPU (2D) = " << Tgpu2D << endl;

  //************************************************************************************************
  // GPU phase (1D Shared Memory)

  // Reservar espacio en memoria para la matriz en DEVICE
  err = cudaMalloc((void **) &d_In_M_1DShared, size);
  if (err != cudaSuccess) {
    cout << "ERROR: Bad Allocation in Device Memory" << endl;
  }

  T = clock();

  // Copiar los datos de la matriz en HOST en la matriz en DEVICE
  err = cudaMemcpy(d_In_M_1DShared, G.Get_Matrix(), size, cudaMemcpyHostToDevice);
  if (err != cudaSuccess) {
    cout << "ERROR: COPY MATRIX TO DEVICE" << endl;
  }

  for (k = 0; k < niters; k++) {
    // Kernel Launch
    floyd_kernel1DShared <<< nblocks1D, blocksize1D >>> (d_In_M_1DShared, nverts, k);
    err = cudaGetLastError();
    if (err != cudaSuccess) {
      fprintf(stderr, "Failed to launch kernel!\n");
      exit(EXIT_FAILURE);
    }
  }

  // Copiar los datos de la matriz en DEVICE en la matriz en HOST
  cudaMemcpy(c_out_M_1DShared, d_In_M_1DShared, size, cudaMemcpyDeviceToHost);

  Tgpu1DShared = clock();
  Tgpu1DShared = (Tgpu1DShared - T) / CLOCKS_PER_SEC;
  cout << "Tiempo gastado GPU (1D Shared) = " << Tgpu1DShared << endl;

#ifndef TIEMPOS
  //************************************************************************************************
  // CPU phase

  T = clock();
  // Bucle ppal del algoritmo
  for (k = 0; k < niters; k++) {
    for (i = 0; i < nverts; i++) {
      for (j = 0; j < nverts; j++) {
        if (i != j && i != k && j != k) {
          int vikj = min(G.arista(i, k) + G.arista(k, j), G.arista(i, j));
          G.inserta_arista(i, j, vikj);
        }
      }
    }
  }

  Tcpu = clock();
  Tcpu = (Tcpu - T) / CLOCKS_PER_SEC;
  //cout << endl << "El Grafo con las distancias de los caminos más cortos es:"
  //     << endl << endl;
  //G.imprime();
  cout << "Tiempo gastado CPU = " << Tcpu << endl;
  cout << "Ganancia (1D) = " << Tcpu / Tgpu1D << endl;
  cout << "Ganancia (2D) = " << Tcpu / Tgpu2D << endl;
  cout << "Ganancia (1D Shared) = " << Tcpu / Tgpu1DShared << endl;

  //************************************************************************************************
  // Comprobar que los resultados en CPU y GPU son los mismos

  for (i = 0; i < nverts; i++) {
    for (j = 0; j < nverts; j++) {
      if (abs(c_out_M_1D[i * nverts + j] - G.arista(i, j)) > 0) {
        cout << "Error 1D (" << i << "," << j << ")   " << c_out_M_1D[i * nverts + j]
             << "..." << G.arista(i, j) << endl;
      }
      if (abs(c_out_M_2D[i * nverts + j] - G.arista(i, j)) > 0) {
        cout << "Error 2D (" << i << "," << j << ")   " << c_out_M_2D[i * nverts + j]
             << "..." << G.arista(i, j) << endl;
      }
      if (abs(c_out_M_1DShared[i * nverts + j] - G.arista(i, j)) > 0) {
        cout << "Error 1D Shared (" << i << "," << j << ")   " << c_out_M_1DShared[i * nverts + j]
             << "..." << G.arista(i, j) << endl;
      }
    }
  }
  //************************************************************************************************
#endif

  // Liberar memoria
  cudaFree(d_In_M_1D);
  cudaFree(d_In_M_2D);
  cudaFree(d_In_M_1DShared);
  delete[] c_out_M_1D;
  delete[] c_out_M_2D;
  delete[] c_out_M_1DShared;
}
