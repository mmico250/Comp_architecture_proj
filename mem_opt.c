#include <stdio.h>
#include <stdlib.h>
#include <time.h>
#include <immintrin.h>
#include <string.h>

#define MATRIX_SIZE 1000
#define ARRAY_SIZE 1000000

float **allocate_matrix(int size) {
  float **matrix;
  matrix = (float **)malloc(size * sizeof(float *));
  for (int i = 0; i < size; i++) {
    matrix[i] = (float *)_mm_malloc(size * sizeof(float), 64);
  }
  return matrix;
}


void free_matrix(float **matrix, int size) {
  for (int i = 0; i < size; i++) {
    _mm_free(matrix[i]);
  }
  free(matrix);
}

void multiply_matrices(float **matrix1, float **matrix2, float **result, int size) {
  for (int i = 0; i < size; i++) {
    for (int j = 0; j < size; j++) {
      result[i][j] = 0.0;
      for (int k = 0; k < size; k++) {
        result[i][j] += matrix1[i][k] * matrix2[k][j];
      }
    }
  }
}

void multiply_matrices_unrolled(float **matrix1, float **matrix2, float **result, int size) {
  for (int i = 0; i < size; i++) {
    for (int j = 0; j < size; j++) {
      float sum1 = 0.0, sum2 = 0.0, sum3 = 0.0, sum4 = 0.0;
      for (int k = 0; k < size; k += 4) {
        sum1 += matrix1[i][k] * matrix2[k][j];
        sum2 += matrix1[i][k + 1] * matrix2[k + 1][j];
        sum3 += matrix1[i][k + 2] * matrix2[k + 2][j];
        sum4 += matrix1[i][k + 3] * matrix2[k + 3][j];
      }
      result[i][j] = sum1 + sum2 + sum3 + sum4;
    }
  }
  //printf("unrolled matrix done \n");
}

/*
void multiply_matrices_aligned(float **matrix1, float **matrix2, float **result, int size) {
  printf("Allocate aligned memory for matrix2 and result matrices \n");
  // Allocate aligned memory for matrix2 and result matrices
  float **matrix2_aligned = (float **)_malloc(size * sizeof(float *), 64);
  float **result_aligned = (float **)malloc(size * sizeof(float *), 64);
  for (int i = 0; i < size; i++) {
    memset(matrix2_aligned[i], 0, size * sizeof(float));
  memset(result_aligned[i], 0, size * sizeof(float));
  }
  printf("Transpose matrix2 to optimize memory access pattern \n");
  // Transpose matrix2 to optimize memory access pattern
  for (int i = 0; i < size; i++) {
    for (int j = 0; j < size; j++) {
      matrix2_aligned[j][i] = matrix2[i][j];
    }
  }
  printf("Perform matrix multiplication using aligned memory access \n");
  // Perform matrix multiplication using aligned memory access
  for (int i = 0; i < size; i++) {
    for (int j = 0; j < size; j++) {
      __m256 sum = _mm256_setzero_ps();
      for (int k = 0; k < size; k += 8) {
        __m256 a = _mm256_load_ps(matrix1[i] + k);
        __m256 b = _mm256_load_ps(matrix2_aligned[j] + k);
        sum = _mm256_add_ps(sum, _mm256_mul_ps(a, b));
      }
      _mm256_store_ps(result_aligned[j] + i * 8, sum);
    }
  }
   printf("Copy result from aligned memory to the output matrix \n");
  // Copy result from aligned memory to the output matrix
  for (int i = 0; i < size; i++) {
    for (int j = 0; j < size; j++) {
      result[i][j] = result_aligned[i][j];
    }
  }
  printf("Free the allocated aligned memory \n");
  // Free the allocated aligned memory
  for (int i = 0; i < size; i++) {
    _mm_free(matrix2_aligned[i]);
    _mm_free(result_aligned[i]);
  }
  _mm_free(matrix2_aligned);
  _mm_free(result_aligned);
}
*/
void print_matrix(float **matrix, int size)
{
for (int i = 0; i < size; i++)
{
    for (int j = 0; j < size; j++)
        {
          printf("%f ", matrix[i][j]);
         }
      printf("\n");
     }
}

int main()
{
      srand(time(NULL));

      float **matrix1 = allocate_matrix(MATRIX_SIZE);
      float **matrix2 = allocate_matrix(MATRIX_SIZE);
      float **result = allocate_matrix(MATRIX_SIZE);

      for (int i = 0; i < MATRIX_SIZE; i++)
      {
            for (int j = 0; j < MATRIX_SIZE; j++) {
            matrix1[i][j] = (float)rand() / (float)RAND_MAX;
            matrix2[i][j] = (float)rand() / (float)RAND_MAX;
            }
      }

      clock_t start, end;
      double cpu_time_used;
      printf("Using Matrix \n");
      start = clock();
      multiply_matrices(matrix1, matrix2, result, MATRIX_SIZE);
      end = clock();
      cpu_time_used = ((double) (end - start)) / CLOCKS_PER_SEC;
      printf("Regular matrix multiplication took %f seconds\n", cpu_time_used);

      start = clock();
      multiply_matrices_unrolled(matrix1, matrix2, result, MATRIX_SIZE);
      end = clock();
      cpu_time_used = ((double) (end - start)) / CLOCKS_PER_SEC;
      printf("Unrolled matrix multiplication took %f seconds\n", cpu_time_used);

      /*
      start = clock();
      multiply_matrices_aligned(matrix1, matrix2, result, MATRIX_SIZE);
      end = clock();
      cpu_time_used = ((double) (end - start)) / CLOCKS_PER_SEC;
      printf("Aligned matrix multiplication took %f seconds\n", cpu_time_used);
      */
      free_matrix(matrix1, MATRIX_SIZE);
      free_matrix(matrix2, MATRIX_SIZE);
      free_matrix(result, MATRIX_SIZE);
      
      int array[ARRAY_SIZE];
  int i;

  // Measure execution time without prefetching or caching
  //clock_t start = clock();

  for (i = 0; i < ARRAY_SIZE; i++) {
    array[i] = i;
  }

  int sum = 0;

  for (i = 0; i < ARRAY_SIZE; i++) {
    sum += array[i];
  }
  
  printf("Using Arrays \n");
  printf("Execution time without prefetching or caching: %f seconds\n",
         (double)(clock() - start) / CLOCKS_PER_SEC);

  // Measure execution time with prefetching
  start = clock();

  for (i = 0; i < ARRAY_SIZE; i++) {
    array[i] = i;
    __builtin_prefetch(&array[i + 1], 0, 3);
  }

  sum = 0;

  for (i = 0; i < ARRAY_SIZE; i++) {
    sum += array[i];
  }

  printf("Execution time with prefetching: %f seconds\n",
         (double)(clock() - start) / CLOCKS_PER_SEC);

  // Measure execution time with caching
  start = clock();

  for (i = 0; i < ARRAY_SIZE; i++) {
    array[i] = i;
  }

  int cache[ARRAY_SIZE];

  for (i = 0; i < ARRAY_SIZE; i++) {
    cache[i] = array[i];
  }

  sum = 0;

  for (i = 0; i < ARRAY_SIZE; i++) {
    sum += cache[i];
  }

  printf("Execution time with caching: %f seconds\n",
         (double)(clock() - start) / CLOCKS_PER_SEC);


return 0;
}

