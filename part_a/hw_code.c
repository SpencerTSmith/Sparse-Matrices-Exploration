/*
  HW5 A: Shared Memory and Sparse Structures

  Instructions: find all instances of "STUDENT_TODO" and replace with
                distributed memory code that makes the test corresponding
                to that function pass.


        To compile and run your code locally use:
          make compile-local
          make run-local

 TODO...
        To access Schooner see: https://www.ou.edu/oscer/support/machine_access
        To compile and run on Schooner use:
          make compile-schooner
          make run-schooner





  Submission: For this assignment you will upload three artifacts to canvas.
              DO NOT INCLUDE YOUR NAME or other identifying information in
              these artifacts.
              1. [figures.pdf] containing pictures describing the movements
                               being performed by the SIMD instructions.
          2. [results.txt] containing the test output of your code.
          3. [code.c] Your modified version of this code.


  Peer Review: Unless you opt out by contacting me, your assignment will be
               reviewed by your peers. They will provide useful feedback, but
               will not determine your grade. I will ultimately determine the
               grade for the assignment.

  - richard.m.veras@ou.edu
*/

/*
  NOTE: To run this code you will have at least three options.
  1. Locally: install mpi on your system
     https://rantahar.github.io/introduction-to-mpi/setup.html
  2. Remote on Schooner



  Problems:
  0. COO --> CSR
  1. COO --> BCSR
  2. CSR --> CSC

  3. Dense matvec 4 ways
  4. CSR matvec 2 ways
  5. BCSR matvec 2 ways


*/

#include <math.h>
#include <stdarg.h>
#include <stdbool.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>

#include <omp.h>

#include "helper.h"
#include "sparse.h"

#define ERROR_THRESHOLD (1e-4)

/*
  SPARSE FORMAT CONVERSION
*/

////////////////
////// PROB01 //
////////////////
void student_coo_matrix_to_csr_matrix(coo_matrix_t *coo_src, csr_matrix_t *csr_dst) {
  // Copy over the metadata from the source to the dest
  csr_dst->m = coo_src->m;
  csr_dst->n = coo_src->n;
  csr_dst->nnz = coo_src->nnz;

  // Initialize the buffers
  // For the sake of the HW we will fill the unitialized values with zeros
  csr_dst->row_idx = (int *)calloc((csr_dst->m + 1), sizeof(int));
  csr_dst->col_idx = (int *)calloc(csr_dst->nnz, sizeof(int));
  csr_dst->values = (float *)calloc(csr_dst->nnz, sizeof(float));

  {
    int row_start = 0;
    for (int r = 0; r <= csr_dst->m; r++) {
      csr_dst->row_idx[r] = row_start;

      // Now loop through this row, checking if we still have elements in it
      while (coo_src->row_idx[row_start] == r)
        row_start++;
    }

    // These are the same
    for (int i = 0; i < csr_dst->nnz; i++) {
      csr_dst->values[i] = coo_src->values[i];
      csr_dst->col_idx[i] = coo_src->col_idx[i];
    }
  }
}

void test_coo_to_csr(coo_matrix_t *coo_src) {

  /*
    Initialize stuff.
  */

  dense_matrix_t *dense_ref = create_dense_matrix_from_coo_matrix(coo_src);
  csr_matrix_t *csr_ref = create_csr_matrix_from_dense_matrix(dense_ref);

  /*
    Student Implementation
  */
  csr_matrix_t *csr_test = (csr_matrix_t *)malloc(sizeof(csr_matrix_t));
  student_coo_matrix_to_csr_matrix(coo_src, csr_test);

  /*
    Compare
  */

  dense_matrix_t *dense_test = create_dense_matrix_from_csr_matrix(csr_test);

  float res = max_pair_wise_diff(coo_src->m * coo_src->n, dense_ref->values, dense_test->values);

  printf("COO to CSR: ");

  if (ERROR_THRESHOLD < res) {
    printf("FAIL with error = %f\n", res);

    printf("======================\n");
    printf("Original Data Stored in Coordinate (COO) Format\n");
    pretty_print_coo_matrix(coo_src);

    printf("======================\n");
    printf("Reference Data Stored in Compressed Sparse Row (CSR) Format\n");
    pretty_print_csr_matrix(csr_ref);

    printf("======================\n");
    printf("Student Data Stored in Compressed Sparse Row (CSR) Format\n");
    pretty_print_csr_matrix(csr_test);

  } else {
    printf("PASS\n");
  }

  destroy_dense_matrix(dense_test);
  destroy_dense_matrix(dense_ref);
  destroy_csr_matrix(csr_ref);
  destroy_csr_matrix(csr_test);
}

////////////////
////// PROB02 //
////////////////
void student_coo_matrix_to_bcsr_matrix(int mb, int nb, int bcs, int brs, coo_matrix_t *coo_src,
                                       bcsr_matrix_t *bcsr_dst) {
  // Copy over the metadata from the source to the dest
  bcsr_dst->m = coo_src->m;
  bcsr_dst->n = coo_src->n;

  bcsr_dst->mb = mb;
  bcsr_dst->nb = nb;
  bcsr_dst->bcs = bcs;
  bcsr_dst->brs = brs;

  bcsr_dst->nnz_blocks = 0;

  {

    // STUDENT_TODO: Do not convert to a dense matrix then back to bcsr.

    // Most important hint you will receive: draw this out on paper first.
    //
    // Minor Hint: Do this in three passes of the coo data. It's not very
    //             efficient, but it's a starting point that will work.
    //             PASS 1: figure out the number of non-zero blocks of
    //                     size MB x NB.
    //
    //             PASS 2: determine the values for block_row_idx
    //
    //             PASS 3: place the appropriate values in block_col_idx
    //                     and block_values. You will need to keep a counter
    //                     for each row block.

    int worst_case_blocks = coo_src->nnz; // Worst case they are each in their own block
    int *block_row_indices = (int *)calloc(worst_case_blocks, sizeof(int));
    int *block_col_indices = (int *)calloc(worst_case_blocks, sizeof(int));

    // PASS 1: ...
    // bcsr_dst->nnz_blocks = ??;

    int n_block_rows = 0;
    int n_block_cols = 0;
    for (int i = 0; i < coo_src->nnz; i++) {
      if (coo_src->values[i] == 0.f)
        continue;

      // Which block would this element go into?
      int block_row = coo_src->row_idx[i] / mb;
      int block_col = coo_src->col_idx[i] / nb;

      // Might be a better way to do this, but i need this info for how i calculate the value
      // indices
      n_block_rows = block_row ? block_row > n_block_rows : n_block_rows;
      n_block_cols = block_col ? block_col > n_block_cols : n_block_cols;

      // Just naively check if we've already collected this block
      bool found = false;
      for (int j = 0; j < bcsr_dst->nnz_blocks; j++) {
        // New block, just checking all of them collected thus far
        if (block_row_indices[j] == block_row && block_col_indices[j] == block_col) {
          found = true;
          break;
        }
      }

      // If not add it to the list
      if (!found) {
        block_row_indices[bcsr_dst->nnz_blocks] = block_row;
        block_col_indices[bcsr_dst->nnz_blocks] = block_col;

        bcsr_dst->nnz_blocks++;
      }
    }

    // Since we want the actual number and not just the indices
    n_block_rows++;
    n_block_cols++;

    // Initialize the buffers
    // For the sake of the HW we will fill the unitialized values with zeros
    bcsr_dst->block_row_idx = (int *)calloc(((bcsr_dst->m / mb) + 1), sizeof(int));
    bcsr_dst->block_col_idx = (int *)calloc(bcsr_dst->nnz_blocks, sizeof(int));
    bcsr_dst->block_values = (float *)calloc(bcsr_dst->nnz_blocks * (mb * nb), sizeof(float));

    // PASS 2: find out how many non-zero blocks are in each block row and modify
    //         bcsr_dst->block_row_idx
    bcsr_dst->block_row_idx[0] = 0; // Always

    // Super inefficient I think
    int *block_row_counts = (int *)calloc(n_block_rows, sizeof(int));

    for (int i = 0; i < bcsr_dst->nnz_blocks; i++) {
      int row = block_row_indices[i];
      block_row_counts[row]++;
    }

    for (int i = 0; i <= n_block_rows; i++) {
      bcsr_dst->block_row_idx[i] = bcsr_dst->block_row_idx[i - 1] + block_row_counts[i - 1];
    }
    free(block_row_indices);
    free(block_col_indices);
    free(block_row_counts);

    // PASS 3: Place the coo values into their right location in the bcsr matrix
    for (int i = 0; i < coo_src->nnz; i++) {
      // Which block would this element go into?
      int block_row = coo_src->row_idx[i] / mb;
      int block_col = coo_src->col_idx[i] / nb;

      int internal_row = coo_src->row_idx[i] % mb;
      int internal_col = coo_src->col_idx[i] % nb;

      int block_offset = (block_row * n_block_cols + block_col);
      bcsr_dst->block_col_idx[block_offset] = block_col;
      int index_into_blocked = block_offset * (mb * nb) + (internal_row * nb + internal_col);
      bcsr_dst->block_values[index_into_blocked] = coo_src->values[i];
    }
  }
}

void test_coo_to_bcsr(coo_matrix_t *coo_src) {

  /*
    Initialize stuff.
  */

  int mb = 3;  // blocksize number of rows
  int nb = 2;  // blocksize number of cols
  int bcs = 2; // column stride in the block
  int brs = 1; // row stride in the block

  dense_matrix_t *dense_ref = create_dense_matrix_from_coo_matrix(coo_src);
  bcsr_matrix_t *bcsr_ref = create_bcsr_matrix_from_dense_matrix(mb, nb, bcs, brs, dense_ref);

  /*
    Student Implementation
  */
  bcsr_matrix_t *bcsr_test = (bcsr_matrix_t *)malloc(sizeof(bcsr_matrix_t));
  student_coo_matrix_to_bcsr_matrix(mb, nb, bcs, brs, coo_src, bcsr_test);

  /*
    Compare
  */

  dense_matrix_t *dense_test = create_dense_matrix_from_bcsr_matrix(bcsr_test);

  float res = max_pair_wise_diff(coo_src->m * coo_src->n, dense_ref->values, dense_test->values);

  printf("COO to BCSR: ");

  if (ERROR_THRESHOLD < res)

  {
    printf("FAIL with error = %f\n", res);

    printf("======================\n");
    printf("Original Data Stored in Coordinate (COO) Format\n");
    pretty_print_coo_matrix(coo_src);

    printf("======================\n");
    printf("Reference Data Stored in Blocked Compressed Sparse Row (BCSR) Format\n");
    pretty_print_bcsr_matrix(bcsr_ref);

    printf("======================\n");
    printf("Student Data Stored in Blocked Compressed Sparse Row (BCSR) Format\n");
    pretty_print_bcsr_matrix(bcsr_test);

  } else {
    printf("PASS\n");
  }

  destroy_dense_matrix(dense_test);
  destroy_dense_matrix(dense_ref);
  destroy_bcsr_matrix(bcsr_ref);
  destroy_bcsr_matrix(bcsr_test);
}

////////////////
////// PROB03 //
////////////////
void student_csr_matrix_to_csc_matrix(csr_matrix_t *csr_src, csc_matrix_t *csc_dst) {
  // Copy over the metadata from the source to the dest
  csc_dst->m = csr_src->m;
  csc_dst->n = csr_src->n;
  csc_dst->nnz = csr_src->nnz;

  // Initialize the buffers
  // For the sake of the HW we will fill the unitialized values with zeros
  csc_dst->row_idx = (int *)calloc((csc_dst->nnz), sizeof(int));
  csc_dst->col_idx = (int *)calloc(csc_dst->n + 1, sizeof(int));
  csc_dst->values = (float *)calloc(csc_dst->nnz, sizeof(float));

  {

    // STUDENT_TODO: Do not convert to a dense matrix then back to csr.

    // Most important hint you will receive: draw this out on paper first.

    // Count how many times each column appears
    for (int i = 0; i < csr_src->nnz; i++) {
      csc_dst->col_idx[csr_src->col_idx[i] + 1]++;
    }

    // And now add up all the occurences, before this col index stored offset 1 to the right
    for (int i = 1; i <= csc_dst->n; i++) {
      csc_dst->col_idx[i] += csc_dst->col_idx[i - 1];
    }

    // Extract values from the csr, going by row
    for (int r = 0; r < csr_src->m; r++) {
      for (int i = csr_src->row_idx[r]; i < csr_src->row_idx[r + 1]; i++) {
        int col = csr_src->col_idx[i];
        int value_index = csc_dst->col_idx[col];

        csc_dst->row_idx[value_index] = r;
        csc_dst->values[value_index] = csr_src->values[i];

        // Can use this temporarily
        csc_dst->col_idx[col]++;
      }
    }

    // Offsets back
    for (int i = csc_dst->n; i > 0; i--) {
      csc_dst->col_idx[i] = csc_dst->col_idx[i - 1];
    }
    csc_dst->col_idx[0] = 0;
  }
}

void test_csr_to_csc(csr_matrix_t *csr_src) {

  /*
    Initialize stuff.
  */

  dense_matrix_t *dense_ref = create_dense_matrix_from_csr_matrix(csr_src);
  csc_matrix_t *csc_ref = create_csc_matrix_from_dense_matrix(dense_ref);

  /*
    Student Implementation
  */
  csc_matrix_t *csc_test = (csc_matrix_t *)malloc(sizeof(csc_matrix_t));
  student_csr_matrix_to_csc_matrix(csr_src, csc_test);

  /*
    Compare
  */

  dense_matrix_t *dense_test = create_dense_matrix_from_csc_matrix(csc_test);

  float res = max_pair_wise_diff(csr_src->m * csr_src->n, dense_ref->values, dense_test->values);

  printf("CSR to CSC: ");

  if (ERROR_THRESHOLD < res) {
    printf("FAIL with error = %f\n", res);

    printf("======================\n");
    printf("Original Data Stored in Coordinate (CSR) Format\n");
    pretty_print_csr_matrix(csr_src);

    printf("======================\n");
    printf("Reference Data Stored in Compressed Sparse Column (CSC) Format\n");
    pretty_print_csc_matrix(csc_ref);

    printf("======================\n");
    printf("Student Data Stored in Compressed Sparse Row (CSC) Format\n");
    pretty_print_csc_matrix(csc_test);

  } else {
    printf("PASS\n");
  }

  destroy_dense_matrix(dense_test);
  destroy_dense_matrix(dense_ref);
  destroy_csc_matrix(csc_ref);
  destroy_csc_matrix(csc_test);
}

int main(int argc, char **argv) {
  /*
    Sparse Conversions.
  */
  float A_buf[] = {1, 0, 0, 2, 0, 3, 4, 0, 5, 0, 6, 0, 0, 0, 0, 7, 0, 8, 0, 9, 10, 11, 0, 0};

  dense_matrix_t *A_dense = create_dense_matrix_and_attach_existing_array(6, 4, 4, 1, A_buf);

  // Print out the dense matrix
  printf("======================\n");
  printf("Original Dense Data\n");
  pretty_print_dense_matrix(A_dense);

  // Create a coo matrix
  coo_matrix_t *A_coo = create_coo_matrix_from_dense_matrix(A_dense);

  //////////////////
  printf("PROB01 -- ");
  test_coo_to_csr(A_coo);
  printf("\n");
  printf("PROB02 -- ");
  test_coo_to_bcsr(A_coo);
  printf("\n");

  // Free up that coo matrix
  destroy_coo_matrix(A_coo);

  csr_matrix_t *A_csr = create_csr_matrix_from_dense_matrix(A_dense);
  printf("PROB03 -- ");
  test_csr_to_csc(A_csr);
  printf("\n");

  // Free up that csr matrix
  destroy_csr_matrix(A_csr);

  // Free up the original dense matrix
  destroy_dense_matrix_and_detach_existing_array(A_dense);

  return 0;
}
