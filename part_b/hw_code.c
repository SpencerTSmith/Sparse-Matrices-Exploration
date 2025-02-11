/*
  HW5 B: Shared Memory and Sparse Structures

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


#include <math.h>
#include <stdarg.h>
#include <stdlib.h>
#include <stdio.h>
#include <time.h>

#include <omp.h>

#include "helper.h"
#include "sparse.h"



#define ERROR_THRESHOLD (1e-4)
#define MAX_PRINT_SIZE (32)


typedef void (*dense_matvec_fp)(dense_matrix_t *A_sequential,
				float *x_sequential,
				float *y_sequential );



/*
  Testing Frameworks
*/

dense_matrix_t *create_dense_matrix_with_uniform_random_data(int m, int n,
							         float density)
{
  int cs = 1;
  int rs = m;

  // Initialize the matrix with zeros
  float *values = (float *)calloc(m*n, sizeof(float));

  // seed the random number generator. This is not the thread safe version
  srand(time(NULL));
  
  for( int i = 0; i < m; ++i )
    for( int j = 0; j < n; ++j )
      {
	int rand_int = rand();

	// This will give us a value between [0,1]
	float rand_float = ((float)rand_int)/((float)RAND_MAX);
	
	if(rand_float < density )
	  values[i*rs+j*cs] = 1.0f;
      }

  dense_matrix_t* A = create_dense_matrix_and_attach_existing_array(m,
								    n,
								    cs,
								    rs,
								    values );

  return A;

}


// SOURCE: https://stackoverflow.com/questions/101439/the-most-efficient-way-to-implement-an-integer-based-power-function-powint-int
int ipow(int base, int exp)
{
    int result = 1;
    for (;;)
    {
        if (exp & 1)
            result *= base;
        exp >>= 1;
        if (!exp)
            break;
        base *= base;
    }

    return result;
}



float get_kron_value_backend( int depth,
			      int i, int j,
			      dense_matrix_t *densities )
{
  int md  = densities->m;
  int nd  = densities->n;
  int rsd = densities->rs;
  int csd = densities->cs;

  float probability;

  if( depth == 0 )
    {
      probability = 1.0f;
    }
  else
    {
      int depth_min_1 = depth - 1;

      int md_pow_depth_min_1 = ipow(md,depth_min_1);
      int nd_pow_depth_min_1 = ipow(nd,depth_min_1);
      
      int cur_i = i/md_pow_depth_min_1;
      int cur_j = j/nd_pow_depth_min_1;

      int next_i = i%md_pow_depth_min_1;
      int next_j = j%nd_pow_depth_min_1;

      probability = densities->values[cur_i*csd + cur_j*rsd] *
	get_kron_value_backend( depth_min_1,
				next_i, next_j,
				densities );


    }

  
  return probability;
}


float get_kron_value(int m, int n,
		     int i, int j,
		     dense_matrix_t *densities )
{
  int md  = densities->m;
  int nd  = densities->n;

  int depth_estimate_m = (int)ceil(log(m)/log(md));
  int depth_estimate_n = (int)ceil(log(n)/log(nd));
  int depth;
  
  if( depth_estimate_m > depth_estimate_n )
      depth = depth_estimate_m;
  else
      depth = depth_estimate_n;

  return get_kron_value_backend( depth,
				 i, j,
				 densities );


  
}


dense_matrix_t *create_dense_matrix_with_kronecker_random_data(int m, int n,
							       dense_matrix_t *densities)
{
  int cs = 1;
  int rs = m;

  // Initialize the matrix with zeros
  float *values = (float *)calloc(m*n, sizeof(float));

  // seed the random number generator. This is not the thread safe version
  srand(time(NULL));
  
  for( int i = 0; i < m; ++i )
    for( int j = 0; j < n; ++j )
      {
	int rand_int = rand();

	// This will give us a value between [0,1]
	float rand_float = ((float)rand_int)/((float)RAND_MAX);


	float kron_density = get_kron_value(m,n,
					    i,j,
					    densities );

	
	if(rand_float < kron_density )
	  values[i*rs+j*cs] = 1.0f;//kron_density;
      }

  dense_matrix_t* A = create_dense_matrix_and_attach_existing_array(m,
								    n,
								    cs,
								    rs,
								    values );

  return A;

}




void reference_dense_matvec_dot_product( dense_matrix_t *A,
					     float *x,
					     float *y )
{
  int m  = A->m;
  int n  = A->n;
  int rs = A->rs;
  int cs = A->cs;

  
  for( int i = 0; i < m; ++i )
    for( int j = 0; j < n; ++j )
      {
	y[i] += A->values[i*cs+j*rs]*x[j];
      }
}


void reference_dense_matvec_axpy( dense_matrix_t *A,
				     float *x,
				     float *y )
{
  int m  = A->m;
  int n  = A->n;
  int rs = A->rs;
  int cs = A->cs;


  for( int j = 0; j < n; ++j )
    for( int i = 0; i < m; ++i )
      {
	y[i] += A->values[i*cs+j*rs]*x[j];
      }
}


void reference_coo_matvec( dense_matrix_t *A,
			     float *x,
			     float *y )
{

  coo_matrix_t *A_coo = create_coo_matrix_from_dense_matrix(A);

  for(int cur_pos = 0; cur_pos < A_coo->nnz; ++cur_pos )
    {
      int i   = A_coo->row_idx[cur_pos];
      int j   = A_coo->col_idx[cur_pos];
      int val = A_coo->values[cur_pos];

      y[i] += val * x[j];
    }

  
  destroy_coo_matrix(A_coo);
}


void reference_csr_matvec( dense_matrix_t *A,
			     float *x,
			     float *y )
{

  csr_matrix_t *A_csr = create_csr_matrix_from_dense_matrix(A);

  for(int i = 0; i < A_csr->m; ++i )
    for(int cur_nz = A_csr->row_idx[i];
	cur_nz < A_csr->row_idx[i+1];
	++cur_nz )
      {
	int j     = A_csr->col_idx[cur_nz];
	float val = A_csr->values[cur_nz];

	y[i] += val * x[j];
      }
  
  destroy_csr_matrix(A_csr);
}


void reference_csc_matvec( dense_matrix_t *A,
			     float *x,
			     float *y )
{

  csc_matrix_t *A_csc = create_csc_matrix_from_dense_matrix(A);

  for(int j = 0; j< A_csc->n; ++j )
    for(int cur_nz = A_csc->col_idx[j];
	cur_nz < A_csc->col_idx[j+1];
	++cur_nz )
      {
	int i     = A_csc->row_idx[cur_nz];
	float val = A_csc->values[cur_nz];

	y[i] += val * x[j];
      }

  
  destroy_csc_matrix(A_csc);
}



void reference_bcsr_matvec( dense_matrix_t *A,
			    float *x,
			    float *y )
{

  int mb  = 4;
  int nb  = 2;
  int bcs = 2;
  int brs = 1;
  
  bcsr_matrix_t *A_bcsr = create_bcsr_matrix_from_dense_matrix(mb,nb,
							       bcs,brs,
							       A);

  // Copy the metadata over
  int m = A_bcsr->m;
  int n = A_bcsr->n;

  for(int io = 0;
      io < m;
      io+=mb )
    {
      int io_ind = io/mb;
      for(int cur_nz = A_bcsr->block_row_idx[io_ind];
	  cur_nz < A_bcsr->block_row_idx[io_ind+1];
	  ++cur_nz )
	{
	  int jo           = A_bcsr->block_col_idx[cur_nz];
	  float *block_val = &A_bcsr->block_values[cur_nz*mb*nb];
	  
	  for( int ii = 0; ii < mb; ++ii )
	    for( int jj = 0; jj < nb; ++jj)
	      {
		int i = ii+io;
		int j = jj+jo;
		float val = block_val[ii*bcs+jj*brs];
		
		y[i] += val * x[j];		    
	      }
	}
    }
  
  destroy_bcsr_matrix(A_bcsr);
}




/*
  Student versions.
*/
void student_dense_matvec_dot_product_2D_parallelism( dense_matrix_t *A,
						           float *x,
						           float *y )
{
  int m  = A->m;
  int n  = A->n;
  int rs = A->rs;
  int cs = A->cs;

  int m_threads = 4;
  int n_threads = 2;
  
  int num_threads = m_threads * n_threads;
  omp_set_num_threads(num_threads);

  // Create and initialize locks
  omp_lock_t *y_locks = (omp_lock_t *)malloc(sizeof(omp_lock_t)*m);

  // we need to initialize the lock before we can use it.
  for(int i = 0; i < m; ++i )
    omp_init_lock(&y_locks[i]);


  #pragma omp parallel
  {

    // Current thread ID
    int tid = omp_get_thread_num();
    int tid_i = tid / n_threads;
    int tid_j = tid % n_threads;

    int mb = m/m_threads;
    int nb = n/n_threads;
    
    for( int ii = 0; ii < mb; ++ii )
      for( int ji = 0; ji < nb; ++ji )
	{
	  /* 
	     Start of STUDENT_TODO 
	     
	     - Only modify the code between these boundaries.
	     - Only use locks for your mutal exclusion.
	     
	  */
	     // int i = tid_i*mb + ii;
	     // int j = tid_j*nb + ji;

	     // y[i] += A->values[i*cs+j*rs]*x[j];

	  /* 
	     End of STUDENT_TODO
	  */
	}
  }

  // Free the locks
  for(int i = 0; i < m; ++i )
    omp_destroy_lock(&y_locks[i]);

  free(y_locks);
}



void student_coo_matvec( dense_matrix_t *A,
			   float *x,
			   float *y )
{

  coo_matrix_t *A_coo = create_coo_matrix_from_dense_matrix(A);
  int m = A_coo->m;
  int n = A_coo->n;

  int num_threads = 8;
  omp_set_num_threads(num_threads);

  // Create and initialize locks
  omp_lock_t *y_locks = (omp_lock_t *)malloc(sizeof(omp_lock_t)*m);

  // we need to initialize the lock before we can use it.
  for(int i = 0; i < m; ++i )
    omp_init_lock(&y_locks[i]);

  #pragma omp parallel for default(none) \
    shared(A_coo,y_locks /* STUDENT_TODO you may need to modify this */)	\
        /* private() STUDENT_TODO you may need to modify this */
  for(int cur_pos = 0; cur_pos < A_coo->nnz; ++cur_pos )
    {
      /* 
	 Start of STUDENT_TODO 
	     
	 - Only modify the code between these boundaries.
	 - Only use locks for your mutal exclusion.
      */

      /*
	int i   = A_coo->row_idx[cur_pos];
	int j   = A_coo->col_idx[cur_pos];
	int val = A_coo->values[cur_pos];

	y[i] += val * x[j];
      */

      /* 
	 End of STUDENT_TODO
      */
    }


  // Free the locks
  for(int i = 0; i < m; ++i )
    omp_destroy_lock(&y_locks[i]);

  free(y_locks);
  
  destroy_coo_matrix(A_coo);
}

void student_csc_matvec( dense_matrix_t *A,
			   float *x,
			   float *y )
{

  csc_matrix_t *A_csc = create_csc_matrix_from_dense_matrix(A);
  int m = A_csc->m;
  int n = A_csc->n;

  int num_threads = 8;
  omp_set_num_threads(num_threads);

  // Create and initialize locks
  omp_lock_t *y_locks = (omp_lock_t *)malloc(sizeof(omp_lock_t)*m);

  // we need to initialize the lock before we can use it.
  for(int i = 0; i < m; ++i )
    omp_init_lock(&y_locks[i]);

  #pragma omp parallel for default(none) \
    shared(A_csc,y_locks /* STUDENT_TODO you may need to modify this */)	\
        /* private() STUDENT_TODO you may need to modify this */
  for(int j = 0; j< A_csc->n; ++j )
    for(int cur_nz = A_csc->col_idx[j];
	cur_nz < A_csc->col_idx[j+1];
	++cur_nz )
      {
	/* 
	   Start of STUDENT_TODO 
	     
	   - Only modify the code between these boundaries.
	   - Only use locks for your mutal exclusion.
	*/
	/*
	  int i     = A_csc->row_idx[cur_nz];
	  float val = A_csc->values[cur_nz];
	
	  y[i] += val * x[j];
	*/
	/* 
	   End of STUDENT_TODO
	*/	
      }

  // Free the locks
  for(int i = 0; i < m; ++i )
    omp_destroy_lock(&y_locks[i]);

  free(y_locks);
  
  destroy_csc_matrix(A_csc);
}

void student_csr_matvec( dense_matrix_t *A,
			   float *x,
			   float *y )
{

  csr_matrix_t *A_csr = create_csr_matrix_from_dense_matrix(A);
  int m = A_csr->m;
  int n = A_csr->n;

  int num_threads = 8;
  omp_set_num_threads(num_threads);

  // Create and initialize locks
  omp_lock_t *y_locks = (omp_lock_t *)malloc(sizeof(omp_lock_t)*m);

  // we need to initialize the lock before we can use it.
  for(int i = 0; i < m; ++i )
    omp_init_lock(&y_locks[i]);

  #pragma omp parallel for default(none) \
    shared(A_csr,y_locks /* STUDENT_TODO you may need to modify this */)	\
        /* private() STUDENT_TODO you may need to modify this */
  for(int i = 0; i < A_csr->m; ++i )
    for(int cur_nz = A_csr->row_idx[i];
	cur_nz < A_csr->row_idx[i+1];
	++cur_nz )
      {
	/* 
	   Start of STUDENT_TODO 
	     
	   - Only modify the code between these boundaries.
	   - Only use locks for your mutal exclusion.
	*/
	/*
	  int j     = A_csr->col_idx[cur_nz];
	  float val = A_csr->values[cur_nz];

	  y[i] += val * x[j];
	*/
	/* 
	   End of STUDENT_TODO
	*/	
      }

  // Free the locks
  for(int i = 0; i < m; ++i )
    omp_destroy_lock(&y_locks[i]);

  free(y_locks);
  
  destroy_csr_matrix(A_csr);
}


void student_bcsr_matvec( dense_matrix_t *A,
			    float *x,
			    float *y )
{

  int mb  = 4;
  int nb  = 2;
  int bcs = 2;
  int brs = 1;
  
  bcsr_matrix_t *A_bcsr = create_bcsr_matrix_from_dense_matrix(mb,nb,
							       bcs,brs,
							       A);

  int m = A_bcsr->m;
  int n = A_bcsr->n;
  
  int num_threads = 8;
  omp_set_num_threads(num_threads);

  // Create and initialize locks
  omp_lock_t *y_locks = (omp_lock_t *)malloc(sizeof(omp_lock_t)*m);

  // we need to initialize the lock before we can use it.
  for(int i = 0; i < m; ++i )
    omp_init_lock(&y_locks[i]);

  #pragma omp parallel for default(none) \
    shared(A_bcsr,y_locks,m,n,mb,nb /* STUDENT_TODO you may need to modify this */) \
        /* private() STUDENT_TODO you may need to modify this */
  for(int io = 0;
      io < m;
      io+=mb )
    {
      int io_ind = io/mb;
      for(int cur_nz = A_bcsr->block_row_idx[io_ind];
	  cur_nz < A_bcsr->block_row_idx[io_ind+1];
	  ++cur_nz )
	{
	  int jo           = A_bcsr->block_col_idx[cur_nz];
	  float *block_val = &A_bcsr->block_values[cur_nz*mb*nb];
	  
	  for( int ii = 0; ii < mb; ++ii )
	    for( int jj = 0; jj < nb; ++jj)
	      {
		/* 
		   Start of STUDENT_TODO 
		   
		   - Only modify the code between these boundaries.
		   - Only use locks for your mutal exclusion.
		*/

		/*
		int i = ii+io;
		int j = jj+jo;
		float val = block_val[ii*bcs+jj*brs];
		
		y[i] += val * x[j];
		*/
		
		/* 
		   End of STUDENT_TODO
		*/	

	      }
	}
    }

  // Free the locks
  for(int i = 0; i < m; ++i )
    omp_destroy_lock(&y_locks[i]);

  free(y_locks);
  
  destroy_bcsr_matrix(A_bcsr);
}







void test_dense_matvec( char *prob,
			dense_matrix_t *A,
			dense_matvec_fp matvec_ref,
			dense_matvec_fp matvec_test )
{

  // We will store A in column major ordering
  int m = A->m;
  int n = A->n;
  
  /*
    Allocate the inputs and outputs
  */
  float *x  = (float *)malloc(sizeof(float)*n);
  float *y_test      = (float *)malloc(sizeof(float)*m);
  float *y_reference = (float *)malloc(sizeof(float)*m);

  /*
    Fill inputs and outputs with data
   */

  fill_sequences( n, 1, x );  
  fill_sequences( m, 10, y_test );
  fill_sequences( m, 10, y_reference );


  /*
    Reference Implementation
  */
  matvec_ref(A,
	     x,
	     y_reference );
  
  /*
    Test Implementation
  */
  
  matvec_test(A,
	      x,
	      y_test );

  /* 
     Test the answer
  */
  float res = max_pair_wise_diff(m,
				 y_test,
				 y_reference);


  

  printf("========================================\n");
  printf("%s: ",prob);
  if( res > 1e-6 )
    {
      printf("FAIL\n");

      if( m > MAX_PRINT_SIZE  || n > MAX_PRINT_SIZE )
	{
	  printf("Too large to print.\n");
	  
	}
      else
	{
	  pretty_print_dense_matrix(A);

	  // Print out the sequential data
	  print_float_mem("    x", n, x);
	  print_float_mem("y_tst", m, y_test);
	  print_float_mem("y_ref", m, y_reference);

	}
	  
    }
  else
    {
      printf( "PASS\n");
    }

  printf("========================================\n");
  printf("\n");

  free(x);
  free(y_test);
  free(y_reference);
}


int main( int argc, char **argv )
{
  /*
    Sparse Conversions.
  */
  float A_buf[] = {1,0,0,2,
		   0,3,4,0,
		   0,0,0,7,
		   0,8,0,9 };

  dense_matrix_t *mat_small_handcooked = create_dense_matrix_and_attach_existing_array(4,4,
								          4,1,
									  A_buf);


  float kron_deterministic_buf[] = {1,1,1,1,
				    1,1,0,0,
				    1,0,1,0,
				    1,0,0,1 };


  float kron_stochastic_buf[] = {0.91f,0.45f,
				 0.53f,0.62f};


  dense_matrix_t *kron_deterministic_densities =
    create_dense_matrix_and_attach_existing_array(4,4,
						    4,1,
						    kron_deterministic_buf);

  dense_matrix_t *kron_stochastic_densities =
    create_dense_matrix_and_attach_existing_array(2,2,
						  2,1,
						  kron_stochastic_buf);

  
  
  dense_matrix_t * mat_small_det_kron
    = create_dense_matrix_with_kronecker_random_data(16,16,
						     kron_deterministic_densities);

  dense_matrix_t * mat_medium_stoch_kron
    = create_dense_matrix_with_kronecker_random_data(64,64,
						     kron_stochastic_densities);


  /*
    printf("======================\n");
    printf("Power Law Random Dense Data\n");
    pretty_print_dense_matrix( kron_deterministic_densities );
    pretty_print_dense_matrix( mat_small_det_kron );
  */

  /*
    printf("======================\n");
    printf("Power Law Random Dense Data\n");
    pretty_print_dense_matrix( kron_stochastic_densities );
    pretty_print_dense_matrix( mat_medium_stoch_kron );
  */
  

  printf("///////////////////////////////////////////////////\n");
  printf("//////////// PROBLEM 01 ///////////////////////////\n");
  printf("//////////// Matrix times Vector Dense ////////////\n");
  printf("///////////////////////////////////////////////////\n");
  /*
  test_dense_matvec( "Sanity Check",
		     mat_small_det_kron,
		     reference_dense_matvec_axpy,
		     reference_dense_matvec_dot_product );
  */

  test_dense_matvec( "Run 01: Dense Matrix time Vector (small handcooked)",
		     mat_small_handcooked,
		     reference_dense_matvec_axpy,
		     student_dense_matvec_dot_product_2D_parallelism );
    
  test_dense_matvec( "Run 02: Dense Matrix time Vector (small deterministic kronecker graph)",
		     mat_small_det_kron,
		     reference_dense_matvec_axpy,
		     student_dense_matvec_dot_product_2D_parallelism );

  test_dense_matvec( "Run 03: Dense Matrix time Vector (medium stochastic kronecker graph)",
		     mat_medium_stoch_kron,
		     reference_dense_matvec_axpy,
		     student_dense_matvec_dot_product_2D_parallelism );

  printf("///////////////////////////////////////////////////\n");
  printf("//////////// PROBLEM 02 ///////////////////////////\n");
  printf("//////////// Matrix times Vector COO //////////////\n");
  printf("///////////////////////////////////////////////////\n");
  /*
  test_dense_matvec( "Sanity Check",
		     mat_small_det_kron,
		     reference_dense_matvec_axpy,
		     reference_coo_matvec );
  */
  test_dense_matvec( "Run 01: COO Matrix time Vector (small handcooked)",
		     mat_small_handcooked,
		     reference_coo_matvec,
		     student_coo_matvec );

  test_dense_matvec( "Run 02: COO Matrix time Vector (small deterministic kronecker graph)",
		     mat_small_det_kron,
		     reference_coo_matvec,
		     student_coo_matvec );

  test_dense_matvec( "Run 03: COO Matrix time Vector (medium stochastic kronecker graph)",
		     mat_medium_stoch_kron,
		     reference_coo_matvec,
		     student_coo_matvec );

  
  printf("///////////////////////////////////////////////////\n");
  printf("//////////// PROBLEM 03 ///////////////////////////\n");
  printf("//////////// Matrix times Vector CSC //////////////\n");
  printf("///////////////////////////////////////////////////\n");
  /*
  test_dense_matvec( "Sanity Check",
		     mat_small_det_kron,
		     reference_dense_matvec_axpy,
		     reference_csc_matvec );
  */
  
  test_dense_matvec( "Run 01: CSC Matrix time Vector (small handcooked)",
		     mat_small_handcooked,
		     reference_csc_matvec,
		     student_csc_matvec );

  test_dense_matvec( "Run 02: CSC Matrix time Vector (small deterministic kronecker graph)",
		     mat_small_det_kron,
		     reference_csc_matvec,
		     student_csc_matvec );

  test_dense_matvec( "Run 03: CSC Matrix time Vector (medium stochastic kronecker graph)",
		     mat_medium_stoch_kron,
		     reference_csc_matvec,
		     student_csc_matvec );

  printf("///////////////////////////////////////////////////\n");
  printf("//////////// PROBLEM 04 ///////////////////////////\n");
  printf("//////////// Matrix times Vector CSR //////////////\n");
  printf("///////////////////////////////////////////////////\n");
  /*
  test_dense_matvec( "Sanity Check",
		     mat_small_det_kron,
		     reference_dense_matvec_axpy,
		     reference_csr_matvec );
  */
  test_dense_matvec( "Run 01: CSR Matrix time Vector (small handcooked)",
		     mat_small_handcooked,
		     reference_csr_matvec,
		     student_csr_matvec );


  test_dense_matvec( "Run 02: CSR Matrix time Vector (small deterministic kronecker graph)",
		     mat_small_det_kron,
		     reference_csr_matvec,
		     student_csr_matvec );

  test_dense_matvec( "Run 03: CSR Matrix time Vector (medium stochastic kronecker graph)",
		     mat_medium_stoch_kron,
		     reference_csr_matvec,
		     student_csr_matvec );

  
  printf("///////////////////////////////////////////////////\n");
  printf("//////////// PROBLEM 05 ///////////////////////////\n");
  printf("//////////// Matrix times Vector BCSR /////////////\n");
  printf("///////////////////////////////////////////////////\n");
  /*
  test_dense_matvec( "Sanity Check",
		     mat_small_det_kron,
		     reference_dense_matvec_axpy,
		     reference_bcsr_matvec );
  */
  test_dense_matvec( "Run 01: BCSR Matrix time Vector (small handcooked)",
		     mat_small_handcooked,
		     reference_bcsr_matvec,
		     student_bcsr_matvec );

  test_dense_matvec( "Run 02: BCSR Matrix time Vector (small deterministic kronecker graph)",
		     mat_small_det_kron,
		     reference_bcsr_matvec,
		     student_bcsr_matvec );

  test_dense_matvec( "Run 03: BCSR Matrix time Vector (medium stochastic kronecker graph)",
		     mat_medium_stoch_kron,
		     reference_bcsr_matvec,
		     student_bcsr_matvec );

  
  // Free the buffers
  printf("Free all buffers: ");
  destroy_dense_matrix( mat_small_det_kron );
  destroy_dense_matrix( mat_medium_stoch_kron );

  destroy_dense_matrix_and_detach_existing_array( mat_small_handcooked );  
  destroy_dense_matrix_and_detach_existing_array( kron_deterministic_densities );
  destroy_dense_matrix_and_detach_existing_array( kron_stochastic_densities );  
  printf("PASS\n");
  
  return 0;
}
