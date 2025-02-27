/*
  General helper functions for PDN class assignments.


  - richard.m.veras@ou.edu

*/


/*
  Helper functions
*/
#include <math.h>
#include <stdarg.h>
#include <stdlib.h>
#include <stdio.h>
#include <time.h>


void print_float_mem(char *name, int vlen, float *src)
{

  printf("%s = [ ",name);
  for( int vid = 0; vid < vlen; ++vid )
    {
      if ( src[vid] < 0.0f )
	printf( " x, ", src[vid] );
      else
	printf( "%2.f, ", src[vid] );
    }
    printf("]\n");
}



void print_float_matrix_mem(char *name, int m, int n, int rs, int cs, float *src)
{

  printf("%s =\n",name);
  for( int i = 0; i < m; ++i )
    {
      for( int j = 0; j < n; ++j )
	{
	  if ( src[i*cs + j*rs] < 0.0f )
	    printf( " x, ", src[i*cs + j*rs] );
	  else
	    printf( "%2.f, ", src[i*cs + j*rs] );
	}
      printf("\n");
    }
  printf("\n");
}







float max_pair_wise_diff(int m, float *a, float *b)
{
  float max_diff = 0.0;
  
  for(int i = 0; i < m; ++i)
    {
      float sum  = fabs(a[i]+b[i]);
      float diff = fabs(a[i]-b[i]);

      float res = 0.0f;

      if(sum == 0.0f)
	res = diff;
      else
	res = 2*diff/sum;

      if( res > max_diff )
	max_diff = res;
    }

  return max_diff;
}



void fill_sequences( int size, int start, float *dst )
{
  for( int i = 0; i < size; ++i )
      dst[i]= (float)(i+start);
}

void neg_xout_sequences( int size, float *dst )
{
  for( int i = 0; i < size; ++i )
    {
      dst[i] = -1.0f;
    }
}

void zero_out_sequences( int size, float *dst )
{
  for( int i = 0; i < size; ++i )
    {
      dst[i] = 0.0f;
    }
}

