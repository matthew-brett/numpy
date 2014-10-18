#ifdef APPLE_ACCELERATE_SGEMV_PATCH  

/* This is an ugly hack to circumvent a bug in Accelerate's cblas_sgemv.
 *
 * See: https://github.com/numpy/numpy/issues/4007
 *
 */ 
  
#define NPY_NO_DEPRECATED_API NPY_API_VERSION
#include "Python.h"
#include "numpy/arrayobject.h"

#include <string.h>
#include <dlfcn.h>
#include <stdio.h>
#include <stdlib.h>
#include <pthread.h>

enum CBLAS_ORDER {CblasRowMajor=101, CblasColMajor=102};
enum CBLAS_TRANSPOSE {CblasNoTrans=111, CblasTrans=112, CblasConjTrans=113};

/* ----------------------------------------------------------------- */
/* Management of aligned memory */

pthread_key_t tls_memory_error;
static int delete_tls_key = 0;

static void *aligned_malloc(size_t size, int align)
/* 
 * aligned malloc
 *
 */
{
    void *ptr;
    if (NPY_UNLIKELY(posix_memalign(&ptr, align, size))) 
        return NULL;
    return ptr;
}

#define BADARRAY(x) (((npy_intp)(void*)x)%32)    
   
static float *aligned_matrix(const enum CBLAS_ORDER order, 
                    const int m, const int n, const float *A, 
                    const int lda)
/* 
 * Return an aligned copy of matrix A
 * if it is misaligned to 32 byte boundary.
 *
 */
{       
    float *alignedA;
    int r, c;
    if (BADARRAY(A)) {
        int sizeA = (order == CblasRowMajor ? m * lda : lda * n);          
        alignedA = (float*)aligned_malloc(m*n*sizeof(float),32);
        if (NPY_UNLIKELY(alignedA == NULL)) return (float*)NULL;       
        if (order == CblasRowMajor) {
            for (r=0; r<m; r++) 
                memcpy((void*)(alignedA+r*n),(void*)(A+r*lda),n*sizeof(float));
        } else {
            for (c=0; c<n; r++) 
                memcpy((void*)(alignedA+c*m),(void*)(A+c*lda),m*sizeof(float));
        }        
    } else {
        alignedA = (float*)A;        
    }
    return alignedA;
}

static float *aligned_vector(const float *V, const int n, const int inc) 
/* 
 * Return an aligned copy of vector V
 * if it is misaligned to 32 byte boundary.
 *
 */ 
{
    float *alignedV, *tmp;
    int i;
    if (BADARRAY(V)) {
        alignedV = (float*)aligned_malloc(n*sizeof(float),32);
        if (NPY_UNLIKELY(alignedV == NULL)) return NULL;
        tmp = alignedV;
        for (i=0; i<n; i++) {
            *tmp++ = *V;
            V += inc;
        }
    } else {
        alignedV = (float*)V;
    }
    return alignedV;
}

/* ----------------------------------------------------------------- */
/* Original cblas_sgemv */

#define VECLIB_FILE "/System/Library/Frameworks/Accelerate.framework/Versions/A/Frameworks/vecLib.framework/vecLib"

typedef int cblas_sgemv_t(const enum CBLAS_ORDER order, 
                          const enum CBLAS_TRANSPOSE trans,
                          const int m, const int n, const float alpha,
                          const float *A, const int lda, 
                          const float *B, const int incB,
                          const float beta,
                          float *C, const int incC);
                          
static void *veclib = NULL;
static cblas_sgemv_t *accelerate_cblas_sgemv = NULL;
static int AVX = 0;

static int cpu_supports_avx(void)
{
    char tmp[1024];
    FILE *p;
    size_t count;
    p = popen("sysctl -a | grep AVX", "r");
    if (p == NULL) return -1;
    count = fread((void *)&tmp, 1, sizeof(tmp)-1, p);
    pclose(p);
    return (count ? 1 : 0);
}

__attribute__((constructor))
static void loadlib()
{
    /* TODO: Better error handling than Py_FatalError */
    
    char errormsg[1024];
    memset((void*)errormsg, 0, sizeof(errormsg));
    delete_tls_key = 0;
    
    /* check if the CPU supports AVX */
    AVX = cpu_supports_avx();
    if (AVX < 0) {
        /* Could not determine if CPU supports AVX,
         * assume for safety that it does */
        AVX = 1; 
    }
    
    /* load vecLib */
    veclib = dlopen(VECLIB_FILE, RTLD_LOCAL | RTLD_FIRST);
    if (!veclib) {
        sprintf(errormsg,"Failed to open vecLib from location '%s'.", VECLIB_FILE);
        Py_FatalError(errormsg); /* calls abort() and dumps core */
    }
    
    /* resolve cblas_sgemv */
    accelerate_cblas_sgemv = (cblas_sgemv_t*) dlsym(veclib, "cblas_sgemv");
    if (!accelerate_cblas_sgemv) {
        sprintf(errormsg,"Failed to resolve symbol 'cblas_sgemv'.");
        Py_FatalError(errormsg);
    }
    
    /* create a pthreads key for tls */
    if (pthread_key_create(&tls_memory_error, NULL)) {
        sprintf(errormsg,"Failed to create TLS key.");
        Py_FatalError(errormsg);
    }
    delete_tls_key = 1;
}

__attribute__((destructor))
static void unloadlib(void)
{
   if (veclib) dlclose(veclib);
   if (delete_tls_key) pthread_key_delete(tls_memory_error);
}

/* ----------------------------------------------------------------- */
/* cblas_sgemv override */
 
void cblas_sgemv(const enum CBLAS_ORDER order, 
                        const enum CBLAS_TRANSPOSE trans,
                        const int m, int n, const float alpha,
                        const float *A, const int lda, 
                        const float *B, const int incB,
                        const float beta,
                        float *C, const int incC)

/* Patch for the cblas_sgemv segfault in Accelerate: 
 * Aligns all input arrays on 32 byte boundaries, then     
 * calls cblas_sgemv in Accelerate. If AVX is not supported
 * by the CPU it just calls cblas_sgemv.
 *
 * If memory allocation fails it associates the
 * value (void*)1 with the TLS key tls_memory_error.  
 *
 * This function will overshadow cblas_sgemv in Accelerate
 * in the link process.
 *
 */
 
{
    float *alignedA = (float*)A, *alignedB = (float*)B, *alignedC = (float*)C;
    int freeA=0, freeB=0, freeC=0;
    int veclen = (trans == CblasTrans ? m : n);
    int _incB = incB, _incC = incC, _lda = lda;
                         
    /* check if arrays are misaligned and the CPU has AVX */
    const int needs_copy = AVX && (BADARRAY(A) || BADARRAY(B) || BADARRAY(C)); 
              
    /* clear memory error tls value */          
    pthread_setspecific(tls_memory_error, (void*)0);              
              
    /* Make temporary copies of misaligned arrays if needed. */
    if (needs_copy) {
        /* aligned copy of A if needed */
        alignedA = aligned_matrix(order, m, n, A, lda);
        if (NPY_UNLIKELY(alignedA == NULL)) goto fail;
        if (alignedA != A) {
            _lda = (order == CblasRowMajor ? n : m);
            freeA = 1;
        }    
        /* aligned copy of B if needed */
        alignedB = aligned_vector(B, veclen, incB);
        if (NPY_UNLIKELY(alignedB == NULL)) goto fail;
        if (alignedB != B) {
            _incB = 1;
            freeB = 1;
        }       
        /* aligned copy of C if needed */
        alignedC = aligned_vector(C, veclen, incC);
        if (NPY_UNLIKELY(alignedC == NULL)) goto fail;
        if (alignedC != C) {
            _incC = 1;
            freeC = 1;
        }  
    }
                
    /* Now we can safely call cblas_sgemv from Accelerate. 
     * Arrays are aligned to 32 byte boundaries if the CPU
     * has AVX. */
     accelerate_cblas_sgemv(order, trans, m, n, alpha, alignedA, _lda, 
                             alignedB, _incB, beta, alignedC, _incC);
                                                        
    /* Clean up temporary arrays if they were made. */
    if (needs_copy) {
        if (freeC) {
            /* copy the result into the output array */
            int i;
            float *tmp1 = C, *tmp2 = alignedC;
            for (i=0; i<veclen; i++) {
                *tmp1 = *tmp2;
                tmp1 += incC;
                tmp2 += _incC;
            }
            free(alignedC);
        }
        if (freeB) free(alignedB);
        if (freeA) free(alignedA);
    }
    return;
fail:
    /* allocation failure */
    pthread_setspecific(tls_memory_error, (void*)1);
    if (freeB) free(alignedB);
    if (freeA) free(alignedA);
}
#endif
