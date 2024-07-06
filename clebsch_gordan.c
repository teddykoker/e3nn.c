#include "clebsch_gordan.h"

#include <math.h>
#include <complex.h>
#include <stdlib.h>
#include <stdio.h>


#define EPS 1e-8

// cache for common factorials - makes initial cg computation a little faster
#define MAX_FACT_CACHE 9
double fact_cache[10] = {
  1.00000000000000000000e+00,
  1.00000000000000000000e+00,
  2.00000000000000000000e+00,
  6.00000000000000000000e+00,
  2.40000000000000000000e+01,
  1.20000000000000000000e+02,
  7.20000000000000000000e+02,
  5.04000000000000000000e+03,
  4.03200000000000000000e+04,
  3.62880000000000000000e+05,
};

#define MAX_DFACT_CACHE 9
double dfact_cache[10] = {
  1.00000000000000000000e+00,
  1.00000000000000000000e+00,
  2.00000000000000000000e+00,
  3.00000000000000000000e+00,
  8.00000000000000000000e+00,
  1.50000000000000000000e+01,
  4.80000000000000000000e+01,
  1.05000000000000000000e+02,
  3.84000000000000000000e+02,
  9.45000000000000000000e+02,
};


typedef float***** ClebschGordanCache;
typedef SparseClebschGordanMatrix** SparseClebschGordanCache;

// TODO: nicer way to avoid globals?
SparseClebschGordanCache* sparse_cg_cache = NULL;
ClebschGordanCache* cg_cache = NULL;

double factorial(int n) {
    if(n < MAX_FACT_CACHE) { return fact_cache[n]; }
    double x = (double) n;
    while(--n > MAX_FACT_CACHE) {
        x *= (double) n;
    }
    return x * fact_cache[n];
}

double dfactorial(int n) {
    if(n < MAX_DFACT_CACHE) { return dfact_cache[n]; }
    double x = (double) n;
    while((n -= 2) > MAX_FACT_CACHE) {
        x *= (double) n;
    }
    return x * dfact_cache[n];
}


double _su2_cg(int j1, int j2, int j3, int m1, int m2, int m3) {
    // calculate the Clebsch-Gordon coefficient
    // for SU(2) coupling (j1,m1) and (j2,m2) to give (j3,m3).
    // based on e3nn.o3._wigner._su2_clebsch_gordan_coeff
    if (m3 != m1 + m2) {
        return 0;
    }
    int vmin = MAX(MAX(-j1 + j2 + m3, -j1 + m1), 0);
    int vmax = MIN(MIN(j2 + j3 + m1, j3 - j1 + j2), j3 + m3);
    
    double C = sqrt(
        (double) (2 * j3 + 1) *
        (
            (factorial(j3 + j1 - j2) *
                factorial(j3 - j1 + j2) *
                factorial(j1 + j2 - j3) *
                factorial(j3 + m3) *
                factorial(j3 - m3)
            ) /
            (factorial(j1 + j2 + j3 + 1) *
                factorial(j1 - m1) *
                factorial(j1 + m1) *
                factorial(j2 - m2) *
                factorial(j2 + m2)
            )
        )
    );
    double S = 0;
    for (int v = vmin; v <= vmax; v++) {
        S += pow(-1, v + j2 + m2) * (
            (factorial(j2 + j3 + m1 - v) * factorial(j1 - m1 + v)) /
            (factorial(v) * factorial(j3 - j1 + j2 - v) * factorial(j3 + m3 - v) * factorial(v + j1 - j2 - m3))
        );
    }
    C = C * S;
    return C;
}


float complex change_basis_real_to_complex(int l, int m1, int m2) {
    // https://en.wikipedia.org/wiki/Spherical_harmonics#Real_form
    // based on:
    // https://github.com/e3nn/e3nn-jax/blob/a2a81ab451b9cd597d7be27b3e1faba79457475d/e3nn_jax/_src/so3.py#L6
    // but instead of returning a matrix, just index at [m1, m2], e.g.:
    // change_basis_real_to_complex(l, m1, m2) = e3nn_jax._src.so3.change_basis_real_to_complex(l)[m1, m2]
    const float complex factor = cpow(-I, l);
    const float inv_sqrt2 = 1 / sqrt(2);
    const float complex I_inv_sqrt2 = I * inv_sqrt2;
    
    if (m1 == m2 && m2 == 0) { return factor; }
    if (m1 < 0) {
        if (m2 == -m1) { return inv_sqrt2 * factor; }
        if (m2 == m1) { return -I_inv_sqrt2 * factor; }
    }
    if (m1 > 0) {
        if (m2 == m1) { return pow(-1, m1) * inv_sqrt2 * factor; }
        if (m2 == -m1) { return pow(-1, m1) * I_inv_sqrt2 * factor; }
    }
    return 0;
}


float compute_clebsch_gordan(int l1, int l2, int l3, int m1, int m2, int m3) {
    // Clebsch-Gordan coefficients of the real irreducible representations of
    // SO3, based on:
    // https://github.com/e3nn/e3nn-jax/blob/a2a81ab451b9cd597d7be27b3e1faba79457475d/e3nn_jax/_src/so3.py#L21
    float c = 0;
    for (int i = -l1; i <= l1; i++) {
        for (int j = -l2; j <= l2; j++) {
            for (int k = -l3; k <= l3; k++) {
                c += creal(
                    change_basis_real_to_complex(l1, i, m1)
                    * change_basis_real_to_complex(l2, j, m2)
                    * conj(change_basis_real_to_complex(l3, k, m3))
                    * _su2_cg(l1, l2, l3, i, j, k)
                );
            }
        }
    }
    // note that this normalization is applied in the su2_clebsch_gordan in the
    // JAX library, however we are not using that function and call _su2_cg directly
    return c / sqrt(2 * l3 + 1);
}


float clebsch_gordan(int l1, int l2, int l3, int m1, int m2, int m3) {
    // Clebsch-Gordan coefficients of the real irreducible representations of SO3
    return cg_cache[l1][l2][l3][m1 + l1][m2 + l2][m3 + l3];
}

SparseClebschGordanMatrix sparse_clebsch_gordan(int l1, int l2, int l3) {
    return sparse_cg_cache[l1][l2][l3];
}


void build_clebsch_gordan_cache(void) {
    // precompute all Clebsch-Gordan coefficients up to L_MAX
    // NOTE: only computing l1 and l2 up to L_MAX / 2 for now to make things faster 
    if (cg_cache) {
        // already built
        return;
    }
    cg_cache = (ClebschGordanCache*) malloc((L_MAX / 2 + 1) * sizeof(float*****));
    for (int l1 = 0; l1 <= L_MAX / 2; l1++) { 
        cg_cache[l1] = (float*****) malloc((L_MAX / 2 + 1) * sizeof(float****));
        for (int l2 = 0; l2 <= L_MAX / 2; l2++) {
            cg_cache[l1][l2] = (float****) malloc((L_MAX + 1) * sizeof(float***));
            for (int l3 = abs(l1 - l2); l3 <= l1 + l2 && l3 <= L_MAX; l3++) {
                cg_cache[l1][l2][l3] = (float***) malloc((2 * l1 + 1) * sizeof(float**));
                for (int m1 = -l1; m1 <= l1; m1++) {
                    cg_cache[l1][l2][l3][m1 + l1] = (float**) malloc((2 * l2 + 1) * sizeof(float*));
                    for (int m2 = -l2; m2 <= l2; m2++) {
                        cg_cache[l1][l2][l3][m1 + l1][m2 + l2] = (float*) malloc((2 * l3 + 1) * sizeof(float));
                        for (int m3 = -l3; m3 <= l3; m3++) {
                            cg_cache[l1][l2][l3][m1 + l1][m2 + l2][m3 + l3] = compute_clebsch_gordan(l1, l2, l3, m1, m2, m3);
                        }
                    }
                }
            }
        }
    }
}

static void free_clebsch_gordan_cache(void) { 
    if (!cg_cache) {
        return;
    }
    for (int l1 = 0; l1 <= L_MAX / 2; l1++) {
        for (int l2 = 0; l2 <= L_MAX / 2; l2++) {
            for (int l3 = abs(l1 - l2); l3 <= l1 + l2 && l3 <= L_MAX; l3++) {
                for (int m1 = -l1; m1 <= l1; m1++) {
                    for (int m2 = -l2; m2 <= l2; m2++) {
                        free(cg_cache[l1][l2][l3][m1 + l1][m2 + l2]);
                    }
                    free(cg_cache[l1][l2][l3][m1 + l1]);
                }
                free(cg_cache[l1][l2][l3]);
            }
            free(cg_cache[l1][l2]);
        }
        free(cg_cache[l1]);
    }
    free(cg_cache);
}

void build_sparse_clebsch_gordan_cache(void) {
    // build sparse Clebsch-Gordan cache
    // NOTE: only computing l1 and l2 up to L_MAX / 2 for now to make things faster 
    if (sparse_cg_cache) {
        // already built
        return;
    }
    build_clebsch_gordan_cache();
    sparse_cg_cache = (SparseClebschGordanMatrix***) malloc((L_MAX / 2 + 1) * sizeof(SparseClebschGordanMatrix**));
    for (int l1 = 0; l1 <= L_MAX / 2; l1++) { 
        sparse_cg_cache[l1] = (SparseClebschGordanMatrix**) malloc((L_MAX / 2 + 1) * sizeof(SparseClebschGordanMatrix*));
        for (int l2 = 0; l2 <= L_MAX / 2; l2++) {
            sparse_cg_cache[l1][l2] = (SparseClebschGordanMatrix*) malloc((L_MAX + 1) * sizeof(SparseClebschGordanMatrix));
            for (int l3 = abs(l1 - l2); l3 <= l1 + l2 && l3 <= L_MAX; l3++) {

                int size = 0;
                for (int m1 = -l1; m1 <= l1; m1++) {
                    for (int m2 = -l2; m2 <= l2; m2++) {
                        for (int m3 = -l3; m3 <= l3; m3++) {
                            float c = clebsch_gordan(l1, l2, l3, m1, m2, m3);
                            if (fabs(c) > EPS) {
                                size++;
                            }
                        }
                    }
                }
                sparse_cg_cache[l1][l2][l3].elements = (SparseClebschGordanElement*) malloc(size * sizeof(SparseClebschGordanElement));
                sparse_cg_cache[l1][l2][l3].size = size;

                int index = 0;
                for (int m1 = -l1; m1 <= l1; m1++) {
                    for (int m2 = -l2; m2 <= l2; m2++) {
                        for (int m3 = -l3; m3 <= l3; m3++) {
                            float c = clebsch_gordan(l1, l2, l3, m1, m2, m3);
                            if (fabs(c) > EPS) {
                                sparse_cg_cache[l1][l2][l3].elements[index].m1 = m1;
                                sparse_cg_cache[l1][l2][l3].elements[index].m2 = m2;
                                sparse_cg_cache[l1][l2][l3].elements[index].m3 = m3;
                                sparse_cg_cache[l1][l2][l3].elements[index].c = c;
                                index++;
                            }
                        }
                    }
                }
            }
        }
    }
}

static void free_sparse_clebsch_gordan_cache(void) {
    if (!sparse_cg_cache) {
        return;
    }
    for (int l1 = 0; l1 <= L_MAX / 2; l1++) {
        for (int l2 = 0; l2 <= L_MAX / 2; l2++) {
            for (int l3 = abs(l1 - l2); l3 <= l1 + l2 && l3 <= L_MAX; l3++) {
                free(sparse_cg_cache[l1][l2][l3].elements);
            }
            free(sparse_cg_cache[l1][l2]);
        }
        free(sparse_cg_cache[l1]);
    }
    free(sparse_cg_cache);
}

__attribute__((destructor))
static void cleanup_clebsch_gordan_cache(void) {
    // ensure caches are freed after program completion
    free_clebsch_gordan_cache();
    free_sparse_clebsch_gordan_cache();
}
