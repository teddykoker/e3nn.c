#ifndef INCLUDED_CLEBSCH_GORDAN_H
#define INCLUDED_CLEBSCH_GORDAN_H

// maximum l value used to precompute Clebsch-Gordan coefficients
// note that L_MAX / 2 is used for the two input l values
// this will effect the startup time as the cache is built before main()
#define L_MAX 14

#define MAX(a,b) ((a) > (b) ? (a) : (b))
#define MIN(a,b) ((a) < (b) ? (a) : (b))

typedef struct {
    int m1;
    int m2;
    int m3;
    float c;
} SparseClebschGordanElement;

typedef struct {
   SparseClebschGordanElement* elements;
   int size; 
} SparseClebschGordanMatrix;


// returns n!
double factorial(int n);

// returns n!!
double dfactorial(int n);

// Clebsch-Gordan coefficients of the real irreducible representations of SO3
// NOTE: build_clebsch_gordan_cache must be called first
float clebsch_gordan(int l1, int l2, int l3, int m1, int m2, int m3);

// Clebsch-Gordan coefficients of the real irreducible representations of SO3
// returned as a list of non-zero elements with specified c at m1, m2, and m3
// NOTE: build_sparse_clebsch_gordan_cache must be called first
SparseClebschGordanMatrix sparse_clebsch_gordan(int l1, int l2, int l3);

// build cache of Clebsch-Gordan coefficients -- will be freed at end of runtime
void build_clebsch_gordan_cache(void);

// build cache of Clebsch-Gordan sparse coefficients -- will be freed at end of
// runtime
void build_sparse_clebsch_gordan_cache(void);


#endif // ifndef INCLUDED_CLEBSCH_GORDAN_H
