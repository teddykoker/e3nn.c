#ifndef INCLUDED_E3NN_H
#define INCLUDED_E3NN_H

#define EVEN 1
#define ODD -1

// use the fastest tensor product
#define tensor_product tensor_product_v3

typedef struct {
    int c; // channels 
    int l; // rotation order
    int p; // parity
} Irrep;


// parse e3nn string into array of Irrep of length size
Irrep* parse_irrep_str(const char* str, int* size);

// tensor product between data1 and data2, written to datao, with respective
// representation strings irrep_str1, irrep_str2, irrep_stro
void tensor_product_v1(const char* irrep_str1, float* data1, const char* irrep_str2, float* data2, const char* irrep_stro, float* datao);

// tensor product between data1 and data2, written to datao, with respective
// representation strings irrep_str1, irrep_str2, irrep_stro
// uses sparse Clebsch-Gordan for faster computation
void tensor_product_v2(const char* irrep_str1, const float* data1, const char* irrep_str2, const float* data2, const char* irrep_stro, float* datao);

// tensor product between data1 and data2, written to datao, with respective
// representation strings irrep_str1, irrep_str2, irrep_stro
// uses precomputed tensor products in tp.c
void tensor_product_v3(const char* irrep_str1, const float* data1, const char* irrep_str2, const float* data2, const char* irrep_stro, float* datao);

#endif // ifndef INCLUDED_E3NN_H
