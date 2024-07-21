#include <stdbool.h>

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

typedef struct {
    Irrep* irreps;
    int size;
} Irreps;


// create Irreps struct from string
Irreps* irreps_create(const char* str);

// create Irreps struct from tensor product of two Irreps
Irreps* irreps_tensor_product(const Irreps*, const Irreps*);

// create Irreps struct from concatenation of two Irreps
Irreps* irreps_concatenate(const Irreps*, const Irreps*);

// creates Irreps of the linear operation, removing unmatching Irreps if
// force_irreps_out==false, but just copies output otherwise
Irreps* irreps_linear(const Irreps* irreps_in, const Irreps* irreps_out, const bool force_irreps_out);

// copies Irreps
Irreps* irreps_copy(const Irreps*);

// free Irreps struct
void irreps_free(Irreps* irreps);

// dimension of irreps
int irreps_dim(const Irreps* irreps);

// print out irreps
void irreps_print(const Irreps* irreps);

// tensor product between data1 and data2, written to datao, with respective
// representation strings irrep_str1, irrep_str2, irrep_stro
void tensor_product_v1(const Irreps* irreps_1, const float* data_1, const Irreps* irreps_2, const float* data_2, const Irreps* irreps_o, float* data_o);

// tensor product between data1 and data2, written to datao, with respective
// representation strings irrep_str1, irrep_str2, irrep_stro
// uses sparse Clebsch-Gordan for faster computation
void tensor_product_v2(const Irreps* irreps_1, const float* data_1, const Irreps* irreps_2, const float* data_2, const Irreps* irreps_o, float* data_o);

// tensor product between data1 and data2, written to datao, with respective
// representation strings irrep_str1, irrep_str2, irrep_stro
// uses precomputed tensor products in tp.c
void tensor_product_v3(const Irreps* irreps_1, const float* data_1, const Irreps* irreps_2, const float* data_2, const Irreps* irreps_o, float* data_o);

// Real spherical harmonics Y_lm(r) of vector (x, y, z) written to out
void spherical_harmonics(const Irreps* irreps, const float x, const float y, const float z, float* out);

// linear or self-interaction operation
// it is assumed that weights are raveled into a single float*, stored in the order they appear in irreps_in
// NOTE: does not support unsimplified irreps
void linear(const Irreps* irreps_in, const float* input, const float* weight, const Irreps* irreps_out, float* out);

// computes size of weights for linear operation
int linear_weight_size(const Irreps* irreps_in, const Irreps* irreps_out);


// concatenates irreps data together
// NOTE: assumes inputs irreps are simplified and sorted, and will maintain
// sorted order for output
void concatenate(const Irreps* irreps_1, const float* data_1, const Irreps* irreps_2, const float* data_2, float* data_o);

// Sigmoid activation function
float sigmoid(float);

// Sigmoid normalized so that \int_{-\infty}^{\infty} \f(x)^2 \frac{e^{-x^2/2}}{\sqrt{2\pi}} dx = 1
float sigmoid_normalized(float);

// SiLU activation function
float silu(float);

// SiLU normalized so that \int_{-\infty}^{\infty} \f(x)^2 \frac{e^{-x^2/2}}{\sqrt{2\pi}} dx = 1
float silu_normalized(float);

// tanh normalized so that \int_{-\infty}^{\infty} \f(x)^2 \frac{e^{-x^2/2}}{\sqrt{2\pi}} dx = 1
float tanh_normalized(float);

// gate activation function
void gate(const Irreps* irreps_in,
          const float* input, 
          float (*even_act)(float),
          float (*odd_act)(float),
          float (*even_gate_act)(float),
          float (*even_odd_act)(float),
          float* out);

#endif // ifndef INCLUDED_E3NN_H
