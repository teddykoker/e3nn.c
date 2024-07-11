#include <stdio.h>

#include "e3nn.h"

int main(void){

    // tensor product
    Irreps* irreps1 = irreps_create("2x0e + 1x1o");
    float input1[] = { 0, 1, 2, 3, 4 };
    Irreps* irreps2 = irreps_create("1x0o + 1x2o");
    float input2[] = { 0, 1, 2, 3, 4, 5 };
    Irreps* irreps_product = irreps_create("2x0o + 2x1e + 1x2e + 2x2o + 1x3e");
    float product[30] = { 0 };
    tensor_product(irreps1, input1, 
                   irreps2, input2, 
                   irreps_product, product);

    printf("product ["); for (int i = 0; i < 30; i++){ printf("%.2f, ", product[i]); } printf("]\n");
    irreps_free(irreps1);
    irreps_free(irreps2);
    irreps_free(irreps_product);


    // spherical harmonics
    Irreps* irreps_sph = irreps_create("1x0e + 1x1o + 1x2e");
    float sph[9] = { 0 };
    spherical_harmonics(irreps_sph, 1.0, 2.0, 3.0, sph);

    printf("sph ["); for (int i = 0; i < 9; i++) { printf("%.2f, ", sph[i]); } printf("]\n");
    irreps_free(irreps_sph);


    // linear/self-interaction
    Irreps* irreps3 = irreps_create("2x0e + 2x1o");
    float input3[] = { 0, 1, 2, 3, 4, 5, 6, 7 };
    //                 [  2 x 3 weight  ][  2 x 3 weight  ]
    float weight[] = { 0, 1, 2, 3, 4, 5, 0, 1, 2, 3, 4, 5 };
    Irreps* irreps_output = irreps_create("3x0e + 3x1o");
    float output[12] = { 0 };
    linear(irreps3, input3, weight,
           irreps_output, output);

    printf("output ["); for (int i = 0; i < 12; i++) { printf("%.2f, ", output[i]); } printf("]\n");
    irreps_free(irreps3);
    irreps_free(irreps_output);

    return 0;
}
