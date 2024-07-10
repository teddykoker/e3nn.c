#include <stdio.h>

#include "e3nn.h"

int main(void){

    // tensor product
    float input1[] = { 0, 1, 2, 3, 4 };
    float input2[] = { 0, 1, 2, 3, 4, 5 };
    float product[30] = { 0 };
    tensor_product("2x0e + 1x1o", input1, 
                   "1x0o + 1x2o", input2, 
                   "2x0o + 2x1e + 1x2e + 2x2o + 1x3e", product);

    printf("product ["); for (int i = 0; i < 30; i++){ printf("%.2f, ", product[i]); } printf("]\n");

    // spherical harmonics
    float sph[9] = { 0 };
    spherical_harmonics("1x0e + 1x1o + 1x2e", 1.0, 2.0, 3.0, sph);

    printf("sph ["); for (int i = 0; i < 9; i++) { printf("%.2f, ", sph[i]); } printf("]\n");

    // linear/self-interaction
    float input3[] = { 0, 1, 2, 3, 4, 5, 6, 7 };
    float weight[] = { 0, 1, 2, 3, 4, 5, 0, 1, 2, 3, 4, 5 };
    float output[12] = { 0 };
    linear("2x0e + 2x1o", input3, weight,
           "3x0e + 3x1o", output);

    printf("output ["); for (int i = 0; i < 12; i++) { printf("%.2f, ", output[i]); } printf("]\n");
    
    return 0;
}
