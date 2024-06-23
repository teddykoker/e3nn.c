#include <stdio.h>

#include "e3nn.h"

int main(void){

    float input1[] = { 0, 1, 2, 3, 4 };
    float input2[] = { 0, 1, 2, 3, 4, 5};
    float output[30] = { 0 };

    tensor_product("2x0e + 1x1o", input1, 
                   "1x0o + 1x2o", input2, 
                   "2x0o + 2x1e + 1x2e + 2x2o + 1x3e", output);

    printf("["); for (int i = 0; i < 30; i++){
        printf("%.2f, ", output[i]);
    }
    printf("]\n");
    
    return 0;
}
