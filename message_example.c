#include <stdio.h>

#include "e3nn.h"

 int main(void){

     float node_position_sh[9] = {0};
     spherical_harmonics("1x0e + 1x1o + 1x2e", 1, 2, 3, node_position_sh);

     printf("sh ["); for (int i = 0; i < 9; i++){ printf("%.2f, ", node_position_sh[i]); } printf("]\n");

     float neighbor_feature[] = {7,8,9};
     float product[27] = { 0 };
     tensor_product("1x0e + 1x1o + 1x2e", node_position_sh, 
                    "1x1e", neighbor_feature, 
                    "1x0o + 1x1o + 2x1e + 1x2e + 1x2o + 1x3e", product);
     printf("product ["); for (int i = 0; i < 27; i++){ printf("%.2f, ", product[i]); } printf("]\n");

     float weights[] = {1, 2, 3, 4, 5, 6, 7, 8, 9};
     ///               [ 1 x 1 weight] [1 x 1 weight] [2 x 2 weight] [1 x 1 weight] [1 x 1 weight] [ 1 x 1 weight]
     float output[27] = { 0 };
     linear("1x0o + 1x1o + 2x1e + 1x2e + 1x2o + 1x3e",
            product,
            weights,
            "1x0o + 1x1o + 2x1e + 1x2e + 1x2o + 1x3e",
            output);

     printf("message ["); for (int i = 0; i < 27; i++) { printf("%.2f, ", output[i]); } printf("]\n");

     return 0;
 }