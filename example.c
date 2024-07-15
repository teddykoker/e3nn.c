#include <stdio.h>

#include "e3nn.h"

int main(void){

    float node_position_sh[9] = { 0 };
    Irreps* node_irreps = irreps_create("1x0e + 1x1o + 1x2e");
    spherical_harmonics(node_irreps, 1, 2, 3, node_position_sh);

    printf("sh ["); for (int i = 0; i < 9; i++){ printf("%.2f, ", node_position_sh[i]); } printf("]\n");
    irreps_free(node_irreps);

    float neighbor_feature[] = { 7, 8, 9 };
    float product[27] = { 0 };
    Irreps* node_sh_irreps = irreps_create("1x0e + 1x1o + 1x2e");
    Irreps* neighbor_feature_irreps = irreps_create("1x1e");
    Irreps* product_irreps = irreps_create("1x0o + 1x1o + 2x1e + 1x2e + 1x2o + 1x3e");
    tensor_product(node_sh_irreps, node_position_sh, 
                   neighbor_feature_irreps, neighbor_feature, 
                   product_irreps, product);

    printf("product ["); for (int i = 0; i < 27; i++){ printf("%.2f, ", product[i]); } printf("]\n");
    irreps_free(node_sh_irreps);
    irreps_free(neighbor_feature_irreps);

    float weights[] = { 1, 2, 3, 4, 5, 6, 7, 8, 9 };
    //                [ 1 x 1 weight] [1 x 1 weight] [2 x 2 weight] [1 x 1 weight] [1 x 1 weight] [ 1 x 1 weight]
    float output[27] = { 0 };
    Irreps* output_irreps = irreps_create("1x0o + 1x1o + 2x1e + 1x2e + 1x2o + 1x3e");
    linear(product_irreps,
           product,
           weights,
           output_irreps,
           output);

    printf("output ["); for (int i = 0; i < 27; i++) { printf("%.2f, ", output[i]); } printf("]\n");
    irreps_free(product_irreps);
    irreps_free(output_irreps);

    return 0;
}