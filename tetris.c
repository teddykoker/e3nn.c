#include <stdio.h>
#include <stdlib.h>

#include "e3nn.h"

typedef struct {
    int* senders;
    int* receivers; 
    int size;
} Neighborlist;

// naive O(n^2) nl creation
Neighborlist* neighborlist_create(float p[][3], int num_pos, float radius) {
    Neighborlist* nl = (Neighborlist*) malloc(sizeof(Neighborlist));
    nl->senders = (int*) malloc(num_pos * num_pos * sizeof(int));
    nl->receivers = (int*) malloc(num_pos * num_pos * sizeof(int));
    nl->size = 0;
    float r2 = radius * radius;
    for (int i = 0; i < num_pos; i++) {
        for (int j = 0; j < num_pos; j++) {
            if (i == j) {continue;}
            if ((p[i][0] - p[j][0]) * (p[i][0] - p[j][0]) +
                (p[i][1] - p[j][1]) * (p[i][1] - p[j][1]) +
                (p[i][2] - p[j][2]) * (p[i][2] - p[j][2]) <= r2
            ) {
                nl->senders[nl->size] = i;
                nl->receivers[nl->size] = j;
                nl->size++;
            }
        }
    }
    nl->senders = (int*) realloc(nl->senders, nl->size * sizeof(int));
    nl->receivers = (int*) realloc(nl->receivers, nl->size * sizeof(int));
    return nl;
}

void neighborlist_free(Neighborlist* nl) {
    free(nl->senders);
    free(nl->receivers);
    free(nl);
}

typedef struct {
    Irreps* irreps_sh;
    Irreps* irreps_sender;
    Irreps* irreps_tp;
    Irreps* irreps_receiver;
    Irreps* irreps_node;
    float* linear_weight;
    float* shortcut_weight;
} Layer;

int main(void){

    float pos[4][3] = { {0, 0, 0}, {1, 0, 0}, {1, 1, 0}, {2, 1, 0}};  // zigszag

    int num_nodes = 4;

    Neighborlist* nl = neighborlist_create(pos, num_nodes, 1.1);

    Layer layer1 = {
        .irreps_sh=irreps_create("1x1o + 1x2e + 1x3o"),
        .irreps_sender=irreps_create("1x0e"),
        .irreps_tp=irreps_create("1x1o + 1x2e + 1x3o"),
        .irreps_receiver=irreps_create("1x0e + 1x1o + 1x2e + 1x3o"),
        .irreps_node=irreps_create("32x0e + 8x1o + 8x2e"),
    };

    float* sender_features   = (float *) malloc(num_nodes * irreps_dim(layer1.irreps_sender) * sizeof(float));
    float* receiver_features = (float *) malloc(nl->size * irreps_dim(layer1.irreps_receiver) * sizeof(float));

    // inital node features are just 1s
    for (int i = 0; i < num_nodes; i++) {
        sender_features[i] = 1.0;
    }
    
    for (int edge = 0; edge < nl->size; edge++) {
        int s = nl->senders[edge];
        int r = nl->receivers[edge];
        float* sh = (float *) malloc(irreps_dim(layer1.irreps_sh) * sizeof(float));
        float* tp = (float *) malloc(irreps_dim(layer1.irreps_tp) * sizeof(float));
        spherical_harmonics(layer1.irreps_sh, 
                            pos[r][0] - pos[s][0], 
                            pos[r][1] - pos[s][1], 
                            pos[r][2] - pos[s][2],
                            sh);
        tensor_product(layer1.irreps_sender, 
                       &sender_features[s * irreps_dim(layer1.irreps_sender)],
                       layer1.irreps_sh,
                       sh,
                       layer1.irreps_tp,
                       tp);
        concatenate(layer1.irreps_sender, 
                    &sender_features[s * irreps_dim(layer1.irreps_sender)],
                    layer1.irreps_tp,
                    tp,
                    &receiver_features[edge * irreps_dim(layer1.irreps_receiver)]);
        free(sh);
        free(tp);
    }


    // TODO: free layer
    free(sender_features);
    free(receiver_features);
    neighborlist_free(nl);

    return 0;
}