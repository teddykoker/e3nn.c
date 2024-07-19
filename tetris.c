#include <stdio.h>
#include <stdlib.h>
#include <fcntl.h>
#include <unistd.h>
#include <sys/mman.h>

#include "e3nn.h"


typedef struct {
    int* senders;
    int* receivers; 
    int size;
} Neighborlist;


// naive O(n^2) neithborlist creation
Neighborlist* neighborlist_create(const float p[][3], int num_pos, float radius) {
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
    int linear_weight_size;
    float* linear_weight;
    int shortcut_weight_size;
    float* shortcut_weight;
    float denominator;
} Layer;


void layer_forward(Layer* layer, const float* node_input, const float pos[][3], int num_node, const Neighborlist* nl, float* node_output) {

    // intermediary storage for spherical harmonics, tensor product, messages,
    // received messages, linear output, and shortcut output
    float* sh           = (float *) malloc(irreps_dim(layer->irreps_sh) * sizeof(float));
    float* tp           = (float *) malloc(irreps_dim(layer->irreps_tp) * sizeof(float));
    float* messages     = (float *) malloc(nl->size * irreps_dim(layer->irreps_receiver) * sizeof(float));
    float* receive      = (float *) malloc(irreps_dim(layer->irreps_receiver) * sizeof(float));
    float* linear_out   = (float *) malloc(irreps_dim(layer->irreps_node) * sizeof(float));
    float* shortcut_out = (float *) malloc(irreps_dim(layer->irreps_node) * sizeof(float));

    // compute messages
    for (int edge = 0; edge < nl->size; edge++) {
        int s = nl->senders[edge];
        int r = nl->receivers[edge];
        spherical_harmonics(layer->irreps_sh, 
                            pos[r][0] - pos[s][0], 
                            pos[r][1] - pos[s][1], 
                            pos[r][2] - pos[s][2],
                            sh);
        tensor_product(layer->irreps_sender, 
                       &node_input[s * irreps_dim(layer->irreps_sender)],
                       layer->irreps_sh,
                       sh,
                       layer->irreps_tp,
                       tp);
        concatenate(layer->irreps_sender, 
                    &node_input[s * irreps_dim(layer->irreps_sender)],
                    layer->irreps_tp,
                    tp,
                    &messages[edge * irreps_dim(layer->irreps_receiver)]);
    }

    // aggregate messages and update nodes
    for (int node = 0; node < num_node; node++ ) {
        // zero out received messages
        for (int i = 0; i < irreps_dim(layer->irreps_receiver); i++) {
            receive[i] = 0;
        }
        // zero linear out and shortcut out
        for (int i = 0; i < irreps_dim(layer->irreps_node); i++) {
            linear_out[i] = 0;
            shortcut_out[i] = 0;
        }

        // sum messages into receive
        for (int edge = 0; edge < nl->size; edge++) {
            if (node == nl->receivers[edge]) {
                for (int i = 0; i < irreps_dim(layer->irreps_receiver); i++) {
                    receive[i] += messages[edge * irreps_dim(layer->irreps_receiver) + i];
                }
            }
        }

        // divide by denominator
        for (int i = 0; i < irreps_dim(layer->irreps_receiver); i++) {
            receive[i] /= layer->denominator;
        }

        linear(layer->irreps_receiver, 
               receive, 
               layer->linear_weight, 
               layer->irreps_node,
               linear_out);

        linear(layer->irreps_sender,
               &node_input[node * irreps_dim(layer->irreps_sender)],
               layer->shortcut_weight,
               layer->irreps_node,
               shortcut_out);

        // add linear + shortcut
        for (int i = 0; i < irreps_dim(layer->irreps_node); i++) {
            node_output[node * irreps_dim(layer->irreps_node) + i] = linear_out[i] + shortcut_out[i];
        }
    }
    free(sh);
    free(tp);
    free(messages);
    free(receive);
    free(linear_out);
    free(shortcut_out);
}

void layer_free(Layer* layer) {
    irreps_free(layer->irreps_sh);
    irreps_free(layer->irreps_sender);
    irreps_free(layer->irreps_tp);
    irreps_free(layer->irreps_receiver);
    irreps_free(layer->irreps_node);
}


typedef struct {
    Layer* layers;
    float* data;      // mmap data pointer
    int fd;           // mmap file descriptor 
    size_t file_size; // mmap file size
} Model;


Model* model_create(void) {
    // TODO: some of these must be manually configured and should be loaded from a config file
    // some irreps can be computed at run time as well
    // weight sizes are also able to be computed at runtime
    Model* model = (Model *) malloc(sizeof(Model));
    model->layers = (Layer *) malloc(3 * sizeof(Layer));

    model->layers[0].irreps_sh=irreps_create("1x1o + 1x2e + 1x3o");
    model->layers[0].irreps_sender=irreps_create("1x0e");
    model->layers[0].irreps_tp=irreps_create("1x1o + 1x2e + 1x3o");
    model->layers[0].irreps_receiver=irreps_create("1x0e + 1x1o + 1x2e + 1x3o");
    model->layers[0].irreps_node=irreps_create("32x0e + 8x1o + 8x2e");
    model->layers[0].linear_weight_size=48;
    model->layers[0].shortcut_weight_size=32;
    model->layers[0].denominator=1.5;

    model->layers[1].irreps_sh=irreps_create("1x1o + 1x2e + 1x3o");
    model->layers[1].irreps_sender=irreps_create("32x0e + 8x1o + 8x2e");
    model->layers[1].irreps_tp=irreps_create("16x0e + 56x1o + 16x1e + 56x2e + 24x2o + 56x3o + 16x3e + 16x4e + 8x4o + 8x5o");
    model->layers[1].irreps_receiver=irreps_create("48x0e + 64x1o + 16x1e + 64x2e + 24x2o + 56x3o + 16x3e + 16x4e + 8x4o + 8x5o");
    model->layers[1].irreps_node=irreps_create("32x0e + 8x1o + 8x1e + 8x2e + 8x2o");
    model->layers[1].linear_weight_size=2880;
    model->layers[1].shortcut_weight_size=1152;
    model->layers[1].denominator=1.5;

    model->layers[2].irreps_sh=irreps_create("1x1o + 1x2e + 1x3o");
    model->layers[2].irreps_sender=irreps_create("32x0e + 8x1o + 8x1e + 8x2e + 8x2o");
    model->layers[2].irreps_tp=irreps_create("16x0e + 16x0o + 72x1o + 40x1e + 80x2e + 48x2o + 72x3o + 40x3e + 24x4e + 24x4o +8x5o + 8x5e");
    model->layers[2].irreps_receiver=irreps_create("48x0e + 16x0o + 80x1o + 48x1e + 88x2e + 56x2o + 72x3o + 40x3e + 24x4e + 24x4o + 8x5o + 8x5e");
    model->layers[2].irreps_node=irreps_create("1x0o + 7x0e");
    model->layers[2].linear_weight_size=352;
    model->layers[2].shortcut_weight_size=224;
    model->layers[2].denominator=1.5;
    return model;
}


void model_mmap_weights(Model* model, const char* filepath) {
    FILE *file = fopen(filepath, "rd");
    fseek(file, 0, SEEK_END);
    model->file_size = ftell(file);
    fclose(file);
    model->fd = open(filepath, O_RDONLY);
    model->data = mmap(NULL, model->file_size, PROT_READ, MAP_PRIVATE, model->fd, 0);
    float* weights = model->data;
    for (int layer = 0; layer < 3; layer++ ) {
        model->layers[layer].linear_weight = weights;
        weights += model->layers[layer].linear_weight_size;
        model->layers[layer].shortcut_weight = weights;
        weights += model->layers[layer].shortcut_weight_size;
    }
}


void model_free(Model* model) {
    munmap(model->data, model->file_size);
    close(model->fd);
    for (int layer = 0; layer < 3; layer++) {
        layer_free(&model->layers[layer]);
    }
    free(model->layers);
    free(model);
}


float* model_forward(Model* model, const float pos[][3], int num_nodes) {
    Neighborlist* nl = neighborlist_create(pos, num_nodes, 1.1);
    float* node_features = (float *) malloc(num_nodes * irreps_dim(model->layers[0].irreps_sender) * sizeof(float));
    float* scatter_sum   = (float *) malloc(irreps_dim(model->layers[2].irreps_node) * sizeof(float));
    float* logits        = (float *) malloc((irreps_dim(model->layers[2].irreps_node) - 1) * sizeof(float));

    // inital node features are just 1s
    for (int i = 0; i < num_nodes; i++) {
        node_features[i] = 1.0;
    }

    // compute node features for each layer
    for (int layer = 0; layer < 3; layer++ ) {
        float* node_features_next = (float *) malloc(num_nodes * irreps_dim(model->layers[layer].irreps_node) * sizeof(float));

        layer_forward(&model->layers[layer], node_features, pos, num_nodes, nl, node_features_next);

        float* tmp = node_features;
        node_features = node_features_next;
        free(tmp);
    }

    // global sum
    for (int i = 0; i < irreps_dim(model->layers[2].irreps_node); i++ ) {
        float sum = 0;
        for (int n = 0; n < num_nodes; n++ ) {
            sum += node_features[n * irreps_dim(model->layers[2].irreps_node) + i];
        }
        scatter_sum[i] = sum;
    }


    logits[0] = scatter_sum[0] * scatter_sum[1];
    logits[1] = -scatter_sum[0] * scatter_sum[1];
    for (int i = 2; i < irreps_dim(model->layers[2].irreps_node); i++ ) {
        logits[i] = scatter_sum[i];
    }

    neighborlist_free(nl);
    free(node_features);
    free(scatter_sum);
    return logits;
}


int main(int argc, char *argv[]) {
    if (argc != 13) {
        fprintf(stderr, "usage: %s x1 y1 z1 x2 y2 z2 x3 y3 z3 x4 y4 z4\n", argv[0]);
        return 1;
    }

    const char* labels[] = {"chiral 1", "chiral 2", "square", "line", "corner", "L", "T", "zigzag"};

    int num_nodes = 4;
    float pos[4][3];
    for (int i = 0; i < 4; i++) {
        pos[i][0] = atof(argv[3 * i + 1]);
        pos[i][1] = atof(argv[3 * i + 2]);
        pos[i][2] = atof(argv[3 * i + 3]);
    }

    Model* model = model_create();
    model_mmap_weights(model, "tetris.bin");
    float* logits = model_forward(model, pos, num_nodes);

    printf("logits:\n");
    for (int i = 0; i < 8; i++) {
        printf("%-12s%.5f\n", labels[i], logits[i]);
    }

    free(logits);
    model_free(model);
    return 0;
}