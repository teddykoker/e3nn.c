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
    Irreps* irreps_in;
    Irreps* irreps_tp;
    Irreps* irreps_message;
    Irreps* irreps_out;
    int linear_weight_size;
    float* linear_weight;
    int shortcut_weight_size;
    float* shortcut_weight;
    float denominator;
} Layer;


Layer layer_create(const Irreps* irreps_in, const Irreps* irreps_sh, const Irreps* irreps_out) {
    // determine intermediary irreps and weight sizes for layer
    Layer layer;
    layer.irreps_sh=irreps_copy(irreps_sh);
    layer.irreps_in=irreps_copy(irreps_in);
    layer.irreps_tp=irreps_tensor_product(layer.irreps_in, layer.irreps_sh);
    layer.irreps_message=irreps_concatenate(layer.irreps_in, layer.irreps_tp);
    layer.irreps_out=irreps_linear(layer.irreps_message, irreps_out, false);
    layer.linear_weight_size=linear_weight_size(layer.irreps_message, layer.irreps_out);
    layer.shortcut_weight_size=linear_weight_size(layer.irreps_in, layer.irreps_out);
    layer.denominator=1.5;
    return layer;
}


// the main forward pass for each layer
void layer_forward(Layer* layer, const float* node_input, const float pos[][3], int num_node, const Neighborlist* nl, float* node_output) {
    // intermediary storage for spherical harmonics, tensor product, messages,
    // received messages, linear output, and shortcut output
    float* sh           = (float *) malloc(irreps_dim(layer->irreps_sh) * sizeof(float));
    float* tp           = (float *) malloc(irreps_dim(layer->irreps_tp) * sizeof(float));
    float* messages     = (float *) malloc(nl->size * irreps_dim(layer->irreps_message) * sizeof(float));
    float* receive      = (float *) malloc(irreps_dim(layer->irreps_message) * sizeof(float));
    float* linear_out   = (float *) malloc(irreps_dim(layer->irreps_out) * sizeof(float));
    float* shortcut_out = (float *) malloc(irreps_dim(layer->irreps_out) * sizeof(float));

    // compute messages
    for (int edge = 0; edge < nl->size; edge++) {
        int s = nl->senders[edge];
        int r = nl->receivers[edge];
        spherical_harmonics(layer->irreps_sh, 
                            pos[r][0] - pos[s][0], 
                            pos[r][1] - pos[s][1], 
                            pos[r][2] - pos[s][2],
                            sh);
        tensor_product(layer->irreps_in, 
                       &node_input[s * irreps_dim(layer->irreps_in)],
                       layer->irreps_sh,
                       sh,
                       layer->irreps_tp,
                       tp);
        concatenate(layer->irreps_in, 
                    &node_input[s * irreps_dim(layer->irreps_in)],
                    layer->irreps_tp,
                    tp,
                    &messages[edge * irreps_dim(layer->irreps_message)]);
    }

    // aggregate messages and update nodes
    for (int node = 0; node < num_node; node++ ) {
        // zero out received messages
        for (int i = 0; i < irreps_dim(layer->irreps_message); i++) {
            receive[i] = 0;
        }
        // zero linear out and shortcut out
        for (int i = 0; i < irreps_dim(layer->irreps_out); i++) {
            linear_out[i] = 0;
            shortcut_out[i] = 0;
        }

        // sum messages into receive
        for (int edge = 0; edge < nl->size; edge++) {
            if (node == nl->receivers[edge]) {
                for (int i = 0; i < irreps_dim(layer->irreps_message); i++) {
                    receive[i] += messages[edge * irreps_dim(layer->irreps_message) + i];
                }
            }
        }

        // divide by denominator
        for (int i = 0; i < irreps_dim(layer->irreps_message); i++) {
            receive[i] /= layer->denominator;
        }

        linear(layer->irreps_message, 
               receive, 
               layer->linear_weight, 
               layer->irreps_out,
               linear_out);

        linear(layer->irreps_in,
               &node_input[node * irreps_dim(layer->irreps_in)],
               layer->shortcut_weight,
               layer->irreps_out,
               shortcut_out);

        // add linear + shortcut
        for (int i = 0; i < irreps_dim(layer->irreps_out); i++) {
            node_output[node * irreps_dim(layer->irreps_out) + i] = linear_out[i] + shortcut_out[i];
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
    irreps_free(layer->irreps_in);
    irreps_free(layer->irreps_tp);
    irreps_free(layer->irreps_message);
    irreps_free(layer->irreps_out);
}


typedef struct {
    Layer* layers;
    float* data;      // mmap data pointer
    int fd;           // mmap file descriptor 
    size_t file_size; // mmap file size
} Model;


Model* model_create(void) {
    Irreps* irreps_hidden = irreps_create("32x0e + 32x0o + 8x1o + 8x1e + 8x2e + 8x2o");
    Irreps* irreps_out = irreps_create("1x0o + 7x0e");
    Irreps* irreps_sh = irreps_create("1x1o + 1x2e + 1x3o");
    Irreps* irreps_input = irreps_create("1x0e");

    Model* model = (Model *) malloc(sizeof(Model));
    model->layers = (Layer *) malloc(3 * sizeof(Layer));

    model->layers[0] = layer_create(irreps_input, irreps_sh, irreps_hidden);
    model->layers[1] = layer_create(model->layers[0].irreps_out, irreps_sh, irreps_hidden);
    model->layers[2] = layer_create(model->layers[1].irreps_out, irreps_sh, irreps_out);

    irreps_free(irreps_hidden);
    irreps_free(irreps_out);
    irreps_free(irreps_sh);
    irreps_free(irreps_input);
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
    float* node_features = (float *) malloc(num_nodes * irreps_dim(model->layers[0].irreps_in) * sizeof(float));
    float* scatter_sum   = (float *) malloc(irreps_dim(model->layers[2].irreps_out) * sizeof(float));
    float* logits        = (float *) malloc((irreps_dim(model->layers[2].irreps_out) - 1) * sizeof(float));

    // inital node features are just 1s
    for (int i = 0; i < num_nodes; i++) {
        node_features[i] = 1.0;
    }

    // compute node features for each layer
    for (int layer = 0; layer < 3; layer++ ) {
        float* node_features_next = (float *) malloc(num_nodes * irreps_dim(model->layers[layer].irreps_out) * sizeof(float));

        layer_forward(&model->layers[layer], node_features, pos, num_nodes, nl, node_features_next);

        float* tmp = node_features;
        node_features = node_features_next;
        free(tmp);
    }

    // global sum
    for (int i = 0; i < irreps_dim(model->layers[2].irreps_out); i++ ) {
        float sum = 0;
        for (int n = 0; n < num_nodes; n++ ) {
            sum += node_features[n * irreps_dim(model->layers[2].irreps_out) + i];
        }
        scatter_sum[i] = sum;
    }


    logits[0] = scatter_sum[0] * scatter_sum[1];
    logits[1] = -scatter_sum[0] * scatter_sum[1];
    for (int i = 2; i < irreps_dim(model->layers[2].irreps_out); i++ ) {
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