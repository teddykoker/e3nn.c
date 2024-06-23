#include <math.h>
#include <stdlib.h>
#include <stdio.h>
#include <string.h>

#include "clebsch_gordan.h"
#include "e3nn.h"
#include "tp.h"

Irrep* parse_irrep_str(const char* str, int* size) {
    // parse str into an array of Irrep with length size
    *size = 1;
    for (const char* p = str; *p; p++) {
        if (*p == '+') {
            (*size)++;
        }
    }
    Irrep* irreps = (Irrep*)malloc(*size * sizeof(Irrep));

    int c, l, index = 0;
    char p;
    const char* start = str;
    while (sscanf(start, "%dx%d%c", &c, &l, &p) == 3) {
        irreps[index++] = (Irrep){ c, l, (p == 'e') ? EVEN : ODD };
        start = strchr(start, '+');
        if (!start) break;
        start++;
    }
    return irreps;
}

void tensor_product_v1(const char* irrep_str1, float* data1, const char* irrep_str2, float* data2, const char* irrep_stro, float* datao) {
    int size1, size2, sizeo;
    Irrep* irreps1 = parse_irrep_str(irrep_str1, &size1);
    Irrep* irreps2 = parse_irrep_str(irrep_str2, &size2);
    Irrep* irrepso = parse_irrep_str(irrep_stro, &sizeo);

    build_clebsch_gordan_cache();

    // Lookup table where the start of each out irrep will be in datao
    // only need to store one location per l/parity pair, it is cheap to hash by
    // l*(p+1)/2
    int out_ptrs[(L_MAX + 1) * 2] = {0};
    int ptr = 0;
    for (int i = 0; i < sizeo; i++) {
        out_ptrs[irrepso[i].l + (irrepso[i].p + 1) / 2 * L_MAX] = ptr;
        ptr += irrepso[i].c * (2 * irrepso[i].l + 1);
    }

    int ptr1 = 0;
    for (int i1 = 0; i1 < size1; i1++) {
        int l1 = irreps1[i1].l;
        int c1 = irreps1[i1].c;
        int p1 = irreps1[i1].p;
        int l1_dim = 2 * l1 + 1;

        int ptr2 = 0;
        for (int i2 = 0; i2 < size2; i2++) {
            int l2 = irreps2[i2].l;
            int c2 = irreps2[i2].c;
            int p2 = irreps2[i2].p;
            int l2_dim = 2 * l2 + 1;

            int po = p1 * p2;
            int lo_min = abs(l1 - l2);
            int lo_max = l1 + l2;

            for (int lo = lo_min; lo <= lo_max; lo++) {
                float normalize = sqrt(2 * lo + 1);
                int out_ptr = out_ptrs[lo + (po + 1) / 2 * L_MAX];

                for (int c1_idx = 0; c1_idx < c1; c1_idx++) {
                    for (int c2_idx = 0; c2_idx < c2; c2_idx++) {
                        for (int m1 = -l1; m1 <= l1; m1++) {
                            int idx1 = ptr1 + c1_idx * l1_dim + m1 + l1;
                            for (int m2 = -l2; m2 <= l2; m2++) {
                                int idx2 = ptr2 + c2_idx * l2_dim + m2 + l2;
                                for (int mo = -lo; mo <= lo; mo++) {
                                    float cg = clebsch_gordan(l1, l2, lo, m1, m2, mo);
                                    datao[out_ptr + mo + lo] += (
                                        cg * data1[idx1] * data2[idx2] * normalize
                                    );
                                } 
                            }
                        }
                        // done writing to this irrep
                        out_ptr += (2 * lo + 1);
                    }
                }
                // done writing this chunk of irreps, shift over pointer so we
                // can write new chunks to the same irrep - this happens when
                // there is more than one path to the same irrep as long as we
                // iterate through the irreps in the same order, this should be
                // functionally equivalent to the e3nn-jax implementation 
                out_ptrs[lo + (po + 1) / 2 * L_MAX] += (2 * lo + 1) * c1 * c2;
            }
            ptr2 += l2_dim * c2;
        }
        ptr1 += l1_dim * c1;
    }
    free(irreps1);
    free(irreps2);
    free(irrepso);
}

void tensor_product_v2(const char* irrep_str1, const float* data1, const char* irrep_str2, const float* data2, const char* irrep_stro, float* datao) {
    // this is the same as tensor_product above, except the inner loops over
    // m1, m2, mo are removed and replaced with lookups into the sparse
    // Clebsch-Gordan coefficients
    int size1, size2, sizeo;
    Irrep* irreps1 = parse_irrep_str(irrep_str1, &size1);
    Irrep* irreps2 = parse_irrep_str(irrep_str2, &size2);
    Irrep* irrepso = parse_irrep_str(irrep_stro, &sizeo);

    build_sparse_clebsch_gordan_cache();

    // Lookup table where the start of each out irrep will be in datao
    // only need to store one location per l/parity pair, it is cheap to hash by
    // l*(p+1)/2
    int out_ptrs[(L_MAX + 1) * 2] = {0};
    int ptr = 0;
    for (int i = 0; i < sizeo; i++) {
        out_ptrs[irrepso[i].l + (irrepso[i].p + 1) / 2 * L_MAX] = ptr;
        ptr += irrepso[i].c * (2 * irrepso[i].l + 1);
    }

    int ptr1 = 0;
    for (int i1 = 0; i1 < size1; i1++) {
        int l1 = irreps1[i1].l;
        int c1 = irreps1[i1].c;
        int p1 = irreps1[i1].p;
        int l1_dim = 2 * l1 + 1;

        int ptr2 = 0;
        for (int i2 = 0; i2 < size2; i2++) {
            int l2 = irreps2[i2].l;
            int c2 = irreps2[i2].c;
            int p2 = irreps2[i2].p;
            int l2_dim = 2 * l2 + 1;

            int po = p1 * p2;
            int lo_min = abs(l1 - l2);
            int lo_max = l1 + l2;

            for (int lo = lo_min; lo <= lo_max; lo++) {
                float normalize = sqrt(2 * lo + 1);
                int out_ptr = out_ptrs[lo + (po + 1) / 2 * L_MAX];
                SparseClebschGordanMatrix cg = sparse_clebsch_gordan(l1, l2, lo);

                for (int c1_idx = 0; c1_idx < c1; c1_idx++) {
                    int idx1 = ptr1 + c1_idx * l1_dim;
                    for (int c2_idx = 0; c2_idx < c2; c2_idx++) {
                        int idx2 = ptr2 + c2_idx * l2_dim;
                        for (int e=0; e<cg.size; e++) {
                            datao[out_ptr + cg.elements[e].m3 + lo] += (
                                cg.elements[e].c 
                                * data1[idx1 + cg.elements[e].m1 + l1] 
                                * data2[idx2 + cg.elements[e].m2 + l2] 
                                * normalize
                            );
                        }
                        // done writing to this irrep
                        out_ptr += (2 * lo + 1);
                    }
                }
                // done writing this chunk of irreps, shift over pointer so we
                // can write new chunks to the same irrep - this happens when
                // there is more than one path to the same irrep as long as we
                // iterate through the irreps in the same order, this should be
                // functionally equivalent to the e3nn-jax implementation 
                out_ptrs[lo + (po + 1) / 2 * L_MAX] += (2 * lo + 1) * c1 * c2;
            }
            ptr2 += l2_dim * c2;
        }
        ptr1 += l1_dim * c1;
    }
    free(irreps1);
    free(irreps2);
    free(irrepso);
}

void tensor_product_v3(const char* irrep_str1, const float* data1, const char* irrep_str2, const float* data2, const char* irrep_stro, float* datao) {
    // this is the same as tensor_product above, except the tensor products for
    // any l1,l2,lo are replace with a call to a precompiled version in tp.c

    int size1, size2, sizeo;
    Irrep* irreps1 = parse_irrep_str(irrep_str1, &size1);
    Irrep* irreps2 = parse_irrep_str(irrep_str2, &size2);
    Irrep* irrepso = parse_irrep_str(irrep_stro, &sizeo);

    // Lookup table where the start of each out irrep will be in datao
    // only need to store one location per l/parity pair, it is cheap to hash by
    // l*(p+1)/2
    int out_ptrs[(L_MAX + 1) * 2] = {0};
    int ptr = 0;
    for (int i = 0; i < sizeo; i++) {
        out_ptrs[irrepso[i].l + (irrepso[i].p + 1) / 2 * L_MAX] = ptr;
        ptr += irrepso[i].c * (2 * irrepso[i].l + 1);
    }

    int ptr1 = 0;
    for (int i1 = 0; i1 < size1; i1++) {
        int l1 = irreps1[i1].l;
        int c1 = irreps1[i1].c;
        int p1 = irreps1[i1].p;
        int l1_dim = 2 * l1 + 1;

        int ptr2 = 0;
        for (int i2 = 0; i2 < size2; i2++) {
            int l2 = irreps2[i2].l;
            int c2 = irreps2[i2].c;
            int p2 = irreps2[i2].p;
            int l2_dim = 2 * l2 + 1;

            int po = p1 * p2;
            int lo_min = abs(l1 - l2);
            int lo_max = l1 + l2;

            for (int lo = lo_min; lo <= lo_max; lo++) {
                // float normalize = sqrt(2 * lo + 1);
                int out_ptr = out_ptrs[lo + (po + 1) / 2 * L_MAX];
                for (int c1_idx = 0; c1_idx < c1; c1_idx++) {
                    int idx1 = ptr1 + c1_idx * l1_dim;
                    for (int c2_idx = 0; c2_idx < c2; c2_idx++) {
                        int idx2 = ptr2 + c2_idx * l2_dim;
                        tp(l1, l2, lo, data1 + idx1, data2 + idx2, datao + out_ptr);
                        // done writing to this irrep
                        out_ptr += (2 * lo + 1);
                    }
                }
                // done writing this chunk of irreps, shift over pointer so we
                // can write new chunks to the same irrep - this happens when
                // there is more than one path to the same irrep as long as we
                // iterate through the irreps in the same order, this should be
                // functionally equivalent to the e3nn-jax implementation 
                out_ptrs[lo + (po + 1) / 2 * L_MAX] += (2 * lo + 1) * c1 * c2;
            }
            ptr2 += l2_dim * c2;
        }
        ptr1 += l1_dim * c1;
    }
    free(irreps1);
    free(irreps2);
    free(irrepso);
}
