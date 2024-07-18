#include <math.h>
#include <stdlib.h>
#include <stdio.h>
#include <string.h>
#include <stdbool.h>
#include <assert.h>

#include "clebsch_gordan.h"
#include "e3nn.h"
#include "tp.h"

// index of spherical harmonic (l, m) in array
#define SH_IDX(l, m) ((l) * (l) + (m) + (l))

Irreps* irreps_create(const char* str) {
    // parse str into an array of Irrep with length size
    Irreps* irreps = (Irreps*) malloc(sizeof(Irreps));
    irreps->size = 1;
    for (const char* p = str; *p; p++) {
        if (*p == '+') {
            irreps->size++;
        }
    }
    irreps->irreps = (Irrep*) malloc(irreps->size * sizeof(Irrep));

    int c, l, index = 0;
    char p;
    const char* start = str;
    while (sscanf(start, "%dx%d%c", &c, &l, &p) == 3) {
        irreps->irreps[index++] = (Irrep){ c, l, (p == 'e') ? EVEN : ODD };
        start = strchr(start, '+');
        if (!start) break;
        start++;
    }
    return irreps;
}


Irreps* irreps_tensor_product(const Irreps* irreps_1, const Irreps* irreps_2) {
    // Lookup table for channel count for each irrep in output
    // indexed by (l + (p+1)/2 * (L_MAX + 1))
    int c_count[(L_MAX + 1) * 2] = {0};
    for (int i1 = 0; i1 < irreps_1->size; i1++) {
        int l1 = irreps_1->irreps[i1].l;
        int c1 = irreps_1->irreps[i1].c;
        int p1 = irreps_1->irreps[i1].p;
        for (int i2 = 0; i2 < irreps_2->size; i2++) {
            int l2 = irreps_2->irreps[i2].l;
            int c2 = irreps_2->irreps[i2].c;
            int p2 = irreps_2->irreps[i2].p;
            int po = p1 * p2;
            int lo_min = abs(l1 - l2);
            int lo_max = l1 + l2;
            for (int lo = lo_min; lo <= lo_max; lo++) {
                c_count[lo + (po + 1) / 2 * (L_MAX + 1)] += c1 * c2;
            }
        }
    }
    Irreps* irreps = (Irreps*) malloc(sizeof(Irreps));
    irreps->size = 0;
    for (int i = 0; i < (L_MAX + 1) * 2; i++) {
        if (c_count[i] != 0) {
            irreps -> size++;
        }
    }
    irreps->irreps = (Irrep*) malloc(irreps->size * sizeof(Irrep));
    int index = 0;
    for (int l = 0; l <= L_MAX; l++) {
        int p = l % 2 == 0 ? EVEN : ODD;
        int c = c_count[l + (p + 1) / 2 * (L_MAX + 1)];
        if (c > 0) {
            irreps->irreps[index++] = (Irrep){ c, l, p };
        }
        p *= -1;
        c = c_count[l + (p + 1) / 2 * (L_MAX + 1)];
        if (c > 0) {
            irreps->irreps[index++] = (Irrep){ c, l, p };
        }
    }
    return irreps;
}


int irrep_compare(const Irrep* i1, const Irrep* i2) {
    if (i1->l == i2->l) {
        if (i1->p == i2->p) {
            return 0;
        } else if ((i1->l % 2 == 0 && i1->p == EVEN) || 
                   (i1->l % 2 == 1 && i1->p == ODD)) {
            return -1;
        } else {
            return 1;
        }
    } else {
        return i1->l - i2->l;
    }
}


bool irreps_is_sorted(const Irreps* irreps) {
    if (irreps->size < 2) { return true; }
    for (int i = 1; i < irreps->size; i++) {
        if (irrep_compare(&irreps->irreps[i-1], &irreps->irreps[i]) >= 0) {
            return false;
        }
    }
    return true;
}


Irreps* irreps_concatenate(const Irreps* irreps_1, const Irreps* irreps_2) {
    assert(irreps_is_sorted(irreps_1));
    assert(irreps_is_sorted(irreps_2));
    Irreps* irreps = (Irreps*) malloc(sizeof(Irreps));
    irreps->size = 0;
    // allocate for worst case and realloc later
    irreps->irreps = (Irrep*) malloc((irreps_1->size + irreps_2->size) * sizeof(Irrep));
    int i1 = 0, i2 = 0;
    while (i1 < irreps_1->size || i2 < irreps_2->size) {
        Irrep write_irr;
        if (
            (i1 < irreps_1->size) &&
            (i2 >= irreps_2->size || irrep_compare(&irreps_1->irreps[i1], &irreps_2->irreps[i2]) <= 0)) {
            write_irr = irreps_1->irreps[i1++];
        } else {
            write_irr = irreps_2->irreps[i2++];
        }

        if (irreps->size == 0 || irrep_compare(&write_irr, &irreps->irreps[irreps->size - 1]) > 0) {
            irreps->irreps[irreps->size] = write_irr;
            irreps->size += 1;
        } else {
            irreps->irreps[irreps->size - 1].c += write_irr.c;
        }
    }
    irreps->irreps = (Irrep*) realloc(irreps->irreps, irreps->size * sizeof(Irrep));
    return irreps;
}


void irreps_free(Irreps* irreps) {
    free(irreps->irreps);
    free(irreps);
}


int irrep_dim(const Irrep* irr) {
    return irr->c * (2 * irr->l + 1);
}


void irreps_print(const Irreps* irreps) {
    for (int i = 0; i < irreps->size; i++) {
        printf("%dx%d%s", irreps->irreps[i].c, irreps->irreps[i].l, irreps->irreps[i].p == EVEN ? "e" : "o");
        if (i < irreps->size - 1) {
            printf(" + ");
        }
    }
    printf("\n");
}

    
int irreps_dim(const Irreps* irreps) {
    int dim = 0;
    for (int i = 0; i < irreps->size; i++) {
        dim += irrep_dim(&irreps->irreps[i]);
    }
    return dim;
}




void tensor_product_v1(const Irreps* irreps_1, const float* data_1, const Irreps* irreps_2, const float* data_2, const Irreps* irreps_o, float* data_o) {
    build_clebsch_gordan_cache();

    // Lookup table where the start of each out irrep will be in datao
    // only need to store one location per l/parity pair, it is cheap to hash
    // by (l + (p+1)/2 * (L_MAX + 1))
    int out_ptrs[(L_MAX + 1) * 2] = {0};
    int ptr = 0;
    for (int i = 0; i < irreps_o->size; i++) {
        out_ptrs[irreps_o->irreps[i].l + (irreps_o->irreps[i].p + 1) / 2 * (L_MAX + 1)] = ptr;
        ptr += irreps_o->irreps[i].c * (2 * irreps_o->irreps[i].l + 1);
    }

    int ptr1 = 0;
    for (int i1 = 0; i1 < irreps_1->size; i1++) {
        int l1 = irreps_1->irreps[i1].l;
        int c1 = irreps_1->irreps[i1].c;
        int p1 = irreps_1->irreps[i1].p;
        int l1_dim = 2 * l1 + 1;

        int ptr2 = 0;
        for (int i2 = 0; i2 < irreps_2->size; i2++) {
            int l2 = irreps_2->irreps[i2].l;
            int c2 = irreps_2->irreps[i2].c;
            int p2 = irreps_2->irreps[i2].p;
            int l2_dim = 2 * l2 + 1;

            int po = p1 * p2;
            int lo_min = abs(l1 - l2);
            int lo_max = l1 + l2;

            for (int lo = lo_min; lo <= lo_max; lo++) {
                float normalize = sqrt(2 * lo + 1);
                int out_ptr = out_ptrs[lo + (po + 1) / 2 * (L_MAX + 1)];

                for (int c1_idx = 0; c1_idx < c1; c1_idx++) {
                    for (int c2_idx = 0; c2_idx < c2; c2_idx++) {
                        for (int m1 = -l1; m1 <= l1; m1++) {
                            int idx1 = ptr1 + c1_idx * l1_dim + m1 + l1;
                            for (int m2 = -l2; m2 <= l2; m2++) {
                                int idx2 = ptr2 + c2_idx * l2_dim + m2 + l2;
                                for (int mo = -lo; mo <= lo; mo++) {
                                    float cg = clebsch_gordan(l1, l2, lo, m1, m2, mo);
                                    data_o[out_ptr + mo + lo] += (
                                        cg * data_1[idx1] * data_2[idx2] * normalize
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
                out_ptrs[lo + (po + 1) / 2 * (L_MAX + 1)] += (2 * lo + 1) * c1 * c2;
            }
            ptr2 += l2_dim * c2;
        }
        ptr1 += l1_dim * c1;
    }
}


void tensor_product_v2(const Irreps* irreps_1, const float* data_1, const Irreps* irreps_2, const float* data_2, const Irreps* irreps_o, float* data_o) {
    // this is the same as tensor_product above, except the inner loops over
    // m1, m2, mo are removed and replaced with lookups into the sparse
    // Clebsch-Gordan coefficients
    build_sparse_clebsch_gordan_cache();

     // Lookup table where the start of each out irrep will be in datao
    // only need to store one location per l/parity pair, it is cheap to hash by
    // by (l+ (p+1)/2 * (L_MAX + 1))
    int out_ptrs[(L_MAX + 1) * 2] = {0};
    int ptr = 0;
    for (int i = 0; i < irreps_o->size; i++) {
        out_ptrs[irreps_o->irreps[i].l + (irreps_o->irreps[i].p + 1) / 2 * (L_MAX + 1)] = ptr;
        ptr += irreps_o->irreps[i].c * (2 * irreps_o->irreps[i].l + 1);
    }

    int ptr1 = 0;
    for (int i1 = 0; i1 < irreps_1->size; i1++) {
        int l1 = irreps_1->irreps[i1].l;
        int c1 = irreps_1->irreps[i1].c;
        int p1 = irreps_1->irreps[i1].p;
        int l1_dim = 2 * l1 + 1;

        int ptr2 = 0;
        for (int i2 = 0; i2 < irreps_2->size; i2++) {
            int l2 = irreps_2->irreps[i2].l;
            int c2 = irreps_2->irreps[i2].c;
            int p2 = irreps_2->irreps[i2].p;
            int l2_dim = 2 * l2 + 1;

            int po = p1 * p2;
            int lo_min = abs(l1 - l2);
            int lo_max = l1 + l2;

            for (int lo = lo_min; lo <= lo_max; lo++) {
                float normalize = sqrt(2 * lo + 1);
                int out_ptr = out_ptrs[lo + (po + 1) / 2 * (L_MAX + 1)];
                SparseClebschGordanMatrix cg = sparse_clebsch_gordan(l1, l2, lo);

                for (int c1_idx = 0; c1_idx < c1; c1_idx++) {
                    int idx1 = ptr1 + c1_idx * l1_dim;
                    for (int c2_idx = 0; c2_idx < c2; c2_idx++) {
                        int idx2 = ptr2 + c2_idx * l2_dim;
                        for (int e=0; e<cg.size; e++) {
                            data_o[out_ptr + cg.elements[e].m3 + lo] += (
                                cg.elements[e].c 
                                * data_1[idx1 + cg.elements[e].m1 + l1] 
                                * data_2[idx2 + cg.elements[e].m2 + l2] 
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
                out_ptrs[lo + (po + 1) / 2 * (L_MAX + 1)] += (2 * lo + 1) * c1 * c2;
            }
            ptr2 += l2_dim * c2;
        }
        ptr1 += l1_dim * c1;
    }
}


void tensor_product_v3(const Irreps* irreps_1, const float* data_1, const Irreps* irreps_2, const float* data_2, const Irreps* irreps_o, float* data_o) {
    // this is the same as tensor_product above, except the tensor products for
    // any l1,l2,lo are replace with a call to a precompiled version in tp.c

    // Lookup table where the start of each out irrep will be in datao
    // only need to store one location per l/parity pair, it is cheap to hash
    // by (l + (p+1)/2 * (L_MAX + 1))
    int out_ptrs[(L_MAX + 1) * 2] = {0};
    int ptr = 0;
    for (int i = 0; i < irreps_o->size; i++) {
        out_ptrs[irreps_o->irreps[i].l + (irreps_o->irreps[i].p + 1) / 2 * (L_MAX + 1)] = ptr;
        ptr += irreps_o->irreps[i].c * (2 * irreps_o->irreps[i].l + 1);
    }

    int ptr1 = 0;
    for (int i1 = 0; i1 < irreps_1->size; i1++) {
        int l1 = irreps_1->irreps[i1].l;
        int c1 = irreps_1->irreps[i1].c;
        int p1 = irreps_1->irreps[i1].p;
        int l1_dim = 2 * l1 + 1;

        int ptr2 = 0;
        for (int i2 = 0; i2 < irreps_2->size; i2++) {
            int l2 = irreps_2->irreps[i2].l;
            int c2 = irreps_2->irreps[i2].c;
            int p2 = irreps_2->irreps[i2].p;
            int l2_dim = 2 * l2 + 1;

            int po = p1 * p2;
            int lo_min = abs(l1 - l2);
            int lo_max = l1 + l2;

            for (int lo = lo_min; lo <= lo_max; lo++) {
                // float normalize = sqrt(2 * lo + 1);
                int out_ptr = out_ptrs[lo + (po + 1) / 2 * (L_MAX + 1)];
                for (int c1_idx = 0; c1_idx < c1; c1_idx++) {
                    int idx1 = ptr1 + c1_idx * l1_dim;
                    for (int c2_idx = 0; c2_idx < c2; c2_idx++) {
                        int idx2 = ptr2 + c2_idx * l2_dim;
                        tp(l1, l2, lo, data_1 + idx1, data_2 + idx2, data_o + out_ptr);
                        // done writing to this irrep
                        out_ptr += (2 * lo + 1);
                    }
                }
                // done writing this chunk of irreps, shift over pointer so we
                // can write new chunks to the same irrep - this happens when
                // there is more than one path to the same irrep as long as we
                // iterate through the irreps in the same order, this should be
                // functionally equivalent to the e3nn-jax implementation 
                out_ptrs[lo + (po + 1) / 2 * (L_MAX + 1)] += (2 * lo + 1) * c1 * c2;
            }
            ptr2 += l2_dim * c2;
        }
        ptr1 += l1_dim * c1;
    }
}


void spherical_harmonics(const Irreps* irreps, const float x, const float y, const float z, float* out) {
    int lmax = 0;
    for (int i = 0; i < irreps->size; i++) { lmax = MAX(lmax, irreps->irreps[i].l); }

    float r = sqrt(x * x + y * y + z * z);

    // NOTE: e3nn uses the following definitions for Euler angles:
    //   theta = acos(y), phi = atan2(x, z)
    float phi = atan2(x, z);

    // See note above, in the equations referenced they use x = cos(theta) which
    // is y in our coordinate system
    float x_ = (r != 0.0) ? (y / r) : 0;

    // Compute Legendre polynomials, see:
    // * Press, W.H., Teukolsky, S.A., Vetterling, W.T. and Flannery, B.P., 1992.
    //   Numerical recipes in C (pp. 252-254). New York, NY: Cambridge university
    //   press.
    // * https://github.com/chrr/libECP/blob/master/src/spherical_harmonics.c
    float* P = (float*) malloc((lmax + 1) * (lmax + 1) * sizeof(float));
    float somx2 = sqrt((1.0 - x_) * (1.0 + x_));
    float somx2m = 1.0; // store \sqrt(1 - x^2)^{1/m}
    for (int l = 0; l <= lmax; l++) {
        int m = l;
        if (l == 0) {
            P[SH_IDX(l, m)] = 1.0;
        } else {
            P[SH_IDX(l, m)] = somx2m * dfactorial(2 * m - 1);
            m = l - 1;
            P[SH_IDX(l, m)] = x_ * (2 * m + 1) * P[SH_IDX(l - 1, m)];
            if (l > 1) {
                for (m = 0; m <= l - 2; m++) {
                    P[SH_IDX(l, m)] = (
                        x_ * (2 * l - 1) * P[SH_IDX(l - 1, m)]
                        - (l + m - 1) * P[SH_IDX(l - 2, m)]
                    ) / (l - m);
                }
            }
        }
        somx2m *= somx2;
    }
    
    // component normalization
    for (int l = 0; l <= lmax; l++) {
        float norm = sqrt(2.0 * (2.0 * l + 1.0));
        // TODO: option for integral normalization, which would be:
        // float norm = sqrt((2.0 * l + 1.0) / (2.0 * M_PI));
        P[SH_IDX(l, 0)] *= norm;
        for (int m = 1; m <= l; m++) {
            P[SH_IDX(l, m)] *= sqrt(factorial(l - m) / factorial(l + m)) * norm;
        }
    }

    // precompute sin(m * phi) and cos(m * phi)
    float sin_mphi[lmax + 1];
    float cos_mphi[lmax + 1];
    if (lmax > 0) {
        sin_mphi[1] = sin(phi);
        cos_mphi[1] = cos(phi);
        for (int m = 2; m <= lmax; m++) {
            sin_mphi[m] = sin_mphi[1] * cos_mphi[m - 1] + cos_mphi[1] * sin_mphi[m - 1];
            cos_mphi[m] = cos_mphi[1] * cos_mphi[m - 1] - sin_mphi[1] * sin_mphi[m - 1];
        }
    }

    int ptr = 0;
    for (int i = 0; i < irreps->size; i++) {
        int l = irreps->irreps[i].l;
        int c = irreps->irreps[i].c;
        for (int cc = 0; cc < c; cc++) {
            for (int m = -l; m <= l; m++) {
                if (m == 0) {
                    out[ptr + l + m] = P[SH_IDX(l, m)] / sqrt(2.0);
                } else if (m < 0) {
                    out[ptr + l + m] = P[SH_IDX(l, -m)] * sin_mphi[-m];
                } else if (m > 0) {
                    out[ptr + l + m] = P[SH_IDX(l, m)] * cos_mphi[m];
                }
            }
            ptr += (2 * l + 1);
        }
    }
    free(P);
}


void linear(const Irreps* irreps_in, const float* input, const float* weight, const Irreps* irreps_out, float* out) {
    int w_ptr = 0;
    int in_ptr = 0;

    for (int i_in = 0; i_in < irreps_in->size; i_in++) {
        int out_ptr = 0;
        for (int i_out = 0; i_out < irreps_out->size; i_out++) {
            // find matching output irrep - could be done in separate loop if too costly
            if (irreps_in->irreps[i_in].l == irreps_out->irreps[i_out].l && irreps_in->irreps[i_in].p == irreps_out->irreps[i_out].p) {
                int l = irreps_in->irreps[i_in].l;
                int dim = 2 * l + 1;
                int in_c = irreps_in->irreps[i_in].c;
                int out_c = irreps_out->irreps[i_out].c;
                float norm = sqrt(1.0 / in_c);

                for (int j = 0; j < in_c; j++) {
                    for (int m = -l; m <= l; m++) {
                        for (int i = 0; i < out_c; i++) {
                            out[out_ptr + m + l + i * dim] += (
                                input[in_ptr + m + l + j * dim]
                                * weight[w_ptr + i + j * out_c]
                                * norm
                            );
                        }
                    }
                }
                // increment weight pointer to next matrix
                w_ptr += in_c * out_c;
                break;
            }
            out_ptr += (irreps_out->irreps[i_out].l * 2 + 1) * irreps_out->irreps[i_out].c;
        }
        in_ptr += (irreps_in->irreps[i_in].l * 2 + 1) * irreps_in->irreps[i_in].c;
    }
}


void concatenate(const Irreps* irreps_1, const float* data_1, const Irreps* irreps_2, const float* data_2, float* data_o) {
    assert(irreps_is_sorted(irreps_1));
    assert(irreps_is_sorted(irreps_2));
    int inc_1 = 0, inc_2 = 0, inc_o = 0;
    int i1 = 0, i2 = 0;
    while (i1 < irreps_1->size || i2 < irreps_2->size) {
        if (
            (i1 < irreps_1->size) &&
            (i2 >= irreps_2->size || irrep_compare(&irreps_1->irreps[i1], &irreps_2->irreps[i2]) <= 0)) {
            int dim = irrep_dim(&irreps_1->irreps[i1]);
            memcpy(data_o + inc_o, data_1 + inc_1, sizeof(float) * dim);
            inc_1 += dim;
            inc_o += dim;
            i1++;
        } else {
            int dim = irrep_dim(&irreps_2->irreps[i2]);
            memcpy(data_o + inc_o, data_2 + inc_2, sizeof(float) * dim);
            inc_2 += dim;
            inc_o += dim;
            i2++;
        }
    }
}