#include <stdlib.h>
#include <math.h>
#include <assert.h>
#include <float.h>
#include "uwnet.h"


// Run a maxpool layer on input
// layer l: pointer to layer to run
// matrix in: input to layer
// returns: the result of running the layer
matrix forward_maxpool_layer(layer l, matrix in)
{
    int outw = (l.width-1)/l.stride + 1;
    int outh = (l.height-1)/l.stride + 1;
    matrix out = make_matrix(in.rows, outw*outh*l.channels);

    // TODO: 6.1 - iterate over the input and fill in the output with max values
    int r, c;
    for (r = 0; r < out.rows; r++) {
        for (c = 0; c < out.cols; c++) {
            // indices into the image
            int i = (c % (outw * outh)) / outw * l.stride;
            int j = (c % outw) * l.stride;
            int channel = c / (outw * outh);
            int channel_offset = channel * l.width * l.height;

            // get the value at the center of the filter
            float max = in.data[r * in.cols + (i * l.width + j) + channel_offset];
            for (int r_off = -((l.size - 1) / 2); r_off <= (l.size - 1) / 2; r_off++) {
                for (int c_off = -((l.size - 1) / 2); c_off <= (l.size - 1) / 2; c_off++) {
                    int curr_i = i + r_off;
                    int curr_j = j + c_off;
                    if (curr_i < l.height && curr_j < l.width) {
                        int col_offset = curr_i * l.width + curr_j;
                        if (in.data[r * in.cols + col_offset + channel_offset] > max) {
                            max = in.data[r * in.cols + col_offset + channel_offset];
                        }
                    }
                }
            }
            out.data[r * out.cols + c] = max;
        }
    }

    l.in[0] = in;
    free_matrix(l.out[0]);
    l.out[0] = out;
    free_matrix(l.delta[0]);
    l.delta[0] = make_matrix(out.rows, out.cols);
    return out;
}

// Run a maxpool layer backward
// layer l: layer to run
// matrix prev_delta: error term for the previous layer
void backward_maxpool_layer(layer l, matrix prev_delta)
{
    matrix in    = l.in[0];
    matrix out   = l.out[0];
    matrix delta = l.delta[0];

    int outw = (l.width-1)/l.stride + 1;
    int outh = (l.height-1)/l.stride + 1;

    // TODO: 6.2 - find the max values in the input again and fill in the
    // corresponding delta with the delta from the output. This should be
    // similar to the forward method in structure.
    int r, c;
    for (r = 0; r < out.rows; r++) {
        for (c = 0; c < out.cols; c++) {
            // indices into the image
            int i = (c % (outw * outh)) / outw * l.stride;
            int j = (c % outw) * l.stride;
            int max_i = i;
            int max_j = j;
            int channel = c / (outw * outh);
            int channel_offset = channel * l.width * l.height;

            // get the value at the center of the filter
            float max = in.data[r * in.cols + (i * l.width + j) + channel_offset];
            for (int r_off = -((l.size - 1) / 2); r_off <= (l.size - 1) / 2; r_off++) {
                for (int c_off = -((l.size - 1) / 2); c_off <= (l.size - 1) / 2; c_off++) {
                    int curr_i = i + r_off;
                    int curr_j = j + c_off;
                    if (curr_i < l.height && curr_j < l.width) {
                        int col_offset = curr_i * l.width + curr_j;
                        if (in.data[r * in.cols + col_offset + channel_offset] > max) {
                            max = in.data[r * in.cols + col_offset + channel_offset];
                            max_i = curr_i;
                            max_j = curr_j;
                        }
                    }
                }
            }
            prev_delta.data[r * prev_delta.cols + (max_i * l.width + max_j) + channel_offset] += delta.data[r * delta.cols + c];
        }
    }

    /*int r, c;
    for (r = 0; r < out.rows; r++) {
        for (c = 0; c < out.cols; c++) {
            int i = (c % (outw * outh)) / outw * l.stride;
            int j = (c % outw) * l.stride;
            int channel = c / (outw * outh);
            float max = in.data[r * in.cols + (i * outw + j) + channel * l.width * l.height];
            int max_i = i;
            int max_j = j;
            //printf("%d\n", r * in.cols + (i * outw + j) + channel * l.width * l.height);
            //printf("starting max: %d pulled from r: %d, c: %d\n", max, r, (i * outw + j));
            //printf("i: %d, j: %d, channel: %d, stride: %d, max: %d, size: %d\n", i, j, channel, l.stride, max, l.size);
            for (int r_off = -((l.size - 1) / 2); r_off <= (l.size - 1) / 2; r_off++) {
                for (int c_off = -((l.size - 1) / 2); c_off <= (l.size - 1) / 2; c_off++) {
                    int curr_i = i + r_off;
                    int curr_j = j + c_off;
                    //printf("\tcurr_i: %d, curr_j: %d\n", curr_i, curr_j);
                    if (curr_i < l.height && curr_j < l.width) {
                        int idx = curr_i * outw + curr_j;
                        if (in.data[r * in.cols + idx + channel * l.width * l.height] > max) {
                            max = in.data[r * in.cols + idx + channel * l.width * l.height];
                            max_i = curr_i;
                            max_j = curr_j;
                        }
                    }
                }
            }
            prev_delta.data[max_i * prev_delta.cols + max_j + channel * outw * outh] += delta.data[r * delta.cols + c];
            //printf("w: %d, h: %d, channels: %d, size: %d\n", l.width, l.height, l.channels, l.size);
            //printf("i: %d, j: %d, channel: %d, stride: %d, max: %d\n", i, j, channel, l.stride, max);
        }
    }*/
}

// Update maxpool layer
// Leave this blank since maxpool layers have no update
void update_maxpool_layer(layer l, float rate, float momentum, float decay)
{
}

// Make a new maxpool layer
// int w: width of input image
// int h: height of input image
// int c: number of channels
// int size: size of maxpool filter to apply
// int stride: stride of operation
layer make_maxpool_layer(int w, int h, int c, int size, int stride)
{
    layer l = {0};
    l.width = w;
    l.height = h;
    l.channels = c;
    l.size = size;
    l.stride = stride;
    l.in = calloc(1, sizeof(matrix));
    l.out = calloc(1, sizeof(matrix));
    l.delta = calloc(1, sizeof(matrix));
    l.forward  = forward_maxpool_layer;
    l.backward = backward_maxpool_layer;
    l.update   = update_maxpool_layer;
    return l;
}

