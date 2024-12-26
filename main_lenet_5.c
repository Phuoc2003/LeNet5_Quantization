#define _CRT_SECURE_NO_WARNINGS
#include <stdio.h>
#include <stdlib.h>
#include "tensor.h"
#include <math.h>
#include <stdint.h>

void relu_int8(int8_t* x, int size) {
    for (int i = 0; i < size; i++) {
        if (x[i] < 0) {
            x[i] = 0;
        }
    }
}

void softmax(float* x, float* output, int size) {
    float max = x[0];
    for (int i = 1; i < size; i++) {
        if (x[i] > max) {
            max = x[i];
        }
    }

    float sum = 0;
    for (int i = 0; i < size; i++) {
        output[i] = exp(x[i] - max);
        sum += output[i];
    }

    for (int i = 0; i < size; i++) {
        output[i] /= sum;
    }
}

void Prediction(float image[28][28],
    int8_t w_conv1[6][1][1],
    int8_t w_conv2[16][6][5][5],
    int8_t w_fc1[120][400],
    int8_t w_fc2[84][120],
    int8_t w_fc3[10][84],
    int32_t b_conv1[6],
    int32_t b_conv2[16],
    int32_t b_fc1[120],
    int32_t b_fc2[84],
    int32_t b_fc3[10],
    float probs[10]) {

    // Convert input image to int8
    int8_t image_int8[28][28];
    for (int i = 0; i < 28; i++) {
        for (int j = 0; j < 28; j++) {
            image_int8[i][j] = (int8_t)(image[i][j] * 127.0f);
        }
    }

    // Conv1 layer
    int8_t conv1_out[6][28][28] = { 0 };
    for (int c = 0; c < 6; c++) {
        for (int h = 0; h < 28; h++) {
            for (int w = 0; w < 28; w++) {
                int32_t temp = (int32_t)image_int8[h][w] * w_conv1[c][0][0] + b_conv1[c];
                temp = temp > 127 ? 127 : (temp < -128 ? -128 : temp);
                conv1_out[c][h][w] = (int8_t)temp;
            }
        }
    }

    // ReLU
    for (int c = 0; c < 6; c++) {
        relu_int8(&conv1_out[c][0][0], 28 * 28);
    }

    // Pool1
    int8_t pool1_out[6][14][14] = { 0 };
    for (int c = 0; c < 6; c++) {
        for (int h = 0; h < 14; h++) {
            for (int w = 0; w < 14; w++) {
                int32_t sum = (int32_t)conv1_out[c][2 * h][2 * w] +
                    conv1_out[c][2 * h + 1][2 * w] +
                    conv1_out[c][2 * h][2 * w + 1] +
                    conv1_out[c][2 * h + 1][2 * w + 1];
                pool1_out[c][h][w] = (int8_t)(sum / 4);
            }
        }
    }

    // Conv2 layer
    int8_t conv2_out[16][14][14] = { 0 };
    for (int c = 0; c < 16; c++) {
        for (int h = 0; h < 14; h++) {
            for (int w = 0; w < 14; w++) {
                int32_t sum = 0;
                for (int k = 0; k < 6; k++) {
                    for (int m = 0; m < 5; m++) {
                        for (int n = 0; n < 5; n++) {
                            sum += (int32_t)pool1_out[k][h + m][w + n] * w_conv2[c][k][m][n];
                        }
                    }
                }
                sum += b_conv2[c];
                sum = sum > 127 ? 127 : (sum < -128 ? -128 : sum);
                conv2_out[c][h][w] = (int8_t)sum;
            }
        }
    }

    // ReLU
    for (int c = 0; c < 16; c++) {
        relu_int8(&conv2_out[c][0][0], 14 * 14);
    }

    // Pool2
    int8_t pool2_out[16][5][5] = { 0 };
    for (int c = 0; c < 16; c++) {
        for (int h = 0; h < 5; h++) {
            for (int w = 0; w < 5; w++) {
                int32_t sum = (int32_t)conv2_out[c][2 * h][2 * w] +
                    conv2_out[c][2 * h + 1][2 * w] +
                    conv2_out[c][2 * h][2 * w + 1] +
                    conv2_out[c][2 * h + 1][2 * w + 1];
                pool2_out[c][h][w] = (int8_t)(sum / 4);
            }
        }
    }

    // Flatten
    int8_t flat_out[400] = { 0 };
    for (int c = 0; c < 16; c++) {
        for (int h = 0; h < 5; h++) {
            for (int w = 0; w < 5; w++) {
                flat_out[c * 25 + h * 5 + w] = pool2_out[c][h][w];
            }
        }
    }

    // FC1
    int8_t fc1_out[120] = { 0 };
    for (int i = 0; i < 120; i++) {
        int32_t sum = 0;
        for (int j = 0; j < 400; j++) {
            sum += (int32_t)flat_out[j] * w_fc1[i][j];
        }
        sum += b_fc1[i];
        sum = sum > 127 ? 127 : (sum < -128 ? -128 : sum);
        fc1_out[i] = (int8_t)sum;
    }
    relu_int8(fc1_out, 120);

    // FC2
    int8_t fc2_out[84] = { 0 };
    for (int i = 0; i < 84; i++) {
        int32_t sum = 0;
        for (int j = 0; j < 120; j++) {
            sum += (int32_t)fc1_out[j] * w_fc2[i][j];
        }
        sum += b_fc2[i];
        sum = sum > 127 ? 127 : (sum < -128 ? -128 : sum);
        fc2_out[i] = (int8_t)sum;
    }
    relu_int8(fc2_out, 84);

    // FC3
    float fc3_out[10] = { 0 };
    for (int i = 0; i < 10; i++) {
        int32_t sum = 0;
        for (int j = 0; j < 84; j++) {
            sum += (int32_t)fc2_out[j] * w_fc3[i][j];
        }
        sum += b_fc3[i];
        // Convert back to float for final layer
        fc3_out[i] = (float)sum / 127.0f;
    }

    // Softmax
    softmax(fc3_out, probs, 10);
}

void print_stats(const char* layer_name, int8_t* data, int size) {
    int8_t min_val = 127;
    int8_t max_val = -128;
    float avg = 0;
    for (int i = 0; i < size; i++) {
        if (data[i] < min_val) min_val = data[i];
        if (data[i] > max_val) max_val = data[i];
        avg += data[i];
    }
    avg /= size;
    printf("%s stats - Min: %d, Max: %d, Avg: %.2f\n",
        layer_name, min_val, max_val, avg);
}


int main(int argc, char** argv) {

    //float image[28][28];
    int8_t w_conv1[6][1][1];
    int8_t w_conv2[16][6][5][5];
    int8_t w_fc1[120][400];
    int8_t w_fc2[84][120];
    int8_t w_fc3[10][84];
    int32_t b_conv1[6];
    int32_t b_conv2[16];
    int32_t b_fc1[120];
    int32_t b_fc2[84];
    int32_t b_fc3[10];
    float probs[10];

    int i, j, m, n, index;
    FILE* fp;

    /* Load Weights from DDR->LMM */
    fp = fopen("data/weights_int8/w_conv1.txt", "r");
    for (i = 0; i < 6; i++) {
        int temp;
        fscanf(fp, "%d ", &temp);
        w_conv1[i][0][0] = (int8_t)temp;
    }
    fclose(fp);

    fp = fopen("data/weights_int8/w_conv2.txt", "r");
    for (i = 0; i < 16; i++) {
        for (j = 0; j < 6; j++) {
            for (m = 0; m < 5; m++) {
                for (n = 0; n < 5; n++) {
                    int temp;
                    fscanf(fp, "%d ", &temp);
                    w_conv2[i][j][m][n] = (int8_t)temp;
                }
            }
        }
    }
    fclose(fp);

    fp = fopen("data/weights_int8/w_fc1.txt", "r");
    for (i = 0; i < 120; i++) {
        for (j = 0; j < 400; j++) {
            int temp;
            fscanf(fp, "%d ", &temp);
            w_fc1[i][j] = (int8_t)temp;
        }
    }
    fclose(fp);

    fp = fopen("data/weights_int8/w_fc2.txt", "r");
    for (i = 0; i < 84; i++) {
        for (j = 0; j < 120; j++) {
            int temp;
            fscanf(fp, "%d ", &temp);
            w_fc2[i][j] = (int8_t)temp;
        }
    }
    fclose(fp);

    fp = fopen("data/weights_int8/w_fc3.txt", "r");
    for (i = 0; i < 10; i++) {
        for (j = 0; j < 84; j++) {
            int temp;
            fscanf(fp, "%d ", &temp);
            w_fc3[i][j] = (int8_t)temp;
        }
    }
    fclose(fp);

    fp = fopen("data/weights_int8/b_conv1.txt", "r");
    for (i = 0; i < 6; i++) {
        int temp;
        fscanf(fp, "%d ", &temp);
        b_conv1[i] = (int8_t)temp;
    }
    fclose(fp);

    fp = fopen("data/weights_int8/b_conv2.txt", "r");
    for (i = 0; i < 16; i++) {
        int temp;
        fscanf(fp, "%d ", &temp);
        b_conv2[i] = (int8_t)temp;
    }
    fclose(fp);

    fp = fopen("data/weights_int8/b_fc1.txt", "r");
    for (i = 0; i < 120; i++) {
        int temp;
        fscanf(fp, "%d ", &temp);
        b_fc1[i] = (int8_t)temp;
    }
    fclose(fp);

    fp = fopen("data/weights_int8/b_fc2.txt", "r");
    for (i = 0; i < 84; i++) {
        int temp;
        fscanf(fp, "%d ", &temp);
        b_fc2[i] = (int8_t)temp;
    }
    fclose(fp);

    fp = fopen("data/weights_int8/b_fc3.txt", "r");
    for (i = 0; i < 10; i++) {
        int temp;
        fscanf(fp, "%d ", &temp);
        b_fc3[i] = (int8_t)temp;
    }
    fclose(fp);


    float* dataset = (float*)malloc(LABEL_LEN * 28 * 28 * sizeof(float));
    int target[LABEL_LEN];

    fp = fopen("mnist-test-target.txt", "r");
    for (i = 0; i < LABEL_LEN; i++)
        fscanf(fp, "%d ", &(target[i]));  fclose(fp);

    fp = fopen("mnist-test-image.txt", "r");
    for (i = 0; i < LABEL_LEN * 28 * 28; i++)
        fscanf(fp, "%f ", &(dataset[i]));  fclose(fp);

    float image[28][28];
    float* datain;
    int acc = 0;
    int mm, nn;
    for (i = 0; i < LABEL_LEN; i++)
    {

        datain = &dataset[i * 28 * 28];
        for (mm = 0; mm < 28; mm++)
            for (nn = 0; nn < 28; nn++)
                image[mm][nn] = *(float*)&datain[28 * mm + nn];

        Prediction(image,
            w_conv1,
            w_conv2,
            w_fc1,
            w_fc2,
            w_fc3,
            b_conv1,
            b_conv2,
            b_fc1,
            b_fc2,
            b_fc3,
            probs
        );

        int index = 0;
        float max = probs[0];
        for (j = 1; j < 10; j++) {
            if (probs[j] > max) {
                index = j;
                max = probs[j];
            }
        }

        if (index == target[i]) acc++;
        printf("Predicted label: %d\n", index);
        printf("Prediction: %d/%d\n", acc, i + 1);
    }
    printf("Accuracy = %f\n", acc * 1.0f / LABEL_LEN);
    free(dataset);
    printf("=== Weight Statistics ===\n");
    print_stats("Conv1", (int8_t*)w_conv1, 6);
    print_stats("Conv2", (int8_t*)w_conv2, 16 * 6 * 5 * 5);
    print_stats("FC1", (int8_t*)w_fc1, 120 * 400);
    print_stats("FC2", (int8_t*)w_fc2, 84 * 120);
    print_stats("FC3", (int8_t*)w_fc3, 10 * 84);
    return 0;
}



