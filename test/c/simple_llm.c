#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>

// ========== 模型超参数（极小模型）==========
#define VOCAB_SIZE  5       // 词汇表大小
#define D_MODEL     4       // 嵌入维度
#define NUM_HEADS   2       // 注意力头数
#define D_K         2       // 每个头的维度 (D_MODEL / NUM_HEADS)
#define SEQ_LEN     3       // 输入序列长度
#define FFN_SIZE    8       // 前馈网络隐藏层大小

// 词汇表
char* vocab[VOCAB_SIZE] = {"<pad>", "我", "爱", "你", "中国"};

// ========== 硬编码的权重（仅供演示）==========
// Token Embedding (VOCAB_SIZE x D_MODEL)
float token_emb[VOCAB_SIZE][D_MODEL] = {
    {0.0, 0.0, 0.0, 0.0},  // <pad>
    {0.1, 0.2, 0.3, 0.4},  // 我
    {0.5, 0.6, 0.7, 0.8},  // 爱
    {0.9, 1.0, 1.1, 1.2},  // 你
    {0.2, 0.4, 0.6, 0.8}   // 中国
};

// 位置嵌入（假设序列最长为3）
float pos_emb[SEQ_LEN][D_MODEL] = {
    {0.0, 0.0, 0.0, 0.0},
    {0.1, 0.1, 0.1, 0.1},
    {0.2, 0.2, 0.2, 0.2}
};

// Q, K, V 投影权重 (D_MODEL x D_MODEL)
float W_Q[D_MODEL][D_MODEL] = {
    {0.5, 0.1, 0.0, 0.0},
    {0.0, 0.5, 0.1, 0.0},
    {0.0, 0.0, 0.5, 0.1},
    {0.1, 0.0, 0.0, 0.5}
};

float W_K[D_MODEL][D_MODEL] = {
    {0.4, 0.0, 0.1, 0.0},
    {0.0, 0.4, 0.0, 0.1},
    {0.1, 0.0, 0.4, 0.0},
    {0.0, 0.1, 0.0, 0.4}
};

float W_V[D_MODEL][D_MODEL] = {
    {0.6, 0.0, 0.0, 0.1},
    {0.1, 0.6, 0.0, 0.0},
    {0.0, 0.1, 0.6, 0.0},
    {0.0, 0.0, 0.1, 0.6}
};

// 输出投影权重 (D_MODEL x D_MODEL)
float W_O[D_MODEL][D_MODEL] = {
    {0.7, 0.1, 0.0, 0.0},
    {0.0, 0.7, 0.1, 0.0},
    {0.0, 0.0, 0.7, 0.1},
    {0.1, 0.0, 0.0, 0.7}
};

// FFN 第一层权重 (D_MODEL x FFN_SIZE)
float W1[D_MODEL][FFN_SIZE] = {
    {0.3, 0.5, -0.2, 0.1, 0.4, -0.3, 0.2, 0.0},
    {-0.1, 0.2, 0.4, -0.3, 0.0, 0.5, -0.2, 0.1},
    {0.2, -0.1, 0.3, 0.5, -0.4, 0.0, 0.1, -0.2},
    {0.0, 0.3, -0.1, 0.2, 0.5, -0.2, -0.3, 0.4}
};

// FFN 第二层权重 (FFN_SIZE x D_MODEL)
float W2[FFN_SIZE][D_MODEL] = {
    {0.2, -0.1, 0.3, 0.0},
    {0.4, 0.1, -0.2, 0.3},
    {-0.1, 0.2, 0.0, -0.3},
    {0.3, -0.3, 0.1, 0.2},
    {0.0, 0.2, 0.4, -0.1},
    {-0.2, 0.3, -0.1, 0.0},
    {0.1, 0.0, -0.3, 0.4},
    {-0.3, 0.1, 0.2, -0.2}
};

// 最终输出投影 (D_MODEL x VOCAB_SIZE)
float lm_head[D_MODEL][VOCAB_SIZE] = {
    {0.8, 0.2, 0.1, 0.0, -0.3},
    {0.1, 0.7, 0.3, -0.1, 0.2},
    {0.0, 0.1, 0.6, 0.4, -0.1},
    {0.2, -0.1, 0.0, 0.5, 0.6}
};

// ========== 辅助函数：矩阵运算 ==========

// 打印矩阵，用于调试
void print_matrix(float* mat, int rows, int cols) {
    for (int i = 0; i < rows; i++) {
        for (int j = 0; j < cols; j++) {
            printf("%.4f ", mat[i * cols + j]);
        }
        printf("\n");
    }
    printf("\n");
}

// 矩阵乘法: C = A x B
// A: m x n, B: n x p, C: m x p
void matmul(float* A, float* B, float* C, int m, int n, int p) {
    for (int i = 0; i < m; i++) {
        for (int j = 0; j < p; j++) {
            C[i * p + j] = 0.0f;
            for (int k = 0; k < n; k++) {
                C[i * p + j] += A[i * n + k] * B[k * p + j];
            }
        }
    }
}

// 矩阵加法: C = A + B (对应元素)
void matadd(float* A, float* B, float* C, int size) {
    for (int i = 0; i < size; i++) {
        C[i] = A[i] + B[i];
    }
}

// Softmax 操作（对行进行，用于注意力分数）
void softmax(float* x, int rows, int cols) {
    for (int i = 0; i < rows; i++) {
        // 找最大值（防止数值溢出）
        float max_val = x[i * cols];
        for (int j = 1; j < cols; j++) {
            if (x[i * cols + j] > max_val) max_val = x[i * cols + j];
        }
        // 计算指数并求和
        float sum = 0.0f;
        for (int j = 0; j < cols; j++) {
            x[i * cols + j] = expf(x[i * cols + j] - max_val);
            sum += x[i * cols + j];
        }
        // 归一化
        for (int j = 0; j < cols; j++) {
            x[i * cols + j] /= sum;
        }
    }
}

// 层归一化（简化版，无缩放和偏移）
void layer_norm(float* x, int size) {
    // 计算均值
    float mean = 0.0f;
    for (int i = 0; i < size; i++) {
        mean += x[i];
    }
    mean /= size;

    // 计算方差
    float var = 0.0f;
    for (int i = 0; i < size; i++) {
        float diff = x[i] - mean;
        var += diff * diff;
    }
    var /= size;

    // 归一化 (加epsilon防除零)
    float eps = 1e-5f;
    for (int i = 0; i < size; i++) {
        x[i] = (x[i] - mean) / sqrtf(var + eps);
    }
}

// ReLU 激活
void relu(float* x, int size) {
    for (int i = 0; i < size; i++) {
        if (x[i] < 0) x[i] = 0;
    }
}

// ========== 核心推理函数 ==========

// 将输入token序列转换为模型输入表示
void prepare_input(int* token_ids, int seq_len, float* input_emb) {
    // input_emb 大小: seq_len x D_MODEL
    for (int i = 0; i < seq_len; i++) {
        int token = token_ids[i];
        for (int j = 0; j < D_MODEL; j++) {
            // 输入 = Token嵌入 + 位置嵌入
            input_emb[i * D_MODEL + j] = token_emb[token][j] + pos_emb[i][j];
        }
    }
}

// 多头注意力机制（简化，只用一个头来演示，实际是多头的组合）
// 输入 x: seq_len x D_MODEL, 输出 output: seq_len x D_MODEL
void multi_head_attention(float* x, int seq_len, float* output) {
    // 为简化，此处直接按单头处理，但保留头数概念。
    // 实际中会将Q,K,V拆成多个头分别计算再拼接。
    
    // 分配临时空间
    float* Q = (float*)malloc(seq_len * D_MODEL * sizeof(float));
    float* K = (float*)malloc(seq_len * D_MODEL * sizeof(float));
    float* V = (float*)malloc(seq_len * D_MODEL * sizeof(float));
    
    // 1. 线性投影得到Q, K, V
    matmul(x, (float*)W_Q, Q, seq_len, D_MODEL, D_MODEL);
    matmul(x, (float*)W_K, K, seq_len, D_MODEL, D_MODEL);
    matmul(x, (float*)W_V, V, seq_len, D_MODEL, D_MODEL);
    
    // 2. 计算注意力分数: Scores = Q * K^T / sqrt(d_k)
    // K^T 的维度: D_MODEL x seq_len
    float* scores = (float*)malloc(seq_len * seq_len * sizeof(float));
    matmul(Q, K, scores, seq_len, D_MODEL, seq_len); // 注意：这里K未转置，实际需要转置
    // 为简化代码，我们手动实现Q*K^T
    // 重新计算正确转置的分数矩阵
    for (int i = 0; i < seq_len; i++) {
        for (int j = 0; j < seq_len; j++) {
            float s = 0.0f;
            for (int k = 0; k < D_MODEL; k++) {
                s += Q[i * D_MODEL + k] * K[j * D_MODEL + k];
            }
            scores[i * seq_len + j] = s / sqrtf((float)D_MODEL);
        }
    }
    
    // 3. Softmax 得到注意力权重
    softmax(scores, seq_len, seq_len);
    
    // 4. 加权求和: Context = Attention * V
    float* context = (float*)malloc(seq_len * D_MODEL * sizeof(float));
    matmul(scores, V, context, seq_len, seq_len, D_MODEL);
    
    // 5. 输出投影
    matmul(context, (float*)W_O, output, seq_len, D_MODEL, D_MODEL);
    
    // 释放内存
    free(Q); free(K); free(V);
    free(scores); free(context);
}

// 前馈网络
void feed_forward(float* x, int seq_len, float* output) {
    float* hidden = (float*)malloc(seq_len * FFN_SIZE * sizeof(float));
    
    // 第一层 + ReLU
    matmul(x, (float*)W1, hidden, seq_len, D_MODEL, FFN_SIZE);
    relu(hidden, seq_len * FFN_SIZE);
    
    // 第二层
    matmul(hidden, (float*)W2, output, seq_len, FFN_SIZE, D_MODEL);
    
    free(hidden);
}

// 单层Transformer Block
void transformer_block(float* x, int seq_len, float* output) {
    // 1. 多头注意力 + 残差连接 + 层归一化
    float* attn_out = (float*)malloc(seq_len * D_MODEL * sizeof(float));
    multi_head_attention(x, seq_len, attn_out);
    
    // 残差连接
    for (int i = 0; i < seq_len * D_MODEL; i++) {
        attn_out[i] += x[i];
    }
    
    // 层归一化（对每个位置单独归一化）
    float* norm1_out = (float*)malloc(seq_len * D_MODEL * sizeof(float));
    memcpy(norm1_out, attn_out, seq_len * D_MODEL * sizeof(float));
    for (int i = 0; i < seq_len; i++) {
        layer_norm(&norm1_out[i * D_MODEL], D_MODEL);
    }
    
    // 2. 前馈网络 + 残差连接 + 层归一化
    float* ffn_out = (float*)malloc(seq_len * D_MODEL * sizeof(float));
    feed_forward(norm1_out, seq_len, ffn_out);
    
    // 残差连接
    for (int i = 0; i < seq_len * D_MODEL; i++) {
        output[i] = ffn_out[i] + norm1_out[i];
    }
    
    // 再次层归一化
    for (int i = 0; i < seq_len; i++) {
        layer_norm(&output[i * D_MODEL], D_MODEL);
    }
    
    free(attn_out);
    free(norm1_out);
    free(ffn_out);
}

// 模型推理主函数
int inference(int* input_ids, int input_len) {
    // 1. 准备输入嵌入
    float* hidden_states = (float*)malloc(input_len * D_MODEL * sizeof(float));
    prepare_input(input_ids, input_len, hidden_states);
    
    printf("输入嵌入:\n");
    print_matrix(hidden_states, input_len, D_MODEL);
    
    // 2. 通过Transformer层
    float* block_output = (float*)malloc(input_len * D_MODEL * sizeof(float));
    transformer_block(hidden_states, input_len, block_output);
    
    printf("Transformer输出:\n");
    print_matrix(block_output, input_len, D_MODEL);
    
    // 3. 取最后一个位置的输出作为下一个词的预测
    float* last_hidden = &block_output[(input_len - 1) * D_MODEL];
    
    // 4. 通过LM Head投影到词汇表
    float* logits = (float*)malloc(VOCAB_SIZE * sizeof(float));
    // 矩阵乘法: 1 x D_MODEL  *  D_MODEL x VOCAB_SIZE  =  1 x VOCAB_SIZE
    for (int i = 0; i < VOCAB_SIZE; i++) {
        logits[i] = 0.0f;
        for (int j = 0; j < D_MODEL; j++) {
            logits[i] += last_hidden[j] * lm_head[j][i];
        }
    }
    
    printf("Logits:\n");
    print_matrix(logits, 1, VOCAB_SIZE);
    
    // 5. Softmax得到概率分布
    softmax(logits, 1, VOCAB_SIZE);
    
    printf("概率分布:\n");
    for (int i = 0; i < VOCAB_SIZE; i++) {
        printf("%s: %.4f ", vocab[i], logits[i]);
    }
    printf("\n");
    
    // 6. 贪心采样：选择概率最大的token
    int next_token = 0;
    float max_prob = logits[0];
    for (int i = 1; i < VOCAB_SIZE; i++) {
        if (logits[i] > max_prob) {
            max_prob = logits[i];
            next_token = i;
        }
    }
    
    // 清理内存
    free(hidden_states);
    free(block_output);
    free(logits);
    
    return next_token;
}

int main() {
    // 输入句子: "我 爱 你" (token IDs: 1, 2, 3)
    int input_ids[] = {1, 2, 3};  // 1:"我", 2:"爱", 3:"你"
    int input_len = 3;
    
    printf("输入序列: 我 爱 你\n");
    printf("正在进行推理...\n\n");
    
    int next_token = inference(input_ids, input_len);
    
    printf("\n预测的下一个词: %s (ID: %d)\n", vocab[next_token], next_token);
    
    return 0;
}
