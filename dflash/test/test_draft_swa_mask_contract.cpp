#include "dflash_graph.h"
#include "internal.h"

#include "ggml.h"

#include <cstdio>
#include <vector>

using namespace dflash27b;

namespace {

struct GraphCase {
    bool is_swa = false;
    int swa_window = 0;
    int ctx_len = 0;
    bool provide_mask = false;
    bool expect_mask = false;
    const char * label = "";
};

ggml_tensor * new_vec(ggml_context * ctx, int64_t n) {
    return ggml_new_tensor_1d(ctx, GGML_TYPE_F32, n);
}

ggml_tensor * new_mat(ggml_context * ctx, int64_t ne0, int64_t ne1) {
    return ggml_new_tensor_2d(ctx, GGML_TYPE_F32, ne0, ne1);
}

bool run_case(const GraphCase & tc) {
    ggml_init_params ip{};
    ip.mem_size = 2 * 1024 * 1024;
    ip.mem_buffer = nullptr;
    ip.no_alloc = true;
    ggml_context * ctx = ggml_init(ip);
    if (!ctx) {
        std::fprintf(stderr, "FAIL %s: ggml_init failed\n", tc.label);
        return false;
    }

    constexpr int hidden = 8;
    constexpr int n_head = 2;
    constexpr int n_kv = 1;
    constexpr int head_dim = 4;
    constexpr int q_len = DFLASH27B_DRAFT_BLOCK_SIZE;
    constexpr int inter = 12;
    constexpr int fc_in = 5 * hidden;
    const int total_k = tc.ctx_len + q_len;

    DraftWeights w{};
    w.n_layer = 1;
    w.n_head = n_head;
    w.n_head_kv = n_kv;
    w.head_dim = head_dim;
    w.swa_window = tc.swa_window;
    w.layers.resize(1);

    w.fc = new_mat(ctx, fc_in, hidden);
    w.hidden_norm = new_vec(ctx, hidden);
    w.out_norm = new_vec(ctx, hidden);

    DraftLayer & layer = w.layers[0];
    layer.attn_norm = new_vec(ctx, hidden);
    layer.ffn_norm = new_vec(ctx, hidden);
    layer.wq = new_mat(ctx, hidden, n_head * head_dim);
    layer.wk = new_mat(ctx, hidden, n_kv * head_dim);
    layer.wv = new_mat(ctx, hidden, n_kv * head_dim);
    layer.wo = new_mat(ctx, n_head * head_dim, hidden);
    layer.q_norm = new_vec(ctx, head_dim);
    layer.k_norm = new_vec(ctx, head_dim);
    layer.w_gate = new_mat(ctx, hidden, inter);
    layer.w_up = new_mat(ctx, hidden, inter);
    layer.w_down = new_mat(ctx, inter, hidden);
    layer.is_swa = tc.is_swa;

    ggml_tensor * noise_embed = ggml_new_tensor_3d(ctx, GGML_TYPE_F32, hidden, q_len, 1);
    ggml_tensor * target_hidden_cat = ggml_new_tensor_3d(ctx, GGML_TYPE_F32, fc_in, tc.ctx_len, 1);
    ggml_tensor * positions_q = ggml_new_tensor_1d(ctx, GGML_TYPE_I32, q_len);
    ggml_tensor * positions_k = ggml_new_tensor_1d(ctx, GGML_TYPE_I32, total_k);
    ggml_tensor * attn_mask = nullptr;
    if (tc.provide_mask) {
        attn_mask = ggml_new_tensor_2d(ctx, GGML_TYPE_F16, total_k, q_len);
        ggml_set_name(attn_mask, "draft_swa_mask");
        ggml_set_input(attn_mask);
    }

    ggml_set_input(noise_embed);
    ggml_set_input(target_hidden_cat);
    ggml_set_input(positions_q);
    ggml_set_input(positions_k);

    DraftGraphInputs in{};
    in.ctx_len = tc.ctx_len;
    in.noise_embed = noise_embed;
    in.target_hidden_cat = target_hidden_cat;
    in.positions_q = positions_q;
    in.positions_k = positions_k;
    in.attn_mask = attn_mask;

    DraftGraphOutputs out = build_draft_graph(ctx, w, in);
    if (!out.hidden_states) {
        std::fprintf(stderr, "FAIL %s: build_draft_graph failed: %s\n",
                     tc.label, dflash27b_last_error());
        ggml_free(ctx);
        return false;
    }

    ggml_cgraph * gf = ggml_new_graph_custom(ctx, 256, false);
    ggml_build_forward_expand(gf, out.hidden_states);

    ggml_tensor * flash = nullptr;
    for (int i = 0; i < ggml_graph_n_nodes(gf); ++i) {
        ggml_tensor * node = ggml_graph_node(gf, i);
        if (node && node->op == GGML_OP_FLASH_ATTN_EXT) {
            flash = node;
            break;
        }
    }

    if (!flash) {
        std::fprintf(stderr, "FAIL %s: no flash_attn_ext node found\n", tc.label);
        ggml_free(ctx);
        return false;
    }

    const bool got_mask = flash->src[3] != nullptr;
    if (got_mask != tc.expect_mask) {
        std::fprintf(stderr, "FAIL %s: expected mask=%d got mask=%d\n",
                     tc.label, tc.expect_mask ? 1 : 0, got_mask ? 1 : 0);
        ggml_free(ctx);
        return false;
    }
    if (tc.expect_mask && flash->src[3] != attn_mask) {
        std::fprintf(stderr, "FAIL %s: flash_attn_ext did not use caller mask tensor\n", tc.label);
        ggml_free(ctx);
        return false;
    }

    std::printf("PASS %s\n", tc.label);
    ggml_free(ctx);
    return true;
}

} // namespace

int main() {
    std::vector<GraphCase> cases(3);
    cases[0].is_swa = true;
    cases[0].swa_window = 8;
    cases[0].ctx_len = 12;
    cases[0].provide_mask = true;
    cases[0].expect_mask = true;
    cases[0].label = "swa-long-context-wires-mask";

    cases[1].is_swa = false;
    cases[1].swa_window = 8;
    cases[1].ctx_len = 12;
    cases[1].provide_mask = true;
    cases[1].expect_mask = false;
    cases[1].label = "non-swa-layer-ignores-mask";

    cases[2].is_swa = true;
    cases[2].swa_window = 64;
    cases[2].ctx_len = 12;
    cases[2].provide_mask = true;
    cases[2].expect_mask = false;
    cases[2].label = "short-context-keeps-full-attn";

    int failed = 0;
    for (const GraphCase & tc : cases) {
        if (!run_case(tc)) {
            ++failed;
        }
    }

    return failed == 0 ? 0 : 1;
}
