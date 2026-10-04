#include <cassert>
#include <cstdint>
#include <cstdlib>
#include <iostream>
#include <string>
#include <unordered_map>
#include <variant>
#include <vector>

#include "checkpoint.hpp"
#include "decoder.hpp"
#include "mappedfile.hpp"
#include "tensor.hpp"
#include "utils.hpp"

using namespace std;
using decoder::Decoder;
using tn::Tensor;

// Format of the file containing the id -> token mappings
// Little endian
// 256 x int32 little-endian header:
//     [0] magic 20240328
//     [1] version 2 (v2 adds the EOT token id at [3])
//     [2] vocab size n
//     [3] EOT token id
// then, for each token id 0..n-1:
//     1 byte length, followed by that many raw bytes
constexpr auto OFFSET_MAGIC = 0;                       // Magic number
constexpr auto OFFSET_VERSION = 1 * sizeof(int32_t);   // Version
constexpr auto OFFSET_VOCAB = 2 * sizeof(int32_t);     // Vocab size
constexpr auto OFFSET_EOT = 3 * sizeof(int32_t);       // End of token ID
constexpr auto OFFSET_TOKENS = 256 * sizeof(int32_t);  // Start of [len][...tokenbytes...]

constexpr auto MAGIC = 20240328;
constexpr auto VERSION = 2;

string idToToken(int id, MappedFile& tk, size_t startofTokens) {
    size_t offset = startofTokens;
    for (int i = 0; i < id; i++) {
        // Read length byte
        uint8_t len = static_cast<uint8_t>(tk.bytes()[offset++]);
        offset += len;
    }
    return string(reinterpret_cast<const char*>(tk.bytes().data() + offset + 1),
                  static_cast<uint8_t>(tk.bytes()[offset]));
}

void parseArgs(unordered_map<string, variant<int, string>>& args, int argc, char** argv) {
    // If fewer args than we are expecting, print out help
    if (argc < 5) {
        cout << "Usage: " << argv[0];
        cout << " [-h | --help] <weights_file> <tokenizer_file> <num_tokens> <prompt>" << "\n";
        cout << "Notes: See karpathy/llm.c to get the GPT2_124M.bin weights file.\n";
        cout << "       This program cannot read the HuggingFace safetensors format.\n";
        exit(0);
    }
    // If any arg is -h or --help, print out help
    for (int i = 1; i < argc; i++) {
        std::string arg = argv[i];
        if (arg == "-h" || arg == "--help") {
            cout << "Usage: " << argv[0];
            cout << " [-h | --help] <weights_file> <tokenizer_file> <num_tokens> <prompt>" << "\n";
            cout << "Notes: See karpathy/llm.c to get the GPT2_124M.bin weights file.\n";
            cout << "       This program cannot read the HuggingFace safetensors format.\n";
            exit(0);
        }
    }
    args["weights_file"] = string(argv[1]);
    args["tokenizer_file"] = string(argv[2]);
    int num_tokens = 0;
    if (convert_arg(string(argv[3]), num_tokens)) args["num_tokens"] = num_tokens;
    args["prompt"] = string(argv[4]);
}

// Usage: generate [-h | --help] <weights_file> <tokenizer_file> <num_tokens> <prompt>
int main(int argc, char** argv) {
    unordered_map<string, variant<int, string>> args;
    parseArgs(args, argc, argv);

    MappedFile tk = MappedFile(get<string>(args["tokenizer_file"]).c_str());  // open the token file
    assert(tk.to_int32(tk.bytes().subspan(OFFSET_MAGIC, sizeof(int32_t))) == MAGIC);
    assert(tk.to_int32(tk.bytes().subspan(OFFSET_VERSION, sizeof(int32_t))) == VERSION);
    int32_t vocab = tk.to_int32(tk.bytes().subspan(OFFSET_VOCAB, sizeof(int32_t)));
    int32_t eot = tk.to_int32(tk.bytes().subspan(OFFSET_EOT, sizeof(int32_t)));

    // "Once upon a time, there was a"
    vector<int32_t> tokens = {7454, 2402, 257, 640, 11, 612, 373, 257};
    vector<int32_t> output;

    Checkpoint ckpt(get<string>(args["weights_file"]).c_str());
    cout << "Loaded checkpoint file. Size: " << ckpt.mp.size() / (1024 * 1024) << "MB\n";
    cout << "Context length:         " << ckpt.maxT << "\n";
    cout << "Vocabulary size:        " << ckpt.vocab << "\n";
    cout << "Layers:                 " << ckpt.layers << "\n";
    cout << "Attention heads:        " << ckpt.nh << "\n";
    cout << "Embedding dimensions:   " << ckpt.embsz << "\n";
    cout << "Padded vocabulary size: " << ckpt.vocab_padded << "\n";
    // instantiate the decoder from the checkpoint
    Decoder gpt2(ckpt);

    // Autoregressive loop
    int32_t pred = tokens[0];
    int tokens_generated = 0;
    int numtokens = get<int>(args["num_tokens"]);

    while (true) {
        Tensor logits = gpt2.forward(tokens, nullptr);             // shape (T, C)
        pred = logits.argmax(0);                                   // simple greedy decoding for now
        cout << idToToken(pred, tk, OFFSET_TOKENS) << std::flush;  // Decode tokenid -> token
        if ((pred != eot) && (tokens_generated++ < numtokens)) {
            assert(pred < vocab);
            tokens.push_back(pred);
            // check number of elements in tokens
            // keep a sliding window of maxT tokens at maximum
            if (tokens.size() > static_cast<size_t>(ckpt.maxT)) {
                tokens.erase(tokens.begin());      // remove front element
                tokens.resize(tokens.size() - 1);  // shrink size
            }
        } else {
            break;
        }
    }
}