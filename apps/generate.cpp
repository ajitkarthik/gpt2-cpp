// // Map the file containing the id -> token mappings
// // Format:
// // 256 x int32 little-endian header:
// //     [0] magic 20240328
// //     [1] version 2 (v2 adds the EOT token id at [3])
// //     [2] vocab size n
// //     [3] EOT token id
// // then, for each token id 0..n-1:
// //     1 byte length, followed by that many raw bytes
// constexpr auto TK_OFFSET_MAGIC = 0;                       // Magic number
// constexpr auto TK_OFFSET_VERSION = 1 * sizeof(int32_t);   // Version
// constexpr auto TK_OFFSET_VOCAB = 2 * sizeof(int32_t);     // Vocab size
// constexpr auto TK_OFFSET_EOT = 3 * sizeof(int32_t);       // End of token ID
// constexpr auto TK_OFFSET_TOKENS = 256 * sizeof(int32_t);  // Start of [len][...tokenbytes...]

// constexpr auto TK_MAGIC = 20240328;
// constexpr auto TK_VERSION = 2;

// constexpr auto TOKENIZERFILE = "gpt2_tokenizer.bin";

// MappedFile tk = MappedFile(TOKENIZERFILE);
// assert(tk.to_int32(tk.bytes().subspan(TK_OFFSET_MAGIC, sizeof(int32_t))) == TK_MAGIC);
// assert(tk.to_int32(tk.bytes().subspan(TK_OFFSET_VERSION, sizeof(int32_t))) == TK_VERSION);
// int32_t tk_vocab = tk.to_int32(tk.bytes().subspan(TK_OFFSET_VOCAB, sizeof(int32_t)));
// int32_t eot = tk.to_int32(tk.bytes().subspan(TK_OFFSET_EOT, sizeof(int32_t)));

// assert(tk_vocab == vocab);  // vocab size from weights file == vocab size from tokenizer file

// string idToToken(int id, MappedFile& tk, size_t startofTokens) {
//     size_t offset = startofTokens;
//     for (int i = 0; i < id; i++) {
//         // Read length byte
//         uint8_t len = static_cast<uint8_t>(tk.bytes()[offset++]);
//         offset += len;
//     }
//     return string(reinterpret_cast<const char*>(offset + 1),
//                   static_cast<uint8_t>(tk.bytes()[offset]));
// }

// vector<int32_t> tokens = {20};
// vector<int32_t> output;
// Tensor x = embed(tokens, wte, wpe);  // shape (T, C)
// Tensor logits = GPT2.forward(x);     // shape (1, V)

// // Autoregressive loop
// int32_t pred = tokens[0];
// while (true) {
//     Tensor x = embed(tokens, wte, wpe);  // shape (T, C)
//     Tensor logits = GPT2.forward(x);     // shape (1, V)
//     pred = logits.argmax(0);
//     if (pred != eot) {
//         assert(pred < tk_vocab);
//         output.push_back(pred);
//         tokens.push_back(pred);
//         // check number of elements in tokens
//         // keep a sliding window of maxT tokens at maximum
//         if (tokens.size() > static_cast<size_t>(maxT)) {
//             tokens.pop_front();
//         }

//     } else {
//         break;
//     }
// }

// // Decode the output tokens
// for (int32_t id : output) {
//     // Find offset into mapped file
//     cout << idToToken(id, tk, TK_OFFSET_TOKENS);
// }

int main(void) { return 0; }