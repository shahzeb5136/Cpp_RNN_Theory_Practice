/**
 * =============================================================================
 * RECURRENT NEURAL NETWORK (RNN) - MAIN PROGRAM
 * =============================================================================
 * 
 * FULLY AUTOREGRESSIVE DEMONSTRATION
 * 
 * The ONLY randomness is in the initial weight/embedding matrices (seed 42).
 * After that, every x_t comes from the embedding matrix E via the previous
 * step's softmax output — no synthetic embeddings, no external randomization.
 * 
 * Flow for each step:
 *   word index -> E[index] = x_t -> forward(x_t) -> y_t -> pick word -> repeat
 * 
 * Every intermediate value is saved to a CSV you can trace by hand.
 * 
 * COMPILE:  g++ -std=c++17 -o rnn_demo main.cpp rnn.cpp
 * RUN:      ./rnn_demo
 * 
 * =============================================================================
 */

#include "rnn.hpp"
#include <iostream>
#include <iomanip>
#include <algorithm>

int main(int /* argc */, char* /* argv */[]) {

    std::cout << R"(
╔══════════════════════════════════════════════════════════════╗
║                                                              ║
║     RECURRENT NEURAL NETWORK (RNN) DEMONSTRATION            ║
║     Fully Autoregressive — Traceable CSV Output              ║
║                                                              ║
╚══════════════════════════════════════════════════════════════╝
)" << std::endl;

    // =========================================================================
    // STEP 1: Define Network Architecture
    // =========================================================================
    std::cout << "═══════════════════════════════════════════" << std::endl;
    std::cout << "STEP 1: DEFINING NETWORK ARCHITECTURE" << std::endl;
    std::cout << "═══════════════════════════════════════════" << std::endl;

    int inputSize  = 4;   // Embedding dimension (size of each x_t)
    int hiddenSize = 5;   // Hidden state dimension
    int outputSize = 6;   // Vocabulary size (also # of rows in E)

    std::cout << "\nNetwork dimensions:" << std::endl;
    std::cout << "  Input size (embedding dim):  " << inputSize << std::endl;
    std::cout << "  Hidden size (memory):        " << hiddenSize << std::endl;
    std::cout << "  Output size (vocab size):    " << outputSize << std::endl;

    std::cout << "\nParameter counts:" << std::endl;
    std::cout << "  W matrix: " << hiddenSize << " x " << inputSize
              << " = " << hiddenSize * inputSize << " parameters" << std::endl;
    std::cout << "  U matrix: " << hiddenSize << " x " << hiddenSize
              << " = " << hiddenSize * hiddenSize << " parameters" << std::endl;
    std::cout << "  V matrix: " << outputSize << " x " << hiddenSize
              << " = " << outputSize * hiddenSize << " parameters" << std::endl;
    std::cout << "  E matrix: " << outputSize << " x " << inputSize
              << " = " << outputSize * inputSize << " parameters" << std::endl;
    std::cout << "  Biases:   " << hiddenSize + outputSize << " parameters" << std::endl;

    int totalParams = hiddenSize * inputSize + hiddenSize * hiddenSize +
                      outputSize * hiddenSize + outputSize * inputSize +
                      hiddenSize + outputSize;
    std::cout << "  TOTAL:    " << totalParams << " parameters" << std::endl;

    // =========================================================================
    // STEP 2: Create the RNN (weights + embeddings initialized once, seed 42)
    // =========================================================================
    std::cout << "\n═══════════════════════════════════════════" << std::endl;
    std::cout << "STEP 2: CREATING THE RNN" << std::endl;
    std::cout << "═══════════════════════════════════════════\n" << std::endl;

    RNN rnn(inputSize, hiddenSize, outputSize, "output");

    // =========================================================================
    // STEP 3: AUTOREGRESSIVE GENERATION
    // =========================================================================
    //
    // This is the entire demo now. We pick a starting word index and let the
    // RNN generate a sequence by feeding its own predictions back through the
    // embedding matrix E.
    //
    //   seed word index
    //        |
    //        v
    //   E[index] = x_0   --->  forward(x_0)  --->  y_0  --->  pick word_1
    //                                                              |
    //   E[word_1] = x_1  <----------------------------------------+
    //        |
    //        v
    //   forward(x_1)  --->  y_1  --->  pick word_2
    //        ...
    //
    // NO other source of input. Every x_t is a row of E.
    // =========================================================================
    std::cout << "\n═══════════════════════════════════════════" << std::endl;
    std::cout << "STEP 3: AUTOREGRESSIVE GENERATION" << std::endl;
    std::cout << "═══════════════════════════════════════════" << std::endl;

    std::cout << "\nThe RNN will generate a sequence purely from its own outputs." << std::endl;
    std::cout << "Only the initial weights/embeddings are random (seed 42)." << std::endl;
    std::cout << "After that, every x_t = E[predicted_word_index].\n" << std::endl;

    int startWord       = 0;   // Seed word index (try 0-5)
    int sequenceLength  = 5;   // How many words to generate after the seed

    std::cout << "Seed word index: " << startWord << std::endl;
    std::cout << "Words to generate: " << sequenceLength << std::endl;
    std::cout << "Selection mode: greedy (argmax)\n" << std::endl;

    // Generate — all CSV tracing happens inside generate()
    std::vector<int> generated = rnn.generate(startWord, sequenceLength);

    // =========================================================================
    // STEP 4: Summary
    // =========================================================================
    std::cout << "\n═══════════════════════════════════════════" << std::endl;
    std::cout << "STEP 4: SUMMARY" << std::endl;
    std::cout << "═══════════════════════════════════════════" << std::endl;

    std::cout << "\nGenerated word indices: [ ";
    for (size_t i = 0; i < generated.size(); ++i) {
        std::cout << generated[i];
        if (i < generated.size() - 1) std::cout << " -> ";
    }
    std::cout << " ]" << std::endl;

    std::cout << "\n--- HOW TO TRACE THROUGH THE CSVs ---" << std::endl;
    std::cout << R"(
  output/00_*.csv   = Initial weights (W, U, V, E) and biases (b, c)
  output/01_*.csv   = Generation step 1 computations
  output/02_*.csv   = Generation step 2 computations
  ...

  For each step N, the files are numbered in order:
    N_1_input_x_t.csv            <-- x_t looked up from E (verify against 00_E)
    N_2_prev_hidden_h_t-1.csv    <-- h from previous step (or zeros if step 1)
    N_3_W_times_x.csv            <-- W * x_t
    N_4_U_times_h.csv            <-- U * h_{t-1}
    N_5_Wx_plus_Uh.csv           <-- W*x_t + U*h_{t-1}
    N_6_pre_activation.csv       <-- + b
    N_7_hidden_state_h_t.csv     <-- ReLU applied = new h_t
    N_8_V_times_h.csv            <-- V * h_t
    N_9_logits.csv               <-- + c
    N_10_output_y_t.csv          <-- softmax = probability distribution
    N_11_selected_word.csv       <-- which word was picked & its embedding

  To verify by hand:
    1. Open 00_E_embedding_matrix.csv -- find the row for the seed word
    2. That row should exactly match 01_1_input_x_t.csv
    3. Multiply 00_W * that x_t vector -- should match 01_3_W_times_x.csv
    4. Continue through each sub-step...
    5. The selected word index in 01_11 tells you which row of E becomes
       the x_t in 02_1_input_x_t.csv -- and so on.
)" << std::endl;

    std::cout << "═══════════════════════════════════════════" << std::endl;
    std::cout << "PROGRAM COMPLETE!" << std::endl;
    std::cout << "═══════════════════════════════════════════" << std::endl;

    return 0;
}
