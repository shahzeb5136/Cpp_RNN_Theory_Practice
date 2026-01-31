/**
 * =============================================================================
 * RECURRENT NEURAL NETWORK (RNN) - MAIN PROGRAM
 * =============================================================================
 * 
 * This program demonstrates how an RNN processes a sequence of inputs.
 * 
 * WHAT THIS PROGRAM DOES:
 * 1. Creates an RNN with specified dimensions
 * 2. Processes a sequence of "word embeddings" (simulated inputs)
 * 3. Shows the step-by-step computation at each time step
 * 4. Saves all intermediate values to CSV files for visualization
 * 
 * COMPILE WITH:
 *   g++ -std=c++17 -o rnn_demo main.cpp rnn.cpp
 * 
 * RUN WITH:
 *   ./rnn_demo
 * 
 * =============================================================================
 */

#include "rnn.hpp"   // Our RNN class
#include <iostream>
#include <iomanip>
#include <algorithm>  // For std::max_element

// -----------------------------------------------------------------------------
// HELPER FUNCTION: Create simulated word embeddings
// -----------------------------------------------------------------------------
/**
 * Creates a simple word embedding vector
 * 
 * In real NLP systems, words are converted to dense vectors using
 * methods like Word2Vec, GloVe, or learned embeddings.
 * 
 * Here we just create simple patterns for demonstration.
 * 
 * @param wordIndex  Index of the "word" (0, 1, 2, etc.)
 * @param embeddingSize  Size of the embedding vector
 * @return  A vector representing the word embedding
 */
Vector createSimpleEmbedding(int wordIndex, int embeddingSize) {
    Vector embedding(embeddingSize, 0.0);
    
    // Create a simple pattern based on word index
    // This is just for demonstration - real embeddings are learned
    for (int i = 0; i < embeddingSize; ++i) {
        // Create some variation based on word and position
        embedding[i] = 0.1 * std::sin(wordIndex * 0.5 + i * 0.3) + 
                       0.2 * std::cos(wordIndex * 0.7 - i * 0.2);
    }
    
    return embedding;
}

// -----------------------------------------------------------------------------
// MAIN FUNCTION
// -----------------------------------------------------------------------------
/**
 * The main() function is the entry point of every C++ program.
 * When you run the program, execution starts here.
 * 
 * argc = argument count (number of command-line arguments)
 * argv = argument values (array of C-strings)
 */
int main(int /* argc */, char* /* argv */[]) {
    
    std::cout << R"(
╔══════════════════════════════════════════════════════════════╗
║                                                              ║
║     RECURRENT NEURAL NETWORK (RNN) DEMONSTRATION            ║
║                                                              ║
║     Learning C++ and Neural Networks Together!               ║
║                                                              ║
╚══════════════════════════════════════════════════════════════╝
)" << std::endl;

    // =========================================================================
    // STEP 1: Define Network Architecture
    // =========================================================================
    std::cout << "═══════════════════════════════════════════" << std::endl;
    std::cout << "STEP 1: DEFINING NETWORK ARCHITECTURE" << std::endl;
    std::cout << "═══════════════════════════════════════════" << std::endl;
    
    // These are hyperparameters - choices we make about network structure
    
    int inputSize = 4;   // Dimension of input vectors (word embedding size)
                         // In real systems, this might be 100-300 or more
    
    int hiddenSize = 5;  // Dimension of hidden state
                         // This is the network's "memory" capacity
                         // More neurons = more capacity but slower
    
    int outputSize = 6;  // Dimension of output (vocabulary size)
                         // In real systems, this could be 30,000+ words
    
    std::cout << "\nNetwork dimensions chosen:" << std::endl;
    std::cout << "  Input size (embedding dim):  " << inputSize << std::endl;
    std::cout << "  Hidden size (memory):        " << hiddenSize << std::endl;
    std::cout << "  Output size (vocab size):    " << outputSize << std::endl;
    
    std::cout << "\nThis means:" << std::endl;
    std::cout << "  W matrix: " << hiddenSize << " x " << inputSize 
              << " = " << hiddenSize * inputSize << " parameters" << std::endl;
    std::cout << "  U matrix: " << hiddenSize << " x " << hiddenSize 
              << " = " << hiddenSize * hiddenSize << " parameters" << std::endl;
    std::cout << "  V matrix: " << outputSize << " x " << hiddenSize 
              << " = " << outputSize * hiddenSize << " parameters" << std::endl;
    std::cout << "  Biases:   " << hiddenSize + outputSize << " parameters" << std::endl;
    
    int totalParams = hiddenSize * inputSize + hiddenSize * hiddenSize + 
                      outputSize * hiddenSize + hiddenSize + outputSize;
    std::cout << "  TOTAL:    " << totalParams << " learnable parameters" << std::endl;
    
    // =========================================================================
    // STEP 2: Create the RNN
    // =========================================================================
    std::cout << "\n═══════════════════════════════════════════" << std::endl;
    std::cout << "STEP 2: CREATING THE RNN" << std::endl;
    std::cout << "═══════════════════════════════════════════\n" << std::endl;
    
    // This calls the constructor, which initializes all weights randomly
    // The output directory "output" will contain all our CSV files
    RNN rnn(inputSize, hiddenSize, outputSize, "output");
    
    // =========================================================================
    // STEP 3: Create Input Sequence
    // =========================================================================
    std::cout << "\n═══════════════════════════════════════════" << std::endl;
    std::cout << "STEP 3: CREATING INPUT SEQUENCE" << std::endl;
    std::cout << "═══════════════════════════════════════════" << std::endl;
    
    // Imagine we have a sentence like "The cat sat"
    // Each word would be converted to an embedding vector
    // Here we simulate 3 "words" in a sequence
    
    int sequenceLength = 3;
    std::vector<Vector> inputSequence;
    
    std::cout << "\nCreating " << sequenceLength << " input embeddings (simulating a sentence):" << std::endl;
    std::cout << "(In reality, these would come from a word embedding lookup table)\n" << std::endl;
    
    for (int t = 0; t < sequenceLength; ++t) {
        Vector embedding = createSimpleEmbedding(t, inputSize);
        inputSequence.push_back(embedding);
        std::cout << "Word " << t << " embedding: ";
        printVector(embedding, "x_" + std::to_string(t));
    }
    
    // =========================================================================
    // STEP 4: Process Sequence Through RNN
    // =========================================================================
    std::cout << "\n═══════════════════════════════════════════" << std::endl;
    std::cout << "STEP 4: PROCESSING SEQUENCE THROUGH RNN" << std::endl;
    std::cout << "═══════════════════════════════════════════" << std::endl;
    
    std::cout << "\nThe RNN will process each input one at a time," << std::endl;
    std::cout << "updating its hidden state at each step." << std::endl;
    std::cout << "\nThis is the core computation:" << std::endl;
    std::cout << "  h_t = ReLU(W * x_t + U * h_{t-1} + b)" << std::endl;
    std::cout << "  y_t = Softmax(V * h_t + c)" << std::endl;
    
    // Store outputs for later analysis
    std::vector<Vector> outputs;
    std::vector<Vector> hiddenStates;
    
    // Save initial hidden state
    hiddenStates.push_back(rnn.getHiddenState());
    
    // Process each input in the sequence
    for (int t = 0; t < sequenceLength; ++t) {
        std::cout << "\n\n╔════════════════════════════════════════════════════════╗" << std::endl;
        std::cout << "║  PROCESSING TIME STEP t = " << t << "                            ║" << std::endl;
        std::cout << "╚════════════════════════════════════════════════════════╝" << std::endl;
        
        // Call forward pass
        // This computes one step of the RNN and returns the output
        Vector output = rnn.forward(inputSequence[t]);
        
        // Store results
        outputs.push_back(output);
        hiddenStates.push_back(rnn.getHiddenState());
    }
    
    // =========================================================================
    // STEP 5: Summary and Analysis
    // =========================================================================
    std::cout << "\n═══════════════════════════════════════════" << std::endl;
    std::cout << "STEP 5: SUMMARY AND ANALYSIS" << std::endl;
    std::cout << "═══════════════════════════════════════════" << std::endl;
    
    std::cout << "\n--- Hidden State Evolution ---" << std::endl;
    std::cout << "Watch how the hidden state changes as we process the sequence:" << std::endl;
    std::cout << "(This is the network's 'memory' evolving)" << std::endl;
    for (size_t t = 0; t < hiddenStates.size(); ++t) {
        std::string label = (t == 0) ? "h_0 (initial)" : "h_" + std::to_string(t);
        printVector(hiddenStates[t], label);
    }
    
    std::cout << "\n--- Output Probabilities at Each Step ---" << std::endl;
    std::cout << "These are probability distributions over the vocabulary:" << std::endl;
    for (size_t t = 0; t < outputs.size(); ++t) {
        std::cout << "\nTime step " << t << ":" << std::endl;
        printVector(outputs[t], "y_" + std::to_string(t));
        
        // Find predicted word (highest probability)
        auto maxIt = std::max_element(outputs[t].begin(), outputs[t].end());
        int predicted = std::distance(outputs[t].begin(), maxIt);
        std::cout << "  -> Predicted 'word' index: " << predicted 
                  << " (probability: " << std::fixed << std::setprecision(4) 
                  << *maxIt << ")" << std::endl;
    }
    
    // =========================================================================
    // STEP 6: File Output Summary
    // =========================================================================
    std::cout << "\n═══════════════════════════════════════════" << std::endl;
    std::cout << "STEP 6: CSV FILES GENERATED" << std::endl;
    std::cout << "═══════════════════════════════════════════" << std::endl;
    
    std::cout << "\nAll intermediate computations have been saved to 'output/' directory." << std::endl;
    std::cout << "\nFile naming convention:" << std::endl;
    std::cout << "  00_*.csv - Initial weights and biases" << std::endl;
    std::cout << "  01_*.csv - Time step 1 computations" << std::endl;
    std::cout << "  02_*.csv - Time step 2 computations" << std::endl;
    std::cout << "  (and so on...)" << std::endl;
    
    std::cout << "\nYou can open these in Excel, Google Sheets, or Python/pandas" << std::endl;
    std::cout << "to visualize the matrices and vectors at each step." << std::endl;
    
    // =========================================================================
    // C++ CONCEPTS DEMONSTRATED
    // =========================================================================
    std::cout << "\n═══════════════════════════════════════════" << std::endl;
    std::cout << "C++ CONCEPTS DEMONSTRATED IN THIS CODE" << std::endl;
    std::cout << "═══════════════════════════════════════════" << std::endl;
    
    std::cout << R"(
1. CLASSES AND OBJECTS
   - The RNN class encapsulates data (weights) and behavior (forward pass)
   - Constructor initializes the object
   - Public/private access specifiers control visibility

2. VECTORS (std::vector)
   - Dynamic arrays that can grow/shrink
   - Used for our neural network vectors and matrices
   - Matrix = vector of vectors

3. FUNCTIONS
   - Defined with return_type name(parameters)
   - 'const' parameters prevent modification
   - References (&) avoid copying large data

4. FILE I/O
   - std::ofstream for writing files
   - std::ifstream for reading files
   - Always check if file opened successfully

5. MEMORY MANAGEMENT
   - std::vector handles memory automatically
   - No need for manual new/delete in this example

6. MODERN C++ FEATURES
   - 'auto' keyword for type inference
   - Range-based for loops
   - Initializer lists in constructors
   - std::filesystem (C++17)

7. CONST CORRECTNESS
   - 'const' functions don't modify object state
   - 'const&' parameters prevent modification

8. NAMESPACES
   - std:: prefix for standard library
   - Keeps code organized, avoids name conflicts

)" << std::endl;

    std::cout << "═══════════════════════════════════════════" << std::endl;
    std::cout << "PROGRAM COMPLETE!" << std::endl;
    std::cout << "═══════════════════════════════════════════" << std::endl;
    
    return 0;  // Return 0 to indicate successful execution
}
