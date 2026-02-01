/**
 * =============================================================================
 * RECURRENT NEURAL NETWORK (RNN) - IMPLEMENTATION FILE
 * =============================================================================
 * 
 * This file contains the implementations of all RNN class methods
 * and utility functions declared in rnn.hpp
 * 
 * =============================================================================
 */

#include "rnn.hpp"     // Our header file
#include <filesystem>  // For creating directories (C++17 feature)
#include <algorithm>   // For std::max_element
#include <numeric>     // For std::accumulate

// Use the filesystem namespace for convenience
namespace fs = std::filesystem;

// =============================================================================
// UTILITY FUNCTION IMPLEMENTATIONS
// =============================================================================

/**
 * Saves a vector to a CSV file
 * 
 * CSV (Comma-Separated Values) is a simple text format where:
 *   - Each line is a row
 *   - Values are separated by commas
 * 
 * For a vector, we save it as a single column (one value per row)
 * This makes it easy to visualize in Excel, Google Sheets, or Python/pandas
 */
void saveVectorToCSV(const Vector& vec, const std::string& filename, 
                     const std::string& label) {
    // Create an output file stream
    // std::ofstream = "output file stream" - used for writing files
    std::ofstream file(filename);
    
    // Check if file opened successfully
    if (!file.is_open()) {
        std::cerr << "Error: Could not open file " << filename << std::endl;
        return;
    }
    
    // Write header/label if provided
    if (!label.empty()) {
        file << label << std::endl;
    }
    
    // Write each value on its own line
    // std::setprecision sets decimal places for floating point output
    file << std::fixed << std::setprecision(6);
    for (size_t i = 0; i < vec.size(); ++i) {
        file << "v[" << i << "]," << vec[i] << std::endl;
    }
    
    file.close();  // Close the file (good practice, though destructor does this)
    std::cout << "  -> Saved vector to: " << filename << std::endl;
}

/**
 * Saves a matrix to a CSV file
 * 
 * Each row of the matrix becomes a row in the CSV
 * Columns are separated by commas
 */
void saveMatrixToCSV(const Matrix& mat, const std::string& filename,
                     const std::string& label) {
    std::ofstream file(filename);
    
    if (!file.is_open()) {
        std::cerr << "Error: Could not open file " << filename << std::endl;
        return;
    }
    
    // Write label if provided
    if (!label.empty()) {
        file << label << std::endl;
    }
    
    // Write column headers (indices)
    file << "row/col";
    if (!mat.empty()) {
        for (size_t j = 0; j < mat[0].size(); ++j) {
            file << ",col_" << j;
        }
    }
    file << std::endl;
    
    // Write each row
    file << std::fixed << std::setprecision(6);
    for (size_t i = 0; i < mat.size(); ++i) {
        file << "row_" << i;  // Row label
        for (size_t j = 0; j < mat[i].size(); ++j) {
            file << "," << mat[i][j];
        }
        file << std::endl;
    }
    
    file.close();
    std::cout << "  -> Saved matrix to: " << filename << std::endl;
}

/**
 * Prints a vector to console with nice formatting
 */
void printVector(const Vector& vec, const std::string& name) {
    std::cout << name << " [" << vec.size() << "]: [ ";
    std::cout << std::fixed << std::setprecision(4);
    for (size_t i = 0; i < vec.size(); ++i) {
        std::cout << vec[i];
        if (i < vec.size() - 1) std::cout << ", ";
    }
    std::cout << " ]" << std::endl;
}

/**
 * Prints a matrix to console with nice formatting
 */
void printMatrix(const Matrix& mat, const std::string& name) {
    std::cout << name << " [" << mat.size() << " x " 
              << (mat.empty() ? 0 : mat[0].size()) << "]:" << std::endl;
    std::cout << std::fixed << std::setprecision(4);
    for (size_t i = 0; i < mat.size(); ++i) {
        std::cout << "  [ ";
        for (size_t j = 0; j < mat[i].size(); ++j) {
            std::cout << std::setw(8) << mat[i][j];
            if (j < mat[i].size() - 1) std::cout << ", ";
        }
        std::cout << " ]" << std::endl;
    }
}

// =============================================================================
// RNN CLASS IMPLEMENTATION
// =============================================================================

/**
 * Constructor Implementation
 * 
 * The syntax ClassName::functionName means "functionName belongs to ClassName"
 * The colon after the parameter list starts the "initialization list"
 * which is an efficient way to initialize member variables
 */
RNN::RNN(int inputSize, int hiddenSize, int outputSize, const std::string& outputDir)
    : inputSize(inputSize),      // Initialize inputSize member with parameter
      hiddenSize(hiddenSize),    // Initialize hiddenSize member
      outputSize(outputSize),    // Initialize outputSize member
      rng(42),                   // Seed random generator with 42 (for reproducibility)
      dist(-0.5, 0.5),          // Uniform distribution between -0.5 and 0.5
      stepCounter(0),            // Start step counter at 0
      outputDir(outputDir)       // Set output directory
{
    // Create output directory if it doesn't exist
    // fs::create_directories creates parent directories too if needed
    fs::create_directories(outputDir);
    
    std::cout << "============================================" << std::endl;
    std::cout << "Initializing RNN with:" << std::endl;
    std::cout << "  Input size:  " << inputSize << std::endl;
    std::cout << "  Hidden size: " << hiddenSize << std::endl;
    std::cout << "  Output size: " << outputSize << std::endl;
    std::cout << "============================================" << std::endl;
    
    // Initialize all weights and biases
    // These would normally be learned through training, but we use random values
    
    std::cout << "\n--- Initializing Weight Matrices ---" << std::endl;
    
    // W: Input-to-hidden weights [hiddenSize x inputSize]
    // Transforms input x_t into hidden space
    W = initializeMatrix(hiddenSize, inputSize);
    std::cout << "W matrix (input->hidden): " << hiddenSize << " x " << inputSize << std::endl;
    saveStepMatrix("00_W_weights_input_to_hidden", W);
    
    // U: Hidden-to-hidden weights [hiddenSize x hiddenSize]
    // This is what makes it "recurrent" - connects previous hidden state to current
    U = initializeMatrix(hiddenSize, hiddenSize);
    std::cout << "U matrix (hidden->hidden): " << hiddenSize << " x " << hiddenSize << std::endl;
    saveStepMatrix("00_U_weights_hidden_to_hidden", U);
    
    // V: Hidden-to-output weights [outputSize x hiddenSize]
    // Transforms hidden state to output (vocabulary) space
    V = initializeMatrix(outputSize, hiddenSize);
    std::cout << "V matrix (hidden->output): " << outputSize << " x " << hiddenSize << std::endl;
    saveStepMatrix("00_V_weights_hidden_to_output", V);
    
    std::cout << "\n--- Initializing Bias Vectors ---" << std::endl;
    
    // b: Hidden layer bias [hiddenSize]
    b = initializeVector(hiddenSize);
    std::cout << "b vector (hidden bias): " << hiddenSize << std::endl;
    saveStepVector("00_b_bias_hidden", b);
    
    // c: Output layer bias [outputSize]  
    c = initializeVector(outputSize);
    std::cout << "c vector (output bias): " << outputSize << std::endl;
    saveStepVector("00_c_bias_output", c);
    
    std::cout << "\n--- Initializing Embedding Matrix ---" << std::endl;

    // E: Embedding matrix [outputSize x inputSize]
    // Each row is one word's embedding vector
    // When we want x_t for word index 3, we grab row 3 of this matrix
    // 
    // In real systems, embeddings are pre-trained (Word2Vec, GloVe)
    // or learned during training. We use random values here.
    E = initializeMatrix(outputSize, inputSize);
    std::cout << "E matrix (word embeddings): " << outputSize << " x " << inputSize << std::endl;
    std::cout << "  Row 0 = embedding for word 0" << std::endl;
    std::cout << "  Row 1 = embedding for word 1" << std::endl;
    std::cout << "  ...and so on" << std::endl;
    saveStepMatrix("00_E_embedding_matrix", E);

    // Initialize hidden state to zeros
    // This is the "memory" at time t=0 (no memory yet)
    hiddenState = initializeVector(hiddenSize);
    std::cout << "\nInitial hidden state set to zeros." << std::endl;
    saveStepVector("00_initial_hidden_state", hiddenState);
    
    std::cout << "\n============================================" << std::endl;
    std::cout << "RNN Initialization Complete!" << std::endl;
    std::cout << "============================================\n" << std::endl;
}

/**
 * Forward Pass Implementation
 * 
 * This is the core of the RNN - processes one input and produces output
 * 
 * Step by step:
 * 1. Compute W * x_t (transform input to hidden space)
 * 2. Compute U * h_{t-1} (transform previous hidden state)
 * 3. Add them together with bias: W*x_t + U*h_{t-1} + b
 * 4. Apply activation function: h_t = ReLU(...)
 * 5. Compute output: y_t = softmax(V * h_t + c)
 */
Vector RNN::forward(const Vector& input) {
    stepCounter++;  // Increment step counter for file naming
    
    std::cout << "\n========== FORWARD PASS - Step " << stepCounter << " ==========" << std::endl;
    
    // Create prefix for this step's files
    std::string prefix = std::to_string(stepCounter);
    if (stepCounter < 10) prefix = "0" + prefix;  // Pad with zero for sorting
    
    // Save the input
    std::cout << "\n[Step " << stepCounter << ".1] Input vector x_t:" << std::endl;
    printVector(input, "x_t");
    saveStepVector(prefix + "_1_input_x_t", input);
    
    // Save current hidden state (from previous step)
    std::cout << "\n[Step " << stepCounter << ".2] Previous hidden state h_{t-1}:" << std::endl;
    printVector(hiddenState, "h_{t-1}");
    saveStepVector(prefix + "_2_prev_hidden_h_t-1", hiddenState);
    
    // ---------- STEP 1: Compute W * x_t ----------
    std::cout << "\n[Step " << stepCounter << ".3] Computing W * x_t:" << std::endl;
    std::cout << "  Matrix W [" << W.size() << "x" << W[0].size() << "] * vector x_t [" << input.size() << "]" << std::endl;
    Vector Wx = matVecMul(W, input);
    printVector(Wx, "W*x_t");
    saveStepVector(prefix + "_3_W_times_x", Wx);
    
    // ---------- STEP 2: Compute U * h_{t-1} ----------
    std::cout << "\n[Step " << stepCounter << ".4] Computing U * h_{t-1}:" << std::endl;
    std::cout << "  Matrix U [" << U.size() << "x" << U[0].size() << "] * vector h_{t-1} [" << hiddenState.size() << "]" << std::endl;
    Vector Uh = matVecMul(U, hiddenState);
    printVector(Uh, "U*h_{t-1}");
    saveStepVector(prefix + "_4_U_times_h", Uh);
    
    // ---------- STEP 3: Add W*x_t + U*h_{t-1} ----------
    std::cout << "\n[Step " << stepCounter << ".5] Computing W*x_t + U*h_{t-1}:" << std::endl;
    Vector sum1 = vecAdd(Wx, Uh);
    printVector(sum1, "W*x_t + U*h_{t-1}");
    saveStepVector(prefix + "_5_Wx_plus_Uh", sum1);
    
    // ---------- STEP 4: Add bias b ----------
    std::cout << "\n[Step " << stepCounter << ".6] Adding bias b:" << std::endl;
    Vector preActivation = vecAdd(sum1, b);
    printVector(preActivation, "W*x_t + U*h_{t-1} + b (pre-activation)");
    saveStepVector(prefix + "_6_pre_activation", preActivation);
    
    // ---------- STEP 5: Apply ReLU activation ----------
    std::cout << "\n[Step " << stepCounter << ".7] Applying ReLU activation:" << std::endl;
    std::cout << "  ReLU(x) = max(0, x) for each element" << std::endl;
    hiddenState = relu(preActivation);  // Update hidden state!
    printVector(hiddenState, "h_t = ReLU(W*x_t + U*h_{t-1} + b)");
    saveStepVector(prefix + "_7_hidden_state_h_t", hiddenState);
    
    // ---------- STEP 6: Compute output V * h_t ----------
    std::cout << "\n[Step " << stepCounter << ".8] Computing V * h_t (output projection):" << std::endl;
    std::cout << "  Matrix V [" << V.size() << "x" << V[0].size() << "] * vector h_t [" << hiddenState.size() << "]" << std::endl;
    Vector Vh = matVecMul(V, hiddenState);
    printVector(Vh, "V*h_t");
    saveStepVector(prefix + "_8_V_times_h", Vh);
    
    // ---------- STEP 7: Add output bias c ----------
    std::cout << "\n[Step " << stepCounter << ".9] Adding output bias c:" << std::endl;
    Vector logits = vecAdd(Vh, c);
    printVector(logits, "logits = V*h_t + c");
    saveStepVector(prefix + "_9_logits", logits);
    
    // ---------- STEP 8: Apply Softmax ----------
    std::cout << "\n[Step " << stepCounter << ".10] Applying Softmax:" << std::endl;
    std::cout << "  softmax(x_i) = exp(x_i) / sum(exp(x_j))" << std::endl;
    Vector output = softmax(logits);
    printVector(output, "y_t = softmax(V*h_t + c)");
    saveStepVector(prefix + "_10_output_y_t", output);
    
    // Verify softmax output sums to 1
    double sum = 0.0;
    for (double val : output) sum += val;
    std::cout << "  Sum of output probabilities: " << sum << " (should be ~1.0)" << std::endl;
    
    // Find the predicted class (highest probability)
    auto maxIt = std::max_element(output.begin(), output.end());
    int predictedClass = std::distance(output.begin(), maxIt);
    std::cout << "  Predicted class: " << predictedClass 
              << " with probability: " << *maxIt << std::endl;
    
    std::cout << "\n========== END FORWARD PASS - Step " << stepCounter << " ==========\n" << std::endl;
    
    return output;
}

/**
 * Reset hidden state to zeros
 * Call this when starting a new sequence
 */
void RNN::resetHiddenState() {
    std::fill(hiddenState.begin(), hiddenState.end(), 0.0);
    std::cout << "Hidden state reset to zeros." << std::endl;
}

/**
 * Get the current hidden state
 * 'const' means this function doesn't modify the object
 */
Vector RNN::getHiddenState() const {
    return hiddenState;
}

// =============================================================================
// PRIVATE HELPER FUNCTION IMPLEMENTATIONS
// =============================================================================

/**
 * Initialize a matrix with random values
 * 
 * We use Xavier/Glorot initialization which helps with training:
 * Values are drawn from a uniform distribution and scaled by
 * sqrt(2 / (fan_in + fan_out)) where fan_in and fan_out are
 * the number of input and output neurons.
 */
Matrix RNN::initializeMatrix(int rows, int cols) {
    // Create a matrix with 'rows' rows, each containing 'cols' zeros
    Matrix mat(rows, Vector(cols, 0.0));
    
    // Xavier initialization scale factor
    double scale = std::sqrt(2.0 / (rows + cols));
    
    // Fill with random values
    for (int i = 0; i < rows; ++i) {
        for (int j = 0; j < cols; ++j) {
            // dist(rng) generates a random number in [-0.5, 0.5]
            // Multiply by 2 to get [-1, 1], then scale
            mat[i][j] = dist(rng) * 2.0 * scale;
        }
    }
    
    return mat;
}

/**
 * Initialize a vector with zeros
 * Biases are typically initialized to 0
 */
Vector RNN::initializeVector(int size) {
    return Vector(size, 0.0);  // Vector of 'size' zeros
}

/**
 * Matrix-Vector Multiplication
 * 
 * For a matrix A of size [m x n] and vector v of size [n],
 * the result is a vector of size [m] where:
 *   result[i] = sum over j from 0 to n-1 of (A[i][j] * v[j])
 * 
 * Visual example for 2x3 matrix and 3-element vector:
 * 
 *   [a00 a01 a02]   [v0]     [a00*v0 + a01*v1 + a02*v2]
 *   [a10 a11 a12] * [v1]  =  [a10*v0 + a11*v1 + a12*v2]
 *                   [v2]
 */
Vector RNN::matVecMul(const Matrix& matrix, const Vector& vec) {
    // Check dimensions
    if (matrix.empty() || matrix[0].size() != vec.size()) {
        std::cerr << "Error: Matrix-vector dimension mismatch!" << std::endl;
        std::cerr << "  Matrix cols: " << (matrix.empty() ? 0 : matrix[0].size()) 
                  << ", Vector size: " << vec.size() << std::endl;
        return Vector();  // Return empty vector on error
    }
    
    int rows = matrix.size();
    int cols = matrix[0].size();
    
    Vector result(rows, 0.0);  // Initialize result vector with zeros
    
    // Perform multiplication
    for (int i = 0; i < rows; ++i) {
        for (int j = 0; j < cols; ++j) {
            result[i] += matrix[i][j] * vec[j];
        }
    }
    
    return result;
}

/**
 * Vector Addition
 * 
 * Adds two vectors element-wise:
 *   result[i] = a[i] + b[i] for all i
 * 
 * Both vectors must have the same length.
 */
Vector RNN::vecAdd(const Vector& a, const Vector& b) {
    // Check dimensions
    if (a.size() != b.size()) {
        std::cerr << "Error: Vector dimension mismatch in addition!" << std::endl;
        std::cerr << "  Vector a size: " << a.size() 
                  << ", Vector b size: " << b.size() << std::endl;
        return Vector();
    }
    
    Vector result(a.size());
    
    for (size_t i = 0; i < a.size(); ++i) {
        result[i] = a[i] + b[i];
    }
    
    return result;
}

/**
 * ReLU (Rectified Linear Unit) Activation Function
 * 
 * ReLU(x) = max(0, x)
 * 
 * Applied element-wise to a vector.
 * 
 * Properties:
 *   - Non-linear (allows network to learn complex patterns)
 *   - Sparse activation (zeros out negative values)
 *   - Computationally efficient
 *   - Helps with vanishing gradient problem (compared to sigmoid/tanh)
 * 
 * Graph:
 *        |    /
 *        |   /
 *        |  /
 *   -----+------ x
 *        |
 *   (output = 0 for x < 0, output = x for x >= 0)
 */
Vector RNN::relu(const Vector& vec) {
    Vector result(vec.size());
    
    for (size_t i = 0; i < vec.size(); ++i) {
        // std::max returns the larger of two values
        result[i] = std::max(0.0, vec[i]);
    }
    
    return result;
}

/**
 * Softmax Function
 * 
 * Converts a vector of arbitrary real numbers into a probability distribution.
 * 
 * softmax(x_i) = exp(x_i) / sum(exp(x_j) for all j)
 * 
 * Properties:
 *   - All outputs are in range (0, 1)
 *   - All outputs sum to exactly 1
 *   - Preserves ordering (larger input -> larger probability)
 *   - Amplifies differences (large values dominate)
 * 
 * Numerical Stability:
 *   Computing exp(x) for large x can overflow. To prevent this,
 *   we subtract the maximum value from all elements first.
 *   This doesn't change the result because:
 *   exp(x_i - c) / sum(exp(x_j - c)) = exp(x_i)/exp(c) / (sum(exp(x_j))/exp(c))
 *                                    = exp(x_i) / sum(exp(x_j))
 */
Vector RNN::softmax(const Vector& vec) {
    Vector result(vec.size());
    
    // Find the maximum value for numerical stability
    double maxVal = *std::max_element(vec.begin(), vec.end());
    
    // Compute exp(x_i - max) for each element and sum them
    double sumExp = 0.0;
    for (size_t i = 0; i < vec.size(); ++i) {
        result[i] = std::exp(vec[i] - maxVal);  // Subtract max for stability
        sumExp += result[i];
    }
    
    // Normalize by the sum to get probabilities
    for (size_t i = 0; i < vec.size(); ++i) {
        result[i] /= sumExp;
    }
    
    return result;
}

/**
 * Save a vector to CSV with step-specific naming
 */
void RNN::saveStepVector(const std::string& stepName, const Vector& data) {
    std::string filename = outputDir + "/" + stepName + ".csv";
    saveVectorToCSV(data, filename, stepName);
}

/**
 * Save a matrix to CSV with step-specific naming
 */
void RNN::saveStepMatrix(const std::string& stepName, const Matrix& data) {
    std::string filename = outputDir + "/" + stepName + ".csv";
    saveMatrixToCSV(data, filename, stepName);
}

// =============================================================================
// WORD SELECTION FROM PROBABILITY DISTRIBUTION
// =============================================================================

/**
 * Pick a word index from softmax output
 *
 * GREEDY (argmax):
 *   Simply pick whichever word has the highest probability.
 *   Deterministic - same input always gives same output.
 *   Pro: Predictable. Con: Boring, repetitive.
 *
 * SAMPLING:
 *   Treat the probabilities as a weighted random choice.
 *   Imagine a spinner wheel where each word gets a slice
 *   proportional to its probability.
 *   Pro: Variety. Con: Can pick unlikely words.
 *
 * Example with probs = [0.1, 0.6, 0.3]:
 *   Greedy always picks index 1 (0.6)
 *   Sampling picks index 1 ~60% of the time, index 2 ~30%, index 0 ~10%
 */
int pickWordFromDistribution(const Vector& probs, bool greedy, std::mt19937& rng) {
    if (greedy) {
        // Argmax: find the index of the largest value
        int bestIndex = 0;
        double bestProb = probs[0];

        for (size_t i = 1; i < probs.size(); ++i) {
            if (probs[i] > bestProb) {
                bestProb = probs[i];
                bestIndex = static_cast<int>(i);
            }
        }
        return bestIndex;
    }
    else {
        // Sampling: generate a random number in [0, 1)
        // then walk through the probabilities until we exceed it
        // 
        // Visual example with probs [0.1, 0.6, 0.3]:
        //   random = 0.35
        //   cumulative: 0.1 (not yet) -> 0.7 (exceeded!) -> pick index 1

        std::uniform_real_distribution<double> uniform(0.0, 1.0);
        double random = uniform(rng);
        double cumulative = 0.0;

        for (size_t i = 0; i < probs.size(); ++i) {
            cumulative += probs[i];
            if (random < cumulative) {
                return static_cast<int>(i);
            }
        }

        // Fallback (shouldn't reach here if probs sum to 1)
        return static_cast<int>(probs.size() - 1);
    }
}

// =============================================================================
// EMBEDDING LOOKUP
// =============================================================================

/**
 * Look up a word's embedding vector from the embedding matrix
 *
 * This is literally just grabbing a row from the matrix.
 * If wordIndex is 3, we return row 3 of the embedding matrix E.
 *
 * This is how word index -> x_t works:
 *   softmax output -> pick word index -> E[index] -> x_t for next step
 */
Vector RNN::getEmbedding(int wordIndex) const {
    // Bounds checking - make sure the word index is valid
    if (wordIndex < 0 || wordIndex >= static_cast<int>(E.size())) {
        std::cerr << "Error: Word index " << wordIndex
            << " out of range [0, " << E.size() - 1 << "]" << std::endl;
        return Vector(inputSize, 0.0);
    }

    // Just return that row of the embedding matrix
    // E[wordIndex] is a Vector of size inputSize
    return E[wordIndex];
}

// =============================================================================
// AUTOREGRESSIVE GENERATION  (fully traced to CSV)
// =============================================================================

/**
 * Generate a sequence of words autoregressively
 *
 * This is the full loop that connects everything:
 *
 *   start word index
 *        |
 *        v
 *   [Embedding Lookup] --> x_t
 *        |
 *        v
 *   [Forward Pass] --> y_t (probabilities)
 *        |
 *        v
 *   [Pick Word] --> next word index
 *        |
 *        v
 *   [Embedding Lookup] --> x_{t+1}
 *        |
 *        v
 *   [Forward Pass] --> y_{t+1}
 *        |
 *       ...continues...
 *
 * The key insight: the output feeds BACK as the input through
 * the embedding lookup table. That's what makes it "generative".
 *
 * CSV TRACING:
 *   For every generation step the following files are produced
 *   (the forward pass files plus an extra _11_selected_word file):
 *
 *     NN_1_input_x_t.csv           -- x_t (should match row of 00_E)
 *     NN_2_prev_hidden_h_t-1.csv   -- h_{t-1}
 *     NN_3_W_times_x.csv           -- W * x_t
 *     NN_4_U_times_h.csv           -- U * h_{t-1}
 *     NN_5_Wx_plus_Uh.csv          -- sum
 *     NN_6_pre_activation.csv      -- + b
 *     NN_7_hidden_state_h_t.csv    -- ReLU => h_t
 *     NN_8_V_times_h.csv           -- V * h_t
 *     NN_9_logits.csv              -- + c
 *     NN_10_output_y_t.csv         -- softmax => y_t
 *     NN_11_selected_word.csv      -- chosen index, prob, and its embedding
 */
std::vector<int> RNN::generate(int startWordIndex, int length) {
    std::cout << "\n╔══════════════════════════════════════════════════════════╗" << std::endl;
    std::cout << "║  AUTOREGRESSIVE GENERATION                               ║" << std::endl;
    std::cout << "║  softmax -> pick word -> embedding lookup -> next input   ║" << std::endl;
    std::cout << "╚══════════════════════════════════════════════════════════╝\n" << std::endl;

    // Reset hidden state for a fresh sequence
    resetHiddenState();

    // Reset step counter so CSV numbering starts at 01
    stepCounter = 0;

    // Store the sequence of generated word indices
    std::vector<int> generatedSequence;
    generatedSequence.push_back(startWordIndex);

    // The current word index - starts with the seed word
    int currentWordIndex = startWordIndex;

    std::cout << "Starting word index: " << startWordIndex << std::endl;
    std::cout << "Generating " << length << " words...\n" << std::endl;

    for (int t = 0; t < length; ++t) {
        std::cout << "\n╔════════════════════════════════════════════════════════╗" << std::endl;
        std::cout << "║  GENERATION STEP " << t + 1 << "                                      ║" << std::endl;
        std::cout << "╚════════════════════════════════════════════════════════╝" << std::endl;

        // =====================================================
        // STEP A: Embedding lookup (word index -> vector)
        // This is the connection: previous output -> current input
        // =====================================================
        std::cout << "\n[A] Embedding lookup: word index " << currentWordIndex
            << " -> row " << currentWordIndex << " of E" << std::endl;

        Vector x_t = getEmbedding(currentWordIndex);
        printVector(x_t, "x_t = E[" + std::to_string(currentWordIndex) + "]");

        // =====================================================
        // STEP B: Forward pass (x_t -> y_t)
        // forward() increments stepCounter and writes all the
        // sub-step CSV files (NN_1 through NN_10)
        // =====================================================
        std::cout << "\n[B] Running forward pass..." << std::endl;
        Vector y_t = forward(x_t);

        // =====================================================
        // STEP C: Pick next word from probabilities (greedy)
        // =====================================================
        std::cout << "\n[C] Picking next word from softmax output (greedy argmax)..." << std::endl;

        int nextWordIndex = pickWordFromDistribution(y_t, true, rng);

        std::cout << "  Selected word index: " << nextWordIndex
            << " (probability: " << std::fixed << std::setprecision(4)
            << y_t[nextWordIndex] << ")" << std::endl;

        // =====================================================
        // STEP D: Save the selection to a CSV for traceability
        //
        // This file records:
        //   - Which word index was selected
        //   - Its probability from y_t
        //   - The full probability distribution
        //   - The embedding vector that will become the NEXT x_t
        // =====================================================
        {
            std::string prefix = std::to_string(stepCounter);
            if (stepCounter < 10) prefix = "0" + prefix;

            std::string filename = outputDir + "/" + prefix + "_11_selected_word.csv";
            std::ofstream file(filename);
            if (file.is_open()) {
                file << "selected_word_index," << nextWordIndex << std::endl;
                file << "selected_probability," << std::fixed << std::setprecision(6)
                     << y_t[nextWordIndex] << std::endl;

                file << std::endl;
                file << "--- full y_t distribution ---" << std::endl;
                for (size_t i = 0; i < y_t.size(); ++i) {
                    file << "y_t[" << i << "]," << y_t[i];
                    if (static_cast<int>(i) == nextWordIndex) file << ",<-- selected";
                    file << std::endl;
                }

                file << std::endl;
                file << "--- embedding for selected word (will become next x_t) ---" << std::endl;
                Vector nextEmb = getEmbedding(nextWordIndex);
                for (size_t i = 0; i < nextEmb.size(); ++i) {
                    file << "E[" << nextWordIndex << "][" << i << "]," << nextEmb[i] << std::endl;
                }

                file.close();
                std::cout << "  -> Saved selection to: " << filename << std::endl;
            }
        }

        // =====================================================
        // STEP E: Feed back — this word becomes the next input
        // =====================================================
        std::cout << "\n[E] Word " << nextWordIndex
            << " will be fed back as input for next step." << std::endl;
        std::cout << "    Next x_t = E[" << nextWordIndex << "]" << std::endl;

        currentWordIndex = nextWordIndex;
        generatedSequence.push_back(currentWordIndex);
    }

    // Print the full generated sequence
    std::cout << "\n════════════════════════════════════" << std::endl;
    std::cout << "Generated sequence of word indices: [ ";
    for (size_t i = 0; i < generatedSequence.size(); ++i) {
        std::cout << generatedSequence[i];
        if (i < generatedSequence.size() - 1) std::cout << " -> ";
    }
    std::cout << " ]" << std::endl;
    std::cout << "════════════════════════════════════\n" << std::endl;

    return generatedSequence;
}
