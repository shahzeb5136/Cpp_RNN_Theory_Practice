/**
 * =============================================================================
 * RECURRENT NEURAL NETWORK (RNN) - HEADER FILE
 * =============================================================================
 * 
 * This file contains the declarations for a simple RNN implementation.
 * Based on the main RNN formula:
 * 
 *     h_t = f(W * x_t + U * h_{t-1} + b)
 * 
 * Where:
 *   - h_t     : Current hidden state vector
 *   - f       : Activation function (ReLU in our case)
 *   - W       : Input-to-hidden weight matrix
 *   - x_t     : Current input vector (word embedding)
 *   - U       : Hidden-to-hidden weight matrix  
 *   - h_{t-1} : Previous hidden state vector
 *   - b       : Bias vector
 * 
 * For output/prediction:
 *     y_t = Softmax(V * h_t)
 * 
 * Where:
 *   - V       : Hidden-to-output (vocabulary) weight matrix
 *   - y_t     : Output probability distribution
 * 
 * =============================================================================
 */

#ifndef RNN_HPP        // Include guard - prevents multiple inclusions
#define RNN_HPP        // Define the guard

// -----------------------------------------------------------------------------
// STANDARD LIBRARY INCLUDES
// -----------------------------------------------------------------------------
#include <vector>      // For dynamic arrays (std::vector)
#include <string>      // For text handling (std::string)
#include <random>      // For random number generation
#include <fstream>     // For file I/O (CSV output)
#include <iostream>    // For console output (std::cout)
#include <cmath>       // For mathematical functions (exp, max)
#include <iomanip>     // For output formatting (std::setprecision)
#include <sstream>     // For string stream operations

// -----------------------------------------------------------------------------
// TYPE ALIASES (using statements make code more readable)
// -----------------------------------------------------------------------------
// A Vector is a 1D array of doubles (e.g., [0.1, 0.2, 0.3])
using Vector = std::vector<double>;

// A Matrix is a 2D array of doubles (vector of vectors)
// e.g., [[0.1, 0.2], [0.3, 0.4]] represents a 2x2 matrix
using Matrix = std::vector<std::vector<double>>;

// -----------------------------------------------------------------------------
// UTILITY FUNCTIONS (declared here, defined in rnn.cpp)
// -----------------------------------------------------------------------------

/**
 * Saves a vector to a CSV file
 * @param vec    The vector to save
 * @param filename  Name of the output file
 * @param label  Optional label/description for the vector
 */
void saveVectorToCSV(const Vector& vec, const std::string& filename, 
                     const std::string& label = "");

/**
 * Saves a matrix to a CSV file
 * @param mat    The matrix to save
 * @param filename  Name of the output file
 * @param label  Optional label/description for the matrix
 */
void saveMatrixToCSV(const Matrix& mat, const std::string& filename,
                     const std::string& label = "");

/**
 * Prints a vector to the console with formatting
 * @param vec    The vector to print
 * @param name   Name/label to display
 */
void printVector(const Vector& vec, const std::string& name);

/**
 * Prints a matrix to the console with formatting
 * @param mat    The matrix to print
 * @param name   Name/label to display
 */
void printMatrix(const Matrix& mat, const std::string& name);

// -----------------------------------------------------------------------------
// RNN CLASS DEFINITION
// -----------------------------------------------------------------------------
/**
 * The RNN class encapsulates all the components of a simple RNN.
 * 
 * MEMBER VARIABLES (the data the class holds):
 *   - Dimensions: inputSize, hiddenSize, outputSize
 *   - Weight matrices: W, U, V
 *   - Bias vectors: b, c
 *   - State: hiddenState (h_t)
 *   - Random generator for initialization
 * 
 * MEMBER FUNCTIONS (what the class can do):
 *   - Constructor: Initialize the RNN with given sizes
 *   - forward(): Process one input and update hidden state
 *   - Various activation functions (ReLU, Softmax)
 *   - Matrix-vector operations
 */
class RNN {
// -----------------------------------------------------------------------------
// PRIVATE MEMBERS (only accessible within the class)
// -----------------------------------------------------------------------------
private:
    // Network dimensions
    int inputSize;    // Size of input vector (e.g., word embedding dimension)
    int hiddenSize;   // Size of hidden state vector
    int outputSize;   // Size of output vector (e.g., vocabulary size)
    
    // Weight matrices (these are learned parameters in real training)
    Matrix W;         // Input-to-hidden weights  [hiddenSize x inputSize]
    Matrix U;         // Hidden-to-hidden weights [hiddenSize x hiddenSize]
    Matrix V;         // Hidden-to-output weights [outputSize x hiddenSize]
    
    // Bias vectors
    Vector b;         // Hidden layer bias  [hiddenSize]
    Vector c;         // Output layer bias  [outputSize]
    
    // Current hidden state (memory of the network)
    Vector hiddenState;  // h_t [hiddenSize]
    
    // Random number generator for weight initialization
    std::mt19937 rng;                    // Mersenne Twister RNG
    std::uniform_real_distribution<double> dist;  // Uniform distribution
    
    // Step counter for CSV file naming
    int stepCounter;
    
    // Output directory for CSV files
    std::string outputDir;

// -----------------------------------------------------------------------------
// PUBLIC MEMBERS (accessible from outside the class)
// -----------------------------------------------------------------------------
public:
    /**
     * Constructor - Creates and initializes the RNN
     * 
     * @param inputSize   Dimension of input vectors
     * @param hiddenSize  Dimension of hidden state
     * @param outputSize  Dimension of output (vocabulary size)
     * @param outputDir   Directory to save CSV files
     * 
     * In C++, the constructor has the same name as the class and no return type.
     * It's called automatically when you create an object of this class.
     */
    RNN(int inputSize, int hiddenSize, int outputSize, 
        const std::string& outputDir = "output");
    
    /**
     * Forward pass - Process one input through the network
     * 
     * This implements: h_t = f(W * x_t + U * h_{t-1} + b)
     *             and: y_t = softmax(V * h_t + c)
     * 
     * @param input  The input vector x_t
     * @return       The output probability distribution y_t
     */
    Vector forward(const Vector& input);
    
    /**
     * Reset the hidden state to zeros
     * Call this when starting a new sequence
     */
    void resetHiddenState();
    
    /**
     * Get the current hidden state (for inspection/debugging)
     * @return  The current hidden state vector
     */
    Vector getHiddenState() const;
    
    /**
     * Get weight matrices (for inspection/debugging)
     * 'const' after the function means it won't modify the object
     */
    Matrix getW() const { return W; }
    Matrix getU() const { return U; }
    Matrix getV() const { return V; }
    Vector getB() const { return b; }
    Vector getC() const { return c; }

// -----------------------------------------------------------------------------
// PRIVATE HELPER FUNCTIONS
// -----------------------------------------------------------------------------
private:
    /**
     * Initialize a matrix with random values
     * Uses Xavier initialization: values scaled by sqrt(2 / (fan_in + fan_out))
     * 
     * @param rows     Number of rows
     * @param cols     Number of columns
     * @return         Initialized matrix
     */
    Matrix initializeMatrix(int rows, int cols);
    
    /**
     * Initialize a vector with zeros
     * @param size  Length of the vector
     * @return      Zero-initialized vector
     */
    Vector initializeVector(int size);
    
    /**
     * Matrix-vector multiplication: result = matrix * vector
     * 
     * If matrix is [m x n] and vector is [n], result is [m]
     * Each element result[i] = sum over j of (matrix[i][j] * vector[j])
     * 
     * @param matrix  The matrix [m x n]
     * @param vec     The vector [n]
     * @return        Result vector [m]
     */
    Vector matVecMul(const Matrix& matrix, const Vector& vec);
    
    /**
     * Vector addition: result = a + b (element-wise)
     * @param a  First vector
     * @param b  Second vector
     * @return   Sum vector
     */
    Vector vecAdd(const Vector& a, const Vector& b);
    
    /**
     * ReLU activation function (element-wise)
     * ReLU(x) = max(0, x)
     * 
     * This introduces non-linearity into the network.
     * Without activation functions, stacking layers would be useless
     * because multiple linear transformations = one linear transformation.
     * 
     * @param vec  Input vector
     * @return     Vector with ReLU applied to each element
     */
    Vector relu(const Vector& vec);
    
    /**
     * Softmax function - converts values to probability distribution
     * 
     * softmax(x_i) = exp(x_i) / sum(exp(x_j) for all j)
     * 
     * Properties:
     *   - All outputs are between 0 and 1
     *   - All outputs sum to 1
     *   - Larger inputs get larger probabilities
     * 
     * @param vec  Input vector (logits)
     * @return     Probability distribution
     */
    Vector softmax(const Vector& vec);
    
    /**
     * Save the current step's computations to CSV files
     * @param stepName  Name/label for this step
     * @param data      Vector to save
     */
    void saveStepVector(const std::string& stepName, const Vector& data);
    
    /**
     * Save matrix data for a step
     * @param stepName  Name/label for this step
     * @param data      Matrix to save
     */
    void saveStepMatrix(const std::string& stepName, const Matrix& data);
};

#endif // RNN_HPP
