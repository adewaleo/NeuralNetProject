/*
 * To change this license header, choose License Headers in Project Properties.
 * To change this template file, choose Tools | Templates
 * and open the template in the editor.
 * Author: Oluwwatosin V. Adewale 
 * Code/logic based on Michael Nielsen's Neural Network code
 * https://github.com/mnielsen/neural-networks-and-deep-learning
 * Also see http://neuralnetworksanddeeplearning.com/chap1.html
 * 
 */
package neuralnet;

// imports
import Jama.Matrix;
import java.util.Map;
import java.util.HashMap;
import java.util.List;
import java.util.Random;

// TODO: matrix based approach.
// TODO: bounds checking and checking of exceptions / stricter?
// TODO: check in place operations, my implementations
// TODO: check proper bounds, particularly for activattions array.
// TODO: check exception.
// TODO: make sure new weights or biases are correct.
/**
 *
 * @author Oluwatosin V. Adewale
 */
public class Network {

    // weights for each layer of the network
    private Matrix[] weights;
    // biases for each layer of the network. It is a row vector
    private Matrix[] biases;
    // number of layers except input layer;
    private final int numLayers;
    // the activation function for each neuron
    private Function activationFunc;
    // model of network for visualizing backpropagation etc.
    private VisualizationModel model;
    // apply the function not its derivative
    private static boolean APPLY = false;
    // apply the function's derivative
    private static boolean APPLYDERIVATIVE = true;
    // constants for biad and gradient keys
    public static boolean BIAS_key = false;
    public static boolean WEIGHT_key = true;

    /**
     * Create a network from a list of layer sizes (input included) and an activation function
     *
     * @param layers
     * @param func
     */
    public Network(int[] layers, Function func) {
        // initialize instance variables
        numLayers = layers.length - 1;
        weights = new Matrix[numLayers];
        biases = new Matrix[numLayers];
        activationFunc = func;

        for (int i = 0; i < numLayers; i++) {
            biases[i] = new Matrix(rand2DArray(layers[i + 1], 1));
            weights[i] = new Matrix(rand2DArray(layers[i + 1], layers[i]));
        }
    }

    // construct a network from another network
    public Network(Network that) {
        this.numLayers = that.numLayers;
        this.weights = new Matrix[numLayers];
        this.biases = new Matrix[numLayers];
        this.activationFunc = that.activationFunc;

        for (int i = 0; i < numLayers; i++) {
            this.biases[i] = that.biases[i].copy();
            this.weights[i] = that.weights[i].copy();
        }

    }

    // return 2D array with values drawn from a guassian random distribution with
    // mean of 0 and standard deviation of 1.
    /**
     *
     * @param row
     * @param col
     * @return
     */
    private double[][] rand2DArray(int row, int col) {
        Random rand = new Random();
        double[][] temp = new double[row][col];
        for (int i = 0; i < row; i++) {
            for (int j = 0; j < col; j++) {
                temp[i][j] = rand.nextGaussian();
            }
        }
        return temp;
    }

    /**
     * Apply given function to Matrix. Returns result in a new matrix.
     * @param m
     * @param applyDerivative
     * @return A new matrix that is the result of applying the appropriate 
     *        function to every element in the Matrix
     */
    private Matrix applyFunction(Matrix m, boolean applyDerivative) {
        int rows, cols;
        rows = m.getRowDimension();
        cols = m.getColumnDimension();

        Matrix result = new Matrix(rows, cols);

        if (applyDerivative) {
            for (int i = 0; i < rows; i++) {
                for (int j = 0; j < cols; j++) {
                    result.set(i, j, activationFunc.applyDerivative(m.get(i, j)));
                }
            }
        } else {
            for (int i = 0; i < rows; i++) {
                for (int j = 0; j < cols; j++) {
                    result.set(i, j, activationFunc.apply(m.get(i, j)));
                }
            }
        }

        return result;
    }

    /**
     * Applies the input a to the neural network and returns the output
     * a should be a column vector (n-by-1)
     * @param a A column Matrix (n-by-1)
     * @return  A
     */
    public Matrix feedForward(Matrix a) {
        // for every layer
        for (int i = 0; i < numLayers; i++) {
            // multiply by weights
            a = weights[i].times(a);
            a = a.plus(biases[i]);
            //apply function to every element.
            a = applyFunction(a, APPLY);
        }
        return a;
    }

    /**
     * Calculate the accuracy of the Network on some list of samples
     * @param data
     * @return 
     */
    public double calculateAccuracy(List<Sample> data) {
        if (data.size() < 1) {
            throw new IllegalArgumentException("Should have at least one sample");
        }
        Matrix testExample = data.get(0).getInput();
        Matrix testLabel = data.get(0).getLabel();
        
        if (testExample.getRowDimension() != this.weights[0].getColumnDimension() ) {
            throw new IllegalArgumentException("Mismatch between example sizes and network input size.");
        }  
        if (testLabel.getRowDimension() != this.weights[this.numLayers - 1].getRowDimension()){
            System.out.printf("Label Rows: %d. Weight row: %d. weight columns %d", 
                    testLabel.getRowDimension(), 
                    this.weights[this.numLayers - 1].getRowDimension(), 
                    this.weights[this.numLayers - 1].getColumnDimension());
            throw new IllegalArgumentException("Mismatch between label size and network output size.");
        }
        
        int count = 0;
        double total = 0;
        for (Sample s : data) {
            count++;
            boolean match = true;
            Matrix label = s.getLabel();
            Matrix output = this.feedForward(s.getInput());

            for (int i = 0; i < label.getRowDimension(); i++) {
                if (label.get(i, 0) < 0.50 && output.get(i, 0) >= 0.50) {
                    match = false;
                    break;
                } else if (label.get(i, 0) >= 0.50 && output.get(i, 0) < 0.50) {
                    match = false;
                    break;
                }

            }

            if (match) {
                total++;
            }
        }

        return (total / count) * 100;
    }

    // derivative of cost function with respect to activation layer. 
    // here cost function is quadratic cost function
    private Matrix costDerivative(Matrix outputActivations, Matrix label) {
        return outputActivations.minus(label);
    }

    /**
     * Carries out back-propagation of error between desired output and actual
     * output. This generates error gradients for weights and biases.
     *
     * @param xInput input Matrix (vector)
     * @param yLabel desired output Matrix (vector)
     * @return Map containing error gradients for weights and biases.
     *
     */
    Map<Boolean, Matrix[]> backpropagation(Matrix xInput, Matrix yLabel) {
        Matrix activation, delta;
        Matrix[] gradBiases, gradWeights, activations, weightedInputs;

        // initialize bias and weight gradient matrixes with all zeros
        gradBiases = new Matrix[numLayers];
        gradWeights = new Matrix[numLayers];
        for (int i = 0; i < numLayers; i++) {
            gradBiases[i] = new Matrix(biases[i].getRowDimension(), biases[i].getColumnDimension());
            gradWeights[i] = new Matrix(weights[i].getRowDimension(), weights[i].getColumnDimension());
        }

        activation = xInput;
        // array of activations includes inputs
        activations = new Matrix[numLayers + 1];
        activations[0] = xInput;
        weightedInputs = new Matrix[numLayers];

        // feedforward and generate weighted inputs and activations
        for (int i = 0; i < numLayers; i++) {
            // multiply activation of weights of prev layer by activation and add biases
            //printMatrixDimensions(activation, "activation matrix");
            //printMatrixDimensions(weights[i], "weight matrix"); 
            Matrix z = weights[i].times(activations[i]);
            z = z.plusEquals(biases[i]);
            //apply function to every element
            activations[i + 1] = applyFunction(z, APPLY);
            weightedInputs[i] = z;
        }

        // get error of output layer by evaluating Hadamard (element-wise) product of 
        // cost derivative and gradient of activation function with respect to input layer
        delta = costDerivative(activations[numLayers], yLabel)
                .arrayTimesEquals(applyFunction(weightedInputs[numLayers - 1], APPLYDERIVATIVE));

        final Matrix outputError = delta.copy();

        gradBiases[numLayers - 1] = delta;
        gradWeights[numLayers - 1] = delta.times(activations[numLayers - 1].transpose());

        // from the second to the last layer to the first layer, do the following
        for (int i = numLayers - 2; i >= 0; i--) {
            Matrix z = weightedInputs[i];
            Matrix deriv = applyFunction(z, APPLYDERIVATIVE);
            delta = weights[i + 1].transpose().times(delta).arrayTimesEquals(deriv);

            gradBiases[i] = delta;
            gradWeights[i] = delta.times(activations[i].transpose());

        }

        this.model = new VisualizationModel(outputError, activations,
                weightedInputs, this.weights, this.biases);

        // map to hold the result. result[0] = biases. result[1] = weights.
        Map<Boolean, Matrix[]> result = new HashMap<>();
        result.put(BIAS_key, gradBiases);
        result.put(WEIGHT_key, gradWeights);
        return result;
    }

    // update network's weights new weights
    void updateWeights(Matrix[] newWeights) {
        if (newWeights == null || newWeights.length != numLayers) {
            throw new IllegalArgumentException("Input is null or has incorrect number of layers");
        }

        this.weights = newWeights;
    }

    // update network's biases with new biases
    void updateBiases(Matrix[] newBiases) {
        if (newBiases == null || newBiases.length != numLayers) {
            throw new IllegalArgumentException("Input is null or has incorrect number of layers");
        }
        this.biases = newBiases;
    }

    /**
     * @return a copy of this Network's weights
     */
    public Matrix[] getWeights() {

        Matrix[] weightsCopy = new Matrix[weights.length];
        for (int i = 0; i < numLayers; i++) {
            weightsCopy[i] = weights[i].copy();
        }
        return weightsCopy;
    }

    /**
     * @return a copy of this Network's biases
     */
    public Matrix[] getBiases() {
        Matrix[] biasesCopy = new Matrix[biases.length];
        for (int i = 0; i < numLayers; i++) {
            biasesCopy[i] = biases[i].copy();
        }
        return biasesCopy;
    }

    static void printMatrixDimensions(Matrix m, String name) {
        System.out.printf("Matrix %s has dimensions row: %d col: %d\n", name, m.getRowDimension(),
                m.getColumnDimension());
    }

    void printNet() {
        System.out.println("Printing weight Matrices");
        int i = 1;
        for (Matrix m : this.weights) {
            System.out.printf("Weights for layer %d", i);
            m.print(5, 3);
            i++;
        }

        System.out.println("Printing bias Matrices");
        i = 1;
        for (Matrix m : this.biases) {
            System.out.printf("Weights for layer %d", i);
            m.print(5, 3);
            i++;
        }
    }

    /**
     * Gets a model of the network for backPropagation
     *
     * @return
     */
    private VisualizationModel getVisualizationModel() {
        return model;
    }

    /**
     * A container of Visualization Information. Contains copy of actual network
     * information. It stores network state post back-propagation and
     * pre-updates
     */
    public static class VisualizationModel {

        // information to visualize backpropagation
        private Matrix outputError;
        private Matrix[] activations, weightedInputs, weights, biases;

        /**
         * Constructs Visualization model using copies of the parameters.
         *
         * @param outputError
         * @param activations
         * @param weightedInputs
         * @param weights
         * @param biases
         */
        public VisualizationModel(Matrix outputError, Matrix[] activations,
                Matrix[] weightedInputs, Matrix[] weights, Matrix[] biases) {
            this.outputError = outputError.copy();
            this.activations = copyMatrixArray(activations);
            this.weightedInputs = copyMatrixArray(weightedInputs);
            this.weights = copyMatrixArray(weights);
            this.biases = copyMatrixArray(biases);
        }

        public Matrix[] getActivations() {
            return activations;
        }

        public Matrix[] getBiases() {
            return biases;
        }

        public Matrix[] getWeights() {
            return weights;
        }

        public Matrix getOutputError() {
            return outputError;
        }

        public Matrix[] getWeightedInputs() {
            return weightedInputs;
        }

        private Matrix[] copyMatrixArray(Matrix[] input) {
            Matrix[] result = new Matrix[input.length];
            for (int i = 0; i < input.length; i++) {
                result[i] = input[i].copy();
            }
            return result;
        }
    }

    /**
     * @param args the command line arguments
     */
    public static void main(String[] args) {
        // TODO Testing Here.
    }
}
