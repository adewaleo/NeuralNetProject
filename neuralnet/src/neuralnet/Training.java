/*
 * To change this license header, choose License Headers in Project Properties.
 * To change this template file, choose Tools | Templates
 * and open the template in the editor.
 */
package neuralnet;

import Jama.Matrix;
import java.util.Collections;
import java.util.Map;
import java.util.List;

// TODO: construct training object from input file?

/**
 *
 * @author tosin
 */
public class Training {
    // network
    private Network net;
    // array of training data;
    private final List<Sample> trainingData;
    // number of epochs and miniBatchSize
    private int epochs, miniBatchSize;
    // learning rate eta
    private double eta;
    
    /**
     * Construct training object from parameters and trains a copy of the input network.
     * @param net
     * @param trainingData
     * @param epochs
     * @param miniBatchSize
     * @param learningRate 
     */
    public Training(Network net, List<Sample> trainingData, int epochs, 
            int miniBatchSize, double learningRate) {
        this.net = new Network(net);
        this.trainingData = trainingData;
        this.epochs = epochs;
        this.miniBatchSize = miniBatchSize;
        this.eta = learningRate;
        stochasticGradientDescent(this.miniBatchSize, this.epochs, this.eta);
    }

    /**
     * Trains the neural network with the training data and current training parameters.
     * @return training neural network
     */
    public Network trainNetwork(){
        stochasticGradientDescent(this.miniBatchSize, this.epochs, this.eta);
        return this.net;
    }
    
    /**
     * Get a copy of the trained Network.
     * @return A copy of the Trained Network.
     */
    public Network getTrainedNet(){
        return new Network(net);
    }
    
    /**
     * Update training parameters
     * @param miniBatchSize
     * @param epochs
     * @param eta 
     */
    public void updateParameters(int miniBatchSize, int epochs, double eta){
        this.miniBatchSize = miniBatchSize;
        this.epochs = epochs;
        this.eta = eta;
    }

    /**
     * Carry out stochastic gradient descent on network being trained, using training parameters.
     * @param miniBatchSize
     * @param epochs
     * @param eta
     */
    private void stochasticGradientDescent(int miniBatchSize, int epochs, double eta){
        Network trainedNet = this.net;
        Matrix[] originalWeights;
        Matrix[] originalbiases;
        
        int numSamples = trainingData.size();
        int numWholeBatches = numSamples/miniBatchSize;
        int numRemainingSamples = numSamples % miniBatchSize;
        
        // for each epoch...
        for (int i = 0; i < epochs; i++) {
            // shuffle the minibatch.
            Collections.shuffle(trainingData);
            
            for (int j = 0; j < numWholeBatches; j++) {
                // get weights and biases
                originalWeights = trainedNet.getWeights();
                originalbiases = trainedNet.getBiases();
                // update minibatches
                updateWithMiniBatch(trainingData.subList(j*miniBatchSize, (j+1)*miniBatchSize), 
                        trainedNet, originalWeights, originalbiases, miniBatchSize);
                // update weights and biases
                trainedNet.updateBiases(originalbiases); 
                trainedNet.updateWeights(originalWeights);
            }
            // update reminder if any
            if (numRemainingSamples != 0) {
                // get start and end index. ensure that this is calculated properly
                int start = numWholeBatches*miniBatchSize;
                int end = numWholeBatches*miniBatchSize + numRemainingSamples;
                assert end == trainingData.size();
                // get old weights and update
                originalWeights = trainedNet.getWeights();
                originalbiases = trainedNet.getBiases();
                updateWithMiniBatch(trainingData.subList(start, end), trainedNet, 
                        originalWeights, originalbiases, numRemainingSamples);    
                trainedNet.updateBiases(originalbiases); 
                trainedNet.updateWeights(originalWeights);
            }          
        }
                    
    }
    
    /**
     * 
     * Update network net's weights and biases using stochastic gradient descent 
     * on mini-batch samples
     * 
     * @param miniBatch
     * @param net
     * @param weights
     * @param biases 
     * @param size
     */
    private void updateWithMiniBatch(List<Sample> miniBatch, Network net, Matrix[] weights, Matrix[] biases, int size) {
        int numLayers = weights.length;
        Matrix[] matrixWeightGradSum, matrixBiasGradSum;
        matrixWeightGradSum =  new Matrix[numLayers];
        matrixBiasGradSum = new Matrix[numLayers];
        
        // initialize matrix values to zero
        for (int i = 0; i < numLayers; i++){
            matrixWeightGradSum[i] = new Matrix(weights[i].getRowDimension(), 
                    weights[i].getColumnDimension(), 0.0);
            matrixBiasGradSum[i] = new Matrix(biases[i].getRowDimension(), 
                    biases[i].getColumnDimension(), 0.0);

        }
        
        // for each sample, calculate gradients and add it to running gradient 
        // total
        for(Sample sample: miniBatch){
            //Network.printMatrixDimensions(sample.getInput(), "input");
            //Network.printMatrixDimensions(sample.getLabel(), "label");

            Map<Boolean, Matrix[]> temp = net.backpropagation(sample.getInput(), sample.getLabel());
            sumMatrixArrays(matrixBiasGradSum, temp.get(Network.BIAS_key));
            sumMatrixArrays(matrixWeightGradSum, temp.get(Network.WEIGHT_key));
        } 
        
        for (int i = 0; i < numLayers; i++) {
            applyUpdate(weights[i], matrixWeightGradSum[i], this.eta, size);
            applyUpdate(biases[i], matrixBiasGradSum[i], this.eta, size);
        }   
    }
    
 
    /**
     * 
     * update old matrix using gradientsum
     * 
     * @param old
     * @param gradSum
     * @param eta
     * @param m 
     * 
     * 
     */
   private void applyUpdate(Matrix old, Matrix gradSum, double eta, int m){
       for (int i = 0; i < old.getRowDimension(); i++){
           for (int j = 0; j< old.getColumnDimension(); j++){
               double oldVal = old.get(i, j);
               double gradVal = gradSum.get(i, j);
               old.set(i, j, oldVal - ( (eta/m) * gradVal));
           }
       }
   }
    
   /**
    * sums up both Matrix Arrays into a
    * 
    * @param a
    * @param b 
    */
    private void sumMatrixArrays(Matrix[] a, Matrix[] b){
        for(int i = 0; i < a.length; i++){
            a[i] = a[i].plusEquals(b[i]);
        }  
    }
    
    /**
     * 
     * @param args 
     */
    public static void main(String[] args) {
        
    }
    
    
    
    
    
    
}
