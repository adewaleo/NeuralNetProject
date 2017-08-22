/*
 * To change this license header, choose License Headers in Project Properties.
 * To change this template file, choose Tools | Templates
 * and open the template in the editor.
 */
package neuralnet;

import Jama.Matrix;
import java.io.FileNotFoundException;
import java.util.logging.Level;
import java.util.logging.Logger;
import java.util.List;

/**
 *
 * @author tosin
 */
public class Test {
    
    private static double calculateAccuracy (List<Sample> data, Network trainedNet){
        
        int count = 0;
        double total = 0;
        for (Sample s: data){
            count++;
            boolean match = true;
            Matrix label =  s.getLabel();
            Matrix output = trainedNet.feedForward(s.getInput());
                      
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
        
        return (total/count)*100;     
    }
    
    
    public static void main(String[] args) {
        
        String trainXFileName;
        trainXFileName = "/Users/tosin/projects/cos583_project/neuralnet/TrainDigitX.csv";
        String trainYFileName;
        trainYFileName = "/Users/tosin/projects/cos583_project/neuralnet/TrainDigitY.csv";

        trainXFileName = "/Users/tosin/projects/cos583_project/neuralnet/TrainHalfAdderX.csv";
        trainYFileName = "/Users/tosin/projects/cos583_project/neuralnet/TrainHalfAdderY.csv";

        
        DataLoader data = null;
        try {
            data = new DataLoader(trainXFileName, trainYFileName);
        } catch (FileNotFoundException ex) {
            Logger.getLogger(Test.class.getName()).log(Level.SEVERE, "File(s) not found", ex);
            System.exit(1);
        }
        
//        
//        int epochs = 25, miniBatchSize = 20;
//        double eta = 3.0;
//        Network testNet = new Network(new int[]{784,30,10}, new Functions.Sigmoid());   
        
        int epochs = 3000, miniBatchSize = 2;
        double eta = 10.0;
        Network testNet = new Network(new int[]{2,4,2}, new Functions.Sigmoid());
        
        Training testTraining = new Training(testNet, data.getDataAsList(), epochs, miniBatchSize, eta);
        testNet = testTraining.getTrainedNet();
        
        if (testNet == null){
            System.out.println("Error, testNet is null");
        }
        
       
        System.out.printf("Accuracy is %f\n", calculateAccuracy(data.getDataAsList(), testNet));
       
        Sample testSample = data.getDataAsList().get(0);
        Matrix output = testNet.feedForward(testSample.getInput());
        Matrix label = testSample.getLabel();
        
        System.out.println("Output of Neuralnet 1");
        output.print(5, 2);
        System.out.println("Correct Labels 1");
        label.print(5, 2); 
        
        testSample = data.getDataAsList().get(1);
        output = testNet.feedForward(testSample.getInput());
        label = testSample.getLabel();
        
        System.out.println("Output of Neuralnet 2");
        output.print(5, 2);
        System.out.println("Correct Labels 2");
        label.print(5, 2); 
        
    }
}
