/*
 * To change this license header, choose License Headers in Project Properties.
 * To change this template file, choose Tools | Templates
 * and open the template in the editor.
 */
package neuralnet;

import Jama.Matrix;
import java.io.File;
import java.io.FileNotFoundException;
import java.io.PrintWriter;
import java.util.Scanner;
import java.util.logging.Level;
import java.util.logging.Logger;

/**
 *
 * @author tosin
 */
public class RewriteData {
    
    
    public static void main(String[] args) {
        String trainYFileName = "/Users/tosin/projects/cos583_project/neuralnet/TrainDigitYOriginal.csv";
        String newTrainYFileName = "/Users/tosin/projects/cos583_project/neuralnet/TrainDigitY.csv";
        
        trainYFileName = "/Users/tosin/TestDigitYOriginal.csv";
        newTrainYFileName = "/Users/tosin/TestDigitY.csv";
        
        Scanner inputScanner = null;
        PrintWriter writer = null;    

        try {
            inputScanner = new Scanner(new File(trainYFileName));
        } catch (FileNotFoundException ex) {
            Logger.getLogger(RewriteData.class.getName()).log(Level.SEVERE,"Opening inFile failed", ex);
            System.exit(1);
        } 
        try {
            writer = new PrintWriter(newTrainYFileName);
        } catch (FileNotFoundException ex) {
            Logger.getLogger(RewriteData.class.getName()).log(Level.SEVERE,"Opening outFile failed", ex);
            System.exit(1);
        }
        
        int min = 0, max = 0;
        
        while(inputScanner.hasNextLine()){
            // read next line of each file.
            int label = Integer.parseInt(inputScanner.nextLine());
            if (label < min) {
               min = label; 
            }
            if (label > max) {
               max = label;
            }
            
            StringBuilder out = new StringBuilder(20);
            
            for(int i = 0; i < 10; i++){
                int val = 0;
                
                if (label == i){
                    val = 1;
                } 
                
                if (i == 9) {
                    out.append(val);
                    out.append(System.lineSeparator());
                } else {
                    out.append(val);
                    out.append(",");
                }
            }
            writer.write(out.toString());       
        }   
        writer.flush();
        writer.close();
        
        System.out.printf("Min is %d. Max is %d\n", min, max);
    }
    
}
