/*
 * To change this license header, choose License Headers in Project Properties.
 * To change this template file, choose Tools | Templates
 * and open the template in the editor.
 */
package neuralnet;
// Import statements
import Jama.Matrix;
import java.io.File;
import java.io.FileNotFoundException;
import java.util.ArrayList;
import java.util.Scanner;
import java.util.List;
import java.util.logging.Level;
import java.util.logging.Logger;

/**
 *
 * @author tosin
 */

// TODO: check if CSV or appropriate input
public class DataLoader {

    private List<Sample> data;
    
    /**
     * Creates a new DataLoader object containing data from files.
     * The input file is expected to be a CSV file.
     * @param inputFilePath 
     * @param labelFilePath
     * @throws FileNotFoundException Throws exception if file not found.
     */
    public DataLoader(String inputFilePath, String labelFilePath) throws FileNotFoundException {
        Scanner inputScanner = new Scanner(new File(inputFilePath));
        Scanner labelScanner = new Scanner(new File(labelFilePath));
        int firstInputLength = 0, firstLabelLength = 0;
        boolean first = true;
        
        data = new ArrayList<Sample>();
        
        // while both files have another line
        int count = 0;
        while(inputScanner.hasNextLine() && labelScanner.hasNextLine()){
            // read nect line of each file.
            count++;
             
            String[] inputs = inputScanner.nextLine().split(",");
            String[] labels = labelScanner.nextLine().split(",");
                        
            int currInputLength = inputs.length;
            int currLabelLength = labels.length;
            
            // if first, store first lengths. Else, check currlength with first length;
            if(first) {
                firstInputLength = inputs.length;
                firstLabelLength = labels.length;
                first = false;
            } else if (firstInputLength != currInputLength || firstLabelLength != currLabelLength) {
                System.out.printf("FirstIn: %d. CurrIn: %d. FirstLab: %d. CurrLab: %d. Line %d\n", 
                        firstInputLength, currInputLength, firstLabelLength, currLabelLength, count);
                throw new IllegalArgumentException("Data in file varies between lines");
            }          
            // inputs/examples and labels are stored as column vectors;
            Matrix tempInput = new Matrix(currInputLength, 1);
            Matrix tempLabel = new Matrix(currLabelLength, 1);
            int i = 0;
            for (String input : inputs) {
                tempInput.set(i, 0, Double.parseDouble(input));
                i++;
            }
            i = 0;
            for (String label : labels) {
                tempLabel.set(i, 0, Double.parseDouble(label));
                i++;
            }
            // add the new sample to the list of samples
            data.add(new Sample(tempInput, tempLabel));
            
        }
        
        if (inputScanner.hasNextLine() || labelScanner.hasNextLine()){
            System.out.println("Next line");
            throw new IllegalArgumentException("Input Files do not have the same lines");
        }
        
        inputScanner.close();
        labelScanner.close();
                       
    }

    /**
     * Returns the list of the samples that have been read in. 
     * @return The underlying list of samples.
     */
    public List<Sample> getDataAsList() {
        
        return data;
//        List<Sample> copy = new ArrayList<>();
//        
//        for (Sample sample: data){
//            copy.add(sample.copy());
//        }
//        
//        return copy;
    }
    
    public static void main(String[] args) {
        String trainXFileName = "/Users/tosin/projects/cos583_project/neuralnet/TrainDigitX.csv";
        String trainYFileName = "/Users/tosin/projects/cos583_project/neuralnet/TrainDigitY.csv";
        
        //trainXFileName = "/Users/tosin/projects/cos583_project/neuralnet/TrainHalfAdderX.csv";
        //trainYFileName = "/Users/tosin/projects/cos583_project/neuralnet/TrainHalfAdderY.csv";
   
        DataLoader loaderTest = null;
        
        try {
            loaderTest = new DataLoader(trainXFileName, trainYFileName);
        } catch (FileNotFoundException ex) {
            Logger.getLogger(DataLoader.class.getName()).log(Level.SEVERE, null, ex);
            System.exit(1);
        }
        
        List<Sample> samples = loaderTest.getDataAsList();

        for (int i = 0; i < 10; i++){
            Sample sample = samples.get(i);
            sample.getLabel().print(5, 2);
        }    
    }
    
}
