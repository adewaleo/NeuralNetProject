/*
 * To change this license header, choose License Headers in Project Properties.
 * To change this template file, choose Tools | Templates
 * and open the template in the editor.
 */
package neuralnet;

import Jama.Matrix;

/**
 * class for holding a sample.
 * @author tosin
 */

public class Sample {
   private final Matrix input;
   private final Matrix label;

   /**
    * Creates a new Sample object from column vector matrices
    * @param input
    * @param label 
    */
   public Sample(Matrix input, Matrix label) {
       this.input = input;
       this.label = label;
   }
   public Matrix getInput(){
       return input;
   }    
   public Matrix getLabel(){
       return label; 
   }  
   public Sample copy(){
       return new Sample(input.copy(), label.copy());
   }
} 
