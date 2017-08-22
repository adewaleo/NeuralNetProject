/*
 * To change this license header, choose License Headers in Project Properties.
 * To change this template file, choose Tools | Templates
 * and open the template in the editor.
 */
package neuralnet;

import java.lang.Math;
/**
 *
 * @author tosin
 */
public class Functions {
    
    public static class Sigmoid implements Function {
        
        public static String Name = "Sigmoid";
    
        // apply the sigmoid function to this imput. return the result
        public double apply (double input) {
            return 1 / (1 + Math.exp(-1 * input));          
        }
        // apply the derivative of the sigmoid function to this input. return the result
        public double applyDerivative (double input) {
            return apply(input) * ( 1 - apply(input) );
        }
    
    }
    
        public static class Relu implements Function {
        
        public static String Name = "Relu";
    
        // apply the sigmoid function to this imput. return the result
        public double apply (double input) {
            return Math.log(1 + Math.exp(input));
        }
        // apply the derivative of the sigmoid function to this input. return the result
        public double applyDerivative (double input) {
            return  1 / (1 + Math.exp(-1 *input));
        }
    
    }
    
    
    // TODO others 
}
