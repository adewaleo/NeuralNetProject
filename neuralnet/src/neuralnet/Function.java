/*
 * To change this license header, choose License Headers in Project Properties.
 * To change this template file, choose Tools | Templates
 * and open the template in the editor.
 */
package neuralnet;

/**
 *
 * @author tosin
 */

public interface Function {
    
    // apply the function to this input. return the result.
    double apply(double input);
    // apply the derivative of this function to this input. return the result.
    public double applyDerivative (double input);
    
}
