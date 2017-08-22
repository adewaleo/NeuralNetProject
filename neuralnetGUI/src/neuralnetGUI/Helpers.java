/*
 * To change this license header, choose License Headers in Project Properties.
 * To change this template file, choose Tools | Templates
 * and open the template in the editor.
 */
package neuralnetGUI;

import java.io.File;
import java.util.List;
import neuralnet.Sample;
import javax.swing.table.AbstractTableModel;

/**
 * Contains Helper Classes
 *
 * @author tosin
 */
public class Helpers {

    /**
     * Container for holding information on loaded data.
     */
    public static class Data {

        public String samplesFileName, labelsFileName;
        public List<Sample> samples;

        public Data() {

        }

        public Data(String samplesFileName, String labelsFileName, List<Sample> trainingData) {
            this.samplesFileName = samplesFileName;
            this.labelsFileName = labelsFileName;
            this.samples = trainingData;
        }

        public void setSamplesFileName(String samplesFileName) {
            this.samplesFileName = samplesFileName;
        }

        public void setLabelsFileName(String labelsFileName) {
            this.labelsFileName = labelsFileName;
        }

        public void setSamples(List<Sample> samples) {
            this.samples = samples;
        }

        public String getSamplesFileName() {
            return samplesFileName;
        }

        public String getLabelsFileName() {
            return labelsFileName;
        }

        public List<Sample> getSamples() {
            return samples;
        }

        public boolean isEmpty() {
            return samples == null || samples.isEmpty();
        }

        public int size() {
            return samples.size();
        }
    }

    /**
     * Container for holding training parameters.
     *
     */
    public static class Parameters {

        public final double eta;
        public final int epochs, minibatchSize;

        public Parameters(double eta, int epochs, int minibatchSize) {
            this.eta = eta;
            this.epochs = epochs;
            this.minibatchSize = minibatchSize;
        }
    }
    
    public static String filePathToName(String filePath) {
        String[] temp = filePath.split(File.separator);
        return temp[temp.length - 1];
    }

}
