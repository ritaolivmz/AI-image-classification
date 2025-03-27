package org.example;

import ai.djl.Application;
import ai.djl.Model;
import ai.djl.ModelException;
import ai.djl.inference.Predictor;
import ai.djl.modality.Classifications;
import ai.djl.modality.cv.Image;
import ai.djl.modality.cv.ImageFactory;
import ai.djl.translate.TranslateException;
import ai.djl.translate.Translator;
import ai.djl.translate.TranslatorContext;
import ai.djl.translate.TranslatorFactory;
import ai.djl.util.Utils;

import java.nio.file.Paths;

public class ImageClassifier {

    public static void main(String[] args) throws Exception {

        // Load the pre-trained ResNet model (or another model)
        String modelPath = "ai.djl.tensorflow:resnet50";  // You can choose a model from the zoo
        Model model = Model.newInstance(modelPath);

        // Create an image loader and load an image for classification
        String imagePath = "path/to/your/image.jpg"; // Path to your image
        Image img = ImageFactory.getInstance().fromFile(Paths.get(imagePath));

        // Create a custom translator (for preprocessing the image and postprocessing the output)
        Translator<Image, Classifications> translator = new ImageTranslator();

        // Create a predictor with the custom translator
        Predictor<Image, Classifications> predictor = model.newPredictor(translator);

        // Perform image classification
        Classifications predictions = predictor.predict(img);

        // Print the classification results
        predictions.items().forEach(item -> {
            System.out.println("Class: " + item.getClassName() + ", Confidence: " + item.getProbability());
        });
    }
}

