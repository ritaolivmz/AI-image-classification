package org.example;

// Example of an Image Translator (used for model inference)
import ai.djl.modality.Classifications;
import ai.djl.modality.cv.Image;
import ai.djl.ndarray.NDArray;
import ai.djl.ndarray.NDList;
import ai.djl.ndarray.NDManager;
import ai.djl.translate.*;
import ai.djl.modality.cv.transform.Resize;
import ai.djl.modality.cv.transform.ToTensor;

import java.util.Arrays;
import java.util.List;
import java.util.stream.Collectors;

public class ImageTranslator implements Translator<Image, Classifications> {

    @Override
    public NDList processInput(TranslatorContext ctx, Image input) {
        NDManager manager = ctx.getNDManager();

        // Convert image to NDArray
        NDArray array = input.toNDArray(manager);

        // Normalize image (divide by 255 to scale pixel values)
        array = array.div(255.0f);

        // Reshape to match model input format (e.g., batch size of 1)
        array = array.expandDims(0);

        return new NDList(array);
    }

    @Override
    public Classifications processOutput(TranslatorContext ctx, NDList list) {
        NDArray output = list.singletonOrThrow(); // Extract model output (NDArray)

        List<String> labels = Arrays.asList("label1", "label2", "label3"); // Update with actual labels
        List<Double> probabilities = Arrays.stream(output.toDoubleArray()).boxed().collect(Collectors.toList());

        return new Classifications(labels, probabilities); // Return proper classification object
    }



    @Override
    public Batchifier getBatchifier() {
        return null; // Not used in this case
    }
}

