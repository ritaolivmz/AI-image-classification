package org.example;

import ai.djl.Application;
import ai.djl.MalformedModelException;
import ai.djl.Model;
import ai.djl.modality.Classifications;
import ai.djl.modality.cv.Image;
import ai.djl.modality.cv.ImageFactory;
import ai.djl.modality.cv.transform.Normalize;
import ai.djl.modality.cv.transform.Resize;
import ai.djl.modality.cv.transform.ToTensor;
import ai.djl.modality.cv.translator.ImageClassificationTranslator;
import ai.djl.repository.zoo.Criteria;
import ai.djl.repository.zoo.ModelNotFoundException;
import ai.djl.repository.zoo.ModelZoo;
import ai.djl.translate.TranslateException;
import ai.djl.inference.Predictor;

import java.io.File;
import java.io.IOException;
import java.nio.file.Paths;

public class ImageFolderClassification {

    public static void main(String[] args) throws ModelNotFoundException, MalformedModelException, IOException, TranslateException {
        String folderPath = "/Users/ruitex23/Desktop/Rita/images"; // Update with your image folder

        // Define a translator with preprocessing steps
        ImageClassificationTranslator translator = ImageClassificationTranslator.builder()
                .addTransform(new Resize(224, 224))  // Resize image
                .addTransform(new ToTensor())        // Convert to tensor format
                .addTransform(new Normalize(new float[]{0.485f, 0.456f, 0.406f},
                        new float[]{0.229f, 0.224f, 0.225f})) // Normalize
                .build();

        // Load model with specified criteria
        Criteria<Image, Classifications> criteria = Criteria.builder()
                .setTypes(Image.class, Classifications.class)
                .optApplication(Application.CV.IMAGE_CLASSIFICATION)
                .optTranslator(translator) // Pass the translator here
                .build();

        try (Model model = ModelZoo.loadModel(criteria);
             Predictor<Image, Classifications> predictor = model.newPredictor(translator)) {

            File folder = new File(folderPath);
            File[] imageFiles = folder.listFiles((dir, name) ->
                    name.toLowerCase().endsWith(".jpg") ||
                            name.toLowerCase().endsWith(".jpeg") ||
                            name.toLowerCase().endsWith(".png"));


            if (imageFiles == null || imageFiles.length == 0) {
                System.out.println("No images found in folder.");
                return;
            }

            for (File file : imageFiles) {
                Image img = ImageFactory.getInstance().fromFile(Paths.get(file.getAbsolutePath()));
                Classifications result = predictor.predict(img);
                System.out.println("Image: " + file.getName() + " -> " + result);
            }
        }
    }
}
