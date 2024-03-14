def inference(model_path, images_dir):
    from glob import glob
    import numpy as np
    import tensorflow as tf
    from tensorflow.keras.preprocessing.image import load_img, img_to_array
    from tensorflow.keras.models import load_model
    import cv2

    # Assuming model is your trained CNN model
    model = load_model(model_path)
    image_filenames = glob(images_dir+'/*')
    images = []
    for filename in image_filenames:
        img = load_img(filename, target_size=(224, 224))  # Adjust target_size as needed
        img_array = img_to_array(img)
        img_array = np.expand_dims(img_array, axis=0)  # Add batch dimension
        img_array /= 255.0  # Normalize pixel values
        images.append(img_array)
    
    # Stack all images into a single numpy array
    input_data = np.vstack(images)
    
    # Perform inference on all images
    predictions = model.predict(input_data)
    predicted_class = tf.argmax(predictions, axis=1).numpy()
    predicted_class = ['meme' if x == 0 else 'non-meme' for x in predicted_class]

    num = 0
    for i,each_file_name in enumerate(image_filenames):
        image = cv2.imread(each_file_name)
        (text_width, text_height), baseline = cv2.getTextSize(predicted_class[i], cv2.FONT_HERSHEY_SIMPLEX, 1, 2)
        x = 10
        y = text_height + 10
        cv2.putText(image, predicted_class[i], (x, y), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)

        image_name = each_file_name.split('/')[-1]
        output_image_path = f'inf_output/{image_name}'
        cv2.imwrite(output_image_path, image)
        num = i+1
        
    return num


if __name__ == "__main__":
    import os
    folder_name = 'inf_output'
    if not os.path.exists(folder_name):
        os.makedirs(folder_name)
        
    num = inference('final101_2classes.h5', 'inf images/memes')
