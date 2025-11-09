# File to handle data preprocessing for face detection and splits faces and non-faces data

from tensorflow.keras.preprocessing.image import ImageDataGenerator # type: ignore

def load_data(img_size=(128,128), batch_size=32):
    """
    Loads and preprocesses training and testing image datasets for mask detection.

    Parameters:
        img_size (tuple): Target size for image resizing.
        batch_size (int): Number of images per batch.
    Returns:
        train_data, test_data: Preprocessed image generators.
    """
    
    # Data augmentation for training images
    train_gen = ImageDataGenerator(
        rescale=1./255,
        rotation_range=15,
        width_shift_range=0.1,
        height_shift_range=0.1,
        shear_range=0.1,
        zoom_range=0.1,
        horizontal_flip=True
    )

    # Only rescale for test images
    test_gen = ImageDataGenerator(rescale=1./255)

    train_data = train_gen.flow_from_directory(
        'data/train',
        target_size=img_size,
        batch_size=batch_size,
        class_mode='binary'
    )

    test_data = test_gen.flow_from_directory(
        'data/test',
        target_size=img_size,
        batch_size=batch_size,
        class_mode='binary'
    )

    print(" Data loaded successfully!")
    return train_data, test_data