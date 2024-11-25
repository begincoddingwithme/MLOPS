import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import classification_report, confusion_matrix
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.image import load_img, img_to_array
from tensorflow.keras.preprocessing.image import ImageDataGenerator
import time

# Define constants
data_dir = r"C:\Users\Shree\Downloads\data"  # Replace with your data directory
classes = ['Blight', 'Gray spot', 'Rust', 'Healthy']
img_size = (224, 224)
batch_size = 32

# Load the model
model = load_model('maize_disease_model.h5')
print("Model loaded successfully.")

# Prepare data generators
datagen = ImageDataGenerator(rescale=1.0 / 255, validation_split=0.2)

validation_generator = datagen.flow_from_directory(
    data_dir,
    target_size=img_size,
    batch_size=batch_size,
    class_mode='categorical',
    subset='validation',
    shuffle=False  # Needed for correct metrics calculation
)

# Evaluate on validation set
print("\nEvaluating on validation set...")
val_loss, val_accuracy = model.evaluate(validation_generator, verbose=1)
print(f"Validation Loss: {val_loss}, Validation Accuracy: {val_accuracy}")

# Test with an individual image
def test_single_image(image_path):
    print(f"\nTesting single image: {image_path}")
    img = load_img(image_path, target_size=img_size)  # Resize image
    img_array = np.expand_dims(img_to_array(img) / 255.0, axis=0)  # Normalize and expand dims
    prediction = model.predict(img_array)
    predicted_class = classes[np.argmax(prediction)]
    print(f"Predicted Class: {predicted_class}")
    return predicted_class

# Test on a specific image (update the path with a test image)
image_path = "C:\\Users\\Shree\\Downloads\\data\\Healthy\\Corn_Health (992).jpg"  # Replace with the test image path

test_single_image(image_path)

# Test on an unrelated/random image (edge case)
random_image_path = "C:\\Users\\Shree\\Downloads\\data\\Common_Rust\\Corn_Common_Rust (834).jpg"
try:
    test_single_image(random_image_path)
except Exception as e:
    print(f"Error processing the image: {e}")

# Measure inference time
def measure_inference_time(image_path):
    img = load_img(image_path, target_size=img_size)
    img_array = np.expand_dims(img_to_array(img) / 255.0, axis=0)
    start_time = time.time()
    model.predict(img_array)
    end_time = time.time()
    print(f"Inference Time: {end_time - start_time:.4f} seconds")

measure_inference_time(image_path)

# Generate detailed metrics
print("\nCalculating detailed metrics...")
y_true = validation_generator.classes  # True labels
y_pred = model.predict(validation_generator)  # Predictions
y_pred_classes = np.argmax(y_pred, axis=1)

# Classification report
print("\nClassification Report:")
print(classification_report(y_true, y_pred_classes, target_names=classes))

# Confusion matrix
print("\nConfusion Matrix:")
cm = confusion_matrix(y_true, y_pred_classes)
plt.figure(figsize=(8, 6))
sns.heatmap(cm, annot=True, fmt='d', xticklabels=classes, yticklabels=classes, cmap="Blues")
plt.xlabel('Predicted')
plt.ylabel('True')
plt.title('Confusion Matrix')
plt.show()
