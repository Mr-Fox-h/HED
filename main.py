import tensorflow as tf
import matplotlib.pyplot as plt
import datasets as dt
import numpy as np

print(tf.__version__)

# loading data
datasets = dt.load_dataset('./archive/')
print(datasets)

# Train/Test split
train_img, train_label = datasets['train'].select_columns('image'), datasets['train'].select_columns('label')
test_img, test_label = datasets['test'].select_columns('image'), datasets['test'].select_columns('label')
train_label = np.array(train_label['label'])
test_label = np.array(test_label['label'])

# Transform
def transforms(examples):
    examples['img'] = [image.convert("RGB").resize((32, 32)) for image in examples['image']]
    return examples

train_img = train_img.map(transforms, remove_columns = ['image'], batched = True)
test_img = test_img.map(transforms, remove_columns = ['image'], batched = True)

# Image processing
train_tensor = []
test_tensor = []

for index in range(len(train_img)):
    train_tensor.append(np.array(train_img[index]['img']))

for index in range(len(test_img)):
    test_tensor.append(np.array(test_img[index]['img']))

train_tensor = np.array(train_tensor)
test_tensor = np.array(test_tensor)

# Normalizing
train_tensor = train_tensor / 255
test_tensor = test_tensor / 255

class_names = ['angry', 'happy', 'noting', 'sad']

# Model
"""
model = tf.keras.Sequential()
model.add(tf.keras.layers.Conv2D(128, (3, 3), activation='relu', input_shape = (32, 32, 3)))
model.add(tf.keras.layers.MaxPooling2D((2, 2)))
model.add(tf.keras.layers.Conv2D(64, (3, 3), activation='relu'))
model.add(tf.keras.layers.Flatten(input_shape = (32, 32, 3)))
model.add(tf.keras.layers.Dense(32, activation='relu'))
model.add(tf.keras.layers.Dense(24, activation='sigmoid'))
model.add(tf.keras.layers.Dense(4, activation='softmax'))

model.compile(optimizer = tf.keras.optimizers.experimental.SGD(learning_rate=0.01), loss = 'sparse_categorical_crossentropy', metrics = ['accuracy'])
model.fit(train_tensor, train_label, epochs = 100)

loss, accuracy = model.evaluate(test_tensor, test_label)
print(f"Loss: {loss} | ACC: {accuracy}")

model.save("HED.pt")
"""

# Load the model

model = tf.keras.models.load_model('HED.pt')

prd = model.predict(test_tensor)

for index in range(20):
    prd_guess = class_names[np.argmax(prd[index])]
    plt.grid(False)
    plt.imshow(test_tensor[index], cmap = plt.cm.binary)
    plt.xlabel(f"Actual: {class_names[test_label[index]]}")
    plt.title(f"Prediction: {prd_guess}")
    plt.show()
