import tensorflow as tf
import tensorflow_datasets as tfds

def preprocess(feature):
  image, label = feature["image"], feature["label"]
  image = tf.cast(image, tf.float32)
  image = image / 255
  shape = tf.shape(image)
  h, w = shape[0], shape[1]
  ratio = w / h
  if ratio >= 1:
      image = tf.image.resize(image, (224, tf.cast(tf.math.round(224 * ratio), dtype=tf.int32)))
      w_offset = tf.cast(tf.math.round(224 * (ratio - 1)/ 2), dtype=tf.int32)
      image = tf.image.crop_to_bounding_box(image, 0, w_offset, 224, 224)
  else:
      image = tf.image.resize(image, (tf.cast(tf.math.round(224 / ratio), dtype=tf.int32), 224))
      h_offset = tf.cast(tf.math.round(224 * (1/ratio - 1) / 2), dtype=tf.int32)
      image = tf.image.crop_to_bounding_box(image, h_offset, 0, 224, 224)
  label = tf.one_hot(label, 2)
  return image, label


if __name__ == "__main__":

  # load data  
  trainset, info = tfds.load(name="covid_ct", split="train", shuffle_files=True, with_info=True)
  print(info)
  valset = tfds.load(name="covid_ct", split="validation", shuffle_files=False)
  testset = tfds.load(name="covid_ct", split="test", shuffle_files=False)
  
  train_ds = trainset.map(preprocess)
  val_ds = valset.map(preprocess)
  test_ds = testset.map(preprocess)
  
  train_batches = train_ds.batch(5)
  val_batches = val_ds.batch(2)
  test_batches = test_ds.batch(1)

  # specify model
  model = tf.keras.Sequential([
      tf.keras.applications.DenseNet169(include_top=True, weights='imagenet'),
      tf.keras.layers.Dense(2, activation="softmax")
      ])

  model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=1e-4), 
                loss=tf.keras.losses.CategoricalCrossentropy(from_logits=True),
                metrics=[tf.keras.metrics.categorical_accuracy])

  print("Fitting model on training data")
  history = model.fit(train_batches.repeat(), epochs=5, steps_per_epoch=85, 
          validation_data=val_batches.repeat(), validation_steps=59)

  print("Evaluating model on the test data")
  results = model.evaluate(test_batches, steps=203)
  print("Test Acc: ", results[1])

  # save model
  model.save("densenet_covid.h5")

