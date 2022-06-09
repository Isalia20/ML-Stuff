import tensorflow as tf
from sklearn.datasets import make_classification
from sklearn.model_selection import train_test_split

X, y = make_classification(n_samples=10000, n_features=30, n_classes=2, random_state=42)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.1)
X_train = tf.convert_to_tensor(X_train)
X_test = tf.convert_to_tensor(X_test)
y_train = tf.convert_to_tensor(y_train)
y_test = tf.convert_to_tensor(y_test)

tf.keras.utils.set_random_seed(42)
inputs = tf.keras.Input(shape=X_train.shape[1])
x = tf.keras.layers.Dense(10, activation="relu")(inputs)
x = tf.keras.layers.Dense(5, activation="relu")(x)
outputs = tf.keras.layers.Dense(1, activation='sigmoid')(x)

model = tf.keras.Model(inputs=inputs, outputs=outputs, name="small_model")
# tf.keras.utils.plot_model(model, "model_architecture.png", show_shapes=True)

model.compile(loss=tf.keras.losses.binary_crossentropy, optimizer=tf.keras.optimizers.Adam(), metrics=["accuracy"])

history = model.fit(X_train, y_train, batch_size=64, epochs=20, validation_split=0.2)

train_scores = model.evaluate(X_train, y_train, verbose=2)
test_scores = model.evaluate(X_test, y_test, verbose=2)
print("Train accuracy:", train_scores[1])
print("Test accuracy:", test_scores[1])

# 93% for both train and test loss which is quite good but let's see if we can improve
#-----------------------------------------------------------------------------------------------------------------------------------------------------------------------------
# Let's deepen network
inputs = tf.keras.Input(shape=X_train.shape[1])
x = tf.keras.layers.Dense(10, activation="relu")(inputs)
x = tf.keras.layers.Dense(5, activation="relu")(x)
x = tf.keras.layers.Dense(3, activation="relu")(x)
x = tf.keras.layers.Dense(2, activation="relu")(x)
outputs = tf.keras.layers.Dense(1, activation='sigmoid')(x)

model = tf.keras.Model(inputs=inputs, outputs=outputs, name="medium_model")

model.compile(loss=tf.keras.losses.binary_crossentropy, optimizer=tf.keras.optimizers.Adam(), metrics=["accuracy"])

# Train for even more epochs hopefully it wont overfit
history = model.fit(X_train, y_train, batch_size=64, epochs=150, validation_split=0.2)

train_scores = model.evaluate(X_train,y_train, verbose=2)
test_scores = model.evaluate(X_test, y_test, verbose=2)
print("Train accuracy:", train_scores[1])
print("Test accuracy:", test_scores[1])
# Well model overfit, not a news as we ran for much larger epochs and deepened the network architecture
#-----------------------------------------------------------------------------------------------------------------------------------------------------------------------------
# Let's try deep network with dropout and maybe some other activation functions as well
inputs = tf.keras.Input(shape=X_train.shape[1])
x = tf.keras.layers.Dense(100, activation="tanh")(inputs)
x = tf.keras.layers.Dense(50, activation="tanh")(x)
x = tf.keras.layers.Dropout(0.5)(x)
x = tf.keras.layers.Dense(30, activation='tanh')(x)
x = tf.keras.layers.Dropout(0.5)(x)
x = tf.keras.layers.Dense(10, activation='tanh')(x)
x = tf.keras.layers.Dropout(0.5)(x)
outputs = tf.keras.layers.Dense(1, activation='sigmoid')(x)

model = tf.keras.Model(inputs=inputs, outputs=outputs, name="small_model")

model.compile(loss=tf.keras.losses.binary_crossentropy, optimizer=tf.keras.optimizers.Adam(), metrics=["accuracy"])
history = model.fit(X_train, y_train, batch_size=64, epochs=50, validation_split=0.2)

train_scores = model.evaluate(X_train,y_train, verbose=2)
test_scores = model.evaluate(X_test, y_test, verbose=2)
print("Train accuracy:", train_scores[1])
print("Test accuracy:", test_scores[1])

# Same model but now with relu activations
inputs = tf.keras.Input(shape=X_train.shape[1])
x = tf.keras.layers.Dense(20, activation="relu")(inputs)
x = tf.keras.layers.Dense(10, activation="relu")(x)
outputs = tf.keras.layers.Dense(1, activation='sigmoid')(x)

model = tf.keras.Model(inputs=inputs, outputs=outputs, name="small_model")

es = tf.keras.callbacks.EarlyStopping(monitor='val_loss', min_delta=0, patience=20,
                                      mode='auto', restore_best_weights=True)

model.compile(loss=tf.keras.losses.binary_crossentropy, optimizer=tf.keras.optimizers.SGD(), metrics=["accuracy"])
history = model.fit(X_train, y_train, batch_size=64, epochs=100, validation_split=0.1, callbacks=[es])

train_scores = model.evaluate(X_train, y_train, verbose=2)
test_scores = model.evaluate(X_test, y_test, verbose=2)
print("Train accuracy:", train_scores[1])
print("Test accuracy:", test_scores[1])
# Okay I think its enough with these easy datasets. Need to move on to more complex ones so actually
# tuning hyperparameters matter more.

