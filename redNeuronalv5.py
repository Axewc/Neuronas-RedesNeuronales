##PATH: C:\Users\eduva\AppData\Local\Microsoft\WindowsApps\PythonSoftwareFoundation.Python.3.11_qbz5n2kfra8p0\python.exe -m pip install --upgrade pip
##pip install --upgrade pip
##import numpy as np
##C:\Users\eduva\AppData\Local\Programs\Python\Python311\python.exe -m pip install --upgrade pip
##c:/Users/eduva/Documents/Amelia/redNeuronal/.conda/python.exe -m pip install numpy
##python -m pip install tensorflow 
##python -m pip install pydot
##python -m pip install graphviz

##from tensorflow.keras.models import Sequencial

import numpy as np
import pandas as pd
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import EarlyStopping
from keras_tuner.tuners import Hyperband
from keras_tuner.engine.hyperparameters import HyperParameters
from keras import callbacks
from sklearn.model_selection import KFold
import matplotlib.pyplot as plt
import seaborn as sns


# Cargar los datos
dataset = pd.read_csv('original_noise10_Train_set.csv')
dataset = dataset.reset_index(drop=True)
dataset = dataset.sample(frac=1).reset_index(drop=True)

# Separar características y objetivos
X = dataset.iloc[:, 3:-2].values   ## cambio por dataset.iloc[:, 3:-2].values, se empieza a contar desde 0, empieza en 2
Y = dataset.iloc[:, -2:].values

print(X.size)  #number of obs for x values

# Establecer la forma de entrada
input_shape = (X.shape[1],)
print(f'Forma de las características: {input_shape}')

def build_model(hp):
    model = Sequential()
    model.add(Dense(units=hp.Int('units1', min_value=3, max_value=50, step=3), input_shape=input_shape, activation='sigmoid'))
    model.add(Dense(units=hp.Int('units2', min_value=3, max_value=50, step=3), activation='sigmoid'))
    model.add(Dense(units=hp.Int('units3', min_value=3, max_value=50, step=3), activation='sigmoid'))
    model.add(Dense(units=hp.Int('units4', min_value=3, max_value=50, step=3), activation='sigmoid'))
    model.add(Dense(units=hp.Int('units5', min_value=3, max_value=50, step=3), activation='sigmoid'))
    model.add(Dense(2, activation='sigmoid'))

    custom_optimizer = Adam(learning_rate=hp.Float('learning_rate', min_value=1e-4, max_value=1e-2, sampling='LOG'))
    model.compile(loss='mean_absolute_error', optimizer=custom_optimizer, metrics=['mean_squared_error'])

    return model

# Definir el tuner
tuner = Hyperband(
    build_model,
    objective='val_mean_squared_error',
    max_epochs=10,
    factor=3,
    directory='keras_tuner_dir',
    project_name='my_tuning_project'
)

# Definir Early Stopping para detener el entrenamiento si no hay mejora en la métrica
early_stopping = EarlyStopping(monitor='val_loss', patience=3)

# Realizar la búsqueda de hiperparámetros
tuner.search(X, Y, epochs=20, validation_split=0.2, callbacks=[early_stopping])

# Obtener el mejor modelo y hiperparámetros
best_model = tuner.get_best_models(num_models=1)[0]
best_hyperparameters = tuner.get_best_hyperparameters(num_trials=1)[0]

# Imprimir los resultados
print("Mejor hiperparámetro learning_rate: {}".format(best_hyperparameters.get('learning_rate')))
print("Mejores hiperparámetros:")
for param, value in best_hyperparameters.values.items():
    print(f"{param}: {value}")
print("Mejor modelo:")
best_model.summary()



##Eligiendo optimo número de epocas con callbacks


# Crear el modelo despues de recibir ayuda de los hiperparametros
model = Sequential()
model.add(Dense(81, input_shape=input_shape, activation='sigmoid'))  # Cambiando la activación a sigmoide
model.add(Dense(33, activation='sigmoid'))
model.add(Dense(39, activation='sigmoid'))
model.add(Dense(9, activation='sigmoid'))
model.add(Dense(2, activation='sigmoid'))  # Dos neuronas de salida para las dos últimas columnas con activación sigmoide

model.summary()

# Especificar el learning rate al crear el optimizador Adam
custom_optimizer = Adam(learning_rate=0.01)  # Puedes ajustar el valor según sea necesario

# Configurar el modelo y comenzar el entrenamiento
model.compile(loss='mean_absolute_error', optimizer=custom_optimizer, metrics=['mean_squared_error'])

earlystopping = callbacks.EarlyStopping(monitor="val_loss",
										mode="min", patience=100,
										restore_best_weights=True)

earlystopping2 = callbacks.EarlyStopping(monitor="accuracy",
										mode="max", patience=100,
										restore_best_weights=True)

history = model.fit(X, Y, epochs=2000, batch_size=32, verbose=1, validation_split=0.2, callbacks=[earlystopping])    #369  validation loss

history2 = model.fit(X, Y, epochs=2000, batch_size=32, verbose=1, validation_split=0.2, callbacks=[earlystopping2])    #2000 validation accuracy


#visualize : val_mean_squared_error, val_loss, val_accuracy  VS epochs






# Cargar el conjunto de datos de prueba
test_dataset = pd.read_csv('original_noise10_test_set.csv')
# Separar características y objetivos del conjunto de prueba
X_test = test_dataset.iloc[:,3:-2].values  # Características (todas las columnas excepto las últimas dos)
Y_test = test_dataset.iloc[:, -2:].values  # Objetivos reales (últimas dos columnas)



# Generate generalization metrics
score = model.evaluate(X_test, Y_test, verbose=0)
print(f'Test loss: {score[0]} / Test accuracy: {score[1]}')
#Test loss: 0.0018360227113589644 / Test accuracy: 1.8790286048897542e-05

# Graficar los datos objetivos reales y las predicciones

# Hacer predicciones en el conjunto de prueba
predictions = model.predict(X_test)

# Graficar los objetivos reales
plt.figure(figsize=(12, 6))

plt.subplot(1, 2, 1)
#plt.scatter(Y_test[:, 1], Y_test[:, 0], color='blue',s=0.1)
sns.scatterplot(x=Y_test[:,1], y=Y_test[:,0], hue=test_dataset['PB'], palette='bright', s=2)
plt.xlabel('ENERGIA')
plt.ylabel('PB)')
plt.legend(title = 'PB')

# Graficar las predicciones
plt.subplot(1, 2, 2)
#plt.scatter(predictions[:, 1], predictions[:, 0], color='red',s=0.1)
sns.scatterplot(x=predictions[:,1], y=predictions[:,0], hue=test_dataset['PB'], palette='bright', s=2)
plt.xlabel('ENERGIA')
plt.ylabel('PB')

plt.tight_layout()
plt.show()

# Visualize history
# Plot history: Loss
plt.plot(history2.history['val_loss'], linewidth=.5)
plt.title('Validation loss history')
plt.ylabel('Loss value')
plt.xlabel('No. epoch')
plt.show()

# Plot history: Accuracy
plt.plot(history2.history['val_accuracy'], linewidth=.5)
plt.title('Validation accuracy history')
plt.ylabel('Accuracy value (%)')
plt.xlabel('No. epoch')
plt.ylim(0, .00003)
plt.show()

plt.plot(history2.history['mean_squared_error'], linewidth=.5)
plt.title('Mean squared error history')
plt.ylabel('Mean squared error')
plt.xlabel('No. epoch')
plt.show()

plt.plot(history2.history['val_mean_squared_error'], linewidth=.5)
plt.title('Val Mean squared error history')
plt.ylabel('Val Mean squared error')
plt.xlabel('No. epoch')
plt.show()


##Adding K-fold Cross Validation

#Define the K-fold Cross Validator
num_folds = 10
#Define pre-fold score containers
acc_per_fold = []
loss_per_fold = []

## merge inputs and targets
inputs = np.concatenate((X, X_test), axis=0)
targets = np.concatenate((Y, Y_test), axis=0)

kfold = KFold(n_splits=num_folds, shuffle=True)

#Code for K-fold Cross validation
fold_no = 1

for train, test in kfold.split(inputs, targets):

    #Define the model architecture
    model = Sequential()
    model.add(Dense(81, input_shape=input_shape, activation='sigmoid'))  # Cambiando la activación a sigmoide
    model.add(Dense(33, activation='sigmoid'))
    model.add(Dense(39, activation='sigmoid'))
    model.add(Dense(9, activation='sigmoid'))
    model.add(Dense(2, activation='sigmoid'))  # Dos neuronas de salida para las dos últimas columnas con activación sigmoide

    #compile the model
    model.compile(loss='mean_absolute_error', optimizer=custom_optimizer, metrics=['mean_squared_error'])

    #Generate a print
    print('--------------------------------------------------------------------')
    print(f'Training for fold {fold_no} ...')

    #Fit data to model
    history3 = model.fit(inputs[train], targets[train],  epochs=2000, batch_size=32, verbose=1, validation_split=0.2)

    #Generate generalization metrics
    scores = model.evaluate(inputs[test], targets[test], verbose = 0)
    print(f'Score for fold {fold_no}: {model.metrics_names[0]} of {scores[0]}; {model.metrics_names[1]} of {scores[1]*100}%')
    acc_per_fold.append(scores[1] * 100)
    loss_per_fold.append(scores[0])

    #Increase fold number
    fold_no = fold_no + 1

#prueba

# == Provide average scores ==
print('------------------------------------------------------------------------')
print('Score per fold')
for i in range(0, len(acc_per_fold)):
  print('------------------------------------------------------------------------')
  print(f'> Fold {i+1} - Loss: {loss_per_fold[i]} - Accuracy: {acc_per_fold[i]}%')
print('------------------------------------------------------------------------')
print('Average scores for all folds:')
print(f'> Accuracy: {np.mean(acc_per_fold)} (+- {np.std(acc_per_fold)})')
print(f'> Loss: {np.mean(loss_per_fold)}')
print('------------------------------------------------------------------------')