library(keras)

#DATA PREPARATION
#load dataset
mnist <- dataset_mnist()

#reshape
x_train_aja = array_reshape(mnist$train$x, c(60000,784))
x_testing_aja = array_reshape(mnist$test$x, c(10000,784))

#rescale by devide by 255
x_train_divide_aja = x_train_aja/255
x_test_divide_aja = x_testing_aja/255

#to categorical
new_y_train = to_categorical(mnist$train$y)
new_y_test = to_categorical(mnist$test$y)

#MODEL DEFINITION
model <- keras_model_sequential()
model %>%
  layer_dense(units = 256, input_shape = c(784)) %>%
  layer_dense(units = 10, activation = 'softmax')

model %>% compile(
  loss = 'categorical_crossentropy',
  optimizer = optimizer_rmsprop(),
  metrics = c('accuracy')
)

#TRAINING AND EVALUATION
# plot history result
history <- model %>% fit(
  x_train_divide_aja, new_y_train,
  batch_size = 128,
  epochs = 12,
  verbose = 1,
  validation_split = 0.2
)

# plot history result
plot(history)

# evaluate the test model
score <- model %>% evaluate(
  x_test_divide_aja, new_y_test,
  verbose = 0
)

# see the test result
score

#CHANGING MODEL PARAMETERS


