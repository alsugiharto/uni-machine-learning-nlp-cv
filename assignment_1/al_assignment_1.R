library(keras)

#DATA PREPARATION
#load dataset
mnist <- dataset_mnist()

#reshape
x_train_aja = array_reshape(mnist$train$x, c(60000,784))
x_testing_aja = array_reshape(mnist$test$x, c(10000,784))

dim(x_train_aja)
dim(x_testing_aja)

#rescale by devide by 255
x_train_divide_aja = x_train_aja/255
x_test_divide_aja = x_testing_aja/255

#to categorical
new_y_train = to_categorical(mnist$train$y)
new_y_test = to_categorical(mnist$test$y)

#==================================================================================================

#Q2 - Q6
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

#==================================================================================================
#TODO: create a function to repeat the different activation
#Q7 - Q10
#CHANGING MODEL PARAMETERS

#MODEL DEFINITION
modelRelu <- keras_model_sequential()
modelRelu %>%
  layer_dense(units = 256, activation= 'relu', input_shape = c(784)) %>%
  layer_dense(units = 10, activation = 'softmax')

modelRelu %>% compile(
  loss = 'categorical_crossentropy',
  optimizer = optimizer_rmsprop(),
  metrics = c('accuracy')
)

#TRAINING AND EVALUATION
# plot history result
historyRelu <- modelRelu %>% fit(
  x_train_divide_aja, new_y_train,
  batch_size = 128,
  epochs = 12,
  verbose = 1,
  validation_split = 0.2
)

# plot history result
plot(historyRelu)

# evaluate the test model
score <- modelRelu %>% evaluate(
  x_test_divide_aja, new_y_test,
  verbose = 0
)

# see the test result
score

#==================================================================================================
#Q11 - Q14
#Deep convolutional networks
#new data preparation

#reshape
x_train_deep = array_reshape(mnist$train$x, c(60000,28, 28, 1))
x_testing_deep = array_reshape(mnist$test$x, c(10000,28, 28, 1))

dim(x_train_deep)
dim(x_testing_deep)

#rescale by devide by 255
x_train_divide_deep = x_train_deep/255
x_test_divide_deep = x_testing_deep/255

#to categorical
new_y_train = to_categorical(mnist$train$y)
new_y_test = to_categorical(mnist$test$y)

#MODEL DEFINITION
model_deep <- keras_model_sequential() %>%
  layer_conv_2d(filters = 32, kernel_size = c(3,3),
                activation = 'relu', input_shape = c(28,28,1)) %>%
  layer_conv_2d(filters = 64, kernel_size = c(3,3),
                activation = 'relu') %>%
  layer_max_pooling_2d(pool_size = c(2,2)) %>%
  layer_flatten() %>%
  layer_dense(units = 128, activation = 'relu') %>%
  layer_dense(units = 10, activation = 'softmax')

model_deep %>% compile(
  loss = 'categorical_crossentropy',
  optimizer = optimizer_adadelta(),
  metrics = c('accuracy')
)

#TRAINING AND EVALUATION
# plot history result
history_deep <- model_deep %>% fit(
  x_train_divide_deep, new_y_train,
  batch_size = 128,
  epochs = 6,
  verbose = 1,
  validation_split = 0.2
)

plot(history_deep)

# evaluate the test model
score_deep <- model_deep %>% evaluate(
  x_test_divide_deep, new_y_test,
  verbose = 0
)

score_deep

#==================================================================================================
#Q15 - Q17
#Deep convolutional networks with dropout

#MODEL DEFINITION
#watch that 2 lines of dropout layes are added
model_deep_dropout <- keras_model_sequential() %>%
  layer_conv_2d(filters = 32, kernel_size = c(3,3),
                activation = 'relu', input_shape = c(28,28,1)) %>%
  layer_conv_2d(filters = 64, kernel_size = c(3,3),
                activation = 'relu') %>%
  layer_max_pooling_2d(pool_size = c(2,2)) %>%
  layer_dropout(rate = 0.25) %>%
  layer_flatten() %>%
  layer_dense(units = 128, activation = 'relu') %>%
  layer_dropout(rate = 0.5) %>%
  layer_dense(units = 10, activation = 'softmax')

model_deep_dropout %>% compile(
  loss = 'categorical_crossentropy',
  optimizer = optimizer_adadelta(),
  metrics = c('accuracy')
)

#TRAINING AND EVALUATION
# plot history result
history_deep_dropout <- model_deep_dropout %>% fit(
  x_train_divide_deep, new_y_train,
  batch_size = 128,
  epochs = 6,
  verbose = 1,
  validation_split = 0.2
)

plot(history_deep_dropout)

# evaluate the test model
score_deep_dropout <- model_deep_dropout %>% evaluate(
  x_test_divide_deep, new_y_test,
  verbose = 0
)

score_deep_dropout

