library(keras)

#LOADING DATASET
mnist <- dataset_mnist()

#Reshaping data for convolutional layers
x_train <- array_reshape(mnist$train$x, c(nrow(mnist$train$x), 28, 28, 1))
x_test <- array_reshape(mnist$test$x, c(nrow(mnist$test$x), 28, 28, 1))

#Rescaling
x_train <- x_train / 255
x_test <- x_test / 255

#Categorising labelings
y_train <- to_categorical(mnist$train$y)
y_test <- to_categorical(mnist$test$y)

#preparing a convolutional model
model <- keras_model_sequential() %>% 
  layer_conv_2d(filters = 32, kernel_size = c(3, 3),
                activation = 'relu', input_shape = c(28, 28, 1)) %>%
  layer_conv_2d(filters = 64, kernel_size = c(3, 3), 
                activation = 'relu') %>%
  layer_max_pooling_2d(pool_size = c(2, 2)) %>%
  layer_flatten() %>%
  layer_dense(units = 128, activation = 'relu') %>%
  layer_dense(units = 10, activation = 'softmax')

#Compiling
model %>% compile(
  loss = 'categorical_crossentropy',
  optimizer = optimizer_rmsprop(),
  metrics = c('accuracy')
)

#Fitting
history <- model %>% fit(
  x_train, y_train,
  batch_size = 128,
  epochs = 6,
  verbose = 1,
  validation_split = 0.2
)

#Evaluation
score <- model %>% evaluate(
  x_test, y_test,
  verbose = 0
)


    