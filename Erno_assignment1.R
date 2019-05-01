library(keras)

#LOADING DATASET
mnist <- dataset_mnist()

##1: PERCEPTRON##

#Reshaping data for perceptron
x_train1 <- array_reshape(mnist$train$x, c(nrow(mnist$train$x), 784))
x_test1 <- array_reshape(mnist$test$x, c(nrow(mnist$test$x), 784))
#Rescaling
x_train1 <- x_train1 / 255
x_test1 <- x_test1 / 255

#Categorising labelings
y_train <- to_categorical(mnist$train$y)
y_test <- to_categorical(mnist$test$y)

#Building, compiling, fitting, evaluation
model1 <- keras_model_sequential() %>%
  layer_dense(units = 256, input_shape = c(784)) %>%
  layer_dense(units = 10, activation = 'softmax')
model1 %>% compile(
  loss = 'categorical_crossentropy',
  optimizer = optimizer_rmsprop(),
  metrics = c('accuracy') 
)

history1 <- model1 %>% fit(
  x_train1, y_train,
  batch_size = 128,
  epochs = 12,
  verbose = 1,
  validation_split = 0.2
)

score1 <- model1 %>% evaluate(
  x_test1, y_test,
  verbose=0
)




##2: CONVOLUTIONAL NETWORK##
#Reshaping data for convolutional layers
x_train2 <- array_reshape(mnist$train$x, c(nrow(mnist$train$x), 28, 28, 1))
x_test2 <- array_reshape(mnist$test$x, c(nrow(mnist$test$x), 28, 28, 1))
#Rescaling
x_train2 <- x_train2 / 255
x_test2 <- x_test2 / 255

#Buildling, compiling, fitting, evaluating
model2 <- keras_model_sequential() %>% 
  layer_conv_2d(filters = 32, kernel_size = c(3, 3),
                activation = 'relu', input_shape = c(28, 28, 1)) %>%
  layer_conv_2d(filters = 64, kernel_size = c(3, 3), 
                activation = 'relu') %>%
  layer_max_pooling_2d(pool_size = c(2, 2)) %>%
  #.25 dropout layer:
  layer_dropout(rate = 0.25) %>%
  layer_flatten() %>%
  layer_dense(units = 128, activation = 'relu') %>%
  #.5 dropout layer
  layer_dropout(rate = 0.5) %>%
  layer_dense(units = 10, activation = 'softmax') 

model2 %>% compile(
  loss = 'categorical_crossentropy',
  optimizer = optimizer_rmsprop(),
  metrics = c('accuracy')
)

history2 <- model2 %>% fit(
  x_train, y_train,
  batch_size = 128,
  epochs = 6,
  verbose = 1,
  validation_split = 0.2
)

score2 <- model2 %>% evaluate(
  x_test, y_test,
  verbose = 0
)

    