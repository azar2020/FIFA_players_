
setwd("C:/Azar_Drive/FIFA")

# Define number of rounding digits
options(digits = 2)

# Required libraries
library(bnlearn)
library(bnstruct)
library(Rgraphviz)
library(caret)
library(forecast)


# Read data
data = read.csv("fiffa_players.csv",  header = TRUE)
dim(data)
colnames(data)

## Select variables
fifa_data = data[c("age", "height", 
                    "weight", "rating", 
                    "value", "heading_accuracy", 
                    "short_passing", "long_passing",
                    "acceleration", "penalties")]

# Check the dimension of data
dim(fifa_data)
colnames(fifa_data)

# Convert to a numeric dataframe
fifa_data = lapply(fifa_data, as.numeric)
fifa_data = as.data.frame(fifa_data) 

# Discretization
discretized_fifa_data = discretize(fifa_data, method = 'quantile', breaks =4)

# Convert discretized_data to a numeric dataframe 
discretized_fifa_data = as.data.frame(discretized_fifa_data)
discretized_fifa_data = sapply(discretized_fifa_data, as.numeric)  


# Divide the data to training (75%) and testing set(25%)
n = dim(discretized_fifa_data)[1]
n_training =214
n_testing = n - n_training
set.seed(6)
training_indices = sample(c(1:n), size = n_training)

training_data = discretized_fifa_data[training_indices,]
testing_data = discretized_fifa_data[-training_indices,]

# Create BNDataset object
BNDataset_object = BNDataset(training_data, discreteness = c( "d4", "d4", "d4", "d4", "d4", "d4", "d4", "d4", "d4", "d4"),
                             variables = c("age", "height", 
                                           "weight", "rating", 
                                           "value", "heading_accuracy",
                                           "short_passing", "long_passing",
                                           "acceleration", "penalties"),
                             node.sizes = c(4, 4, 4, 4, 4, 4, 4, 4, 4, 4))
BN_structure = learn.network(BNDataset_object, algo = "sm", scoring.func = "BIC")
BN_structure

# Plot the graph
# Convert DAG to PDAG
pdag = dag.to.cpdag(dag(BN_structure))

# Create weighted partially dag
wpdag(BN_structure) = pdag

plot(BN_structure,plot.wpdag=TRUE,node.col =c("cadetblue1", "cadetblue1","cadetblue1", "green3",
                                              "pink2",
                                              "green3",
                                              "green3","green3",
                                              "green3","green3"),
     node.lab.cex=24)



############ PREDICTION
## Makeb a bn object
adjacency_matrix = dag(BN_structure)
e = empty.graph(c("age", "height", 
                        "weight", "rating", 
                        "value", "heading_accuracy", 
                        "short_passing", "long_passing",
                        "acceleration", "penalties"))
node_sizes = node.sizes(BN_structure)
colnames(fifa_data) 
amat(e) = adjacency_matrix
e
bn_object <- bn.fit(e, data = data.frame(training_data))

#Predict "value" of a player
pred = predict(bn_object, node = "value", data = data.frame(testing_data), method = "parents")

# Compute error
error = accuracy(object = pred, x = testing_data[, "value"])
mean_absolute_error = error[3]
mean_absolute_error

## Remove "penalties" from the list of the variables
testing_data_without_penalty <- testing_data[, -10]


# Check the dimension of data
dim(testing_data_without_penalty)
colnames(testing_data_without_penalty)


# Predict "value" again
pred = predict(bn_object, node = "value", data = data.frame(testing_data_without_penalty), method = "parents")
error = accuracy(object = pred, x = testing_data[, "value"])
mean_absolute_error = error[3]
mean_absolute_error

