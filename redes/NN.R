#### Neural Networks in R  ######


# Aplicaremos redes neuronales a un ejemplo de ingenieria, en donde es clave predecir el desempeno
# de los materiales
# Pronosticaremos la fortaleza del concreto
# Con base en las caracteristicas de un conjunto de mezclas
# Trabajaremos con el archivo concrete.csv

concrete <- read.csv("concrete.csv")

# Veamos la estructura de los datos

str(concrete)

# 1030 mezclas de concreto y 9 variables
# El outcome de interes es strength. El resto son caracteristicas de la mezcla


# En redes neuronales, por las caracteristicas de las funciones de activacion, es ideal normalizar

# Definamos una funcion de normalizacion

normalize <- function(x) {
  return((x - min(x)) / (max(x) - min(x)))
}

# Y apliquemosla a nuestros datos

concrete_norm <- as.data.frame(lapply(concrete, normalize))

# Podemos verificar que se hayan normalizado los datos examinando el outcome de interes

summary(concrete_norm$strength)

# En efecto esta entre 0 y 1

# Dividamos la base en dos: entrenamiento y prueba. La base ya esta ordenada aleatoriamente

concrete_train <- concrete_norm[1:773, ]

concrete_test <- concrete_norm[774:1030, ]

# Construiremos una red neuronal para predecir la fortaleza del concreto en funcion de las demas
# caracteristicas

# Usaremos el paquete neuralnet para ello

# Instalemoslo
install.packages("neuralnet")

# Y carguemoslo

library(neuralnet)

# El paquete tiene la funcion neuralnet, para hace prediccion numerica

# Cuya sintaxis es m <- neuralnet(target~predictors, data=mydata, hidden=1)

# target es la variable que queremos predecir
# predictors los predictores que usaremos
# data la base a usar
# hidden especifica el numero de nodos ocultos. El default es 1


# Entrenemos un modelo sencillo con solo una capa oculta en nuestro caso

set.seed(123)
concrete_model <- neuralnet(strength ~ cement + slag
                            + ash + water + superplastic + coarseagg + fineagg + age,
                            data = concrete_train)


# Podemos graficar la topologia de la red estimada

plot(concrete_model)

# El modelo tiene un nodo de entrada para cada caracteristica
# Un nodo oculto y uno nodo de salida
# Los numeros en la lineas azules son los terminos de sesgo (similares a los interceptos)
# Notese que abajo tenemos el numero de pasos empleado para entrenar la red
# Y la suma de errores cudraticos


# Veamos ahora el desempeno de la red en los datos de entrenamiento
# Para eso usamos la funcion compute()

model_results <- compute(concrete_model, concrete_test[1:8])


# Podemos recuperar el vector de predicciones de la siguiente forma:

predicted_strength <- model_results$net.result

# La prediccion es numerica, luego no podemos construir la matriz de confusion
# Queremos saber que tanto se parece la prediccion con lo que observamos

# Empecemos viendo la correlacion:

cor(predicted_strength, concrete_test$strength)

# Una correlacion de 0.72, relativamente alta. El resultado es moderadamente bueno

# Podemos calcular la suma de los residuos al cuadrado:

SSR <- function(x,xhat){
  sum((x-xhat)^2)
}

SSR(concrete_test$strength, predicted_strength)

# Nos da 3.25

# Que pasa si estimamos una red mas compleja?
# Estimemos una con 5 nodos ocultos


set.seed(123)
concrete_model2 <- neuralnet(strength ~ cement + slag
                            + ash + water + superplastic + coarseagg + fineagg + age,
                            data = concrete_train, hidden=5)

# Como antes, podemos visualizar la red
plot(concrete_model2)

# Mucho mas compleja, naturalmente

# Hacemos las predicciones

model_results2 <- compute(concrete_model2, concrete_test[1:8])
predicted_strength2 <- model_results2$net.result

# Y evaluamos el desempeno

cor(predicted_strength2, concrete_test$strength)
SSR(concrete_test$strength, predicted_strength2)

# Mejoramos en correlacion, empeoramos en SSR
