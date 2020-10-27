# Conjunto de dados de tipos vegetacionais dos EUA
dataset = foreign::read.arff("../ex05/covtypeNorm/covtypeNorm.arff")

# Testar a dependência de um atributo
# (1) Avaliar a ACF
#	atributo Elevation -> existe dependência
#	atributo Horizontal_Distance_To_Hydrology -> existe dependência
#
# (2) Statistical Learning Theory (SLT) --> Teoria do Aprendizado Estatístico
#	Garantir aprendizado de máquina --> provas de aprendizado
#		
#	Modelo f -> conjunto de dados
#
#		f:Teste -> Respostas tão corretas 
#			   quanto as produzidas no 
#			   conjunto de treinamento
#
#	Construímos o modelo f usando uma subamostra do dataset 75% -> 0.75
#	Testar f nas amostras restantes que nunca foram vistas 25%  -> 0.755
#
#		Poder de generalização: |(1-0.75) - (1-0.755)| = 
#					|0.25 - 0.245|	= 0.005
#
#	Quão mais próximo de zero, maior a generalização
#
#
#	SLT:
#		(1) Deve-se generalizar!!! |R_emp1(f) - R_emp2(f)| -> 0
#				conforme o tamanho das amostras -> inf
#		(2) O risco na amostra de treinamento deve tender a zero
#				conforme o tamanho da amostra -> inf
#		(3) Isto somente funciona SE E SOMENTE SE:
#			- os exemplos forem independentes entre si

require(randomForest)
size = floor(0.1 * nrow(dataset))
cat("Tamanho das amostras de treino e teste = ", size, "\n")

all.generalizations = c()
for (i in 1:10) {
	cat("Iteration ", i, "\n")
	train.id = sample(1:nrow(dataset), size=size)
	test.id = sample(1:nrow(dataset), size=size)

	f = randomForest(x=dataset[train.id,1:10], y=dataset[train.id,55], ntree=100)

	train.y = predict(f, dataset[train.id,1:10])
	test.y = predict(f, dataset[test.id,1:10])

	train.acc = sum(dataset[train.id,55] == train.y) / size
	test.acc = sum(dataset[test.id,55] == test.y) / size

	cat("Riscos nas amostras de treinamento e teste: ",
		(1-train.acc), " ", (1-test.acc), "\n")

	generalization = abs((1-train.acc) - (1-test.acc))

	all.generalizations = c(all.generalizations, generalization)
}
# all.generalizations
# 0.1228722 0.1196193 0.1200151 0.1178293 0.1171581 0.1194300 0.1202905 0.1189308 0.1198430 0.1236812
#

# Comparando com a generalização para o conjunto de dados original (sem aleatorização)
f = randomForest(x=dataset[1:size,1:10], y=dataset[1:size,55], ntree=100)
train.y = predict(f, dataset[1:size,1:10])
test.y = predict(f, dataset[(size+1):(2*size),1:10])
train.acc = sum(dataset[1:size,55] == train.y) / size
test.acc = sum(dataset[(size+1):(2*size),55] == test.y) / size
cat("Riscos nas amostras de treinamento e teste: ", 
    (1-train.acc), " ", (1-test.acc), "\n")
cat("Generalização: ", abs((1-train.acc) - (1-test.acc)), "\n")
# Riscos nas amostras de treinamento e teste:  0   0.2653311 
# Generalização:  0.2653311
