# Classification

source("ibl.r")

require(dismo)
require(foreign)
dataset = read.arff("covtypeNorm/covtypeNorm.arff")

# Para testar bem rápido :D
#dataset = dataset[sample(1:nrow(dataset), size=10000),]

print(names(dataset))
print(dim(dataset))

# Estudo do espaço a fim de saber quais e quantas dimensões são úteis para meu problema
for (i in 11:54) {
	dataset[,i] = as.numeric(dataset[,i])-1
}
# 11:54 -> {0, 1}
# for (col in 11:54) {
#	dataset[,col] = dataset[,col] + rnorm(mean=0, sd=1e-3, n=nrow(dataset))
# }
C = cov(dataset[,1:54])
E = eigen(C)

plot(cumsum(E$values/sum(E$values)), pch=20, cex=2)

new.space = as.matrix(dataset[,1:54])%*%E$vectors
new.dataset = new.space[,1:20]

##########################################################################
# Dividir meu conjunto em treinamento e teste
#	Hold-out: Separar o conjunto original em 2 subconjuntos disjuntos:
#			um treinamento e outro para teste
#train.size = 0.8
#ids = sample(1:nrow(new.dataset), size=ceiling(nrow(new.dataset)*train.size))
#X.train = new.dataset[ids,1:20]
#Y.train = dataset[ids,55]
#X.test = new.dataset[-ids,1:20]
#Y.test = dataset[-ids,55]
#
#correct = 0
#for (i in 1:nrow(X.test)) {
#	cat(i, " out of ", nrow(X.test), "\n")
#	x = X.test[i,]
#	y = knn(query=x, k=3, X=X.train, Y=Y.train)$max.prob.class
#	if (y == Y.test[i]) { correct = correct + 1 }
#	cat("Partial hold-out accuracy: ", correct / i, "\n")
#}
#cat("Hold-out accuracy = ", correct / nrow(X.test), "\n")
##########################################################################

##########################################################################
#	Leave-one-out: Treine com todos exceto um elemento e 
#		teste com esse elemento que ficou fora do treinamento
#correct = 0
#for (i in 1:nrow(new.dataset)) {
#	X.train = new.dataset[-i,1:20]
#	Y.train = dataset[-i,55]
#
#	cat(i, " out of ", nrow(new.dataset), "\n")
#	x = new.dataset[i,1:20]
#	y = knn(query=x, k=3, X=X.train, Y=Y.train)$max.prob.class
#	if (y == dataset[i,55]) { correct = correct + 1 }
#	cat("Partial leave-one-out accuracy: ", correct / i, "\n")
#}
#cat("Leave-one-out accuracy = ", correct / nrow(X.test), "\n")
##########################################################################

##########################################################################
#	K-Fold Cross Validation: dividir meu conjunto em k subconjuntos.
#		Realizo o treinamento com k-1 folds e testo com o k-ésimo
#
#> sum(dataset[,55] == 1)
#[1] 211840
#> sum(dataset[,55] == 2)
#[1] 283301
#> sum(dataset[,55] == 3)
#[1] 35754
#> sum(dataset[,55] == 4)
#[1] 2747
#> sum(dataset[,55] == 5)
#[1] 9493
#> sum(dataset[,55] == 6)
#[1] 17367
#> sum(dataset[,55] == 7)
#[1] 20510

X.folds = list()
Y.folds = list()

for (i in 1:10) {
	X.folds[[i]] = NA
	Y.folds[[i]] = NA
}

for (i in 1:7) {
	ids = which(dataset[,55] == i)
	foldId = kfold(ids, k=10)
	for (j in 1:10) {
	   if (is.na(X.folds[[j]])) {
		   X.folds[[j]] = as.matrix(new.dataset[ids[which(foldId == j)], 1:20])
		   Y.folds[[j]] = as.vector(dataset[ids[which(foldId == j)],55])
	   } else {
		   X.folds[[j]] = 
			   rbind(X.folds[[j]], 
				 as.matrix(new.dataset[ids[which(foldId == j)], 1:20]))
		   Y.folds[[j]] = c(Y.folds[[j]], 
				    as.vector(dataset[ids[which(foldId == j)],55]))
	   }
	}
}

all.accuracies = c()
for (i in 1:10) {
	X.train = NULL
	Y.train = c()

	# Treinar com todos os folds exceto o fold i
	for (j in setdiff(1:10, i)) {
		X.train = rbind(X.train, X.folds[[j]])
		Y.train = c(Y.train, Y.folds[[j]])
	}

	# Testar com o fold = i
	correct = 0
	for (j in 1:nrow(X.folds[[i]])) {
		x = X.folds[[i]][j,1:20]
		y = knn(query=x, k=3, X=X.train, Y=Y.train)$max.prob.class
		if (y == Y.folds[[i]][j]) {
			correct = correct + 1
			#cat("Partial 10-fold cross validation = ", correct / j, "\n")
		}
	}
	acc = correct / nrow(X.folds[[i]])
	all.accuracies = c(all.accuracies, acc)
	print(all.accuracies)
}
cat("10-fold cross validation accuracies: ")
print(all.accuracies)
##########################################################################

# Visualizar os resultados de classificação
