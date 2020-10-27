
############################
# Aprendizado Supervisionado
############################
# Qualquer modelo f induzido à partir de um conjunto de exemplos 
# {(x1,y1), ..., (xn,yn)} a fim de:
#	
#	f : X -> Y
#
# x1, ..., xn in X, espaço de entradas
# y1, ..., yn in Y, espaço de saídas
#
# Subtipos de aprendizado supervisionados:
#	- classificação:
#		Espaço de saída Y é finito e discreto
#		Ex: KNN
#	- regressão:
#		Espaço de saída Y é contínuo e infinito
#		Ex: DWNN
############################

############################
# Como garantir o sucesso do aprendizado supervisionado?
############################
#
# Vapnik propôs a Teoria do Aprendizado Estatístico
#
#	Conceito de Generalização -> ajuda nas provas de aprendizado
#
#		|Remp(f) - R(f)| -> 0, n -> Inf
#
#	Risco Empírico/Risco Amostral/Erro Amostral
#
#		Remp(f) = 1/n Sum(i, 1, n, Loss(xi, yi, f(xi)))
#
#	0-1 Loss:			  -
#		0-1 Loss(xi, yi, f(xi)) = | 0, f(xi) = yi
#					  | 1, caso contrário
#					  -
#	Risco/Valor Esperado do Risco/Valor Esperado do Erro
#
#		R(f) = E(Loss(X, Y, f(X))), todas as possíveis entradas
#
#			Não é calculado sobre uma amostra, mas sim
#			sobre todo o espaço de possíveis entradas e
#			suas respectivas saídas
############################

############################
# Instanciação do Conceito de Generalização 
############################
#
#	Seja um grupo de estudantes denominado Classe a qual
#	deve estudar e APRENDER um assunto qualquer
#
#	Seja um professor que seja responsável pelo ensino de
#	tal assunto perante esta Classe
#
#	É responsabilidade do professor criar um conjunto de exemplos
#	representativos para o problema em questão
#
#		- esses exemplos correspondem à realidade ou são
#		  um subconjunto que não contempla todas as possíveis
#		  classes/rótulos (classificação) ou espaço de saídas
#		  (regressão)?
#		- motivar os alunos a procurar por materiais de viéses
#		  distintos (codificar, ler livros ou tutoriais com 
#		  características distintas e diversas)
#
#	É responsabilidade de cada aluno:
#
#		- não decorar (overfitting) um material único
#			- overfitting -> a representação perfeita de 
#				uma amostra / um modelo perfeito para uma
#				amostra / meu erro é zero nesta amostra
#
#				Remp(f)	= 0 (f produz overfitting)
#				R(f) = 0 ---> apenas para problemas simplórios
#					ou
#				R(f) = max
#
#	Instância:
#
#		|Remp(f) - R(f)| -> 0, n -> Inf
#
#		Nota prova (acurácia in [0,1]):
#			1 - [1/n Sum(i, 1, n, 0-1 Loss(xi,yi,f(xi)))]
#
#			Nota 	Remp
#			  0.9	 0.1
#			  0.7	 0.3
#			  0.2	 0.8
#			  1.0    0.0
#
#		Contato com o mundo real (Generalização/Vapnik)
#
#			Nota 	| Remp	| Mundo Real (Remp') | Generalização
#			  0.9	|  0.1	|	       0.15  | |0.1-0.15|=0.05
#			  0.7	|  0.3  |	       0.3   | |0.3-0.3|=0
#			  0.2	|  0.8  | 	       0.8   | |0.8-0.8|=0
#			  1.0   |  0.0  |              0.25  | |0.0-0.25|=0.25
#
#	O que é APRENDER para Vapnik??? (Convergências)
#
#		(1) Generalização |Remp(f)-R(f)| -> 0, n -> Inf
#		(2) Remp(f) -> 0, n -> Inf
#
#	Overfitting			Underfitting
#	###########			############
#
#	- Representação perfeita	- O modelo f nem mesmo representa
#	  de uma amostra		  a própria amostra escolhida
#	- Grande erro quando exposto	- Grande erro quando exposto a todas
#	  a todas as possibilidades	  as possibilidades para o problema
#	  para o problema em questão	  em questão
#
#		Remp(f) = 0			Remp(f) ~ max
#		R(f) = max			R(f) = max
#
#
#	Generalização é apenas um critério de seleção de modelos (Estimador)
#       ####################################################################
#
#	Modelos
#		f1, f2, f3, ..., fm
#
#	Remp	R	G	--->   Subseleção dos melhores estimadores
#					  (Critério G <= 0.05)
# f1	0.05	0.07	0.02			X
# f2	0.1	0.11	0.01			X
# f3    0.5	0.5	0.0			X
# f4	0.01	0.5	0.49			Nope
# ...
# fm    0.4	0.45	0.05			X
#
#	Encontrar os melhores estimadores para R(f)
#	###########################################
# 	
#		Remp    R
#	  f1	0.05	0.07	<<<<---------
#	  f2	0.1	0.11
#	  f3    0.5	0.5 
#	  fm    0.4	0.45
#
#####################################################################

#####################################################################
# Vapnik 1966 ~ 1999 : Teoria do Aprendizado Estatístico
#####################################################################
#
#	Lei dos Grandes Números	(A Probabilistic Theory of 
#					Pattern Recognition)
#
#	P(|Remp(f) - R(f)| > eps) <= 2 exp(-2 n eps^2), n -> Inf
#
#	em que:
#		Remp(f) in [0,1]
#		R(f) in [0,1]
#		n tamanho da amostra, um número natural
#
#	P(sup_{f in F} |Remp(f) - R(f)| > eps) <= 
#		2 P(sup_{f in F} |Remp(f) - Remp'(f)| > eps) <= 
#			2 m exp(-n eps^2 / 4)
#
#		F é o viés ou bias do algoritmo de aprendizado
#
# Para refletir:
#
#	- O que ocorre se o número de funções contidas em F for infinito?
#	- O que ocorre se o número de funções contidas em F for finito?
#		- E se m for uma função do tamanho da amostra n?
#			m(n) = n^2
#			m(n) = n^3
#			m(n) = n^5
#			m(n) = log(n)
######################################################################

######################################################################
#
# 2 P(sup_{f in F} |Remp(f) - Remp'(f)| > eps) <= 2 m exp(-n eps^2 / 4)
#
#
#					Um grande conjunto de exemplos (n)
#					##############################
#
#	Hold-Out		----> 	Amostra 1 (treino), Amostra 2 (teste)
#					A1 (treino), A2 (Validação), A3(teste)
#
#	Subsampling		---->  	A1 (n) reamostrados com reposição 68%
#					A2 disjunto de A1 para testar
#					(exemplos nunca vistos em treinamento)
#
#	K-Fold Cross Validation	---->	K folds (k <= n)
#		quando k = n, leave-one-out
#
#					fold 1 (1/n)
#					fold 2 (1/n)
#					fold 3 (1/n)
#					fold 4 (1/n)
#					...
#					fold n-1 (1/n)
#					fold n (1/n)
#
#				------------------------------------
#				Remp_{fold1}(f), ..., Remp_{foldn}(f)
#########################################################################

#########################################################################
# Qualquer analista faria
#########################################################################
# Seja A um algoritmo da lista A1, A2, ..., Al
#
# 	Para A
#
#		(1) Calcule os Remps para K-Fold Cros Validation
#		(2) Calcule:
#		  mean(Remp_{fold1}(f), ..., Remp_{foldn}(f))
#		  sd(Remp_{fold1}(f), ..., Remp_{foldn}(f))
#
#	Selecione o Algoritmo A com menor média e sd
#########################################################################

#########################################################################
#
#	P(sup_{f in F} |Remp(f) - R(f)| > eps) <= 
#		2 P(sup_{f in F} |Remp(f) - Remp'(f)| > eps) <= 
#			2 m(2n) exp(-n eps^2 / 4)
#
# 2 P(sup_{f in F} |Remp(f) - Remp'(f)| > eps) <= 2 m(2n) exp(-n eps^2 / 4)
# delta = 2 m(2n) exp(-n eps^2 / 4)
# delta / (2 m(2n)) = exp(-n eps^2 / 4)
# log(delta) - log(2 m(2n)) = -n eps^2 / 4
# 4/n (log(2 m(2n) - log(delta)) = eps^2
# eps = sqrt(4/n (log(2 m(2n) - log(delta)))
#
# Quando eu erro:
#################
# sup_{f in F} |Remp(f) - Remp'(f)| > sqrt(4/n (log(2 m(2n) - log(delta)))
#
# Quando eu acerto:
###################
# sup_{f in F} |Remp(f) - Remp'(f)| <= sqrt(4/n (log(2 m(2n) - log(delta)))
#
#########################################################################

require(class)

############################
# iris
#	setosa		50
#	virginica	50
#	versicolor	50
#
# K-Fold Cross Validation
#	K=10 ~ leave-one-out
############################

require(dismo)

folds = list()

# Criar folds estratificados, i.e., contendo a mesma representatividade
# das classes existentes no nosso conjunto de dados original
for (class in unique(iris[,5])) {
	ids = which(iris[,5] == class)
	folds.ids = kfold(ids, k=10)

	for (fold in 1:10) {
		sel = which(fold == folds.ids)
		if (length(folds) >= fold) {
			folds[[fold]] = rbind(folds[[fold]], iris[ids[sel],])
		} else {
			folds[[fold]] = iris[ids[sel],]
		}
	}
}

# Randomizar cada fold
for (fold in 1:10) {
	folds[[fold]] = folds[[fold]][sample(1:nrow(folds[[fold]])),]
}

Remp.folds = c()

for (fold in 1:10) {

	X.train = NULL
	Y.train = c()

	for (f in setdiff(1:10, fold)) {
		X.train = rbind(X.train, folds[[f]][,1:4])
		Y.train = c(Y.train, folds[[f]][,5])
	}

	y = knn(train=X.train, test=folds[[fold]][,1:4], cl=Y.train, k = 3)

	acc = sum(as.numeric(y) == as.numeric(folds[[fold]][,5])) / nrow(folds[[fold]])
	Remp = 1 - acc
	Remp.folds = c(Remp.folds, Remp)
}

cat("Remp médio = ", mean(Remp.folds), "\n")

# Se o desvio for alto
#
# - cada modelo apresenta um resultado bastante diferente do outro
# - intuições sobre meu conjunto de dados:
#	- eventualmente os exemplos de meu conjunto de dados
#		apresentam algum grau dependência entre si
#	- o desbalanceamento de classes por fold afeta o aprendizado
cat("Remp stdev = ", sd(Remp.folds), "\n")

