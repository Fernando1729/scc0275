
##############################
# IBL: Instance-Based Learning
##############################

#################################################################
# K-Nearest Neighbors (Classificação/Y é finito e discreto)
# query = consulta ou ponto de consulta sobre o espaço de entrada
# k = define o número de vizinhos mais próximos considerados na
#     votação
# X = base de conhecimento, instâncias conhecidas, knowledge base
# Y = as classes de cada instância (ou exemplo) contida em X
#################################################################
knn <- function(query, k, X, Y) {
	E = apply(X, 1, function(row) { sqrt(sum((row - query)^2)) })
	row.ids = sort.list(E,dec=F)[1:k]
	classes = unique(Y)
	count = rep(0, length(classes))
	i = 1
	for (class in classes) {
		count[i] = sum(class == Y[row.ids])
		i = i + 1
	}
	ret = list()
	ret$classes = classes
	ret$count = count
	ret$max.prob.class = classes[which.max(count)]
	return (ret)
}

############################################################################
# Distance-Weighted Nearest Neighbors (Regressão/Y seja infinito e contínuo)
# query = consulta ou ponto de consulta sobre o espaço de entrada
# X = base de conhecimento, instâncias conhecidas, knowledge base
# Y = as classes de cada instância (ou exemplo) contida em X
# sigma = abertura ou o desvio padrão da Gaussiana utilizada para determinar 
#	  vizinhanças
############################################################################
dwnn <- function(query, X, Y, sigma) {
	E = apply(X, 1, function(row) { sqrt(sum((row - query)^2)) })
	weight = exp(-E^2 / (2*sigma^2))
	return (weight %*% Y / sum(weight))
}

estacionariedade <- function(series, length=250, order=1) {
	require(moments)

	# Média 	-> primeiro momento estatístico
	# Variância 	-> segundo momento estatístico
	# Skewness	-> terceiro momento estatístico
	# Curtose	-> quarto momento estatístico
	len = length(series)
	div = sqrt((moment(series[1:floor(len / 2)], order=order) - 
		    	moment(series[(floor(len / 2)+1):len], order=order))^2)
	if (len > length) {
		div = c(div, estacionariedade(series[1:floor(len / 2)]) + 
			estacionariedade(series[(floor(len / 2)+1):len]))
	}
	return (div)
}

seno <- function(sigma=0.5, sd=0.5, mi.epsilon=0.05) {

	# x(t) = sin(2*pi*t) + N(u=0, sd=0.5)
	tempo = seq(0,5, length=1000)
	valores = sin(2*pi*tempo) + rnorm(mean=0, sd=sd, n=1000)

	# Decomposição: Applying Empirical Mode Decomposition and mutual information 
	# to separate stochastic and deterministic influences embedded in signals
	# -> considera a adição entre componente determinístico e estocástico
	require(EMD)
	decomposicao = emd(valores)

	# Analisar as fases das Intrinsic Mode Functions para verificar
	# a ocorrência de congruências (congruência de fase)
	par(mfrow=c(3,ceiling((decomposicao$nimf + 2) / 3)))
	plot(valores, pch=20, cex=2, main="Série original", cex.axis=2, cex.main=2)
	for (i in 1:decomposicao$nimf) {
		plot(decomposicao$imf[,i], 
		     pch=20, cex=2, main=paste("IMF", i), cex.axis=2, cex.main=2)
	}
	plot(decomposicao$residue, pch=20, cex=2, main="Resíduo", cex.axis=2, cex.main=2)
	locator(1)

	par(mfrow=c(3,ceiling((decomposicao$nimf + 2) / 3)))
	plot(valores, pch=20, cex=2, main="Série original", cex.axis=2, cex.main=2)
	for (i in 1:decomposicao$nimf) {
		coeff = fft(decomposicao$imf[,i])
		plot(coeff, asp=1, pch=20, cex=2, main=paste("IMF", i), cex.axis=2, cex.main=2)
	}
	coeff = fft(decomposicao$residue)
	plot(coeff, asp=1, pch=20, cex=2, main="Resíduo", cex.axis=2, cex.main=2)
	locator(1)

	par(mfrow=c(3,ceiling((decomposicao$nimf + 2) / 3)))
	plot(valores, pch=20, cex=2, main="Série original", cex.axis=2, cex.main=2)
	phases.vector = list()
	for (i in 1:decomposicao$nimf) {
		coeff = fft(decomposicao$imf[,i])
		phases = atan(Im(coeff) / Re(coeff))
		plot(phases, pch=20, cex=2, main=paste("IMF", i), cex.axis=2, cex.main=2)
		phases.vector[[i]] = phases
	}
	coeff = fft(decomposicao$residue)
	phases = atan(Im(coeff) / Re(coeff))
	plot(phases, pch=20, cex=2, main="Resíduo", cex.axis=2, cex.main=2)
	phases.vector[[i+1]] = phases
	locator(1)

	require(c3net)
	mutual.information = rep(0, length(phases.vector)-1)
	for (i in 1:(length(phases.vector)-1)) {
		mutual.information[i] = makemim(rbind(phases.vector[[i]], phases.vector[[i+1]]))[1,2]
	}
	par(mfrow=c(1,1))
	plot(mutual.information, cex=3, cex.axis=3, cex.main=3, main="Mutual Information", pch=20)
	locator(1)
	det.start = which(mutual.information > mi.epsilon)[1]

	# Componente estocástico
	# e(t) = N(u=0, sd=0.5)
	stochastic = rowSums(decomposicao$imf[,1:(det.start-1)])

	# Componente determinístico
	# d(t) = sin(2*pi*t)
	deterministic = rowSums(decomposicao$imf[,det.start:decomposicao$nimf]) +
					decomposicao$residue

	# Portanto:
	# x(t) = d(t) + e(t)
	par(mfrow=c(3,1))
	plot(valores, pch=20, cex=2, cex.axis=3, main="Série temporal original")
	plot(stochastic, pch=20, cex=2, cex.axis=3, main="Stochastic")
	plot(deterministic, pch=20, cex=2, cex.axis=3, main="Deterministic")

	###########################################################################################
	# Modelo ARMA proposto por Box & Jenkins --> Somente se a série temporal é estacionária
	###########################################################################################
	#
	# X[t] = a[1]X[t-1] + ... + a[p]X[t-p] + e[t] + b[1]e[t-1] + ... + b[q]e[t-q]
	#
	# p = ordem do modelo Auto-Regressivo (AR: Autoregressive) 
	#			-> modelar dependências com observações passadas
	#
	# q = ordem do modelo de Média Móvel (MA: Moving Average)
	#			-> modelar as dependências com erros de modelagem passados
	#
	# X[t] = 0.05 * X[t-1] + 0.15 * X[t-2] + e[t] + 0.1 * e[t-1] + 0.5 * e[t-2] + 0.01 * e[t-3]
	#	 -----------------------------   --------------------------------------------------
	#		     v					       v
	#		Auto-regressiva				Média Móvel
	#
	###########################################################################################
	# Modelo ARIMA proposto por Box & Jenkins
	###########################################################################################
	#
	# Utiliza o modelo ARMA, porém, antes ele realiza uma decomposição da série temporal
	#	por meio de uma análise de estacionariedade
	#
	#	Estacionariedade def: é aquela cujos momentos estatísticos permanecem estáveis
	#					ao longo do tempo
	#
	#		- Divida sua série ao meio, calcule o primeiro momento estatístico
	#			para cada metade e compare esses valores pela distância Euclidiana
	#		- Quão mais próximos essa distância Euclidiana for de zero, 
	#			maior a estabilidade
	#
	#	Não-Estacionariedade def: complemento da definição acima

	ret = list()
	ret$series = valores
	ret$stochastic = stochastic
	ret$deterministic = deterministic

	return (ret)

#	locator(1)
#
#	####################################################################
#	# Modelagem baseada em Processos Estocásticos
#	####################################################################
#	# stochastic
#	require(forecast)
#	# ACF, PACF, Cálculo de resíduos (erros do modelo)
#	# Necessidade de remoção de tendências (não estacionariedades presentes)
#	model = auto.arima(stochastic)
#	stochastic.pred = as.numeric(predict(model, n.ahead = 300)$pred)
#
#	####################################################################
#	# Modelagem baseada em Sistemas Dinâmicos
#	####################################################################
#	# deterministic
#	par(mfrow=c(2,2))
#	# Estimando o time lag para aplicar no Takens' embedding theorem
#	ami = tseriesChaos::mutual(deterministic, lag.max=100)
#	v = diff(ami)
#	d = as.numeric(which(v > 0)[1])
#	cat("Time lag = ", d, "\n")
#
#	# Estimando a embedding dimension para aplicar no Takens' embedding theorem
#	fnn = tseriesChaos::false.nearest(deterministic, m=10, d=d, t=10)
#	plot(fnn)
#	m = as.numeric(which.min(fnn[1,]))
#	cat("Embedding dimension = ", m, "\n")
#
#	# Aplicação do Takens' embedding theorem
#	dataset = tseriesChaos::embedd(deterministic, m=m, d=d)
#
#	# Modelagem e Predição
#	labelId = ncol(dataset)
#	X = matrix(dataset[,1:(labelId-1)], ncol=labelId-1)
#	Y = dataset[,labelId]
#	buffer = dataset
#
#	for (new.row in (nrow(buffer)+1):(nrow(buffer)+300)) {
#		x = as.numeric(buffer[new.row - d, 2:ncol(buffer)])
#		y = dwnn(query=x, X=X, Y=Y, sigma=sigma)
#		buffer = rbind(buffer, c(x, y))
#	}
#	deterministic.pred = buffer[(nrow(X)+1):nrow(buffer),labelId]
#
#	plot(Y, t="l", cex.lab=2, cex=2, cex.axis=2, 
#	     xlab="Tempo", ylab="Valores", xlim=c(1,nrow(buffer)))
#	stochastic.pred = c(rep(NA, nrow(X)), stochastic.pred)
#	points(stochastic.pred, col=2, pch=20, cex=2)
#
#	plot(Y, t="l", cex.lab=2, cex=2, cex.axis=2, 
#	     xlab="Tempo", ylab="Valores", xlim=c(1,nrow(buffer)))
#	deterministic.pred = c(rep(NA, nrow(X)), deterministic.pred)
#	points(deterministic.pred, col=2, pch=20, cex=2)
}

test <- function() {

	#	c(1,2,0) significa:
	#		- a ordem do AR é 1
	#		- a ordem da integração é 2
	#		- a ordem do MA é 0
	#
	#	antes da modelagem preciso computar uma diferença de primeira ordem
	#	e em seguida, preciso computar outra diferença de primeira ordem
	#		pois a integração vale 2 --> não estacionária
	#
	#	series[t] = 0.7 * series[t-1] + b
	series = arima.sim(list(order = c(1,2,0), ar = 0.7), n = 1000)

	cat("Calculando estacionariedades:")
	print(estacionariedade(series, order=1))
	print(estacionariedade(series, order=2))
	print(estacionariedade(series, order=3))
	print(estacionariedade(series, order=4))

	par(mfrow=c(3,1))
	plot(series)
	diff.primeira.ordem = diff(series)
	plot(diff.primeira.ordem)

	cat("Calculando estacionariedades:")
	print(estacionariedade(diff.primeira.ordem, order=1))
	print(estacionariedade(diff.primeira.ordem, order=2))
	print(estacionariedade(diff.primeira.ordem, order=3))
	print(estacionariedade(diff.primeira.ordem, order=4))

	diff2 = diff(diff.primeira.ordem)
	plot(diff2)

	cat("Calculando estacionariedades:")
	print(estacionariedade(diff2, order=1))
	print(estacionariedade(diff2, order=2))
	print(estacionariedade(diff2, order=3))
	print(estacionariedade(diff2, order=4))

	### MODELAR USANDO ARMA -> AR e MA propostas por Box & Jenkins

	pacf(diff2, main="Quantos e quais coeficientes importam para AR")
	# Ordem do AR é 1 ---> X[t-1]

	acf(diff2, main="Quantos e quais coeficientes importam para MA")
	# Ordem do MA é 5 -> e(t) + a[1]*e(t-1) + a[2]*e(t-2)+ ... + a[5]*e(t-5)

	# Testar uma lista de modelos candidatos
	#
	# AR = 1				MA = 5
	#
	# modelagem com (0,2,0)
	# modelagem com (0,2,1)
	# modelagem com (0,2,2)
	# modelagem com (0,2,3)
	# modelagem com (0,2,4)
	# modelagem com (0,2,5)
	#
	# modelagem com (1,2,0)
	# modelagem com (1,2,1)
	# modelagem com (1,2,2)
	# modelagem com (1,2,3)
	# modelagem com (1,2,4)
	# modelagem com (1,2,5)
	#
	#
	# auto.arima
	model = list()
	model[[1]] = arima(x = series, order = c(0, 2, 0))
	model[[2]] = arima(x = series, order = c(0, 2, 1))
	model[[3]] = arima(x = series, order = c(0, 2, 2))
	model[[4]] = arima(x = series, order = c(0, 2, 3))
	model[[5]] = arima(x = series, order = c(0, 2, 4))
	model[[6]] = arima(x = series, order = c(0, 2, 5))
	model[[7]] = arima(x = series, order = c(1, 2, 0))
	model[[8]] = arima(x = series, order = c(1, 2, 1))
	model[[9]] = arima(x = series, order = c(1, 2, 2))
	model[[10]] = arima(x = series, order = c(1, 2, 3))
	model[[11]] = arima(x = series, order = c(1, 2, 4))
	model[[12]] = arima(x = series, order = c(1, 2, 5))

	model.id = -1
	akaike.information = Inf
	for (i in 1:length(model)) {
		if (model[[i]]$aic < akaike.information) {
			model.id = i
			akaike.information = model[[i]]$aic
		}
	}

	cat("Model id:")
	print(model.id)
	selected.model = model[[model.id]]

	d = c(series, predict(selected.model, n.ahead=100)$pred)
	plot(c(series, rep(NA, 100)), ylim=range(d))
	points(c(rep(NA, length(series)), predict(selected.model, n.ahead=100)$pred), col=2)

	return (series)
}
