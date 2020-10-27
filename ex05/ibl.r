
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

test.sin <- function() {
	series = sin(2*pi*seq(0,9,len=1000))
	plot(series)

	require(rgl)
	require(tseriesChaos)
	# expectativa de obter um espaço mais representativo para realizar a regressão
	# x(t)    x(t+2)    x(t+4)    x(t+6) ---> reconstrução espacial (kernel)
	#
	########################################################
	# Teorema 1932 por Whitney (Whitney's embedding theorem)
	########################################################
	# m=5
	# 2m = 10
	# x(t) x(t+1) x(t+2) x(t+3) ... x(t+9)
	#
	#####################################################
	# Teorema 1981 por Takens (Takens' embedding theorem)
	#####################################################
	#
	# X = {x(1), x(2), ..., x(t), x(t+1), ..., x(n)}
	#
	#
	# Phi(x(t), m, d) = (x(t), x(t+d), x(t+2d), ..., x(t+(m-1)*d))
	#
	# Kennel -> estimar a dimensão embutida m
	# Swinney & Fraser -> estimar a dimensão de sepração ou time lag ou time delay d
	# Lucas de Carvalho Pagliosa, Rodrigo Fernandes de Mello:
	# Applying a kernel function on time-dependent data to provide 
	#	supervised-learning guarantees. Expert Syst. Appl. 71: 216-229 (2017)

	dataset = embedd(series, m=3, d=25)
	plot3d(dataset, cex.lab=2, cex=2, pch=20, 
	       xlab="x(t)", ylab="x(t+25)", zlab="x(t+50)")

	X = dataset[,1:2]
	Y = dataset[,3]

	#
	# x(t)   x(t+25)  x(t+50)
	# x(t+1) x(t+26)  x(t+51)
	# x(t+2) x(t+27)  x(t+52)
	# ...
	# x(t+876) x(t+901) x(t+926)
	# ...
	# x(t+900) x(t+925)  x(t+950)
	# x(t+901) x(t+926) x(t+951)

	window=250
	buffer = dataset
	for (i in (nrow(buffer)+1):(nrow(buffer)+window)) {
		x = buffer[i-25,2:ncol(buffer)]
		y = dwnn(query=x, X=X, Y=Y, sigma=0.2)
		buffer = rbind(buffer, c(x, y))
	}

	print(buffer)

	plot(dataset[,3], xlim=c(1, nrow(buffer)), col=1, pch=20)
	pred = c(rep(NA, nrow(dataset)), buffer[(nrow(X)+1):nrow(buffer),3])
	points(pred, col=2, pch=20)

}

