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
	ret$ids = row.ids
	return (ret)
}

require(rgl)
require(tseriesChaos)

dataset = cbind(rnorm(mean=0,sd=0.1,n=1000), rnorm(mean=0,sd=0.1,n=1000), rep(1,1000))
dataset = rbind(dataset, cbind(embedd(sin(2*pi*seq(0,9,length=1025))+
				rnorm(mean=0, sd=0.1, n=1025), m=2, d=25), rep(2,1000)))
plot(dataset[,1:2], col=dataset[,3], pch=20, cex=2)
locator(1)

closest.ids = knn(query=c(0,0), X=dataset[,1:2], Y=dataset[,3], k=1500)$ids
points(dataset[closest.ids,1:2], col=3, cex=2, pch=20)

# Kernel polinomial de ordem 2 homogêneo
#[ dataset[,1] dataset[,2] ] = [ x1 x2 ] ---> Norma L2 ???
#	sqrt(x1^2 + x2^2) ---> Kernelização
#
#
# Phi: R^2 -> R^3
# Phi([x1 x2]) = [ x1^2  sqrt(2)*x1*x2   x2^2]

new.space = cbind(dataset[,1]^2, sqrt(2)*dataset[,1]*dataset[,2], dataset[,2]^2)
plot3d(new.space, col=dataset[,3])
