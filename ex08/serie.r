
dwnn <- function(query, X, Y, sigma) {
	E = apply(X, 1, function(row) { sqrt(sum((row - query)^2)) })
	weight = exp(-E^2 / (2*sigma^2))
	return (weight %*% Y / sum(weight))
}

x = sin(2*pi*seq(0,9,len=1000))
x_new = sample(x)

par(mfrow=c(3,2))
plot(x,pch=20,cex=2)
acf(x)
plot(x_new,pch=20,cex=2)
acf(x_new)

x.phase.space = cbind(x[1:999], x[2:1000])
x_new.phase.space = cbind(x_new[1:999], x_new[2:1000])

x.buffer = x.phase.space[1:500,]
x_new.buffer = x_new.phase.space[1:500,]

sigma = 0.00025

# Modelagem e predição
for (i in (nrow(x.buffer)+1):nrow(x.phase.space)) {
	x = x.buffer[i-1,2:ncol(x.buffer)]
	y = dwnn(query=x, X=matrix(x.buffer[,1], ncol=1), 
	     	Y=matrix(x.buffer[,2], ncol=1), sigma=sigma)
	x.buffer = rbind(x.buffer, c(x, y))
}

for (i in (nrow(x_new.buffer)+1):nrow(x_new.phase.space)) {
	x = x_new.buffer[i-1,2:ncol(x_new.buffer)]
	y = dwnn(query=x, X=matrix(x_new.buffer[,1], ncol=1), 
	     	Y=matrix(x_new.buffer[,2], ncol=1), sigma=sigma)
	x_new.buffer = rbind(x_new.buffer, c(x, y))
}

plot(x.phase.space[,2],pch=20,cex=2)
lines(x.buffer[,2], col=2)

plot(x_new.phase.space[,2],pch=20,cex=2)
lines(x_new.buffer[,2], col=2)

x.Error = NULL
x_new.Error = NULL

for (sigma in seq(0.0001, 0.0005, length=15)) {

	print(sigma)
	x.buffer = x.phase.space[1:500,]
	x_new.buffer = x_new.phase.space[1:500,]

	for (i in (nrow(x.buffer)+1):nrow(x.phase.space)) {
		x = x.buffer[i-1,2:ncol(x.buffer)]
		y = dwnn(query=x, X=matrix(x.buffer[,1], ncol=1), 
			Y=matrix(x.buffer[,2], ncol=1), sigma=sigma)
		x.buffer = rbind(x.buffer, c(x, y))
	}

	Error = sqrt(sum((x.phase.space[501:nrow(x.phase.space),2] - x.buffer[501:nrow(x.phase.space),2])^2))
	x.Error = rbind(x.Error, c(sigma, Error))

	for (i in (nrow(x_new.buffer)+1):nrow(x_new.phase.space)) {
		x = x_new.buffer[i-1,2:ncol(x_new.buffer)]
		y = dwnn(query=x, X=matrix(x_new.buffer[,1], ncol=1), 
			Y=matrix(x_new.buffer[,2], ncol=1), sigma=sigma)
		x_new.buffer = rbind(x_new.buffer, c(x, y))
	}

	Error = sqrt(sum((x_new.phase.space[501:nrow(x_new.phase.space),2] - x_new.buffer[501:nrow(x_new.phase.space),2])^2))
	x_new.Error = rbind(x_new.Error, c(sigma, Error))
}

par(mfrow=c(1,1))
x.ids = which(!is.nan(x.Error[,2]))
x_new.ids = which(!is.nan(x_new.Error[,2]))
plot(x.Error, ylim=range(c(x.Error[x.ids,2], x_new.Error[x_new.ids,2])), t="l")
lines(x_new.Error, col=2)
