dwnn <- function(query, X, Y, sigma) {
	E = apply(X, 1, function(row) { sqrt(sum((row - query)^2)) })
	weight = exp(-E^2 / (2*sigma^2))
	return (weight %*% Y / sum(weight))
}

# datetime, price(view up)
dataset = read.csv("petr4.csv")
series = dataset[,2]

# Selecionando até 3 dias de dados passados
start.series = length(series) - 60*7*3
end.series = length(series)
series = series[start.series:end.series]

require(EMD)
decomposition = emd(series)

# Plotando as IMFs (Intrinsic Mode Functions)
par(mfrow=c(3,ceiling((decomposition$nimf + 2) / 3)))
plot(series, pch=20, cex=2, main="Série original", cex.axis=2, cex.main=2)
for (i in 1:decomposition$nimf) {
	plot(decomposition$imf[,i], pch=20, cex=2, main=paste("IMF", i), cex.axis=2, cex.main=2)
}
plot(decomposition$residue, pch=20, cex=2, main="Resíduo", cex.axis=2, cex.main=2)
locator(1)

# Plotando os coeficientes complexos de Fourier de cada IMF
par(mfrow=c(3,ceiling((decomposition$nimf + 2) / 3)))
plot(series, pch=20, cex=2, main="Série original", cex.axis=2, cex.main=2)
for (i in 1:decomposition$nimf) {
	coeff = fft(decomposition$imf[,i])
	plot(coeff, asp=1, pch=20, cex=2, main=paste("IMF", i), cex.axis=2, cex.main=2)
}
coeff = fft(decomposition$residue)
plot(coeff, asp=1, pch=20, cex=2, main="Resíduo", cex.axis=2, cex.main=2)
locator(1)

# Plotando as fases dos coeficientes complexos de Fourier das IMFs
par(mfrow=c(3,ceiling((decomposition$nimf + 2) / 3)))
plot(series, pch=20, cex=2, main="Série original", cex.axis=2, cex.main=2)
phases.vector = list()
for (i in 1:decomposition$nimf) {
	coeff = fft(decomposition$imf[,i])
	phases = atan(Im(coeff) / Re(coeff))
	plot(phases, pch=20, cex=2, main=paste("IMF", i), cex.axis=2, cex.main=2)
	phases.vector[[i]] = phases
}
coeff = fft(decomposition$residue)
phases = atan(Im(coeff) / Re(coeff))
plot(phases, pch=20, cex=2, main="Resíduo", cex.axis=2, cex.main=2)
phases.vector[[i+1]] = phases
locator(1)

# Calculando a informação mútua entre pares sucessivos de IMFs (fases)
require(c3net)
mutual.information = rep(0, length(phases.vector)-1)
for (i in 1:(length(phases.vector)-1)) {
	mutual.information[i] = makemim(rbind(phases.vector[[i]], phases.vector[[i+1]]))[1,2]
}

cutoff = which(mutual.information > 0.01)[1]
stochastic = rowSums(decomposition$imf[,1:(cutoff-1)])
deterministic = rowSums(decomposition$imf[,cutoff:decomposition$nimf]) + decomposition$residue

########################################
# Estudo sobre o componente estocástico
########################################
par(mfrow=c(3,1))
plot(stochastic, cex=2, cex.axis=2, pch=20)
pacf(stochastic) # AR -> 1, 2, 3, ..., 15
acf(stochastic)  # MA -> 1, 2, 3, ..., 15

# Avaliar a modelagem do componente estocástico utilizando AR (0, 1, ..., 15)
#	e MA (0, 1, ..., 15)
counter = 1
model = list()
for (AR in 0:5) { # 0:15
	for (MA in 0:5) { # 0:15
		cat("Counter = ", counter, "\n")
		model[[counter]] = arima(x = stochastic, order = c(AR, 0, MA))
		counter = counter + 1
	}
}
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
stochastic.pred = predict(selected.model, n.ahead=300)$pred

###########################################
# Estudo sobre o componente determinístico
###########################################
par(mfrow=c(3,1))
plot(deterministic, cex=2, cex.axis=2, pch=20)
require(tseriesChaos)
mi = mutual(deterministic, plot=F, lag.max=5000)
plot(mi)
v = diff(mi)
d = as.numeric(which(v > 0)[1]) # d = time lag para o Takens' embedding theorem
cat("Time lag  = ", d, "\n")

fnn = false.nearest(deterministic, d=d, m=10, t=10)
plot(fnn)
m = as.numeric(which.min(fnn[1,]))
cat("Embedding dimension = ", m, "\n")

phase.space = embedd(deterministic, m=m, d=d)

#sigma = as.numeric(quantile(dist(phase.space), 0.0001)) # Sèrra: cover song identification
sigma = 0.25

# Modelagem e Predição
labelId = ncol(phase.space)
X = matrix(phase.space[,1:(labelId-1)], ncol=labelId-1)
Y = phase.space[,labelId]
buffer = phase.space

for (new.row in (nrow(buffer)+1):(nrow(buffer)+300)) {
	x = as.numeric(buffer[new.row - d, 2:ncol(buffer)])
	y = dwnn(query=x, X=X, Y=Y, sigma=sigma)
	buffer = rbind(buffer, c(x, y))
}
deterministic.pred = buffer[(nrow(X)+1):nrow(buffer),labelId]

# Plotando...
par(mfrow=c(2,1))
plot(stochastic, t="l", cex.lab=2, 
     cex=2, cex.axis=2, xlab="Tempo", 
     	ylab="Valores", xlim=c(1,length(stochastic)+300))
stochastic.pred = c(rep(NA, length(stochastic)), stochastic.pred)
points(stochastic.pred, col=2, pch=20, cex=2)

plot(Y, t="l", cex.lab=2, cex=2, cex.axis=2, xlab="Tempo", ylab="Valores", xlim=c(1,nrow(buffer)))
deterministic.pred = c(rep(NA, nrow(X)), deterministic.pred)
points(deterministic.pred, col=2, pch=20, cex=2)

