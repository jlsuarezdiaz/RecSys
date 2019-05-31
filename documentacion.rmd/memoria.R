# Juan Luis Suárez Díaz, DNI: 77148642-H
# E-mail: jlsuarezdiaz@correo.ugr.es
# Ejercicio de trabajo autónomo. Series temporales. Curso 2018-2019.

## ---- message=F, warning=F-----------------------------------------------
CODIGO.ESTACION <- "6258X"
datos.estacion <- read.csv(paste0("DatosEstaciones - 2018-02/",CODIGO.ESTACION,".csv"), 
                           sep = ";")


## ---- message=F, warning=F-----------------------------------------------
tiempo.fechas <- as.Date(datos.estacion$Fecha) # Fechas
tiempo.seq <- 1:length(tiempo.fechas) # Índices correspondientes a las fechas
tmax <- datos.estacion$Tmax          # Datos de la temperatura máxima
# Temperatura como objeto serie temporal (asumiendo una estacionalidad de 365 días)
tmax.ts <- ts(tmax, frequency = 365) 


## ---- message=F, warning=F-----------------------------------------------
plot(tmax ~ tiempo.fechas, type="l")


## ---- message=F, warning=F-----------------------------------------------
month <- function(d) format(d, "%b") # Función para recuperar el mes de una fecha
year <- function(d) format(d, "%Y")  # Función para recuperar el año de una fecha
day <- function(d) format(d, "%d")   # Función para recuperar el día de una fecha
table(year(tiempo.fechas)) # Número de datos por año


## ---- message=F, warning=F-----------------------------------------------
library(tidyr)
library(dplyr)
# Creamos un data.frame con la serie y las fechas
datos.incompletos <- data.frame(tiempo.fechas=tiempo.fechas, tmax = tmax)
# Completamos las fechas con todas las posibles fechas del período 
# en el que se han tomado medidas.
datos.completados <- datos.incompletos %>% 
  mutate(completed.dates = as.Date(tiempo.fechas)) %>%
  complete(completed.dates = seq.Date(from = min(tiempo.fechas), to = max(tiempo.fechas),
                                      by="day"))
  # Las fechas completadas que no estaban ya presentes añaden un valor
  # perdido a la temperatura máxima.
# Recuperamos las fechas completas y las temperaturas asociadas 
# a todas las fechas (con o sin NAs)
tiempo.fechas <- datos.completados$completed.dates
tmax <- datos.completados$tmax
tiempo.seq <- 1:length(tiempo.fechas)

table(year(tiempo.fechas))
length(tiempo.fechas)
length(tmax)


## ---- message=F, warning=F-----------------------------------------------
tiempo.nas.seq <- which(is.na(tmax.ts))
tiempo.nas.fechas <- tiempo.fechas[tiempo.nas.seq]


## ---- message=F, warning=F-----------------------------------------------

barplot(table(year(tiempo.nas.fechas)))  # Valores perdidos por año
barplot(table(month(tiempo.nas.fechas))) # Valores perdidos por mes
# La serie junto con los valores perdidos
plot(tiempo.nas.fechas, rep(0, length(tiempo.nas.fechas)), 
     ylim = c(-0.1, 40), pch=16, col="red") 
# Posiciones de los valores perdidos
points(tiempo.fechas, tmax, pch=16, col="blue")




## ---- message=F, warning=F-----------------------------------------------
library(imputeTS)
imputation.interp <- na.interpolation(tmax) # Imputación mediante interpolación lineal.
plot(tiempo.fechas, tmax, pch=16, col="blue")
points(tiempo.nas.fechas, imputation.interp[tiempo.nas.seq], pch=16, col="green")
plot(tiempo.fechas, imputation.interp, type="l")
points(tiempo.nas.fechas, imputation.interp[tiempo.nas.seq], pch=16, col="green")


## ---- message=F, warning=F-----------------------------------------------
imputation.ma <- na.ma(tmax) # Imputación mediante medias móviles.
plot(tiempo.fechas, tmax, pch=16, col="blue")
points(tiempo.nas.fechas, imputation.ma[tiempo.nas.seq], pch=16, col="green")
plot(tiempo.fechas, imputation.ma, type="l")
points(tiempo.nas.fechas, imputation.ma[tiempo.nas.seq], pch=16, col="green")


## ---- message=F, warning=F-----------------------------------------------
# Imputación LOCF (se imputa utilizando el valor más reciente no perdido)
imputation.locf <- na.locf(tmax, option = "locf") 
plot(tiempo.fechas, tmax, pch=16, col="blue")
points(tiempo.nas.fechas, imputation.locf[tiempo.nas.seq], pch=16, col="green")
plot(tiempo.fechas, imputation.locf, type="l")
points(tiempo.nas.fechas, imputation.locf[tiempo.nas.seq], pch=16, col="green")


## ---- message=F, warning=F-----------------------------------------------
tmax <- imputation.ma
tmax.ts <- ts(tmax, frequency = 365)


## ---- message=F, warning=F-----------------------------------------------
plot(tiempo.fechas, tmax, type="l")


## ---- message=F, warning=F-----------------------------------------------
plot(tiempo.fechas, tmax, type="l")
lines(tiempo.fechas, cummax(tmax), col="red")
lines(tiempo.fechas, cummin(tmax), col="blue")


## ---- include=F, eval=F--------------------------------------------------
## # ampl <- cummax(tmax) - cummin(tmax)
## # ampl.seq <- tiempo.seq[ampl > 24]
## # ampl <- ampl[ampl > 24]
## # ampl.model <- lm(ampl ~ ampl.seq)
## # ampl.preds <- ampl.model$coefficients[1] + tiempo.seq * ampl.model$coefficients[2]
## # summary(ampl.model)
## # tmax <- (tmax-mean(tmax))/ampl.preds + mean(tmax)
## # plot(tiempo.fechas, tmax, type="l")
## # lines(tiempo.fechas, cummax(tmax), col="red")
## # lines(tiempo.fechas, cummin(tmax), col="blue")


## ---- message=F, warning=F-----------------------------------------------
# Transformación logarítmica
plot(tiempo.fechas, log(tmax), type="l")
lines(tiempo.fechas, cummax(log(tmax)), col="red")
lines(tiempo.fechas, cummin(log(tmax)), col="blue")


## ---- message=F, warning=F-----------------------------------------------
# Raíz cuadrada
plot(tiempo.fechas, sqrt(tmax), type="l")
lines(tiempo.fechas, cummax(sqrt(tmax)), col="red")
lines(tiempo.fechas, cummin(sqrt(tmax)), col="blue")


## ---- message=F, warning=F-----------------------------------------------
# Descomposición por medias móviles
plot(decompose(tmax.ts))


## ---- message=F, warning=F-----------------------------------------------
# Descomposición STL
plot(stl(tmax.ts, s.window = "periodic"))


## ---- message=F, warning=F-----------------------------------------------
# Modelo de regresión lineal para la serie
summary(lm(tmax ~ tiempo.seq))


## ---- message=F, warning=F-----------------------------------------------
# Modelo de regresión lineal para la serie a partir de 2014
summary(lm(tmax[year(tiempo.fechas) >= "2014"] ~ tiempo.seq[year(tiempo.fechas) >= "2014"]))


## ---- message=F, warning=F-----------------------------------------------
Ntest <- 7 # Número de valores a predecir, y de valores para test

# Conjunto de entrenamiento
tiempo.seq.tr <- tiempo.seq[1:(length(tiempo.seq) - Ntest)]
tmax.tr <- tmax[tiempo.seq.tr]
tiempo.fechas.tr <- tiempo.fechas[tiempo.seq.tr]

# Conjunto de test
tiempo.seq.ts <- tiempo.seq[(length(tiempo.seq.tr)+1):length(tiempo.seq)]
tmax.ts <- tmax[tiempo.seq.ts]
tiempo.fechas.ts <- tiempo.fechas[tiempo.seq.ts]

# Dibujamos ambos conjuntos (desde 2016 para visualizar mejor el test)
plot(tiempo.fechas.tr, tmax.tr, type="l",
     xlim=c(min(tiempo.fechas[year(tiempo.fechas)>="2016"]), max(tiempo.fechas)))
lines(tiempo.fechas.ts, tmax.ts, col="red")


## ---- message=F, warning=F-----------------------------------------------
# Cálculo de la estacionalidad.
# Para cada i entre 1 y 365, nos quedamos con el dato de esa posición para cada
# período anual de 365 días, empezando desde el inicio de la serie. Hacemos
# la media para cada i, obteniendo la estacionalidad resultante.
estacionalidad <- sapply(1:365,function(i) mean(tmax.tr[seq(i, length(tmax.tr), by=365)]))
plot.ts(estacionalidad)


## ---- message=F, warning=F-----------------------------------------------
# Función que ajusta un polinomio trigonométrico a datos periódicos.
# - series: La serie periódica a ajustar.
# - time: Intervalo de tiempo de los datos de la serie.
# - degree: Grado del polinomio trigonométrico.
# - period: Período de la serie.
fit.trigonometric.polynomial <- function(series, time, degree, period){
  coefficients <- c()
  data <- data.frame(series)
  
  for(i in 1:degree){
    coefficients <- c(coefficients, paste0("s",i), paste0("c",i))
    data[[paste0("s",i)]] <- sin(2*pi*i*time/period)
    data[[paste0("c",i)]] <- cos(2*pi*i*time/period)
  }
  formula <- as.formula(paste("series ~", paste(coefficients,collapse="+")))
  model <- lm(formula, data)
  model
}

# Función que predice un modelo de polinomio trigonométrico ajustado para nuevos datos.
# model: Modelo entrenado con fit.trigonometric.polynomial
# time: Tiempos a predecir.
# period: Periodo de la serie.
predict.trigonometric.polynomial <- function(model, time, period){
  coefs <- model$coefficients
  degree <- (length(coefs)-1)/2
  pred <- rep(0, length(time))
  pred <- pred + coefs[1]
  for(i in 1:degree){
    coef.s <- coefs[2*i]
    coef.c <- coefs[2*i+1]
    pred <- pred + coef.s * sin(2*pi*i*time/period) + coef.c * cos(2*pi*i*time/period)
  }
  pred
}

# Ajustamos un polinomio trigonométrico de grado 1
model.tri.1 <- fit.trigonometric.polynomial(estacionalidad, 1:365, 1, 365)
preds.tri.1 <- predict.trigonometric.polynomial(model.tri.1, 1:365, 365)

# Ajustamos un polinomio trigonométrico de grado 2
model.tri.2 <- fit.trigonometric.polynomial(estacionalidad, 1:365, 2, 365)
preds.tri.2 <- predict.trigonometric.polynomial(model.tri.2, 1:365, 365)

# Ajustamos un polinomio trigonométrico de grado 3
model.tri.3 <- fit.trigonometric.polynomial(estacionalidad, 1:365, 3, 365)
preds.tri.3 <- predict.trigonometric.polynomial(model.tri.3, 1:365, 365)

# Ajustamos un polinomio trigonométrico de grado 4
model.tri.4 <- fit.trigonometric.polynomial(estacionalidad, 1:365, 4, 365)
preds.tri.4 <- predict.trigonometric.polynomial(model.tri.4, 1:365, 365)

# Mostramos los ajustes obtenidos
library(scales)
plot(estacionalidad, type="l", col=alpha("cyan", 0.4))
lines(preds.tri.1, col="red")
lines(preds.tri.2, col="blue")
lines(preds.tri.3, col="orange")


## ---- message=F, warning=F-----------------------------------------------
# Predicciones de cada modelo sobre train
estacionalidad.tr.1 <- predict.trigonometric.polynomial(model.tri.1, tiempo.seq.tr, 365)
estacionalidad.tr.2 <- predict.trigonometric.polynomial(model.tri.2, tiempo.seq.tr, 365)
estacionalidad.tr.3 <- predict.trigonometric.polynomial(model.tri.3, tiempo.seq.tr, 365)
# Predicciones de cada modelo sobre test
estacionalidad.ts.1 <- predict.trigonometric.polynomial(model.tri.1, tiempo.seq.ts, 365)
estacionalidad.ts.2 <- predict.trigonometric.polynomial(model.tri.2, tiempo.seq.ts, 365)
estacionalidad.ts.3 <- predict.trigonometric.polynomial(model.tri.3, tiempo.seq.ts, 365)

plot(tmax.tr,  col=alpha("cyan",0.4), type="l", pch=16)
lines(tiempo.seq.ts, tmax.ts, col="green")
lines(estacionalidad.tr.1, type="l", col="red")
lines(estacionalidad.tr.2, type="l", col="blue")
lines(estacionalidad.tr.3, type="l", col="orange")
lines(tiempo.seq.ts, estacionalidad.ts.1, type="l", col="red")
lines(tiempo.seq.ts, estacionalidad.ts.2, type="l", col="blue")
lines(tiempo.seq.ts, estacionalidad.ts.3, type="l", col="orange")



## ---- message=F, warning=F-----------------------------------------------
# Resumen del modelo de grado 1
summary(model.tri.1)
# Resumen del modelo de grado 2
summary(model.tri.2)
# Resumen del modelo de grado 3
summary(model.tri.3)
# RMSE sobre test
cat("RMSE grado 1:", sqrt(mean((estacionalidad.ts.1-tmax.ts)^2)),"\n")
cat("RMSE grado 2:", sqrt(mean((estacionalidad.ts.2-tmax.ts)^2)),"\n")
cat("RMSE grado 3:", sqrt(mean((estacionalidad.ts.3-tmax.ts)^2)),"\n")



## ---- message=F, warning=F-----------------------------------------------
# Estacionalidad en train y test
estacionalidad.tr <- estacionalidad.tr.1
estacionalidad.ts <- estacionalidad.ts.1
# Eliminamos estacionalidad
tmax.tr.no.est <- tmax.tr - estacionalidad.tr
tmax.ts.no.est <- tmax.ts - estacionalidad.ts
# Dibujamos la serie sin estacionalidad
plot(tiempo.fechas.tr, tmax.tr.no.est, type="l")
lines(tiempo.fechas.ts, tmax.ts.no.est, col="blue")


## ---- message=F, warning=F-----------------------------------------------
# ACF
acf.values <- acf(tmax.tr.no.est)


## ---- message=F, warning=F-----------------------------------------------
# Test de Dickey-Fuller aumentado
library(tseries)
adf.test(tmax.tr.no.est)


## ---- message=F, warning=F-----------------------------------------------
# PACF
pacf.values <- pacf(tmax.tr.no.est)


## ---- message=F, warning=F-----------------------------------------------
# Estimamos los parámetros del modelo arima usando el
# RMSE sobre el conjunto de test.
# Matriz con los parámetros (p,d,q) a evaluar.
parametros.arima <- matrix(c(0,0,5,
                             0,0,9,
                             0,0,13,
                             0,0,17,
                             0,0,20,
                             0,0,23,
                             1,0,0
                             ), ncol = 3, byrow = T)
modelos.arima <- list()
rmse <- data.frame()
# Entrenamiento y test de los distintos modelos
for(i in 1:nrow(parametros.arima)){
  val <- parametros.arima[i,]
  # Ajuste
  modelos.arima[[toString(val)]] <- arima(tmax.tr.no.est, order = val)
  # Cálculo de predicciones y errores
  pred.arima.tr <- tmax.tr.no.est - modelos.arima[[toString(val)]]$residuals
  pred.arima.ts <- predict(modelos.arima[[toString(val)]], n.ahead = Ntest)$pred
  rmse[toString(val), "train"] <- sqrt(mean((modelos.arima[[toString(val)]]$residuals)^2))
  rmse[toString(val), "test"] <- sqrt(mean((tmax.ts.no.est - pred.arima.ts)^2))
}

# Tabla de resultados
print(rmse)
library(reshape2)
library(ggplot2)
# Mostramos gráficamente los resultados
nrmse <- data.frame(name=rownames(rmse), melt(rmse))
ggplot(nrmse, aes(x=name, y=value, fill=variable)) + 
  geom_bar(stat = "identity", position = "dodge")




## ------------------------------------------------------------------------
# Resultados del mejor modelo
modelo.arima <- modelos.arima[["0, 0, 23"]]
pred.arima.tr <- tmax.tr.no.est - modelo.arima$residuals
pred.arima.ts <- predict(modelo.arima, n.ahead = Ntest)$pred
# Dibujamos las predicciones del mejor modelo
plot(tiempo.fechas.tr, tmax.tr.no.est, type="l", 
     xlim=c(max(tiempo.fechas)-50, max(tiempo.fechas)))
lines(tiempo.fechas.tr, pred.arima.tr, col="blue")
lines(tiempo.fechas.ts, tmax.ts.no.est, col="red")
lines(tiempo.fechas.ts, pred.arima.ts, col="green")


## ------------------------------------------------------------------------
# Test de Box-Pierce
boxtest <- Box.test(modelo.arima$residuals)
boxtest


## ------------------------------------------------------------------------
# Tests de normalidad
jarque.bera.test(modelo.arima$residuals)
shapiro.test(modelo.arima$residuals)
# Histograma de los datos
hist(modelo.arima$residuals, prob=T)
lines(density(modelo.arima$residuals))
# QQplot de los datos
qqnorm(modelo.arima$residuals)
qqline(modelo.arima$residuals)


## ---- message=F, warning=F-----------------------------------------------
# Añadimos la estacionalidad a las predicciones
pred.tr <- pred.arima.tr + estacionalidad.tr
pred.ts <- pred.arima.ts + estacionalidad.ts

# Serie completa
plot(tiempo.fechas.tr, tmax.tr, type="l")
lines(tiempo.fechas.tr, pred.tr, col="blue")
lines(tiempo.fechas.ts, tmax.ts, col="red")
lines(tiempo.fechas.ts, pred.ts, col="green")

# Ampliación en test
plot(tiempo.fechas.tr, tmax.tr, type="l", xlim=c(max(tiempo.fechas)-250, max(tiempo.fechas)))
lines(tiempo.fechas.tr, pred.tr, col="blue")
lines(tiempo.fechas.ts, tmax.ts, col="red")
lines(tiempo.fechas.ts, pred.ts, col="green")

cat("Error medio: ", mean(abs(pred.ts - tmax.ts)))


## ---- message=F, warning=F-----------------------------------------------
# Calculamos la estacionalidad.
estacionalidad.all <- sapply(1:365,function(i) mean(tmax[seq(i, length(tmax), by=365)]))
# Ajustamos el modelo trigonométrico para la estacionalidad
model.est.all <- fit.trigonometric.polynomial(estacionalidad, 1:365, 1, 365)
# Eliminamos la estacionalidad
estacionalidad.all <- predict.trigonometric.polynomial(model.est.all, tiempo.seq, 365)
tmax.no.est <- tmax - estacionalidad.all
# Ajustamos modelo de medias móviles a la serie sin estacionalidad
modelo.arima <- arima(tmax.no.est, order = c(0,0,23))
# Valores ajustados para la serie estacionaria
fit.arima.all <- tmax.no.est - modelo.arima$residuals
# Predicciones para la próxima semana de la serie estacionaria
preds.arima <- predict(modelo.arima, n.ahead = Ntest)$pred
# Reconstruimos el ajuste para la serie original
tmax.fit <- fit.arima.all + estacionalidad.all
# Reconstruimos las predicciones para la serie original
preds.estacionalidad <- predict.trigonometric.polynomial(model.est.all,
                                                         max(tiempo.seq)+1:Ntest, 365)
tmax.preds <- preds.estacionalidad + preds.arima
# Mostramos los resultados de la predicción

# Dibujamos la serie completa
plot(tiempo.fechas, tmax, type="l")
lines(tiempo.fechas, tmax.fit, col="blue")
lines(max(tiempo.fechas) + 1:Ntest, tmax.preds, col="red")

# Ampliación en las predicciones
plot(tiempo.fechas, tmax, type="l", xlim=c(max(tiempo.fechas)-50, max(tiempo.fechas)+Ntest))
lines(tiempo.fechas, tmax.fit, col="blue")
lines(max(tiempo.fechas) + 1:Ntest, tmax.preds, col="red")
points(max(tiempo.fechas) + 1:Ntest, tmax.preds, col="red", pch=16)
text(max(tiempo.fechas) + 1:Ntest, tmax.preds, 
     labels=sprintf("%4.1f", tmax.preds), col="red", cex=0.5, pos=1)


## ---- message=F, warning=F-----------------------------------------------
CODIGO.ESTACION <- "6258X"
datos.estacion <- read.csv(paste0("DatosEstaciones - 2018-02/",CODIGO.ESTACION,".csv"),
                           sep = ";")


## ---- message=F, warning=F-----------------------------------------------
tiempo.fechas <- as.Date(datos.estacion$Fecha) # Fechas
tiempo.seq <- 1:length(tiempo.fechas) # Índices correspondientes a las fechas
tmax <- datos.estacion$Tmax          # Datos de la temperatura máxima
# Temperatura como objeto serie temporal (asumiendo una estacionalidad de 365 días)
tmax.ts <- ts(tmax, frequency = 365) 


## ---- message=F, warning=F-----------------------------------------------
library(zoo)
# Valores perdidos por par (mes, año)
table(as.yearmon(tiempo.nas.fechas))
plot(table(as.yearmon(tiempo.nas.fechas)))


## ---- message=F, warning=F-----------------------------------------------
# Datos de los meses con mayor cantidad de valores perdidos
meses.con.muchos.nas <- c("mar 2014", "nov 2014", "mar 2015", 
                          "oct 2015", "nov 2015", "dic 2015")
plot(tiempo.fechas[as.yearmon(tiempo.fechas) %in% as.yearmon(meses.con.muchos.nas)],
              tmax[as.yearmon(tiempo.fechas) %in% as.yearmon(meses.con.muchos.nas)],
     xlab = "Tiempo", ylab = "Temperatura", pch=16)


## ---- message=F, warning=F-----------------------------------------------
# Dataframe con los datos diarios y temperaturas
datos.diarios <- data.frame(tiempo.mes=as.yearmon(tiempo.fechas), tmax=tmax)
datos.mensuales <- datos.diarios %>%
    group_by(tiempo.mes) %>% # Agrupamos por mes y año
    summarise(tmax.mes=max(tmax, na.rm = T)) # Nos quedamos con la temperatura máxima
tmax.mes <- datos.mensuales$tmax.mes # Temperatura máxima mensual
tiempo.mes <- datos.mensuales$tiempo.mes # Fechas a nivel de mes (mes y año)
tiempo.seq.mes <- 1:length(tmax.mes) # Tiempo en valores enteros
# Objeto ts con la temperatura (en este caso la estacionalidad es en periodos de 12 meses)
tmax.mes.ts <- ts(tmax.mes, frequency = 12) 
# Dibujamos la serie mensual
plot(tiempo.mes, tmax.mes, type="l")


## ---- message=F, warning=F-----------------------------------------------
# Evolución de las temperaturas para cada mes
ggplot(datos.mensuales, aes(x=tiempo.mes, y=tmax.mes, 
                            col=as.factor(month(tiempo.mes)))) + geom_line()


## ---- message=F, warning=F-----------------------------------------------
# Evolución de las temperaturas por mes tras transformación logarítmica
tmax.log <- log(tmax.mes)
ggplot(datos.mensuales, aes(x=tiempo.mes, y=log(tmax.mes),
                            col=as.factor(month(tiempo.mes)))) + geom_line()
plot(tmax.log, type="l")


## ---- message=F, warning=F-----------------------------------------------
# Serie logarítmica convertida a objeto ts.
tmax.log.ts <- ts(tmax.log, frequency = 12)
# Descomposición por medias móviles
plot(decompose(tmax.log.ts))
# Descomposición STL
plot(stl(tmax.log.ts, s.window = "periodic"))


## ---- message=F, warning=F-----------------------------------------------
# Longitud del conjunto test
Ntest <- 6
# Tiempo (entero) para train y test
tiempo.seq.mes.tra <- 1:(length(tiempo.seq.mes)-Ntest)
tiempo.seq.mes.tst <- (length(tiempo.seq.mes.tra)+1):length(tiempo.seq.mes)
# Temperaturas mensuales para train y test
tmax.mes.tra <- tmax.log[tiempo.seq.mes.tra]
tmax.mes.tst <- tmax.log[tiempo.seq.mes.tst]
# Tiempo (mes y año) para train y test
tiempo.mes.tra <- tiempo.mes[tiempo.seq.mes.tra]
tiempo.mes.tst <- tiempo.mes[tiempo.seq.mes.tst]
# Dibujamos los conjuntos
plot(tiempo.mes.tra, tmax.mes.tra, type="l", xlim=c(min(tiempo.mes), max(tiempo.mes)))
lines(tiempo.mes.tst, tmax.mes.tst, col="red")


## ---- message=F, warning=F-----------------------------------------------
# Modelo lineal de tendencia
trend.model <- lm(tmax.mes.tra ~ tiempo.seq.mes.tra)
summary(trend.model)


## ---- message=F, warning=F-----------------------------------------------
# Cálculo de la estacionalidad.
# Para cada i entre 1 y 12, nos quedamos con el dato de esa posición para cada
# período anual de 12 meses, empezando desde el inicio de la serie. Hacemos
# la media para cada i, obteniendo la estacionalidad resultante.
estacionalidad <- sapply(1:12,function(i) mean(tmax.mes.tra[seq(i, length(tmax.mes.tra), by=12)]))
plot.ts(estacionalidad)


## ---- message=F, warning=F-----------------------------------------------
# Repetimos la curva de estacionalidad durante 5 años.
estacionalidad.rep <- rep(estacionalidad, 5)
# Componente estacional de train
estacionalidad.tra <- estacionalidad.rep[tiempo.seq.mes.tra]
# Componente estacional de test
estacionalidad.tst <- estacionalidad.rep[tiempo.seq.mes.tst]
# Eliminamos la estacionalidad
tmax.mes.tra.no.est <- tmax.mes.tra - estacionalidad.tra
tmax.mes.tst.no.est <- tmax.mes.tst - estacionalidad.tst
# Dibujamos la serie sin estacionalidad
plot(tiempo.mes.tra, tmax.mes.tra.no.est, type="l", xlim=c(min(tiempo.mes), max(tiempo.mes)))
lines(tiempo.mes.tst, tmax.mes.tst.no.est, col="red")


## ---- message=F, warning=F-----------------------------------------------
# Test de Dickey-Fuller aumentado
adf.test(tmax.mes.tra.no.est)


## ---- message=F, warning=F-----------------------------------------------
# Autocorrelaciones
acf(tmax.mes.tra.no.est)


## ---- message=F, warning=F-----------------------------------------------
# Autocorrelaciones parciales
pacf(tmax.mes.tra.no.est)


## ---- message=F, warning=F-----------------------------------------------
# Modelo de ruido blanco
modelo.arima.0 <- arima(tmax.mes.tra.no.est, order = c(0,0,0))
# Dibujamos los resultados
plot(tiempo.mes.tra, tmax.mes.tra.no.est, type="l", xlim=c(min(tiempo.mes), max(tiempo.mes)))
lines(tiempo.mes.tra, tmax.mes.tra.no.est - modelo.arima.0$residuals, col="blue")
lines(tiempo.mes.tst, tmax.mes.tst.no.est, col="red")
lines(tiempo.mes.tst, predict(modelo.arima.0, n.ahead = Ntest)$pred, col="green")


## ---- message=F, warning=F-----------------------------------------------
# Test de Box-Pierce para el modelo de ruido blanco
Box.test(modelo.arima.0$residuals)


## ---- message=F, warning=F-----------------------------------------------
# Modelo ARIMA(0,0,1)
modelo.arima.1 <- arima(tmax.mes.tra.no.est, order = c(0,0,1))
# Dibujamos los resultados
plot(tiempo.mes.tra, tmax.mes.tra.no.est, type="l", xlim=c(min(tiempo.mes), max(tiempo.mes)))
lines(tiempo.mes.tra, tmax.mes.tra.no.est - modelo.arima.1$residuals, col="blue")
lines(tiempo.mes.tst, tmax.mes.tst.no.est, col="red")
lines(tiempo.mes.tst, predict(modelo.arima.1, n.ahead = Ntest)$pred, col="green")


## ---- message=F, warning=F-----------------------------------------------
# Tests de aleatoriedad y normalidad
Box.test(modelo.arima.1$residuals)
shapiro.test(modelo.arima.1$residuals)


## ---- message=F, warning=F-----------------------------------------------
# Modelo ARIMA(0,0,12)
modelo.arima.12 <- arima(tmax.mes.tra.no.est, order = c(0,0,12))
# Dibujamos los resultados
plot(tiempo.mes.tra, tmax.mes.tra.no.est, type="l", xlim=c(min(tiempo.mes), max(tiempo.mes)))
lines(tiempo.mes.tra, tmax.mes.tra.no.est - modelo.arima.12$residuals, col="blue")
lines(tiempo.mes.tst, tmax.mes.tst.no.est, col="red")
lines(tiempo.mes.tst, predict(modelo.arima.12, n.ahead = Ntest)$pred, col="green")


## ---- message=F, warning=F-----------------------------------------------
# Tests de aleatoriedad y normalidad
Box.test(modelo.arima.12$residuals)
shapiro.test(modelo.arima.12$residuals)


## ---- message=F, warning=F-----------------------------------------------
# AIC para los 3 modelos
AIC(modelo.arima.0, modelo.arima.1, modelo.arima.12)


## ---- message=F, warning=F-----------------------------------------------
# Mejor modelo
modelo.arima <- modelo.arima.1
pred.arima.tra <- tmax.mes.tra.no.est - modelo.arima$residuals
pred.arima.tst <- predict(modelo.arima, n.ahead = Ntest)$pred


## ---- message=F, warning=F-----------------------------------------------
# Añadimos la estacionalidad
pred.est.tra <- pred.arima.tra + estacionalidad.tra
pred.est.tst <- pred.arima.tst + estacionalidad.tst
# Invertimos la transformación logarítmica en las predicciones
pred.mes.tra <- exp(pred.est.tra)
pred.mes.tst <- exp(pred.est.tst)
# Invertimos la transformación logarítmica en la serie
tmax.exp.tra <- exp(tmax.mes.tra)
tmax.exp.tst <- exp(tmax.mes.tst)
# Dibujamos la serie y predicciones obtenidas
plot(tiempo.mes.tra, tmax.exp.tra, type="l", xlim=c(min(tiempo.mes), max(tiempo.mes)))
lines(tiempo.mes.tst, tmax.exp.tst, col="red")
lines(tiempo.mes.tra, pred.mes.tra, col="blue")
lines(tiempo.mes.tst, pred.mes.tst, col="green")


## ---- message=F, warning=F-----------------------------------------------
# Errores
cat("Error en test: ",mean(abs(pred.mes.tst - tmax.exp.tst)),"\n")
cat("Error los 2 primeros meses de test: ",
    mean(abs(pred.mes.tst[1:2] - tmax.exp.tst[1:2])),"\n")


## ---- message=F, warning=F-----------------------------------------------
# Predicciones finales
Npreds <- 2
# Eliminamos la estacionalidad
tmax.no.est <- tmax.log - estacionalidad.rep[tiempo.seq.mes]
# Ajustamos el modelo arima
modelo.arima <- arima(tmax.no.est, order = c(0,0,1))
# Ajuste sobre train
fit.arima <- tmax.no.est - modelo.arima$residuals
# Predicciones de la serie estacionaria los 2 proximos meses
preds.arima <- predict(modelo.arima, n.ahead = Npreds)$pred
# Recuperamos las predicciones de la serie estacional
fit.est <- fit.arima + estacionalidad.rep[tiempo.seq.mes]
preds.est <- preds.arima + estacionalidad.rep[max(tiempo.seq.mes) + 1:Npreds]
# Deshacemos la transformación logarítmica
fit.final <- exp(fit.est)
preds.final <- exp(preds.est)


## ---- message=F, warning=F-----------------------------------------------
# Mostramos las predicciones finales
plot(tiempo.mes, tmax.mes, type="l", xlim=c(min(tiempo.mes), max(tiempo.mes)+2/12))
lines(tiempo.mes, fit.final, col="blue")
lines(max(tiempo.mes) + 1:Npreds/12, preds.final, col="red")
points(max(tiempo.mes) + 1:Npreds/12, preds.final, col="red", pch=16)
text(max(tiempo.mes) + 1:Npreds/12, preds.final, 
     labels=sprintf("%4.2f", preds.final), col="red", cex=0.7, pos=2)

