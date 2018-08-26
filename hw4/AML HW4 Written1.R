#Maximum-margin classifiers
#https://rpubs.com/ppaquay/65566

x1 = c(3, 2, 4, 1, 2, 4, 4)
x2 = c(4, 2, 4, 4, 1, 3, 1)
colors = c("red", "red", "red", "red", "blue", "blue", "blue")
plot(x1, x2, col = colors, xlim = c(0, 5), ylim = c(0, 5))

abline(-0.5, 1)

plot(x1, x2, col = colors, xlim = c(0, 5), ylim = c(0, 5))
abline(-0.5, 1)
abline(-1, 1, lty = 2)
abline(0, 1, lty = 2)

plot(x1, x2, col = colors, xlim = c(0, 5), ylim = c(0, 5))
abline(-0.2, 1)

plot(x1, x2, col = colors, xlim = c(0, 5), ylim = c(0, 5))
points(c(2), c(4), col = c("blue"))
