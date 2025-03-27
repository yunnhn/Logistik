sigmoid <- function(z) {
  1 / (1 + exp(-z))
}

newton_raphson_logistic <- function(X, y, tol = 1e-6, max_iter = 100) {
  n <- nrow(X)
  p <- ncol(X)
  beta <- rep(0, p)

  for (i in 1:max_iter) {
    p_hat <- sigmoid(X %*% beta)
    W <- diag(as.vector(p_hat * (1 - p_hat)))
    gradient <- t(X) %*% (y - p_hat)
    hessian <- t(X) %*% W %*% X

    delta <- solve(hessian, gradient)
    beta <- beta + delta

    if (sum(abs(delta)) < tol) {
      break
    }
  }
  return(list(beta = beta, fit = p_hat))
}

irls_logistic <- function(X, y, tol = 1e-6, max_iter = 100) {
  n <- nrow(X)
  p <- ncol(X)
  beta <- rep(0, p)

  for (i in 1:max_iter) {
    p_hat <- sigmoid(X %*% beta)
    W <- diag(as.vector(p_hat * (1 - p_hat)))
    z <- X %*% beta + solve(W) %*% (y - p_hat)
    beta_new <- solve(t(X) %*% W %*% X) %*% t(X) %*% W %*% z

    if (sum(abs(beta_new - beta)) < tol) {
      break
    }
    beta <- beta_new
  }
  return(list(beta = beta, fit = p_hat))
}

# Contoh penggunaan
set.seed(42)
X <- cbind(1, matrix(rnorm(200), ncol = 2))  # Menambahkan intercept
y <- as.numeric(runif(100) > 0.5)

nr_result <- newton_raphson_logistic(X, y)
irls_result <- irls_logistic(X, y)

print("Newton-Raphson:")
print(nr_result$beta)
print(head(nr_result$fit, 5))  # Menampilkan 5 data pertama

print("\nIRLS:")
print(irls_result$beta)
print(head(irls_result$fit, 5))
