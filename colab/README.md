# Math4AI Colab notebooks

Runnable PyTorch code for the interactive demos of *Mathematics for AI and Machine Learning* by Fuheng Wu ([math4ai.org](https://math4ai.org)). Click a badge to open the notebook in Google Colab; nothing to install.

These notebooks are generated from the book's companion snippets; edits made here are overwritten on the next build.


## Chapter 1: Vector Space and Inner Product

| Demo | What it shows | |
|---|---|---|
| [Norm Unit Balls](chapter-01/norms.ipynb) | Interactive L1, L2, Lp, L∞, Mahalanobis, and standardized Euclidean unit-ball contours. | [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/xuanxinjushi/mathematics-for-ai-machine-learning/blob/main/colab/chapter-01/norms.ipynb) |
| [Vector Addition Parallelogram](chapter-01/vector-addition.ipynb) | Drag u and v; see c = u + v and coordinate projections. | [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/xuanxinjushi/mathematics-for-ai-machine-learning/blob/main/colab/chapter-01/vector-addition.ipynb) |
| [Outer Product u vᵀ](chapter-01/outer-product.ipynb) | Heatmap of u vᵀ with highlighted entry uᵢ vⱼ. | [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/xuanxinjushi/mathematics-for-ai-machine-learning/blob/main/colab/chapter-01/outer-product.ipynb) |
| [Inner Product and Angle Explorer](chapter-01/inner-product.ipynb) | Interactively connect inner product, norms, and angle between two vectors. | [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/xuanxinjushi/mathematics-for-ai-machine-learning/blob/main/colab/chapter-01/inner-product.ipynb) |
| [Orthogonal Projection onto Span](chapter-01/projection-span.ipynb) | Project a vector onto a one-dimensional subspace and inspect residual orthogonality. | [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/xuanxinjushi/mathematics-for-ai-machine-learning/blob/main/colab/chapter-01/projection-span.ipynb) |
| [Hyperplane Distance](chapter-01/hyperplane-distance.ipynb) | 3D hyperplane θᵀx = b, foot of perpendicular, and signed distance d⊥. | [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/xuanxinjushi/mathematics-for-ai-machine-learning/blob/main/colab/chapter-01/hyperplane-distance.ipynb) |
| [Linear Regression Geometry](chapter-01/linear-regression.ipynb) | Hyperplane of predictions, data point, orthogonal foot, and residual in augmented space. | [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/xuanxinjushi/mathematics-for-ai-machine-learning/blob/main/colab/chapter-01/linear-regression.ipynb) |
| [Weight Decay Bowl](chapter-01/weight-decay-bowl.ipynb) | Quadratic potential (λ/2)||W||², trajectory, W* vs W_opt with weight decay. | [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/xuanxinjushi/mathematics-for-ai-machine-learning/blob/main/colab/chapter-01/weight-decay-bowl.ipynb) |
| [L1 vs L2 Regularization](chapter-01/l1-l2-geometry.ipynb) | Constraint geometry and lasso/ridge paths—why L1 hits axes. | [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/xuanxinjushi/mathematics-for-ai-machine-learning/blob/main/colab/chapter-01/l1-l2-geometry.ipynb) |
| [Cauchy-Schwarz Bound Visualizer](chapter-01/cauchy-schwarz.ipynb) | See dot-product magnitude compared with product of norms. | [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/xuanxinjushi/mathematics-for-ai-machine-learning/blob/main/colab/chapter-01/cauchy-schwarz.ipynb) |
| [Gram-Schmidt Orthonormalization](chapter-01/gram-schmidt.ipynb) | Build an orthonormal basis from two input vectors step by step. | [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/xuanxinjushi/mathematics-for-ai-machine-learning/blob/main/colab/chapter-01/gram-schmidt.ipynb) |

## Chapter 2: Matrix

| Demo | What it shows | |
|---|---|---|
| [Matrix-Vector Multiplication Views](chapter-02/matvec-views.ipynb) | Compare row-dot-product and column-linear-combination interpretations. | [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/xuanxinjushi/mathematics-for-ai-machine-learning/blob/main/colab/chapter-02/matvec-views.ipynb) |
| [Outer Product Matrix Build](chapter-02/outer-product.ipynb) | Construct a rank-1 matrix from two vectors and inspect structure. | [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/xuanxinjushi/mathematics-for-ai-machine-learning/blob/main/colab/chapter-02/outer-product.ipynb) |
| [Matrix Multiplication Four Views](chapter-02/matmul-views.ipynb) | Track equivalent entries from row-column, row-wise, column-wise, and outer-product views. | [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/xuanxinjushi/mathematics-for-ai-machine-learning/blob/main/colab/chapter-02/matmul-views.ipynb) |
| [Condition Number and Sensitivity](chapter-02/condition-number.ipynb) | See how near-singular matrices amplify perturbations in linear solves. | [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/xuanxinjushi/mathematics-for-ai-machine-learning/blob/main/colab/chapter-02/condition-number.ipynb) |

## Chapter 3: Subspaces and Orthogonality

| Demo | What it shows | |
|---|---|---|
| [Subspace Membership Checker](chapter-03/subspace-membership.ipynb) | Check if a target vector lies in the span of two basis vectors in 2D. | [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/xuanxinjushi/mathematics-for-ai-machine-learning/blob/main/colab/chapter-03/subspace-membership.ipynb) |
| [Orthogonal Decomposition Explorer](chapter-03/orthogonal-decomposition.ipynb) | Split a vector into projection and orthogonal residual components. | [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/xuanxinjushi/mathematics-for-ai-machine-learning/blob/main/colab/chapter-03/orthogonal-decomposition.ipynb) |
| [Least Squares Line Fit](chapter-03/least-squares-line.ipynb) | Fit y=ax+b by minimizing squared residuals and visualize errors. | [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/xuanxinjushi/mathematics-for-ai-machine-learning/blob/main/colab/chapter-03/least-squares-line.ipynb) |
| [Null Space and Column Space View](chapter-03/null-column-space.ipynb) | Inspect rank, nullity, and geometric directions for a 2x2 matrix. | [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/xuanxinjushi/mathematics-for-ai-machine-learning/blob/main/colab/chapter-03/null-column-space.ipynb) |
| [Rotary Position Embedding (RoPE)](chapter-03/rope.ipynb) | Rotate query/key pairs by position; the attention score depends only on the relative distance n − m. | [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/xuanxinjushi/mathematics-for-ai-machine-learning/blob/main/colab/chapter-03/rope.ipynb) |

## Chapter 4: QR Decomposition and Numerical Rank

| Demo | What it shows | |
|---|---|---|
| [Classical Gram-Schmidt to QR](chapter-04/gs-qr.ipynb) | Compute Q and R for a 2x2 matrix and verify A=QR. | [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/xuanxinjushi/mathematics-for-ai-machine-learning/blob/main/colab/chapter-04/gs-qr.ipynb) |
| [Householder Reflection Geometry](chapter-04/householder-geometry.ipynb) | Reflect vectors across a hyperplane and observe norm preservation. | [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/xuanxinjushi/mathematics-for-ai-machine-learning/blob/main/colab/chapter-04/householder-geometry.ipynb) |
| [Numerical Rank via Singular Values](chapter-04/numerical-rank.ipynb) | Explore threshold-dependent effective rank from singular values. | [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/xuanxinjushi/mathematics-for-ai-machine-learning/blob/main/colab/chapter-04/numerical-rank.ipynb) |
| [Orthogonality Loss Under Perturbation](chapter-04/orthogonality-loss.ipynb) | See how nearly collinear columns challenge numerical orthogonalization. | [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/xuanxinjushi/mathematics-for-ai-machine-learning/blob/main/colab/chapter-04/orthogonality-loss.ipynb) |

## Chapter 5: Square Matrix and LU Decomposition

| Demo | What it shows | |
|---|---|---|
| [LU Factorization Step Viewer](chapter-05/lu-steps.ipynb) | Factor a 2x2 matrix into L and U and verify the reconstruction. | [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/xuanxinjushi/mathematics-for-ai-machine-learning/blob/main/colab/chapter-05/lu-steps.ipynb) |
| [Pivoting Stability Explorer](chapter-05/pivoting-stability.ipynb) | Compare elimination with and without partial pivoting on near-singular systems. | [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/xuanxinjushi/mathematics-for-ai-machine-learning/blob/main/colab/chapter-05/pivoting-stability.ipynb) |
| [Determinant from Triangular Factors](chapter-05/det-triangular.ipynb) | Connect det(A) with diagonal products of triangular factors. | [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/xuanxinjushi/mathematics-for-ai-machine-learning/blob/main/colab/chapter-05/det-triangular.ipynb) |
| [Forward and Back Substitution](chapter-05/forward-backward.ipynb) | Solve Ly=b then Ux=y and compare with direct solve. | [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/xuanxinjushi/mathematics-for-ai-machine-learning/blob/main/colab/chapter-05/forward-backward.ipynb) |

## Chapter 6: Geometric Transformation and Eigen Decomposition

| Demo | What it shows | |
|---|---|---|
| [Linear Transform of Unit Circle](chapter-06/unit-circle-transform.ipynb) | Map the unit circle by a 2x2 matrix and inspect stretching and rotation. | [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/xuanxinjushi/mathematics-for-ai-machine-learning/blob/main/colab/chapter-06/unit-circle-transform.ipynb) |
| [Eigenvector Direction Check](chapter-06/eigenvector-check.ipynb) | Test candidate directions and see when Av stays collinear with v. | [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/xuanxinjushi/mathematics-for-ai-machine-learning/blob/main/colab/chapter-06/eigenvector-check.ipynb) |
| [Power Iteration Convergence](chapter-06/power-iteration.ipynb) | Iterate x_{k+1}=Ax_k/||Ax_k|| and track dominant eigenvector convergence. | [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/xuanxinjushi/mathematics-for-ai-machine-learning/blob/main/colab/chapter-06/power-iteration.ipynb) |
| [Diagonalization Recomposition](chapter-06/diagonalization.ipynb) | Rebuild A as PDP^{-1} and inspect reconstruction error. | [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/xuanxinjushi/mathematics-for-ai-machine-learning/blob/main/colab/chapter-06/diagonalization.ipynb) |

## Chapter 7: Symmetric Matrix

| Demo | What it shows | |
|---|---|---|
| [Quadratic Form Geometry](chapter-07/quadratic-form.ipynb) | Visualize x^T A x level behavior for 2x2 symmetric matrices. | [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/xuanxinjushi/mathematics-for-ai-machine-learning/blob/main/colab/chapter-07/quadratic-form.ipynb) |
| [Spectral Theorem Explorer](chapter-07/spectral-theorem.ipynb) | Inspect orthonormal eigenvectors and diagonalization of symmetric matrices. | [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/xuanxinjushi/mathematics-for-ai-machine-learning/blob/main/colab/chapter-07/spectral-theorem.ipynb) |
| [Positive Definiteness Check](chapter-07/positive-definite.ipynb) | Track eigenvalue and principal-minor conditions for SPD behavior. | [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/xuanxinjushi/mathematics-for-ai-machine-learning/blob/main/colab/chapter-07/positive-definite.ipynb) |
| [Rayleigh Quotient Bounds](chapter-07/rayleigh-bounds.ipynb) | See how x^T A x / x^T x stays between minimum and maximum eigenvalues. | [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/xuanxinjushi/mathematics-for-ai-machine-learning/blob/main/colab/chapter-07/rayleigh-bounds.ipynb) |
| [Spectral Clustering and the Fiedler Vector](chapter-07/spectral-clustering.ipynb) | Normalized graph Laplacian of two linked clusters; the Fiedler vector's signs split the graph. | [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/xuanxinjushi/mathematics-for-ai-machine-learning/blob/main/colab/chapter-07/spectral-clustering.ipynb) |

## Chapter 8: Singular Value Decomposition (SVD)

| Demo | What it shows | |
|---|---|---|
| [SVD Geometry of Unit Circle](chapter-08/svd-geometry.ipynb) | Decompose a 2x2 map into rotation-stretch-rotation and track image axes. | [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/xuanxinjushi/mathematics-for-ai-machine-learning/blob/main/colab/chapter-08/svd-geometry.ipynb) |
| [Low-Rank Approximation Error](chapter-08/low-rank-error.ipynb) | Compare Frobenius error of rank-k truncations from singular values. | [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/xuanxinjushi/mathematics-for-ai-machine-learning/blob/main/colab/chapter-08/low-rank-error.ipynb) |
| [Pseudo-Inverse and Least Squares](chapter-08/pseudoinverse-ls.ipynb) | Solve overdetermined systems via A^+ and visualize residuals. | [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/xuanxinjushi/mathematics-for-ai-machine-learning/blob/main/colab/chapter-08/pseudoinverse-ls.ipynb) |
| [Conditioning Through Singular Values](chapter-08/singular-conditioning.ipynb) | Inspect condition number and sensitivity as smallest singular value changes. | [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/xuanxinjushi/mathematics-for-ai-machine-learning/blob/main/colab/chapter-08/singular-conditioning.ipynb) |

## Chapter 9: Matrix Decompositions and Beyond

| Demo | What it shows | |
|---|---|---|
| [Polar Decomposition Explorer](chapter-09/polar-decomposition.ipynb) | Split a matrix into orthogonal and symmetric factors and inspect reconstruction. | [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/xuanxinjushi/mathematics-for-ai-machine-learning/blob/main/colab/chapter-09/polar-decomposition.ipynb) |
| [Schur Form Intuition](chapter-09/schur-form.ipynb) | Relate eigenvalues to triangular Schur form entries for 2x2 examples. | [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/xuanxinjushi/mathematics-for-ai-machine-learning/blob/main/colab/chapter-09/schur-form.ipynb) |
| [Jordan Sensitivity Toy Model](chapter-09/jordan-sensitivity.ipynb) | Show how nearly defective matrices react strongly to small perturbations. | [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/xuanxinjushi/mathematics-for-ai-machine-learning/blob/main/colab/chapter-09/jordan-sensitivity.ipynb) |
| [Matrix Function via Eigenvalues](chapter-09/matrix-function-eigs.ipynb) | Compare f(A) from spectral mapping with direct series-style intuition. | [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/xuanxinjushi/mathematics-for-ai-machine-learning/blob/main/colab/chapter-09/matrix-function-eigs.ipynb) |

## Chapter 10: Matrix Block Partitioning

| Demo | What it shows | |
|---|---|---|
| [Block Matrix Multiplication](chapter-10/block-multiplication.ipynb) | Verify block-product rules against elementwise multiplication results. | [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/xuanxinjushi/mathematics-for-ai-machine-learning/blob/main/colab/chapter-10/block-multiplication.ipynb) |
| [Schur Complement Lens](chapter-10/schur-complement.ipynb) | Inspect how eliminating one block yields Schur complement updates. | [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/xuanxinjushi/mathematics-for-ai-machine-learning/blob/main/colab/chapter-10/schur-complement.ipynb) |
| [Block Gaussian Elimination](chapter-10/block-elimination.ipynb) | Follow elimination steps on a 2x2 block system and track residuals. | [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/xuanxinjushi/mathematics-for-ai-machine-learning/blob/main/colab/chapter-10/block-elimination.ipynb) |
| [Woodbury Identity Playground](chapter-10/woodbury.ipynb) | Compare direct inverse update with Woodbury low-rank formula. | [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/xuanxinjushi/mathematics-for-ai-machine-learning/blob/main/colab/chapter-10/woodbury.ipynb) |
| [Tiled Softmax Attention](chapter-10/tiled-softmax.ipynb) | Online softmax over key/value blocks (the FlashAttention idea) matches full attention exactly. | [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/xuanxinjushi/mathematics-for-ai-machine-learning/blob/main/colab/chapter-10/tiled-softmax.ipynb) |

## Chapter 11: Matrix Calculus

| Demo | What it shows | |
|---|---|---|
| [Gradient Field and Descent Direction](chapter-11/gradient-field.ipynb) | Visualize gradients as vectors and compare with steepest-descent steps. | [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/xuanxinjushi/mathematics-for-ai-machine-learning/blob/main/colab/chapter-11/gradient-field.ipynb) |
| [Jacobian Chain Rule Explorer](chapter-11/jacobian-chain.ipynb) | Compose two vector maps and compare direct Jacobian with chain-rule product. | [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/xuanxinjushi/mathematics-for-ai-machine-learning/blob/main/colab/chapter-11/jacobian-chain.ipynb) |
| [Hessian Curvature Lens](chapter-11/hessian-curvature.ipynb) | Inspect eigenvalues of Hessian to classify local curvature and conditioning. | [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/xuanxinjushi/mathematics-for-ai-machine-learning/blob/main/colab/chapter-11/hessian-curvature.ipynb) |
| [Second-Order Taylor Approximation](chapter-11/taylor-second-order.ipynb) | Compare exact function values with local second-order Taylor expansion around a chosen point. | [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/xuanxinjushi/mathematics-for-ai-machine-learning/blob/main/colab/chapter-11/taylor-second-order.ipynb) |

## Chapter 12: Optimization Methods

| Demo | What it shows | |
|---|---|---|
| [Gradient Descent Step Size Explorer](chapter-12/gd-stepsize.ipynb) | Inspect convergence or divergence of gradient descent as learning rate changes. | [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/xuanxinjushi/mathematics-for-ai-machine-learning/blob/main/colab/chapter-12/gd-stepsize.ipynb) |
| [Newton vs Gradient Descent](chapter-12/newton-vs-gd.ipynb) | Compare first-order and second-order updates on a curved one-dimensional objective. | [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/xuanxinjushi/mathematics-for-ai-machine-learning/blob/main/colab/chapter-12/newton-vs-gd.ipynb) |
| [Projected Gradient on Box Constraint](chapter-12/projected-gradient.ipynb) | Show unconstrained steps followed by projection onto a feasible interval. | [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/xuanxinjushi/mathematics-for-ai-machine-learning/blob/main/colab/chapter-12/projected-gradient.ipynb) |
| [Duality Gap Toy Problem](chapter-12/duality-gap.ipynb) | Visualize primal and dual objectives and the effect of strong duality conditions. | [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/xuanxinjushi/mathematics-for-ai-machine-learning/blob/main/colab/chapter-12/duality-gap.ipynb) |

## Chapter 13: Probability and Random Variables

| Demo | What it shows | |
|---|---|---|
| [Multivariate Gaussian Visualizer](chapter-13/multivariate-gaussian.ipynb) | Explore how mean and covariance shape a 2D Gaussian density. | [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/xuanxinjushi/mathematics-for-ai-machine-learning/blob/main/colab/chapter-13/multivariate-gaussian.ipynb) |
| [Distribution Gallery](chapter-13/distribution-gallery.ipynb) | Interactive PMFs and PDFs for Uniform, Bernoulli, Binomial, Categorical, Gaussian, Beta, Dirichlet, and GMM. | [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/xuanxinjushi/mathematics-for-ai-machine-learning/blob/main/colab/chapter-13/distribution-gallery.ipynb) |
| [Bayes Rule Diagnostic Test](chapter-13/bayes-test.ipynb) | See how base rate, sensitivity, and specificity determine posterior probability. | [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/xuanxinjushi/mathematics-for-ai-machine-learning/blob/main/colab/chapter-13/bayes-test.ipynb) |
| [Law of Large Numbers Simulator](chapter-13/lln.ipynb) | Track empirical means converging to expectation as sample size grows. | [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/xuanxinjushi/mathematics-for-ai-machine-learning/blob/main/colab/chapter-13/lln.ipynb) |
| [Central Limit Theorem Demo](chapter-13/clt.ipynb) | Observe normalized sample means approach a Gaussian distribution. | [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/xuanxinjushi/mathematics-for-ai-machine-learning/blob/main/colab/chapter-13/clt.ipynb) |
| [ROC Curve & AUC Explorer](chapter-13/roc-auc.ipynb) | Sweep a classifier threshold to trace TPR vs FPR and read AUC. | [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/xuanxinjushi/mathematics-for-ai-machine-learning/blob/main/colab/chapter-13/roc-auc.ipynb) |
| [Binomial PMF & CDF](chapter-13/pmf-cdf.ipynb) | Compare binomial probability mass with the cumulative step function. | [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/xuanxinjushi/mathematics-for-ai-machine-learning/blob/main/colab/chapter-13/pmf-cdf.ipynb) |
| [Joint vs Marginal PMF](chapter-13/joint-marginal.ipynb) | Heatmap a discrete joint table and inspect row/column marginals. | [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/xuanxinjushi/mathematics-for-ai-machine-learning/blob/main/colab/chapter-13/joint-marginal.ipynb) |
| [Independence Checker](chapter-13/independence.ipynb) | Contrast a 2×2 joint PMF with the independence product P(X)P(Y). | [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/xuanxinjushi/mathematics-for-ai-machine-learning/blob/main/colab/chapter-13/independence.ipynb) |
| [Change of Variables](chapter-13/change-of-variables.ipynb) | Uniform→Exponential (1D) and linear pushforward of a bivariate Gaussian with Jacobian factor. | [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/xuanxinjushi/mathematics-for-ai-machine-learning/blob/main/colab/chapter-13/change-of-variables.ipynb) |
| [Normalizing Flow (1D Affine)](chapter-13/normalizing-flow.ipynb) | Chain two affine layers on N(0,1); exact density via cumulative log|det J|. | [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/xuanxinjushi/mathematics-for-ai-machine-learning/blob/main/colab/chapter-13/normalizing-flow.ipynb) |

## Chapter 14: Information Theory

| Demo | What it shows | |
|---|---|---|
| [Entropy of a Categorical Distribution](chapter-14/entropy-categorical.ipynb) | Simplex heatmap, probability partition, and −p_i log p_i decomposition for 3-class entropy. | [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/xuanxinjushi/mathematics-for-ai-machine-learning/blob/main/colab/chapter-14/entropy-categorical.ipynb) |
| [KL Direction Explorer](chapter-14/kl-direction.ipynb) | Compare KL(p||q) and KL(q||p) for two one-dimensional Gaussian distributions. | [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/xuanxinjushi/mathematics-for-ai-machine-learning/blob/main/colab/chapter-14/kl-direction.ipynb) |
| [Mutual Information in Correlated Gaussian](chapter-14/mutual-info-gaussian.ipynb) | See mutual information increase with correlation strength. | [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/xuanxinjushi/mathematics-for-ai-machine-learning/blob/main/colab/chapter-14/mutual-info-gaussian.ipynb) |
| [ELBO Components and Beta Weight](chapter-14/elbo-components.ipynb) | Explore the β-VAE trade-off: reconstruction vs KL penalty, with waterfall decomposition and ELBO vs β. | [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/xuanxinjushi/mathematics-for-ai-machine-learning/blob/main/colab/chapter-14/elbo-components.ipynb) |

## Chapter 15: Bellman Equations and Operators

| Demo | What it shows | |
|---|---|---|
| [Bellman Backup on 3-State MDP](chapter-15/bellman-backup.ipynb) | Apply one-step Bellman expectation updates and inspect residuals state-by-state. | [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/xuanxinjushi/mathematics-for-ai-machine-learning/blob/main/colab/chapter-15/bellman-backup.ipynb) |
| [Contraction Mapping Visualizer](chapter-15/contraction.ipynb) | See geometric error decay under repeated Bellman operator applications. | [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/xuanxinjushi/mathematics-for-ai-machine-learning/blob/main/colab/chapter-15/contraction.ipynb) |
| [Value Iteration vs Policy Iteration](chapter-15/vi-vs-pi.ipynb) | Compare convergence speed and policy stability for value iteration and policy iteration. | [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/xuanxinjushi/mathematics-for-ai-machine-learning/blob/main/colab/chapter-15/vi-vs-pi.ipynb) |
| [Policy Evaluation Linear Solve](chapter-15/policy-evaluation.ipynb) | Contrast iterative policy evaluation with closed-form linear-system solution. | [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/xuanxinjushi/mathematics-for-ai-machine-learning/blob/main/colab/chapter-15/policy-evaluation.ipynb) |

## Chapter 16: Variational Inference and Latent Variables

| Demo | What it shows | |
|---|---|---|
| [ELBO Decomposition Explorer](chapter-16/elbo-decomposition.ipynb) | Decompose ELBO into expected log-likelihood and KL with a beta weight. | [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/xuanxinjushi/mathematics-for-ai-machine-learning/blob/main/colab/chapter-16/elbo-decomposition.ipynb) |
| [KL Direction: Mode-Seeking vs Mode-Covering](chapter-16/kl-direction.ipynb) | Compare reverse KL and forward KL on a bimodal target with a unimodal variational family. | [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/xuanxinjushi/mathematics-for-ai-machine-learning/blob/main/colab/chapter-16/kl-direction.ipynb) |
| [Reparameterization Trick Sampler](chapter-16/reparameterization.ipynb) | Sample z = mu + sigma * epsilon and observe how gradients can flow through mu and sigma. | [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/xuanxinjushi/mathematics-for-ai-machine-learning/blob/main/colab/chapter-16/reparameterization.ipynb) |
| [Posterior Collapse Dial](chapter-16/posterior-collapse.ipynb) | See how increasing beta can shrink latent usage toward the prior in a toy VAE objective. | [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/xuanxinjushi/mathematics-for-ai-machine-learning/blob/main/colab/chapter-16/posterior-collapse.ipynb) |

## Chapter 17: Score Function and Energy-Based Models

| Demo | What it shows | |
|---|---|---|
| [Univariate Gaussian Score](chapter-17/gaussian-score-1d.ipynb) | PDF, log PDF tangents, and score s(x) for a 1D Gaussian. | [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/xuanxinjushi/mathematics-for-ai-machine-learning/blob/main/colab/chapter-17/gaussian-score-1d.ipynb) |
| [2D Gaussian Score Field](chapter-17/gaussian-score.ipynb) | Score vectors as gradients of log-density for a 2D Gaussian. | [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/xuanxinjushi/mathematics-for-ai-machine-learning/blob/main/colab/chapter-17/gaussian-score.ipynb) |
| [Energy, Density, and Score](chapter-17/three-views.ipynb) | Linked triple view: double-well energy, p∝e^{−E}, and score field. | [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/xuanxinjushi/mathematics-for-ai-machine-learning/blob/main/colab/chapter-17/three-views.ipynb) |
| [Circle Manifold Density](chapter-17/manifold-density.ipynb) | Density concentrated on the unit circle; on vs off manifold. | [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/xuanxinjushi/mathematics-for-ai-machine-learning/blob/main/colab/chapter-17/manifold-density.ipynb) |
| [Mixture Score Field](chapter-17/mixture-score.ipynb) | Gaussian-mixture score field bending between two modes. | [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/xuanxinjushi/mathematics-for-ai-machine-learning/blob/main/colab/chapter-17/mixture-score.ipynb) |
| [DSM S-Curve Manifold](chapter-17/dsm-scurve.ipynb) | Noisy samples and denoising score vectors on an S-curve. | [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/xuanxinjushi/mathematics-for-ai-machine-learning/blob/main/colab/chapter-17/dsm-scurve.ipynb) |
| [DSM Circle Manifold](chapter-17/dsm-circle.ipynb) | Denoising score matching on the unit circle. | [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/xuanxinjushi/mathematics-for-ai-machine-learning/blob/main/colab/chapter-17/dsm-circle.ipynb) |
| [DSM Univariate Gaussian](chapter-17/dsm-gaussian.ipynb) | Clean vs noisy marginal and closed-form denoising score. | [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/xuanxinjushi/mathematics-for-ai-machine-learning/blob/main/colab/chapter-17/dsm-gaussian.ipynb) |
| [Energy to Probability](chapter-17/energy-to-probability.ipynb) | Convert a customizable 1D energy landscape to normalized density. | [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/xuanxinjushi/mathematics-for-ai-machine-learning/blob/main/colab/chapter-17/energy-to-probability.ipynb) |
| [Score Matching Toy Objective](chapter-17/score-matching.ipynb) | Compare true vs model score and the squared mismatch objective. | [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/xuanxinjushi/mathematics-for-ai-machine-learning/blob/main/colab/chapter-17/score-matching.ipynb) |

## Chapter 18: Langevin Dynamics and Sampling

| Demo | What it shows | |
|---|---|---|
| [Optimization vs Sampling](chapter-18/opt-vs-sampling.ipynb) | Gradient ascent vs Langevin dynamics on a 2D Gaussian mixture. | [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/xuanxinjushi/mathematics-for-ai-machine-learning/blob/main/colab/chapter-18/opt-vs-sampling.ipynb) |
| [1D Langevin Gaussian](chapter-18/langevin-1d.ipynb) | Convergence of Langevin dynamics to N(0,1) with live histogram. | [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/xuanxinjushi/mathematics-for-ai-machine-learning/blob/main/colab/chapter-18/langevin-1d.ipynb) |
| [Metropolis–Hastings MCMC](chapter-18/metropolis-hastings.ipynb) | Sample a bimodal distribution with random-walk Metropolis–Hastings. | [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/xuanxinjushi/mathematics-for-ai-machine-learning/blob/main/colab/chapter-18/metropolis-hastings.ipynb) |
| [ULA vs MALA](chapter-18/mala.ipynb) | Unadjusted vs Metropolis-adjusted Langevin on the same target. | [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/xuanxinjushi/mathematics-for-ai-machine-learning/blob/main/colab/chapter-18/mala.ipynb) |
| [SGLD vs SGD](chapter-18/sgld.ipynb) | Stochastic gradient Langevin dynamics for mini-batch Bayesian learning. | [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/xuanxinjushi/mathematics-for-ai-machine-learning/blob/main/colab/chapter-18/sgld.ipynb) |
| [Trace Plot & Histogram](chapter-18/trace-histogram.ipynb) | Burn-in, mixing, and empirical density for a double-well target. | [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/xuanxinjushi/mathematics-for-ai-machine-learning/blob/main/colab/chapter-18/trace-histogram.ipynb) |
| [2D Gaussian Mixture Langevin](chapter-18/mixture-2d.ipynb) | Score-based Langevin sampling on a two-mode 2D density. | [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/xuanxinjushi/mathematics-for-ai-machine-learning/blob/main/colab/chapter-18/mixture-2d.ipynb) |
| [Particle Cloud Evolution](chapter-18/particle-cloud.ipynb) | Many particles spreading under Langevin dynamics (Fokker–Planck preview). | [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/xuanxinjushi/mathematics-for-ai-machine-learning/blob/main/colab/chapter-18/particle-cloud.ipynb) |

## Chapter 19: Stochastic Differential Equations

| Demo | What it shows | |
|---|---|---|
| [ODE vs SDE Roughness](chapter-19/ode-vs-sde.ipynb) | Zoom into a smooth sine curve vs a Brownian path. | [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/xuanxinjushi/mathematics-for-ai-machine-learning/blob/main/colab/chapter-19/ode-vs-sde.ipynb) |
| [Random Walk → Brownian Motion](chapter-19/random-walk.ipynb) | Compare discrete random walks at different Δt with the same seed. | [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/xuanxinjushi/mathematics-for-ai-machine-learning/blob/main/colab/chapter-19/random-walk.ipynb) |
| [2D Langevin Dynamics](chapter-19/langevin-2d.ipynb) | Particles relaxing to a 2D Gaussian target under Langevin SDE. | [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/xuanxinjushi/mathematics-for-ai-machine-learning/blob/main/colab/chapter-19/langevin-2d.ipynb) |
| [OU Path Streaming Demo](chapter-19/ou-path-stream.ipynb) | Stream an Ornstein-Uhlenbeck path in real time with browser-side simulation. | [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/xuanxinjushi/mathematics-for-ai-machine-learning/blob/main/colab/chapter-19/ou-path-stream.ipynb) |
| [OU Terminal Distribution](chapter-19/ou-distribution.ipynb) | Many OU paths with a live histogram of X_T vs the stationary Gaussian. | [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/xuanxinjushi/mathematics-for-ai-machine-learning/blob/main/colab/chapter-19/ou-distribution.ipynb) |
| [GBM & Itô Correction](chapter-19/gbm-ito.ipynb) | Geometric Brownian motion paths vs e^{μt} and the Itô drift correction. | [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/xuanxinjushi/mathematics-for-ai-machine-learning/blob/main/colab/chapter-19/gbm-ito.ipynb) |
| [Itô vs Stratonovich](chapter-19/ito-stratonovich.ipynb) | One-step schematic: left endpoint vs midpoint stochastic integral. | [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/xuanxinjushi/mathematics-for-ai-machine-learning/blob/main/colab/chapter-19/ito-stratonovich.ipynb) |
| [Euler–Maruyama Convergence](chapter-19/em-convergence.ipynb) | Compare OU discretizations at several Δt values against the exact mean. | [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/xuanxinjushi/mathematics-for-ai-machine-learning/blob/main/colab/chapter-19/em-convergence.ipynb) |
| [Forward SDE Funnel](chapter-19/forward-sde.ipynb) | Two data clusters spread into noise under a forward OU SDE. | [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/xuanxinjushi/mathematics-for-ai-machine-learning/blob/main/colab/chapter-19/forward-sde.ipynb) |
| [Reverse Diffusion: Noise to Data](chapter-19/reverse-diffusion.ipynb) | Run the reverse-time SDE (or probability flow ODE) with the exact score from noise back to a two-mode distribution. | [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/xuanxinjushi/mathematics-for-ai-machine-learning/blob/main/colab/chapter-19/reverse-diffusion.ipynb) |

## Chapter 20: ODE, SDE, and Continuous Limits of Algorithms

| Demo | What it shows | |
|---|---|---|
| [Gradient Descent to ODE](chapter-20/gd-to-ode.ipynb) | Compare discrete gradient descent iterates with the continuous gradient-flow trajectory. | [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/xuanxinjushi/mathematics-for-ai-machine-learning/blob/main/colab/chapter-20/gd-to-ode.ipynb) |
| [Probability Flow ODE vs SDE](chapter-20/pf-ode-vs-sde.ipynb) | One deterministic and one stochastic evolution under shared drift parameters. | [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/xuanxinjushi/mathematics-for-ai-machine-learning/blob/main/colab/chapter-20/pf-ode-vs-sde.ipynb) |
| [Linear ODE Discretization Methods](chapter-20/discretization-methods.ipynb) | Euler, exact zero-order hold, and Tustin updates for the same stable linear system. | [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/xuanxinjushi/mathematics-for-ai-machine-learning/blob/main/colab/chapter-20/discretization-methods.ipynb) |
| [Stability Region Explorer](chapter-20/stability-region.ipynb) | See when discrete step size destabilizes a continuous stable mode. | [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/xuanxinjushi/mathematics-for-ai-machine-learning/blob/main/colab/chapter-20/stability-region.ipynb) |
| [Flow Matching](chapter-20/flow-matching.ipynb) | Straight training paths, curved non-crossing flow: the exact flow matching velocity carries noise to a 3-mode target. | [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/xuanxinjushi/mathematics-for-ai-machine-learning/blob/main/colab/chapter-20/flow-matching.ipynb) |

## Chapter 21: Fokker-Planck and Distribution Dynamics

| Demo | What it shows | |
|---|---|---|
| [Probability Current Decomposition](chapter-21/probability-current.ipynb) | Visualize drift and diffusion contributions to Fokker-Planck probability current. | [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/xuanxinjushi/mathematics-for-ai-machine-learning/blob/main/colab/chapter-21/probability-current.ipynb) |
| [Langevin to Stationary Density](chapter-21/langevin-stationary.ipynb) | Simulate many particles and compare terminal histogram with Boltzmann stationary density. | [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/xuanxinjushi/mathematics-for-ai-machine-learning/blob/main/colab/chapter-21/langevin-stationary.ipynb) |
| [Heat Equation Smoothing](chapter-21/heat-smoothing.ipynb) | Forward diffusion broadens density and increases entropy over time. | [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/xuanxinjushi/mathematics-for-ai-machine-learning/blob/main/colab/chapter-21/heat-smoothing.ipynb) |
| [SDE and Probability Flow ODE Match](chapter-21/sde-ode-match.ipynb) | Compare distribution evolution under stochastic SDE and deterministic probability-flow ODE. | [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/xuanxinjushi/mathematics-for-ai-machine-learning/blob/main/colab/chapter-21/sde-ode-match.ipynb) |

## AI & ML Foundations: Neural networks and backpropagation

| Demo | What it shows | |
|---|---|---|
| [Neuron and Activation Functions](ml/neuron-activation.ipynb) | One neuron y = phi(w x + b): how weight, bias and the activation (sigmoid, tanh, ReLU, GELU) shape the output and its gradient. | [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/xuanxinjushi/mathematics-for-ai-machine-learning/blob/main/colab/ml/neuron-activation.ipynb) |
| [MLP Curve Fitting](ml/mlp-fit.ipynb) | Train a one-hidden-layer network in the browser and watch wider layers fit a curve better. | [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/xuanxinjushi/mathematics-for-ai-machine-learning/blob/main/colab/ml/mlp-fit.ipynb) |
| [Backpropagation on a Computation Graph](ml/backprop-graph.ipynb) | Forward values and backward gradients node by node for a tiny network, checked against finite differences. | [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/xuanxinjushi/mathematics-for-ai-machine-learning/blob/main/colab/ml/backprop-graph.ipynb) |
| [Vanishing and Exploding Gradients](ml/vanishing-gradients.ipynb) | Gradient size layer by layer in a deep network: sigmoid vs ReLU, and how the initialization scale decides whether signals survive. | [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/xuanxinjushi/mathematics-for-ai-machine-learning/blob/main/colab/ml/vanishing-gradients.ipynb) |

## AI & ML Foundations: Attention and Transformers

| Demo | What it shows | |
|---|---|---|
| [Scaled Dot-Product Attention](ml/attention.ipynb) | softmax(Q K^T / sqrt(d)) V as a heatmap, and why the sqrt(d) keeps softmax from saturating. | [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/xuanxinjushi/mathematics-for-ai-machine-learning/blob/main/colab/ml/attention.ipynb) |
| [Causal Masking](ml/causal-mask.ipynb) | Hide future tokens with a -inf mask and see each row of attention renormalize over the past only. | [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/xuanxinjushi/mathematics-for-ai-machine-learning/blob/main/colab/ml/causal-mask.ipynb) |
| [Multi-Head Attention](ml/multi-head.ipynb) | Split the model width into heads: each head attends with its own projections and finds a different pattern. | [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/xuanxinjushi/mathematics-for-ai-machine-learning/blob/main/colab/ml/multi-head.ipynb) |
| [Transformer Block: Residuals and LayerNorm](ml/transformer-block.ipynb) | Stack blocks with and without residual connections and LayerNorm and track how the signal and its gradient survive depth. | [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/xuanxinjushi/mathematics-for-ai-machine-learning/blob/main/colab/ml/transformer-block.ipynb) |

## AI & ML Foundations: Embeddings and retrieval

| Demo | What it shows | |
|---|---|---|
| [Similarity Search and Ranking Metrics](ml/similarity-ranking.ipynb) | Rank items by cosine similarity to a query and score the ranking with P@K, R@K and Average Precision (Chapters 1 and 14). | [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/xuanxinjushi/mathematics-for-ai-machine-learning/blob/main/colab/ml/similarity-ranking.ipynb) |
| [SVD Embeddings](ml/svd-embedding.ipynb) | Embed the rows of a rating matrix as U_k Sigma_k: similar items land together and inner products are kept (Chapter 8). | [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/xuanxinjushi/mathematics-for-ai-machine-learning/blob/main/colab/ml/svd-embedding.ipynb) |

## AI & ML Foundations: Convolutional networks

| Demo | What it shows | |
|---|---|---|
| [Convolution and Translation Equivariance](ml/conv-equivariance.ipynb) | Shared weights make convolution commute with shifts; per-position weights break it (Chapter 6). | [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/xuanxinjushi/mathematics-for-ai-machine-learning/blob/main/colab/ml/conv-equivariance.ipynb) |
| [Pooling Gradients](ml/pooling-gradients.ipynb) | Max pooling sends the gradient only to each window's maximum; average pooling spreads it evenly (Chapter 11). | [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/xuanxinjushi/mathematics-for-ai-machine-learning/blob/main/colab/ml/pooling-gradients.ipynb) |

## AI & ML Foundations: Optimizers

| Demo | What it shows | |
|---|---|---|
| [Optimizers in an Ill-Conditioned Valley](ml/optimizers-valley.ipynb) | Gradient descent, momentum and Adam on a narrow valley; rotate it and Adam's per-axis scaling stops helping (Chapters 1, 11, 20). | [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/xuanxinjushi/mathematics-for-ai-machine-learning/blob/main/colab/ml/optimizers-valley.ipynb) |
| [Muon's Newton-Schulz Orthogonalization](ml/muon-newton-schulz.ipynb) | Q <- Q(3I - Q^T Q)/2 drives every singular value to 1, giving the polar factor Muon uses as its update (Chapters 8, 11). | [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/xuanxinjushi/mathematics-for-ai-machine-learning/blob/main/colab/ml/muon-newton-schulz.ipynb) |
