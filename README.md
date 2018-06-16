# matrix

Matrices library in Scala

This is a small library about conducting matrix calculations in Scala and
strongly inspired by the Haskell library [Data.Matrix](https://hackage.haskell.org/package/matrix-0.3.6.1/docs/Data-Matrix.html); in reality i needed a small library and was looking for a FP implementation.

Basically, i ported most of the functionality i found in that Haskell package and they are:

- Matrix builders (including special matrices)
- List conversions
- Accessing and manipulating matrices (e.g. applying scalar operations)
- Submatrices (e.g. splitting and joining blocks)
- Matrix operations 
- Matrix multiplication (e.g. vanilla and strassen's)
- Linear transformations (e.g. scale matrices, rows, columns)
- Decompositions (e.g. cholesky, laplace and lu)
- Determinant

The only dependency we rely on is [Scalaz](https://github.com/scalaz/scalaz)

# benchmarks

Planning to do thi, stay tuned.
