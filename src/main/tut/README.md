Here is how you would combine matrices
```tut
import scalaz._, Scalaz._
import m.ops.{matrixM,matrixS}
val a = m.ops.fromList(3)(3)(1 to 100 toList)
val b = m.ops.fromList(3)(3)(1 to 100 toList)
(a |+| b).shows
```
Here is how you would join matrices vertically and horizontally
```tut
import scalaz._, Scalaz._
import m.ops.{<->,<|>, matrixS}
val a = m.ops.fromList(3)(3)(1 to 100 toList)
val b = m.ops.fromList(3)(3)(1 to 100 toList)
(<|>(a)(b)).shows

(<->(a)(b)).shows
```

# Add to one row a scalar multiple of another row
```tut
import scalaz._, Scalaz._
import m.ops.{combineRows, matrixS}
val a = m.ops.fromList(3)(3)(1 to 100 toList)
combineRows(2,2,1)(a).shows
```

# Scale a row of the matrix by a scalar
```tut
import scalaz._, Scalaz._
import m.ops.{matrixS, matrixF}
val a = m.ops.fromList(3)(3)(1 to 100 toList)
m.ops.scaleRow(2)(3)(a).shows
```

# Scale a matrix by a scalar
```tut
import scalaz._, Scalaz._
import m.ops.{matrixS, matrixF}
val a = m.ops.fromList(3)(3)(1 to 100 toList)
m.ops.scaleMatrix(2)(a).shows
```

# Swap rows of a matrix
```tut
import scalaz._, Scalaz._
import m.ops.{matrixS, matrixF}
val a = m.ops.fromList(3)(3)(1 to 100 toList)
a.shows

m.ops.switchRows(1)(2)(a).shows
```

# Swap columns of a matrix
```tut
import scalaz._, Scalaz._
import m.ops.{matrixS, matrixF}
val a = m.ops.fromList(3)(3)(1 to 100 toList)
a.shows

m.ops.switchCols(1)(2)(a).shows
```

# Minor matrix
```tut
import scalaz._, Scalaz._
import m.ops.{matrixS, matrixF}
val a = m.ops.fromList(3)(3)(1 to 100 toList)
a.shows

m.ops.minorMatrix(1)(2)(a).shows

m.ops.minorMatrix(2)(2)(a).shows

m.ops.minorMatrix(3)(3)(a).shows
```

# Splitting matrix into 4-blocks from an element
```tut
import scalaz._, Scalaz._
import m.ops.{matrixS, matrixF}
val a = m.ops.fromList(3)(3)(1 to 100 toList)
a.shows

m.ops.splitBlocks(2)(2)(a).shows

val b = m.ops.fromList(4)(4)(1 to 100 toList)
b.shows

m.ops.splitBlocks(3)(3)(b).shows
```

# Standard matrix multiplication
```tut
import scalaz._, Scalaz._
import m.ops.{mult, matrixS, matrixF}
val a = m.ops.fromList(3)(3)(1 to 10 toList)
val b = m.ops.fromList(3)(3)(10 to 30 toList)
a.shows

b.shows
val x = m.ops.mult(a, b)
x.shows
val c = m.ops.fromList(4)(3)(-1.2f to 19.6f by 1.2f toList)
val d = m.ops.fromList(3)(4)(-1.2f to 19.6f by 1.2f toList)
val y = m.ops.mult(d,c)
val z = m.ops.mult(c,d)
y.shows

z.shows
```

# Matrix multiplication via Strassen's method
```tut
import scalaz._, Scalaz._
import m.ops.{multStrassen, matrixS, matrixF}
val a = m.ops.fromList(3)(3)(1 to 10 toList)
val b = m.ops.fromList(3)(3)(10 to 30 toList)
a.shows

b.shows
val m1 = mult(a, b)
val m2 = multStrassen(a, b)

m1.shows
m2.shows

import m.ops.matrixEQ
m2.map(m => matrixEQ.equal(m, m1))

val c = m.ops.fromList(4)(4)(-1.2f to 19.6f by 1.2f toList)
val d = m.ops.fromList(4)(4)(-1.2f to 19.6f by 1.2f toList)
val y = multStrassen(d, c)

y.shows
```


# Splitting matrix into 4-blocks from an element, re-joining them
```tut
import scalaz._, Scalaz._
import m.ops.{joinBlocks, matrixS, matrixF}
val a = m.ops.fromList(3)(3)(1 to 100 toList)
a.shows
m.ops.splitBlocks(2)(2)(a).shows
val b = m.ops.fromList(4)(4)(1 to 100 toList)
val splitted = m.ops.splitBlocks(3)(3)(b)
splitted.map(tuple => joinBlocks(tuple)).shows

val c = m.ops.fromList(14)(4)(1 to 100 toList)
val splittedC = m.ops.splitBlocks(3)(3)(c)
splittedC.map(tuple => joinBlocks(tuple)).shows
```


# LU decomposition
```tut
import scalaz._, Scalaz._
import m.ops.{matrixS, fromList, lu}
val a = lu(fromList(3)(3)(List(1,3,5,2,4,7,1,1,0.0)))
```

# Cholesky decomposition
```tut
import scalaz._, Scalaz._
import m.ops.{matrixS, fromList, cholesky}
val a = cholesky(fromList(3)(3)(List(25,15,-5,15,18,0,-5,0,11.0)))
a.shows
val b =
cholesky(fromList(4)(4)(List(18,22,54,42,22,70,86,62,54,86,174,134,42,62,134,106)))
b.shows
```

# Laplace Expansion
```tut
import scalaz._, Scalaz._
import m.ops.{matrixS, fromList, laplaceDet, luDet}
val a = laplaceDet(fromList(3)(3)((1.0 to 10.0 by 1.0).toList))
val b = luDet(fromList(3)(3)((BigDecimal(1.0).to(10.0, 1.0)).toList))
a equals b
```

