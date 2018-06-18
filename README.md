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

Planning to do this, stay tuned.

# Examples

## Combine matrices

Here is how you would combine matrices
```scala
scala> import scalaz._, Scalaz._
import scalaz._
import Scalaz._

scala> import m.ops.{matrixMonoid,matrixS}
import m.ops.{matrixMonoid, matrixS}

scala> val a = m.ops.fromList(3)(3)(1 to 100 toList)
a: m.Matrix[Int] = Matrix(3,3,0,0,3,Vector(1, 2, 3, 4, 5, 6, 7, 8, 9))

scala> val b = m.ops.fromList(3)(3)(1 to 100 toList)
b: m.Matrix[Int] = Matrix(3,3,0,0,3,Vector(1, 2, 3, 4, 5, 6, 7, 8, 9))

scala> (a |+| b).shows
res0: String =
(2 4 6)
(8 10 12)
(14 16 18)
```

## Join matrices "vertically" and "horizontally"

Here is how you would join matrices vertically and horizontally
```scala
scala> import scalaz._, Scalaz._
import scalaz._
import Scalaz._

scala> import m.ops.{<->,<|>, matrixS}
import m.ops.{$less$minus$greater, $less$bar$greater, matrixS}

scala> val a = m.ops.fromList(3)(3)(1 to 100 toList)
a: m.Matrix[Int] = Matrix(3,3,0,0,3,Vector(1, 2, 3, 4, 5, 6, 7, 8, 9))

scala> val b = m.ops.fromList(3)(3)(1 to 100 toList)
b: m.Matrix[Int] = Matrix(3,3,0,0,3,Vector(1, 2, 3, 4, 5, 6, 7, 8, 9))

scala> (<|>(a)(b)).shows
res1: String =
(1 2 3 1 2 3)
(4 5 6 4 5 6)
(7 8 9 7 8 9)

scala> (<->(a)(b)).shows
res2: String =
(1 2 3)
(4 5 6)
(7 8 9)
(1 2 3)
(4 5 6)
(7 8 9)
```

## Add to one row a scalar multiple of another row

```scala
scala> import scalaz._, Scalaz._
import scalaz._
import Scalaz._

scala> import m.ops.{combineRows, matrixS}
import m.ops.{combineRows, matrixS}

scala> val a = m.ops.fromList(3)(3)(1 to 100 toList)
a: m.Matrix[Int] = Matrix(3,3,0,0,3,Vector(1, 2, 3, 4, 5, 6, 7, 8, 9))

scala> combineRows(2,2,1)(a).shows
res3: String =
(1 2 3)
(6 9 12)
(7 8 9)
```

## Scale a row of the matrix by a scalar

```scala
scala> import scalaz._, Scalaz._
import scalaz._
import Scalaz._

scala> import m.ops.{matrixS, matrixFunctor}
import m.ops.{matrixS, matrixFunctor}

scala> val a = m.ops.fromList(3)(3)(1 to 100 toList)
a: m.Matrix[Int] = Matrix(3,3,0,0,3,Vector(1, 2, 3, 4, 5, 6, 7, 8, 9))

scala> m.ops.scaleRow(2)(3)(a).shows
res4: String =
(1 2 3)
(4 5 6)
(14 16 18)
```

## Scale a matrix by a scalar

```scala
scala> import scalaz._, Scalaz._
import scalaz._
import Scalaz._

scala> import m.ops.{matrixS, matrixFunctor}
import m.ops.{matrixS, matrixFunctor}

scala> val a = m.ops.fromList(3)(3)(1 to 100 toList)
a: m.Matrix[Int] = Matrix(3,3,0,0,3,Vector(1, 2, 3, 4, 5, 6, 7, 8, 9))

scala> m.ops.scaleMatrix(2)(a).shows
res5: String =
(2 4 6)
(8 10 12)
(14 16 18)
```

## Swap rows of a matrix

```scala
scala> import scalaz._, Scalaz._
import scalaz._
import Scalaz._

scala> import m.ops.{matrixS, matrixFunctor}
import m.ops.{matrixS, matrixFunctor}

scala> val a = m.ops.fromList(3)(3)(1 to 100 toList)
a: m.Matrix[Int] = Matrix(3,3,0,0,3,Vector(1, 2, 3, 4, 5, 6, 7, 8, 9))

scala> a.shows
res6: String =
(1 2 3)
(4 5 6)
(7 8 9)

scala> m.ops.switchRows(1)(2)(a).shows
res7: String =
(4 5 6)
(1 2 3)
(7 8 9)
```

## Swap columns of a matrix

```scala
scala> import scalaz._, Scalaz._
import scalaz._
import Scalaz._

scala> import m.ops.{matrixS, matrixFunctor}
import m.ops.{matrixS, matrixFunctor}

scala> val a = m.ops.fromList(3)(3)(1 to 100 toList)
a: m.Matrix[Int] = Matrix(3,3,0,0,3,Vector(1, 2, 3, 4, 5, 6, 7, 8, 9))

scala> a.shows
res8: String =
(1 2 3)
(4 5 6)
(7 8 9)

scala> m.ops.switchCols(1)(2)(a).shows
res9: String =
(2 1 3)
(5 4 6)
(8 7 9)
```

## Minor matrix

```scala
scala> import scalaz._, Scalaz._
import scalaz._
import Scalaz._

scala> import m.ops.{matrixS, matrixFunctor}
import m.ops.{matrixS, matrixFunctor}

scala> val a = m.ops.fromList(3)(3)(1 to 100 toList)
a: m.Matrix[Int] = Matrix(3,3,0,0,3,Vector(1, 2, 3, 4, 5, 6, 7, 8, 9))

scala> a.shows
res10: String =
(1 2 3)
(4 5 6)
(7 8 9)

scala> m.ops.minorMatrix(1)(2)(a).shows
res11: String =
(4 6)
(7 9)

scala> m.ops.minorMatrix(2)(2)(a).shows
res12: String =
(1 3)
(7 9)

scala> m.ops.minorMatrix(3)(3)(a).shows
res13: String =
(1 2)
(4 5)
```

## Splitting matrix into 4-blocks from an element

```scala
scala> import scalaz._, Scalaz._
import scalaz._
import Scalaz._

scala> import m.ops.{matrixS, matrixFunctor}
import m.ops.{matrixS, matrixFunctor}

scala> val a = m.ops.fromList(3)(3)(1 to 100 toList)
a: m.Matrix[Int] = Matrix(3,3,0,0,3,Vector(1, 2, 3, 4, 5, 6, 7, 8, 9))

scala> a.shows
res14: String =
(1 2 3)
(4 5 6)
(7 8 9)

scala> m.ops.splitBlocks(2)(2)(a).shows
res15: String =
\/-(((1 2)
(4 5),(3)
(6),(7 8),(9)))

scala> val b = m.ops.fromList(4)(4)(1 to 100 toList)
b: m.Matrix[Int] = Matrix(4,4,0,0,4,Vector(1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16))

scala> b.shows
res16: String =
(1 2 3 4)
(5 6 7 8)
(9 10 11 12)
(13 14 15 16)

scala> m.ops.splitBlocks(3)(3)(b).shows
res17: String =
\/-(((1 2 3)
(5 6 7)
(9 10 11),(4)
(8)
(12),(13 14 15),(16)))
```

## Standard matrix multiplication

```scala
scala> import scalaz._, Scalaz._
import scalaz._
import Scalaz._

scala> import m.ops.{mult, matrixS, matrixFunctor}
import m.ops.{mult, matrixS, matrixFunctor}

scala> val a = m.ops.fromList(3)(3)(1 to 10 toList)
a: m.Matrix[Int] = Matrix(3,3,0,0,3,Vector(1, 2, 3, 4, 5, 6, 7, 8, 9))

scala> val b = m.ops.fromList(3)(3)(10 to 30 toList)
b: m.Matrix[Int] = Matrix(3,3,0,0,3,Vector(10, 11, 12, 13, 14, 15, 16, 17, 18))

scala> a.shows
res18: String =
(1 2 3)
(4 5 6)
(7 8 9)

scala> b.shows
res19: String =
(10 11 12)
(13 14 15)
(16 17 18)

scala> val x = m.ops.mult(a, b)
x: m.Matrix[Int] = Matrix(3,3,0,0,3,Vector(84, 90, 96, 201, 216, 231, 318, 342, 366))

scala> x.shows
res20: String =
(84 90 96)
(201 216 231)
(318 342 366)

scala> val c = m.ops.fromList(4)(3)(-1.2f to 19.6f by 1.2f toList)
<console>:82: warning: method to in trait FractionalProxy is deprecated (since 2.12.6): use BigDecimal range instead
       val c = m.ops.fromList(4)(3)(-1.2f to 19.6f by 1.2f toList)
                                          ^
c: m.Matrix[Float] = Matrix(4,3,0,0,3,Vector(-1.2, 0.0, 1.2, 2.4, 3.6000001, 4.8, 6.0, 7.2, 8.4, 9.599999, 10.799999, 11.999999))

scala> val d = m.ops.fromList(3)(4)(-1.2f to 19.6f by 1.2f toList)
<console>:82: warning: method to in trait FractionalProxy is deprecated (since 2.12.6): use BigDecimal range instead
       val d = m.ops.fromList(3)(4)(-1.2f to 19.6f by 1.2f toList)
                                          ^
d: m.Matrix[Float] = Matrix(3,4,0,0,4,Vector(-1.2, 0.0, 1.2, 2.4, 3.6000001, 4.8, 6.0, 7.2, 8.4, 9.599999, 10.799999, 11.999999))

scala> val y = m.ops.mult(d,c)
y: m.Matrix[Float] = Matrix(3,3,0,0,3,Vector(31.68, 34.56, 37.44, 112.31999, 138.23999, 164.15999, 192.95998, 241.91997, 290.87994))

scala> val z = m.ops.mult(c,d)
z: m.Matrix[Float] = Matrix(4,4,0,0,4,Vector(11.52, 11.5199995, 11.52, 11.5199995, 50.4, 63.36, 76.32, 89.28, 89.27999, 115.2, 141.11998, 167.03998, 128.15999, 167.03998, 205.91997, 244.79999))

scala> y.shows
res21: String =
(31.68 34.56 37.44)
(112.31999 138.23999 164.15999)
(192.95998 241.91997 290.87994)

scala> z.shows
res22: String =
(11.52 11.5199995 11.52 11.5199995)
(50.4 63.36 76.32 89.28)
(89.27999 115.2 141.11998 167.03998)
(128.15999 167.03998 205.91997 244.79999)
```

## Matrix multiplication via Strassen's method

```scala
scala> import scalaz._, Scalaz._
import scalaz._
import Scalaz._

scala> import m.ops.{multStrassen, matrixS, matrixFunctor}
import m.ops.{multStrassen, matrixS, matrixFunctor}

scala> val a = m.ops.fromList(3)(3)(1 to 10 toList)
a: m.Matrix[Int] = Matrix(3,3,0,0,3,Vector(1, 2, 3, 4, 5, 6, 7, 8, 9))

scala> val b = m.ops.fromList(3)(3)(10 to 30 toList)
b: m.Matrix[Int] = Matrix(3,3,0,0,3,Vector(10, 11, 12, 13, 14, 15, 16, 17, 18))

scala> a.shows
res23: String =
(1 2 3)
(4 5 6)
(7 8 9)

scala> b.shows
res24: String =
(10 11 12)
(13 14 15)
(16 17 18)

scala> val m1 = mult(a, b)
m1: m.Matrix[Int] = Matrix(3,3,0,0,3,Vector(84, 90, 96, 201, 216, 231, 318, 342, 366))

scala> val m2 = multStrassen(a, b)
m2: String \/ m.Matrix[Int] = \/-(Matrix(3,3,0,0,4,Vector(84, 90, 96, 0, 201, 216, 231, 0, 318, 342, 366, 0, 0, 0, 0, 0)))

scala> m1.shows
res25: String =
(84 90 96)
(201 216 231)
(318 342 366)

scala> m2.shows
res26: String =
\/-((84 90 96)
(201 216 231)
(318 342 366))

scala> import m.ops.matrixEqual
import m.ops.matrixEqual

scala> m2.map(m => matrixEqual.equal(m, m1))
res27: String \/ Boolean = \/-(false)

scala> val c = m.ops.fromList(4)(4)(-1.2f to 19.6f by 1.2f toList)
<console>:90: warning: method to in trait FractionalProxy is deprecated (since 2.12.6): use BigDecimal range instead
       val c = m.ops.fromList(4)(4)(-1.2f to 19.6f by 1.2f toList)
                                          ^
c: m.Matrix[Float] = Matrix(4,4,0,0,4,Vector(-1.2, 0.0, 1.2, 2.4, 3.6000001, 4.8, 6.0, 7.2, 8.4, 9.599999, 10.799999, 11.999999, 13.199999, 14.399999, 15.599998, 16.8))

scala> val d = m.ops.fromList(4)(4)(-1.2f to 19.6f by 1.2f toList)
<console>:90: warning: method to in trait FractionalProxy is deprecated (since 2.12.6): use BigDecimal range instead
       val d = m.ops.fromList(4)(4)(-1.2f to 19.6f by 1.2f toList)
                                          ^
d: m.Matrix[Float] = Matrix(4,4,0,0,4,Vector(-1.2, 0.0, 1.2, 2.4, 3.6000001, 4.8, 6.0, 7.2, 8.4, 9.599999, 10.799999, 11.999999, 13.199999, 14.399999, 15.599998, 16.8))

scala> val y = multStrassen(d, c)
y: String \/ m.Matrix[Float] = \/-(Matrix(4,4,0,0,4,Vector(43.19995, 46.080017, 48.960007, 51.839996, 158.39993, 184.31992, 210.23999, 236.16, 273.59995, 322.55997, 371.5199, 420.47995, 388.79993, 460.79987, 532.7999, 604.7999)))

scala> y.shows
res28: String =
\/-((43.19995 46.080017 48.960007 51.839996)
(158.39993 184.31992 210.23999 236.16)
(273.59995 322.55997 371.5199 420.47995)
(388.79993 460.79987 532.7999 604.7999))
```


## Splitting matrix into 4-blocks from an element, re-joining them

```scala
scala> import scalaz._, Scalaz._
import scalaz._
import Scalaz._

scala> import m.ops.{joinBlocks, matrixS, matrixFunctor}
import m.ops.{joinBlocks, matrixS, matrixFunctor}

scala> val a = m.ops.fromList(3)(3)(1 to 100 toList)
a: m.Matrix[Int] = Matrix(3,3,0,0,3,Vector(1, 2, 3, 4, 5, 6, 7, 8, 9))

scala> a.shows
res29: String =
(1 2 3)
(4 5 6)
(7 8 9)

scala> m.ops.splitBlocks(2)(2)(a).shows
res30: String =
\/-(((1 2)
(4 5),(3)
(6),(7 8),(9)))

scala> val b = m.ops.fromList(4)(4)(1 to 100 toList)
b: m.Matrix[Int] = Matrix(4,4,0,0,4,Vector(1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16))

scala> val splitted = m.ops.splitBlocks(3)(3)(b)
splitted: String \/ (m.Matrix[Int], m.Matrix[Int], m.Matrix[Int], m.Matrix[Int]) = \/-((Matrix(3,3,0,0,4,Vector(1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16)),Matrix(3,1,0,3,4,Vector(1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16)),Matrix(1,3,3,0,4,Vector(1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16)),Matrix(1,1,3,3,4,Vector(1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16))))

scala> splitted.map(tuple => joinBlocks(tuple)).shows
res31: String =
\/-((1 2 3 4)
(5 6 7 8)
(9 10 11 12)
(13 14 15 16))

scala> val c = m.ops.fromList(14)(4)(1 to 100 toList)
c: m.Matrix[Int] = Matrix(14,4,0,0,4,Vector(1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23, 24, 25, 26, 27, 28, 29, 30, 31, 32, 33, 34, 35, 36, 37, 38, 39, 40, 41, 42, 43, 44, 45, 46, 47, 48, 49, 50, 51, 52, 53, 54, 55, 56))

scala> val splittedC = m.ops.splitBlocks(3)(3)(c)
splittedC: String \/ (m.Matrix[Int], m.Matrix[Int], m.Matrix[Int], m.Matrix[Int]) = \/-((Matrix(3,3,0,0,4,Vector(1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23, 24, 25, 26, 27, 28, 29, 30, 31, 32, 33, 34, 35, 36, 37, 38, 39, 40, 41, 42, 43, 44, 45, 46, 47, 48, 49, 50, 51, 52, 53, 54, 55, 56)),Matrix(3,1,0,3,4,Vector(1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23, 24, 25, 26, 27, 28, 29, 30, 31, 32, 33, 34, 35, 36, 37, 38, 39, 40, 41, 42, 43, 44, 45, 46, 47, 48, 49, 50, 51, 52, 53, 54, 55, 56)),Matrix(11,3,3,0,4,Vector(1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23, 24, 25, 26, 27, 28, 29, 30, 31, 32, 33, 34, 35, 36, 37, 38, 39, 40, 41, 42, 43, 44, 45, 46, 47...

scala> splittedC.map(tuple => joinBlocks(tuple)).shows
res32: String =
\/-((1 2 3 4)
(5 6 7 8)
(9 10 11 12)
(13 14 15 16)
(17 18 19 20)
(21 22 23 24)
(25 26 27 28)
(29 30 31 32)
(33 34 35 36)
(37 38 39 40)
(41 42 43 44)
(45 46 47 48)
(49 50 51 52)
(53 54 55 56))
```


## LU decomposition

```scala
scala> import scalaz._, Scalaz._
import scalaz._
import Scalaz._

scala> import m.ops.{matrixS, fromList, lu}
import m.ops.{matrixS, fromList, lu}

scala> val a = lu(fromList(3)(3)(List(1,3,5,2,4,7,1,1,0.0)))
a: Option[(m.Matrix[Double], m.Matrix[Double], m.Matrix[Double], Double)] = Some((Matrix(3,3,0,0,3,Vector(2.0, 4.0, 7.0, 0.0, -1.0, -3.5, 0.0, 0.0, -2.0)),Matrix(3,3,0,0,3,Vector(1.0, 0.0, 0.0, 0.5, 1.0, 0.0, 0.5, -1.0, 1.0)),Matrix(3,3,0,0,3,Vector(0.0, 1.0, 0.0, 0.0, 0.0, 1.0, 1.0, 0.0, 0.0)),1))
```

## Cholesky decomposition

```scala
scala> import scalaz._, Scalaz._
import scalaz._
import Scalaz._

scala> import m.ops.{matrixS, fromList, cholesky}
import m.ops.{matrixS, fromList, cholesky}

scala> val a = cholesky(fromList(3)(3)(List(25,15,-5,15,18,0,-5,0,11.0)))
a: m.Matrix[Double] = Matrix(3,3,0,0,3,Vector(5.0, 0, 0, 3.0, 3.0, 0, -1.0, 1.0, 3.0))

scala> a.shows
res33: String =
(5.0 0 0)
(3.0 3.0 0)
(-1.0 1.0 3.0)

scala> val b =
     | cholesky(fromList(4)(4)(List(18,22,54,42,22,70,86,62,54,86,174,134,42,62,134,106)))
b: m.Matrix[Double] = Matrix(4,4,0,0,4,Vector(4.242640687119285, 0, 0, 0, 5.185449728701349, 6.565905201197403, 0, 0, 12.727922061357857, 3.0460384954008553, 1.6497422479090682, 0, 9.899494936611667, 1.624553864213788, 1.8497110052313714, 1.3926212476455935))

scala> b.shows
res34: String =
(4.242640687119285 0 0 0)
(5.185449728701349 6.565905201197403 0 0)
(12.727922061357857 3.0460384954008553 1.6497422479090682 0)
(9.899494936611667 1.624553864213788 1.8497110052313714 1.3926212476455935)
```

## Laplace Expansion

```scala
scala> import scalaz._, Scalaz._
import scalaz._
import Scalaz._

scala> import m.ops.{matrixS, fromList, laplaceDet, luDet}
import m.ops.{matrixS, fromList, laplaceDet, luDet}

scala> val a = laplaceDet(fromList(3)(3)((1.0 to 10.0 by 1.0).toList))
<console>:118: warning: method to in trait FractionalProxy is deprecated (since 2.12.6): use BigDecimal range instead
       val a = laplaceDet(fromList(3)(3)((1.0 to 10.0 by 1.0).toList))
                                              ^
a: Option[Double] = Some(0.0)

scala> val b = luDet(fromList(3)(3)((BigDecimal(1.0).to(10.0, 1.0)).toList))
b: BigDecimal = -6.0E-33

scala> a equals b
res35: Boolean = false
```

