// MIT License
// 
// Copyright (c) 2018 Raymond Tay
// 
// Permission is hereby granted, free of charge, to any person obtaining a copy
// of this software and associated documentation files (the "Software"), to deal
// in the Software without restriction, including without limitation the rights
// to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
// copies of the Software, and to permit persons to whom the Software is
// furnished to do so, subject to the following conditions:
// 
// The above copyright notice and this permission notice shall be included in all
// copies or substantial portions of the Software.
// 
// THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
// IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
// FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
// AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
// LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
// OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
// SOFTWARE.

package m

import scala.language.postfixOps

import scalaz._, Scalaz._

case class Matrix[@specialized(Float,Double,Int) +A](
  nrows : Int,
  ncols: Int,
  rowOffset: Int,
  colOffset : Int,
  vcols: Int,
  mvect : Vector[A])

trait MatrixImplicits extends Base {

  implicit def matrixSemigroup[A:Semigroup] : Semigroup[Matrix[A]] = new Semigroup[Matrix[A]] {
    def append(f1: Matrix[A], f2: ⇒ Matrix[A]) : Matrix[A] = {
      matrix(math.max(f1.nrows, f2.nrows))(math.max(f1.ncols, f2.ncols))((i: Int, j:Int) ⇒
        (safeGet(i,j)(f1) |+| safeGet(i, j)(f2)).getOrElse(0.asInstanceOf[A])
      )
    }
  }

  implicit def matrixMonoid[A:Monoid] : Monoid[Matrix[A]] = new Monoid[Matrix[A]] {
    def zero = fromList(1)(1)(0.asInstanceOf[A] ::Nil)
    def append(f1: Matrix[A], f2: ⇒ Matrix[A]) : Matrix[A] = {
      matrix(math.max(f1.nrows, f2.nrows))(math.max(f1.ncols, f2.ncols))((i: Int, j:Int) ⇒
        (safeGet(i,j)(f1) |+| safeGet(i, j)(f2)).getOrElse(0.asInstanceOf[A])
      )
    }
  }

  implicit val matrixFunctor : Functor[Matrix] = new Functor[Matrix] {
    def map[A,B](fa: Matrix[A])(f : A ⇒ B) : Matrix[B] = fa.copy(mvect = fa.mvect.map(f(_)))
  }

  implicit val matrixAp : Applicative[Matrix] = new Applicative[Matrix] {
    def point[A](a: ⇒ A) = fromList(1)(1)(a :: Nil)

    def ap[A,B](fa: ⇒ Matrix[A])(f: ⇒ Matrix[A ⇒ B]) : Matrix[B] = {
      // TODO: val F = implicitly[Functor[Matrix]] appears to be calling itself
      val F = matrixFunctor
      flatten(F.map(f)(g ⇒ F.map(fa)(a ⇒ g(a))))
    }
  }

  implicit def matrixEqual[A] : Equal[Matrix[A]] = new Equal[Matrix[A]] {
    def equal(a1: Matrix[A], a2: Matrix[A]) : Boolean = {
      if (a1.nrows == a2.nrows && a1.ncols == a2.ncols && a1.mvect.equals(a2.mvect))
        true else false
    }
  }


  implicit val matrixFoldable = new Foldable1[Matrix] {
    def foldMap1[A, B](fa: Matrix[A])(f: A ⇒ B)(implicit F: scalaz.Semigroup[B]): B =
      fa.mvect.map(f(_)).reduce((x,y) ⇒ F.append(x, y))

    def foldMapRight1[A, B](fa: Matrix[A])(z: A ⇒ B)(f: (A, ⇒ B) ⇒ B): B = {
      val r = fa.mvect.reverse
      r.tail.foldLeft(z(r.reverse.head))((x,y) ⇒ f(y,x))
    }
  }

  private
  def flatten[A](mm : Matrix[Matrix[A]]) : Matrix[A] = {
    val F = implicitly[Foldable1[Matrix]]
    F.foldl1(mm)((m: Matrix[A]) ⇒ (n: Matrix[A]) ⇒ <->(m)(n))
  }

  private[m]
  def encode = (m: Int) ⇒ (p: (Int,Int)) ⇒
    (p._1 - 1)*m + p._2 - 1

  def unsafeGet[A](i: Int, j: Int)(m: Matrix[A]) : A =
    m.mvect( encode(m.vcols)(i+m.rowOffset,j+m.colOffset) )

  def safeGet[A](i: Int, j: Int)(m: Matrix[A]) : Option[A] =
    if (i > m.nrows || j > m.ncols || i < 1 || j < 1) none
    else unsafeGet(i, j)(m).some

  implicit def matrixS[A:Numeric] = new Show[Matrix[A]] {
    val M = implicitly[Numeric[A]]
    override def show(m: Matrix[A]) = {
      val rows =
      for {
        i ← 1 to m.nrows
      } yield getRow(i)(m).map(row ⇒ row.mkString(" "))
      s"""${rows.map(row ⇒ "(" + row.mkString(" ") + ")\n").mkString.trim}"""
    }
  }

}

class MatrixOps extends MatrixImplicits {

  private
  def decode = (m:Int) ⇒ (k: Int) ⇒ {
    val (q, r) = if (k == 0) (0,0) else (k/m, k % m)
    (q+1,r+1)
  }

  override def unsafeMatrix[A](m: Matrix[A]) : Matrix[A] =
    matrix(m.nrows)(m.ncols)((i:Int,j:Int) ⇒ unsafeGet(i, j)(m))

  override def <->[A](m: Matrix[A])(n: Matrix[A]) : Matrix[A] = {
    assert(m.nrows == n.nrows, "For the horizontal cat operation, both matrices must have the same number of rows.")
    matrix(m.nrows + n.nrows)(m.ncols)((i: Int, j: Int) ⇒ if (i <= m.nrows) unsafeGet(i, j)(m) else unsafeGet(i - m.nrows, j)(n))
  }

  override def <|>[A](m: Matrix[A])(n: Matrix[A]) : Matrix[A] = {
    assert(m.ncols == n.ncols, "For the vertical cat operation, both matrices must have the same number of columns.")
    matrix(m.nrows)(m.ncols + n.ncols)((i: Int, j:Int) ⇒ if (j <= n.ncols) unsafeGet(i, j)(m) else unsafeGet(i, j - n.ncols)(n))
  }

  /*
   * Map a function over a row.
   * @param f function to map over
   * @param c value that is matched, upon which 'f' is applied
   * @return matrix with the function applied for matching value, 'c'.
   */
  def mapRow[A](f: (Int, A) ⇒ A)(r: Int)(m: Matrix[A]) : Matrix[A] =
    matrix(m.nrows)(m.ncols){(x: Int, y: Int) ⇒
      val ele = unsafeGet(x, y)(m)
      if (x == r) f(y,ele) else ele
    }

  /*
   * Map a function over a column.
   * @param f function to map over
   * @param c value that is matched, upon which 'f' is applied
   * @return matrix with the function applied for matching value, 'c'.
   */
  def mapCol[A](f: (Int, A) ⇒ A)(c: Int)(m: Matrix[A]) : Matrix[A] =
    matrix(m.nrows)(m.ncols){(x: Int, y: Int) ⇒
      val ele = unsafeGet(x, y)(m)
      if (y == c) f(y, ele) else ele
    }

  def mapPos[A,B](f:((Int,Int)) ⇒ A ⇒ B)(m: Matrix[A]): Matrix[B] =
    m.copy(mvect = m.mvect.zipWithIndex.collect{ case (ele, i) ⇒ f(decode(m.ncols)(i))(ele) })

  /*
   * The zero matrix of the given size.
   * @param n size of square matrix
   * @return zero-matrix
   */
  def zero[A](n: Int)(m: Int) : Matrix[A] =
    Matrix(n, m, 0, 0, m, Vector.fill[A](n*m)(0.asInstanceOf[A]))

  /**
   * Generate a matrix from a generator function:
   * Example usage: matrix(4)(4)((x:Int, y:Int) ⇒ 2*x - y)
   * matrix.Matrix[Int] = Matrix(4,4,0,0,4,Vector(1, 0, -1, -2, 3, 2, 1, 0, 5, 4, 3, 2, 7, 6, 5, 4))
   * @param n rows
   * @param m cols
   * @param generator function
   * @return a matrix
   */
  override def matrix[@specialized(Int,Double,Float) A](n: Int)(m: Int)(f: (Int, Int) ⇒ A) : Matrix[A] = {
    var v = collection.mutable.ArrayBuffer.fill(n*m)(0.asInstanceOf[A])
    val en = encode(m)
    for {
      i ← 1 to n
      j ← 1 to m
    } yield {
      v.update(en(i,j), f(i, j))
    }
    Matrix(n, m, 0, 0, m, Vector(v:_*))
  }

  /**
   * Identity matrix
   * @param n dimension of matrix (n,n)
   * @return a matrix of (n, n)
   */
  def identity[A:Numeric](n: Int) = {
    val mm = implicitly[Numeric[A]]
    matrix(n)(n)((x:Int, y:Int) ⇒ if (x == y) mm.one else mm.zero)
  }

  /**
    * Creates diagonal matrix
    * @param ele
    * @param vector the vector whose elements will be placed on the diagonal
    * of the generated matrix
    */
  def diagonal[A](ele: A)(vector: Vector[A]) : Matrix[A] =
    matrix(vector.size)(vector.size)((x: Int, y: Int) ⇒ if (x == y) vector(x-1) else ele)

  /*
   * Diagonal matrix from a non-empty list for the desired size and
   * non-diagonal elements will be filled with the given default element.
   *                   n
   *   1 ( 1 0 ... 0   0 )
   *   2 ( 0 2 ... 0   0 )
   *     (     ...       )
   *     ( 0 0 ... n-1 0 )
   *   n ( 0 0 ... 0   n )
   * @param n dimension of matrix (n,n)
   * @return a matrix of (n, n)
   */
  def diagonalList[A](n: Int)(ele: A)(xs: Vector[A]) : Matrix[A] = {
    assert(n == xs.size, "size of vector should be equal to elements")
    matrix(n)(n)((i: Int, j: Int) ⇒ if (i == j) xs(i-1) else ele)
  }

  override def fromList[A](n: Int)(m: Int)(xs: List[A]) : Matrix[A] = {
    def fetch : List[A] = {
      for {
        x ← 0 until n
        y ← 0 until m
      } yield {
        xs(x*m + y)
      }}.toList
    Matrix(n, m, 0, 0, m, Vector(fetch:_*))
  }

  def toList[A](m: Matrix[A]) = m.mvect

  def toLists[A](m : Matrix[A]) : Vector[Vector[A]] =
    m.mvect.grouped(m.ncols).toVector

  /*
   * Create a matrix from a non-empty list of non-empty lists.
   * Dimension of the matrix is determined by the first sub-list
   * @param xss list of lists
   * @return matrix
   */
  def fromLists[A](xss: List[List[A]]) : Either[String, Matrix[A]] = {
    xss match {
      case Nil ⇒ Left("fromLists: empty list.")
      case xs :: xss ⇒ Right(fromList(1 + xss.size)(xs.size)(xs ++ xss.map(xx ⇒ xx.take(xs.size)).flatten))
    }
  }

  // Represent a vector as a one row matrix.
  def rowVector[A](xs: Vector[A]) = Matrix(1, xs.size, 0, 0, xs.size, xs)

  // Represent a vector as a one column matrix.
  def colVector[A](xs: Vector[A]) = Matrix(xs.size, 1, 0, 0, xs.size, xs)

  // Permutation Matrix
  def permutationMatrix[A:Numeric](n: Int)(row1: Int)(row2: Int) = {
    def decide = (i: Int, j: Int) ⇒ {
      if (i == row1) { if (j == row2) 1 else 0 }
      else if (i == row2) { if (j == row1) 1 else 0 }
      else if (j == i) 1 else 0
    }
    if(row1 == row2) identity(n) else matrix(n)(n)(decide)
  }

  // safe get of an element in the matrix
  def getElem[A](i: Int)(j: Int)(m: Matrix[A]) : Option[A] =
    if (m.nrows < i || m.ncols < j) none else unsafeGet(i,j)(m).some

  // safe get of an row from the matrix
  override def getRow[A](r: Int)(m: Matrix[A]) : Option[Vector[A]] =
    if (m.nrows < r) none else {
      val start = m.vcols*(r-1+m.rowOffset) + m.colOffset
      m.mvect.slice(start, start + m.ncols).some
    }

  // safe get of an column from the matrix
  def getCol[A](c: Int)(m: Matrix[A]) : Option[Vector[A]] =
    if (m.ncols < c || 1 > c) none else {
     (for {
        r ← 0 until m.nrows
      } yield m.mvect( encode(m.vcols)(r+1+m.rowOffset,c+m.colOffset))).toVector.some
    }

  def getDiag[A](m: Matrix[A]) : Vector[A] = {
    val s = scala.math.min(m.nrows,m.ncols)
    val v = collection.mutable.ListBuffer.fill[A](s)(0.asInstanceOf[A])
    for { i ← 0 until s } getElem(i+1)(i+1)(m).map(e ⇒ v.update(i, e))
    Vector(v:_*)
  }

  def getMatrixAsVector[A] = (m: Matrix[A]) ⇒ unsafeMatrix(m).mvect

  def setElem[A](ele: A)(position: (Int,Int))(m: Matrix[A]) : Matrix[A] =
    m.copy(mvect = m.mvect.updated(encode(m.ncols)(position._1,position._2), ele))

  def transpose[A](m: Matrix[A]) : Matrix[A] = matrix(m.ncols)(m.nrows)((i:Int, j: Int) ⇒ unsafeGet(j, i)(m))

  @inline
  def submatrix[A](r1: Int, r2: Int, c1: Int, c2: Int)(m: Matrix[A]) : \/[String, Matrix[A]] = {
    if (r1 < 1 || r1 > m.nrows) "Invalid rows requested".left else
    if (r2 < 1 || r2 < r1) "Second row requested cannot be smaller than first row".left else
    if (c1 < 1 || c1 > m.ncols) "Invalid columns requested".left else
    if (c2 < 1 || c2 < c1) "Second column requested cannot be smaller than first column".left else
    m.copy(nrows = r2-r1+1, ncols = c2-c1+1, rowOffset = m.rowOffset + r1 -1, colOffset = m.colOffset + c1 - 1).right
  }

  /*
   * Remove a row and a column from a matrix.
   * @param r row to remove
   * @param c column to remove
   * @return matrix with (r,c) removed
   */
  def minorMatrix[A](r: Int)(c: Int)(m: Matrix[A]) : Matrix[A] = {
    val _r = r + m.rowOffset
    val _c = c + m.colOffset
    m.copy(
      nrows = m.nrows-1,
      ncols = m.ncols-1,
      vcols = m.vcols-1,
      mvect = m.mvect.zipWithIndex.filter(p ⇒ { val (i,j) = decode(m.vcols)(p._2); (_r != i && _c != j) }).collect{ case (e,idx) ⇒ e }
    )
  }

   /**
    * Splits the given block into 4 sub-blocks.
    * @param r row of the splitting element
    * @param c col of the splitting element
    * @param m matrix in question
    * @return a 4-tuple (i,j,k,l) where i = top-left matrix, j = top-right
    *         matrix, k = bottom-left matrix, l = bottom-right matrix
    */
  @inline
  def splitBlocks[A](r: Int)(c: Int)(m: Matrix[A]) : \/[String, (Matrix[A],Matrix[A],Matrix[A],Matrix[A])] = {
    for {
      tl ← submatrix(1  , r      , 1  , c      )(m)
      tr ← submatrix(1  , r      , c+1, m.ncols)(m)
      bl ← submatrix(r+1, m.nrows, 1  , c      )(m)
      br ← submatrix(r+1, m.nrows, c+1, m.ncols)(m)
    } yield (tl, tr, bl, br)
  }

  /**
    * Primarily meant for joining equally sized blocks and potentially
    * parallelizable.
    * @param 4-tuple of equally sized blocks
    * @return matrix where the (4-tuple) blocks are joined 
    */
  @inline
  def joinBlocks[A](blocks: (Matrix[A],Matrix[A],Matrix[A],Matrix[A])): Matrix[A] = {
    val tl = blocks._1
    val tr = blocks._2
    val bl = blocks._3
    val br = blocks._4
    val R = tl.nrows + bl.nrows
    val C = tl.ncols + tr.ncols
    val en = encode(C)
    val mutV = collection.mutable.ListBuffer.fill[A](R*C)(0.asInstanceOf[A])
    for {
      i ← 1 to tl.nrows
      j ← 1 to tl.ncols
    } {
     mutV.update(en(i,j), getElem(i)(j)(tl).getOrElse(0.asInstanceOf[A])) 
    }
    for {
      i ← 1 to tl.nrows
      j ← 1 to tr.ncols
    } {
     mutV.update(en(i,j+tl.ncols), getElem(i)(j)(tr).getOrElse(0.asInstanceOf[A])) 
    }
     for {
      i ← 1 to bl.nrows
      j ← 1 to tl.ncols
    } {
     val ii = i + tl.nrows
     mutV.update(en(ii,j), getElem(i)(j)(bl).getOrElse(0.asInstanceOf[A])) 
    }
    for {
      i ← 1 to bl.nrows
      j ← 1 to tr.ncols
    } {
     val ii = i + tl.nrows
     mutV.update(en(ii,j+tl.ncols), getElem(i)(j)(br).getOrElse(0.asInstanceOf[A])) 
    }

    fromList(R)(C)(mutV.toList)
  }

  /*
   * Scale a matrix by a given factor.
   * @param factor what factor to scale
   * @param m matrix to scale
   * @return matrix where values are scaled.
   */
  def scaleMatrix[A:Numeric](factor: A)(m: Matrix[A])(implicit F: Functor[Matrix]) : Matrix[A] = {
    val num = implicitly[Numeric[A]]
    F.map(m)((x: A) ⇒ num.times(x, factor))
  }

  /*
   * Scale a row by a given factor.
   * @param factor what factor to scale
   * @param r row to scale
   * @return matrix with the row "r" scaled by "factor"
   */
  def scaleRow[A:Numeric](factor: A)(r: Int)(m: Matrix[A]) : Matrix[A] = {
    val M = implicitly[Numeric[A]]
    mapRow((i: Int, x:A) ⇒ M.times(x, factor))(r)(m)
  }

  /*
   * Add to one row a scalar multiple of another row.
   * @param r1 lower row bound
   * @param l value to scale
   * @param r2 upper bound
   * @return matrix with the scalar multiple applied
   */
  def combineRows[A:Numeric](r1: Int, l: A, r2: Int)(m: Matrix[A]) : Matrix[A] = {
    val M = implicitly[Numeric[A]]
    mapRow((j:Int,x:A) ⇒ M.plus(M.times(getElem(r2)(j)(m).get, l), x))(r1)(m)
  }

  /*
   * Switch two rows of a matrix.
   * @param r1 lower bound
   * @param r2 upper bound
   * @return matrix with the rows r1 and r2 swapped.
   */
  def switchRows[A](r1: Int)(r2: Int)(m: Matrix[A]) : Matrix[A] = {
    assert(r1 <= m.nrows && r1 >= 1 && r2 >= 1, "Out of bounds")
    assert(r1 <= r2, "You have requested invalid bounds.")
    val mutV = collection.mutable.ListBuffer(m.mvect: _*)
    for {
      c ← 0 until m.vcols
    } {
      val e = mutV((r1-1)*m.ncols+c)
      mutV.update((r1-1)*m.ncols+c, mutV((r2-1)*m.ncols+c))
      mutV.update((r2-1)*m.ncols+c, e)
    }
    m.copy(mvect = Vector(mutV:_*))
  }

  /*
   * Switch two columns of a matrix.
   *  Example:
   * 
   *                 ( 1 2 3 )   ( 2 1 3 )
   *                 ( 4 5 6 )   ( 5 4 6 )
   *  switchCols 1 2 ( 7 8 9 ) = ( 8 7 9 )
   * @param c1 lower bound column
   * @param c2 upper bounc column
   * @return returns the matrix where c1 and c2 are swapped
   */
  def switchCols[A:scala.reflect.ClassTag](c1: Int)(c2: Int)(m: Matrix[A]) : Matrix[A] = {
    assert(c1 <= m.ncols , "Out of bounds")
    assert(c2 <= m.ncols && c1 < c2, "Out of bounds or you have requested invalid bounds.")
    val mutV = m.mvect.toArray
    for {
      j ← 1 to m.nrows
    } {
      val e = mutV(encode(m.ncols)(j+m.rowOffset,c1+m.colOffset))
      mutV.update(encode(m.ncols)(j+m.rowOffset,c1+m.colOffset), mutV(encode(m.ncols)(j+m.rowOffset,c2+m.colOffset)))
      mutV.update(encode(m.ncols)(j+m.rowOffset,c2+m.colOffset), e)
    }
    m.copy(mvect = Vector(mutV:_*))
  }

  /** 
    * Standard matrix-matrix multiplication
    * @param m
    * @paran n
    * @return
    */
  def mult[@specialized (Float,Double,Int) A:Numeric](m : Matrix[A], n : Matrix[A]) : Matrix[A] = {
    assert(m.ncols == n.nrows, "you cannot have non-compatible matrices")
    val M = implicitly[Numeric[A]]
    matrix(m.nrows)(n.ncols)((i,j) ⇒ 
      getCol(j)(n).map(
        col ⇒ getRow(i)(m).map(
                row ⇒ row.zip(col).map(p ⇒ M.times(p._1 , p._2))
              )
      ).flatten.get.reduce((a,b) ⇒ M.plus(a, b))
    )
  }

  def -[A:Numeric](m: Matrix[A])(n: Matrix[A]) : Matrix[A] = elementwise((x:A) ⇒ (y: A) ⇒ implicitly[Numeric[A]].minus(x, y))(m)(n)
  def +[A:Numeric](m: Matrix[A])(n: Matrix[A]) : Matrix[A] = elementwise((x:A) ⇒ (y: A) ⇒ implicitly[Numeric[A]].plus(x, y))(m)(n)

  def elementwise[A:Numeric,B:Numeric,C:Numeric](f: A ⇒ B ⇒ C)(m: Matrix[A])(n: Matrix[B]) : Matrix[C] = {
    matrix(m.nrows)(m.nrows)((x,y) ⇒ (getElem(x)(y)(m) |@| getElem(x)(y)(n))((x1,y1) ⇒ f(x1)(y1)).get)
  }

  /**
    * Volker Strassen's matrix multiplication https://en.wikipedia.org/wiki/Strassen_algorithm
    * which is clever considering that it reduces 1 matrix-matrix
    * multiplication (performs 7 instead of the usual 8).
    * @param m matrix
    * @param n matrix
    * @return applies strassen's algorithm
    */
  def multStrassen[@specialized (Float,Double,Int) A:Numeric](m: Matrix[A], n: Matrix[A]) : \/[String,Matrix[A]] = {
    def find(v: Double, threshold: Double) : Stream[Int] = 
      if (scala.math.pow(2,v) >= threshold) Stream(scala.math.pow(2,v).toInt) else find(v+1, threshold)

    if (m.ncols != n.nrows) "Non-compatible matrices".left[Matrix[A]]
    else {
      val max = List(m.nrows,m.ncols,n.nrows,n.ncols).reduce(scala.math.max)
      val n2 = find(0.0, max.toDouble).head
      val b1 : Matrix[A] = setSize(0.asInstanceOf[A])(n2)(n2)(m)
      val b2 : Matrix[A] = setSize(0.asInstanceOf[A])(n2)(n2)(n)
      submatrix(1, m.nrows, 1, n.ncols)(strassen(b1)(b2))
    }
  }

  private
  def setSize[A](e: A)(r:Int)(c:Int)(m: Matrix[A]) : Matrix[A] =
    matrix(r)(c)((i,j) ⇒ if (i <= m.nrows && j <= m.ncols) unsafeGet(i,j)(m) else e)

  // Strassen's algorithm over square matrices of order 2^n.
  private
  def strassen[@specialized (Float,Double,Int) A:Numeric](m: Matrix[A])(n: Matrix[A]) : Matrix[A] = {
    val M = implicitly[Numeric[A]]
    if (m.nrows == 1 && m.ncols == 1 && n.nrows == 1 && n.ncols == 1)
      matrix(1)(1)((i,j) ⇒ (getElem(1)(1)(m) |@| getElem(1)(1)(n))((x,y) ⇒ M.times(x,y)).get)
    else {
      val x = m.nrows / 2
      (for {
        abcd              ← splitBlocks(x)(x)(m)
        (a11,a12,a21,a22) = abcd
        defg              ← splitBlocks(x)(x)(n)
        (b11,b12,b21,b22) = defg
        p1                ← strassen( (+)(a11)(a22) )( (+)(b11)(b22) ).right[String]
        p2                ← strassen( (+)(a21)(a22) )( b11           ).right[String]
        p3                ← strassen( a11           )( (-)(b12)(b22) ).right[String]
        p4                ← strassen( a22           )( (-)(b21)(b11) ).right[String]
        p5                ← strassen( (+)(a11)(a12) )( b22           ).right[String]
        p6                ← strassen( (-)(a21)(a11) )( (+)(b11)(b12) ).right[String]
        p7                ← strassen( (-)(a12)(a22) )( (+)(b21)(b22) ).right[String]
      } yield {
        val c11 = (+)((-)((+)(p1)(p4))(p5))(p7) // p1 + p4 - p5 + p7
        val c12 = (+)(p3)(p5)                   // p3 + p5
        val c21 = (+)(p2)(p4)                   // p2 + p4
        val c22 = (+)((+)((-)(p1)(p2))(p3))(p6) // p1 - p2 + p3 + p6
        joinBlocks( (c11, c12, c21, c22) )
      }).getOrElse(m)
    }
  }

  /**
    * LU decomposition where the pivot strategy is picking the maximum value of
    * the current row of "coefficients".
    * @return if a LU decomposition exists, then it would Some(a,b,c) where
    * a,b,c ∈ matrices else a None
    */
  def lu[@specialized (Float,Double,Int) A:scalaz.Order : Fractional](m : Matrix[A]) : Option[(Matrix[A], Matrix[A], Matrix[A], A)] = {
    val i = identity(m.nrows)
    val n = scala.math.min(m.nrows, m.ncols)
    reclu(m, i, i, 1, 1, n)
  }

  private
  def reclu[A:scalaz.Order:Fractional](u : Matrix[A], l: Matrix[A], p: Matrix[A], d: Int, currentRow: Int, totalRows: Int) : Option[(Matrix[A], Matrix[A], Matrix[A], A)] = {

    if (currentRow > totalRows) Some((u,l,p,d.asInstanceOf[A])) else {
      val F = implicitly[scalaz.Foldable[List]]
      val i = F.maximumBy((currentRow to totalRows).toList)((col:Int) ⇒ implicitly[Numeric[A]].abs(getElem(col)(currentRow)(u).get)).get
      val uu = switchRows(currentRow)(i)(u) // switch rows 
      val ll = {
        val lw = l.vcols
        val en = encode(lw)
        val lro = l.rowOffset
        val lco = l.colOffset
        if (i == currentRow) l else {
          val mutV = collection.mutable.ListBuffer(l.mvect: _*)
          for {
            idx ← 1 to (currentRow-1)
          } {
              val t = mutV( en(i+lro, idx+lco) ) 
              mutV(en(i+lro, idx+lco)) = mutV(en(currentRow+lro, idx+lco))
              mutV(en(currentRow+lro, idx+lco)) = t
          }
          fromList(l.nrows)(l.ncols)(mutV.toList).copy(rowOffset = lro, colOffset = lco, vcols = lw)
        }
      }
      val pp = switchRows(currentRow)(i)(p)
      val dd = if (i == currentRow) d else -d // next row to process
      val ukk = getElem(currentRow)(currentRow)(uu).get
      
      def process(upper: Matrix[A], lower: Matrix[A], i: Int) : (Matrix[A], Matrix[A]) = 
        if (i > upper.nrows) (upper, lower) else {
          val x = implicitly[Fractional[A]].div(getElem(i)(currentRow)(upper).get,ukk)
          process(combineRows(i, implicitly[Numeric[A]].negate(x), currentRow)(upper),
                  setElem(x)(i,currentRow)(lower), 
                  i+1)
        }

      val (uuu,lll) = process(uu, ll, currentRow+1)

      if (ukk == 0) none else reclu(uuu,lll,pp,dd,currentRow+1,totalRows)
    }
  }

  /**
    * Cholesky decomposition
    * @param a matrix of real numbers
    * @return result of applying cholesky's decomposition
    */
  def cholesky(m : Matrix[Double]) : Matrix[Double] = {
    import scala.math.sqrt
    if (m.nrows == 1 && m.ncols == 1) implicitly[Functor[Matrix]].map(m)(e ⇒ sqrt(e))
    else {
      val xs = splitBlocks(1)(1)(m).toList.head
      val (a11, a12, a21, a22) = splitBlocks(1)(1)(m).toList.head
      val l111 = sqrt(getElem(1)(1)(a11).get)
      val l11  = fromList(1)(1)(l111::Nil)
      val l12  = zero(a12.nrows)(a12.ncols)
      val l21  = scaleMatrix(1/l111)(a21)
      val a222 = (-)(a22)(mult(l21, transpose(l21)))
      val l22  = cholesky(a222)
      joinBlocks((l11, l12, l21, l22))
    }
  }

  def isSquare[A](m : Matrix[A]) : Boolean = m.nrows == m.ncols

  def diagonalAsProduct[A:Numeric](m: Matrix[A]) : A = {
    val mm = implicitly[Numeric[A]]
    if (!isSquare(m)) mm.negate(mm.one)
    else (for {
      i ← 1 to m.nrows
    } yield getElem(i)(i)(m)).toList.sequence.fold(mm.zero)(xs ⇒ xs.reduce(mm.times(_,_)))
  }

  /**
    * Laplace Expansion for discovering the determinant
    * @param m matrix
    * @return some kind of number which is the determinant or none
    */
  def laplaceDet[A:Numeric](m: Matrix[A]) : Option[A] = {
    val mm = implicitly[Numeric[A]]
    if (m.nrows == 1 && m.ncols == 1) getElem(1)(1)(m)
    else 
      (for {
        i ← 1 to m.nrows
      } yield {
        laplaceDet(minorMatrix(i)(1)(m)).map(
          (e:A) ⇒ mm.times(mm.times(scala.math.pow(mm.toDouble(mm.negate(mm.one)), i-1).asInstanceOf[A], unsafeGet(i,1)(m)) ,e))
      }).toList.sequence.fold(mm.zero)(xs ⇒ xs.reduce(mm.plus(_,_))).some
    
  }


  /**
    * Using the LU-decomposition to obtain the matrix's determinant.
    * @param m matrix 
    * @return the determinant of the matrix using LU-decomposition
    */
  def luDet[@specialized (Float,Double,Int) A:scalaz.Order:Fractional](m: Matrix[A]) : A = {
    lu(m) match {
      case Some((u,_,_,d)) ⇒
        val N = implicitly[Fractional[A]]
        (BigDecimal(d.asInstanceOf[Int]) * BigDecimal(N.toDouble(diagonalAsProduct(u)))).asInstanceOf[A]
      case None            ⇒
        val N = implicitly[Numeric[A]]
        N.zero
    }
  }

}

object ops extends MatrixOps

