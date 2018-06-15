package object m {

  trait Base {

    def getRow[A](r: Int)(m: Matrix[A]) : Option[Vector[A]]
    def fromList[A](n: Int)(m: Int)(xs: List[A]) : Matrix[A]
    def unsafeMatrix[A](m: Matrix[A]) : Matrix[A]
    def matrix[@specialized(Int,Double,Float) A](n: Int)(m: Int)(f: (Int, Int) ⇒ A) : Matrix[A]
    def <->[A](m: Matrix[A])(n: Matrix[A]) : Matrix[A]
    def <|>[A](m: Matrix[A])(n: Matrix[A]) : Matrix[A]

  }

}
