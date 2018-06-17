package m.bench

import m.ops

import org.openjdk.jmh.annotations.Benchmark

class MatrixStd {

  import ops._

  @Benchmark
  def createMatrices = matrix(1000)(1000)((i, j) => i+j)

}
