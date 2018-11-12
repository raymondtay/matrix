import xerial.sbt.Sonatype._
name := "matrix"

version := "0.5"

scalaVersion := "2.12.6"

libraryDependencies += "org.scalaz" %% "scalaz-core" % "7.2.21"

resolvers += Resolver.sonatypeRepo("releases")

addCompilerPlugin("org.spire-math" %% "kind-projector" % "0.9.6")

scalacOptions ++= Seq("-deprecation", "-feature", "-language:postfixOps")

// if your project uses multiple Scala versions, use this for cross building
addCompilerPlugin("org.spire-math" % "kind-projector" % "0.9.6" cross CrossVersion.binary)

enablePlugins(TutPlugin)
enablePlugins(MicrositesPlugin)
enablePlugins(JmhPlugin)

// if your project uses both 2.10 and polymorphic lambdas
libraryDependencies ++= (scalaBinaryVersion.value match {
  case "2.10" =>
    compilerPlugin("org.scalamacros" % "paradise" % "2.1.0" cross CrossVersion.full) :: Nil
  case _ =>
    Nil
})

publishTo := sonatypePublishTo.value
sonatypeProfileName := "com.github.raymondtay"
publishMavenStyle := true
licenses := Seq("MIT" -> url("https://github.com/raymondtay/matrix/LICENSE"))
sonatypeProjectHosting := Some(GitHubHosting("raymondtay", "matrix", "raymond.tay@yahoo.com"))
homepage := Some(url("https://github.com/raymondtay/matrix/wiki"))
scmInfo := Some(
  ScmInfo(
    url("https://github.com/raymondtay/matrix"),
    "scm:git@github.com:raymondtay/matrix.git"
  )
)
developers := List(
  Developer(id="tayboonl",
            name="Raymond Tay",
            email="raymond.tay@yahoo.com",
            url=url("https://twitter.com/RaymondTayBL"))
)

useGpg := true
publishConfiguration := publishConfiguration.value.withOverwrite(true)
publishLocalConfiguration := publishLocalConfiguration.value.withOverwrite(true)

pgpPublicRing := file("/Users/raymondtay/.sbt/gpg/pubring.asc")

