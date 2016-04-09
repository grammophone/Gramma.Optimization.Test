using System;
using System.Text;
using System.Collections.Generic;
using System.Linq;
using Microsoft.VisualStudio.TestTools.UnitTesting;

using Grammophone.Vectors;
using Grammophone.Optimization;

namespace OptimizationTest
{
	/// <summary>
	/// Summary description for UnitTest1
	/// </summary>
	[TestClass]
	public class KrylovSolverTest
	{
		#region Construction

		public KrylovSolverTest()
		{
			//
			// TODO: Add constructor logic here
			//
		}

		#endregion

		#region Private fields

		private TestContext testContextInstance;

		private static double ε = 1e-5;

		#endregion

		#region Public properties

		/// <summary>
		///Gets or sets the test context which provides
		///information about and functionality for the current test run.
		///</summary>
		public TestContext TestContext
		{
			get
			{
				return testContextInstance;
			}
			set
			{
				testContextInstance = value;
			}
		}

		#endregion

		#region Additional test attributes
		//
		// You can use the following additional attributes as you write your tests:
		//
		// Use ClassInitialize to run code before running the first test in the class
		// [ClassInitialize()]
		// public static void MyClassInitialize(TestContext testContext) { }
		//
		// Use ClassCleanup to run code after all tests in a class have run
		// [ClassCleanup()]
		// public static void MyClassCleanup() { }
		//
		// Use TestInitialize to run code before running each test 
		// [TestInitialize()]
		// public void MyTestInitialize() { }
		//
		// Use TestCleanup to run code after each test has run
		// [TestCleanup()]
		// public void MyTestCleanup() { }
		//
		#endregion

		[TestMethod]
		public void IdentitySystem()
		{
			var A = Vector.GetTensor(
				(i, j) => i == j ? 1.0 : 0.0
			);

			var b = new Vector(new double[] { 1, 2, 3 });

			var x = KrylovSolver.LinearSolve(
				A,
				-b,
				new Vector(b.Length),
				new KrylovSolver.LinearSolveOptions());

			Assert.IsTrue((x - b).Norm2 < ε, "x should be equal to b.");

		}

		[TestMethod]
		public void Trivial()
		{
			var A = Vector.GetTensor(
				(i, j) => 1.0
			);

			var b = new Vector(new double[] { 3 });

			var x = KrylovSolver.LinearSolve(
				A,
				-b,
				new Vector(b.Length),
				new KrylovSolver.LinearSolveOptions(),
				b.Length);

			Assert.IsTrue((x - b).Norm2 < ε, "x should be equal to b.");

		}

		[TestMethod]
		public void ThreeByThree()
		{
			var m = new double[,]
			{
				{4, 2, 2},
				{2, 4, 2},
				{2, 2, 4}
			};

			var A = Vector.GetTensor(m);

			var B = new Vector(new double[] { 14, 16, 18 });

			var X = KrylovSolver.LinearSolve(
				A,
				-B,
				new Vector(B.Length),
				new KrylovSolver.LinearSolveOptions());

			var Xc = new Vector(new double[] { 1, 2, 3 });

			Assert.IsTrue((X - Xc).Norm2 < ε, "Expected sulution was not found.");

		}

		[TestMethod]
		public void TwoByTwo()
		{
			var A = Vector.GetTensor(new double[,]
			{
				{3, 1},
				{1, 2}
			});

			var B = new Vector(new double[] { 11, 12 });

			var X = KrylovSolver.LinearSolve(
				A,
				-B,
				new Vector(B.Length),
				new KrylovSolver.LinearSolveOptions());

			var Xc = new Vector(new double[] { 2, 5 });

			Assert.IsTrue((X - Xc).Norm2 < ε, "Expected sulution was not found.");
		}
		
	}
}
