using System;
using System.Text;
using System.Collections.Generic;
using System.Linq;
using Microsoft.VisualStudio.TestTools.UnitTesting;

using Grammophone.Linq;
using Grammophone.Vectors;
using Grammophone.Vectors.ExtraExtensions;
using Grammophone.Optimization;

namespace OptimizationTest
{
	[TestClass]
	public class LineSearchOptimizationTest
	{
		private static double ε = 1e-5;

		[TestMethod]
		public void BiQuadratic()
		{
			ScalarFunction f =
				x => 0.5 * ((x[0] - 0.5) * (x[0] - 0.5) + (x[1] - 0.8) * (x[1] - 0.8));

			VectorFunction df =
				x => new Vector(new double[] { x[0] - 0.5, x[1] - 0.8 });

			var w0 = new Vector(new double[] { 1.6, -0.2 });

			var w = ConjugateGradient.LineSearchMinimize(
				f,
				df,
				w0,
				new ConjugateGradient.LineSearchMinimizeOptions());

			var we = new Vector(new double[] { 0.5, 0.8 });

			Assert.IsTrue((we - w).Norm2 < ε, "Solution should have been [0.5 0.8].");
		}

		[TestMethod]
		public void MultiQuadratic()
		{
			int N = 1024;

			var range = Enumerable.Range(0, N);

			ScalarFunction f =
				x =>
					0.5 * range.Sum(i => (x[i] - 0.1 * (double)i) * (x[i] - 0.1 * (double)i));

			VectorFunction df =
				x =>
					range.Select(i => x[i] - 0.1 * (double)i);

			var w0 = new Vector(N);

			var w = ConjugateGradient.LineSearchMinimize(
				f,
				df,
				w0,
				new ConjugateGradient.LineSearchMinimizeOptions());

			Vector we =
				range.Select(i => 0.1 * (double)i);

			Assert.IsTrue((we - w).Norm2 < ε, "Solution should have been [0.1i].");
		}

		[TestMethod]
		public void MultiQuartic()
		{
			int N = 1024;

			var range = Enumerable.Range(0, N);

			ScalarFunction f =
				x =>
					0.25 * range.Sum(i => Math.Pow(x[i] - 0.1 * (double)i, 2));

			VectorFunction df =
				x =>
					range.Select(i => Math.Pow(x[i] - 0.1 * (double)i, 3));

			var w0 = new Vector(N);

			var w = ConjugateGradient.LineSearchMinimize(
				f,
				df,
				w0,
				new ConjugateGradient.LineSearchMinimizeOptions()
				{
					StopCriterion = ConjugateGradient.LineSearchMinimizeOptions.GetGradientNormCriterion(1e-13),
					LineSearchThreshold = 1e-4
				});

			Vector we =
				range.Select(i => 0.1 * (double)i);

			Assert.IsTrue((we - w).Norm2 < 0.01 * w.Length, "Solution should have been [0.1i].");
		}

		[TestMethod]
		public void SimpleConstrained()
		{
			ScalarFunction f = // Goal function.
				x => 0.5 * ((x[0] - 0.5).Squared() + (x[1] - 0.8).Squared());

			// Gradient of the goal function.
			// Shows implicit conversion from double[] to Vector.
			VectorFunction df = 
				x => new double[] { x[0] - 0.5, x[1] - 0.8 };

			int constraintsCount = 1;

			Func<int, ScalarFunction> fc = // Constraints fc(i)(x) <= 0 
				i =>
					x =>
						x[0];

			// Gradients of the constraints.
			// Implicit conversion from double[] to Vector again.
			Func<int, VectorFunction> dfc = 
				i =>
					x =>
						new double[] { 1, 0 };


			var options = // Solver options.
				new ConjugateGradient.LineSearchConstrainedMinimizeOptions
				{
					DualityGap = 1e-7,
					BarrierScaleFactor = 10.0,
					BarrierInitialScale = 100.0
				};

			// Initial point.
			var w0 = new Vector(new double[] { -5, 1 });

			// Solution certificate. 
			// Contains the optimum point found and the Lagrange multipliers.
			var certificate = ConjugateGradient.LineSearchConstrainedMinimize(
				f,
				df,
				w0,
				constraintsCount,
				fc,
				dfc,
				options);

			// Should we trust the solution?
			// Let us form the Lagrange dual problem, 
			// the infimum g of the Lagrangian L over x,
			// then compare the primal goal at the optimum found
			// against g at the lagrange multipliers vector λ found.
			// The difference should bebelow the specified duality gap.
			var constraintsRange = Enumerable.Range(0, constraintsCount);

			Func<Vector, Vector, double> L = // The Lagrangian.
				(x, λ) =>
					f(x) + constraintsRange.Sum(i => λ[i] * fc(i)(x));

			ScalarFunction g = // g is the infimum of L over x.
				λ =>
					L(new double[] { 0.5 - λ[0], 0.8 }, λ);

			var dualityGap = f(certificate.Optimum) - g(certificate.λ);

			Assert.IsTrue(
			  dualityGap < options.DualityGap, 
			  "Duality gap too big.");

			// What we expect.
			Vector we = new double[] { 0.0, 0.8 };

			Assert.IsTrue(
				(we - certificate.Optimum).Norm2 < ε, 
				"Solution should have been [0.0 0.8].");
		}

		[TestMethod]
		public void ConstrainedQuadratic()
		{
			var g = new Vector(new double[] { -1, -1, -1, -1, -1 });

			var gramMatrix = 
				new double[,]
				{
					{ 3, 1, 3, 0, 1 },
					{ 1, 3, 3, 0, 1 },
					{ 3, 3, 5, 1, 3 },
					{ 0, 0, 1, 2, 3 },
					{ 1, 1, 3, 3, 5 }
				};

			var Q = Vector.GetTensor(gramMatrix);

			ScalarFunction f = // Goal function.
				x => 0.5 * x * Q(x) + g * x;

			VectorFunction df = // Gradient of the goal function.
				x => Q(x) + g;

			TensorFunction d2f = // Hessian of the goal function.
				x =>
					y =>
						Q(y);

			int P = g.Length;

			var range = Enumerable.Range(0, P);

			VectorFunction d2fd = // Diagonal of the Hessian of the goal.
				x =>
					range.Select(i => gramMatrix[i, i]);

			int constraintsCount = 2 * P;

			// Constraint functions: 0 <= fc(i)(x) <= 10
			Func<int, ScalarFunction> fc =
				i =>
					x =>
						i < P ? 
						-x[i] : 
						x[i - P] - 10.0;

			Func<int, VectorFunction> dfc = // Gradients of constraints.
				i =>
					x =>
						i < P ?
						range.Select(j => (j == i) ? -1.0 : 0.0) :
						range.Select(j => (j == i - P) ? 1.0 : 0.0);

			Func<int, TensorFunction> d2fc =
				i =>
					x =>
						Vector.ZeroTensor;

			var zeroVector = new Vector(P);

			Func<int, VectorFunction> d2fcd = // Diagonals of the constraints Hessians.
				i =>
					x => zeroVector;

			// Unility function that returns Jacobi 
			// preconditioner Tensor function.
			var M = ConjugateGradient.ConstrainedMinimizeOptions.GetJacobiPreconditioner(
				d2fd,
				P,
				fc,
				dfc,
				d2fcd);

			var certificate = ConjugateGradient.LineSearchConstrainedMinimize(
				f,
				df,
				new Vector(new double[] { 0.5, 0.5, 0.5, 0.5, 0.5 }), // Initial point.
				constraintsCount,
				fc,
				dfc,
				new ConjugateGradient.LineSearchConstrainedMinimizeOptions
				{
					DualityGap = 1e-10
				},
				M);


			// What we expect.
			var xe = new Vector(new double[] { 0.25, 0.25, 0.0, 0.5, 0.0 });

			Assert.IsTrue(
				(certificate.Optimum - xe).Norm2 < ε, 
				String.Format(
					"Should have found {0}, found {1} instead", 
					xe, 
					certificate.Optimum));
		}

		[TestMethod]
		public void ManualConstrainedQuadratic()
		{
			var g = new Vector(new double[] { -1, -1, -1, -1, -1 });

			int P = g.Length;

			var range = Enumerable.Range(0, P);

			var gramMatrix =
				new double[,]
				{
					{ 3, 1, 3, 0, 1 },
					{ 1, 3, 3, 0, 1 },
					{ 3, 3, 5, 1, 3 },
					{ 0, 0, 1, 2, 3 },
					{ 1, 1, 3, 3, 5 }
				};

			var Q = Vector.GetTensor(gramMatrix);

			ScalarFunction f =
				x => 0.5 * x * Q(x) + g * x;

			VectorFunction df = // Gradient of the goal function.
				x => Q(x) + g;

			TensorFunction d2f = // Hessian of the goal function.
				x =>
					y =>
						Q(y);

			VectorFunction d2fd = // Diagonal of the Hessian of the goal.
				x =>
					range.Select(i => gramMatrix[i, i]);

			ScalarFunction φ =
				x =>
					- x.Sum(xi => Math.Log(xi) + Math.Log(10.0 - xi));

			VectorFunction dφ =
				x =>
					-x.Select(xi => 1.0 / xi + 1.0 / (xi - 10));

			int constraintsCount = 2 * P;

			var constraintsRange = Enumerable.Range(0, constraintsCount);

			Func<double, VectorFunction> λ =
				t =>
					x =>
						constraintsRange.Select(i =>
							i < P ?
							1.0 / (t * x[i]) :
							1.0 / (t * (10.0 - x[i - P]))
						);

			Func<Vector, bool> outOfDomainIndicator =
				x =>
					x.Any(xi => xi < 0 || xi > 10.0);

			Func<int, ScalarFunction> fc =
				i =>
					x =>
						i < P ?
						-x[i] :
						x[i - P] - 10.0;

			Func<int, VectorFunction> dfc =
				i =>
					x =>
						i < P ?
						range.Select(j => (j == i) ? -1.0 : 0.0) :
						range.Select(j => (j == i - P) ? 1.0 : 0.0);

			Func<int, TensorFunction> d2fc =
				i =>
					x =>
						Vector.ZeroTensor;

			Func<int, VectorFunction> d2fcd =
				i =>
					x => new Vector(P);

			// Hessian diagonal of the total unconstrained goal tf + φ
			VectorFunction d2φd =
				x =>
					x.Select(xi => 1.0 / (xi * xi) + 1.0 / ((xi - 10.0) * (xi - 10.0)));

			ConjugateGradient.ConstrainedMinimizePreconditioner M =
				t =>
					x =>
						Vector.GetDiagonalTensor((t * d2fd(x) + d2φd(x)).Select(Hii => 1.0 / Hii));

			var certificate = ConjugateGradient.LineSearchConstrainedMinimize(
				f,
				df,
				new Vector(new double[] { 0.5, 0.5, 0.5, 0.5, 0.5 }),
				φ,
				dφ,
				λ,
				new ConjugateGradient.LineSearchConstrainedMinimizeOptions()
				{
					DualityGap = 1e-10
				},
				outOfDomainIndicator,
				M);

			var xe = new Vector(new double[] { 0.25, 0.25, 0.0, 0.5, 0.0 });

			Assert.IsTrue(
				(certificate.Optimum - xe).Norm2 < ε,
				String.Format("Should have found {0}, found {1} instead", xe, certificate.Optimum));
		}
	}
}
