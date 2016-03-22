using System;
using System.Text;
using System.Collections.Generic;
using System.Linq;
using Microsoft.VisualStudio.TestTools.UnitTesting;

using Gramma.Vectors;
using Gramma.Vectors.ExtraExtensions;
using Gramma.Optimization;

namespace OptimizationTest
{
	[TestClass]
	public class TruncatedNewtonTest
	{
		private static double ε = 1e-5;

		[TestMethod]
		public void BiQuadratic()
		{
			ScalarFunction f =
				x => 0.5 * ((x[0] - 0.5) * (x[0] - 0.5) + (x[1] - 0.8) * (x[1] - 0.8));

			VectorFunction df =
				x => new Vector(new double[] { x[0] - 0.5, x[1] - 0.8 });

			TensorFunction d2f =
				x => Vector.GetTensor(
					new double[,]
					{
						{ 1.0, 0.0 },
						{ 0.0, 1.0 }
					}
				);

			var w0 = new Vector(new double[] { -5, 1.3 } );

			var w = ConjugateGradient.TruncatedNewtonMinimize(
				df,
				d2f,
				w0,
				x => false,
				new ConjugateGradient.TruncatedNewtonMinimizeOptions());

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

			TensorFunction d2f = 
				x => Vector.IdentityTensor;

			var w0 = new Vector(N);

			var w = ConjugateGradient.TruncatedNewtonMinimize(
				df,
				d2f,
				w0,
				x => false,
				new ConjugateGradient.TruncatedNewtonMinimizeOptions());

			Vector we =
				range.Select(i => 0.1 * (double)i);

			Assert.IsTrue((we - w).Norm2 < ε, "Solution should have been [0.1i].");
		}

		[TestMethod]
		public void SimpleConstrained()
		{
			ScalarFunction f =
				x => 0.5 * ((x[0] - 0.5) * (x[0] - 0.5) + (x[1] - 0.8) * (x[1] - 0.8));

			VectorFunction df =
				x => new Vector(new double[] { x[0] - 0.5, x[1] - 0.8 });

			TensorFunction d2f =
				x => Vector.IdentityTensor;

			int constraintsCount = 1;

			Func<int, ScalarFunction> fc =
				i =>
					x =>
						x[0];

			Func<int, VectorFunction> dfc =
				i =>
					x =>
						new Vector(new double[] { 1, 0 });

			Func<int, TensorFunction> d2fc =
				i =>
					x =>
						Vector.ZeroTensor;

			var w0 = new Vector(new double[] { -5, 1 });

			var options = 
				new ConjugateGradient.TruncatedNewtonConstrainedMinimizeOptions
				{
					DualityGap = 1e-7
				};

			var certificate = ConjugateGradient.TruncatedNewtonConstrainedMinimize(
				df,
				d2f,
				w0,
				constraintsCount,
				fc,
				dfc,
				d2fc,
				options);

			var constraintsRange = Enumerable.Range(0, constraintsCount);

			Func<Vector, Vector, double> L =
				(x, λ) =>
					f(x) + constraintsRange.Sum(i => λ[i] * fc(i)(x));

			ScalarFunction g =
				λ =>
					L(new Vector(new double[] { 0.5 - λ[0], 0.8 }), λ);

			var dualityGap = Math.Abs(f(certificate.Optimum) - g(certificate.λ));

			Assert.IsTrue(
				dualityGap <= options.DualityGap,
				"Duality gap too big.");

			var we = new Vector(new double[] { 0.0, 0.8 });

			Assert.IsTrue((we - certificate.Optimum).Norm2 < ε, "Solution should have been [0.0 0.8].");
		}

		[TestMethod]
		public void ConstrainedQuadratic()
		{
			var g = new Vector(new double[] { -1, -1, -1, -1, -1 });

			int P = g.Length;

			int constraintsCount = 2 * P;

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

			VectorFunction df =
				x => Q(x) + g;

			TensorFunction d2f =
				x =>
					y =>
						Q(y);

			VectorFunction d2fd =
				x =>
					range.Select(i => gramMatrix[i, i]);

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

			var M = ConjugateGradient.ConstrainedMinimizeOptions.GetJacobiPreconditioner(
				d2fd,
				P,
				fc,
				dfc,
				d2fcd);

			var certificate = ConjugateGradient.TruncatedNewtonConstrainedMinimize(
				df,
				d2f,
				new Vector(new double[] { 0.5, 0.5, 0.5, 0.5, 0.5 }),
				constraintsCount,
				fc,
				dfc,
				d2fc,
				new ConjugateGradient.TruncatedNewtonConstrainedMinimizeOptions
				{
					DualityGap = 1e-15
				},
				M);

			var xe = new Vector(new double[] { 0.25, 0.25, 0.0, 0.5, 0.0 });

			Assert.IsTrue(
				(certificate.Optimum - xe).Norm2 < ε,
				String.Format("Should have found {0}, found {1} instead", xe, certificate.Optimum));
		}

	}
}
