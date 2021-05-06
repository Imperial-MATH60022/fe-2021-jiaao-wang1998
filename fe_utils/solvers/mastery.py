"""Solve a nonlinear problem using the finite element method.
If run as a script, the result is plotted. This file can also be
imported as a module and convergence tests run on the solver.
"""
from fe_utils import *
import numpy as np
from numpy import cos, pi, sin
import scipy.sparse as sp
import scipy.sparse.linalg as splinalg
from argparse import ArgumentParser

def assemble(fs, f):
    """Assemble the finite element system for the Helmholtz problem given
    the function space in which to solve and the right hand side
    function."""

    fe = fs.element
    mesh = fs.mesh

    # number of edges and vertices
    v = mesh.entity_counts[0]
    e = mesh.entity_counts[1]
    m = 2*(v+e)
    n = v

    # Create an appropriate (complete) quadrature rule.
    Q = gauss_quadrature(fe.cell, 2*fe.degree)

    # Tabulate the basis functions and their gradients at the quadrature points.
    phi = fe.tabulate(Q.points)
    psi = fe.tabulate(Q.points, grad=True)

    # Create the left hand side matrix and right hand side vector.
    # This creates a sparse matrix because creating a dense one may
    # well run your machine out of memory!
    A = sp.lil_matrix((m, m))
    B = sp.lil_matrix((n, m))
    F_hat = np.zeros(m+n)
    F = F_hat[:m]

    # Now loop over all the cells and assemble A, B and f
    for c in range(mesh.entity_counts[-1]):
        # Find the appropriate global node numbers for this cell.
        nodes = fs.cell_nodes[c, :]

        # Compute the change of coordinates.
        J = mesh.jacobian(c)
        detJ = np.abs(np.linalg.det(J))

        print(phi.shape)

        print(f.values[nodes].shape)
        print(Q.weights.shape)
        print(F[nodes].shape)
        print()


        # Compute the actual cell quadrature for rignt-hand side
        F[nodes] += np.einsum("qi, k, qk, q -> i", phi, f.values[nodes], phi, Q.weights) * detJ

        # Compute the actual cell quadrature for A
        A[np.ix_(nodes, nodes)] +=  (1/4) * (np.einsum("ba, qib, ca, qjc, q -> ij", np.linalg.inv(J), psi, np.linalg.inv(J), psi, Q.weights)
                                + np.einsum("ba, qib, ca, qjc, q -> ij", np.linalg.inv(J), psi.T, np.linalg.inv(J), psi, Q.weights)
                                + np.einsum("ba, qib, ca, qjc, q -> ij", np.linalg.inv(J), psi, np.linalg.inv(J), psi.T, Q.weights)
                                + np.einsum("ba, qib, ca, qjc, q -> ij", np.linalg.inv(J), psi.T, np.linalg.inv(J), psi.T, Q.weights) ) * detJ
        
        # Compute the actual cell quadrature for B
        B[np.ix_(nodes, nodes)] += np.einsum("ba, qib, ca, qjc, q -> ij", np.linalg.inv(J), psi, np.linalg.inv(J), psi, Q.weights) * detJ

    # Stitch sparse matrix from blocks
    A = sp.bmat([[A, B.T], [B, None]], format='lil').toarray()

    # set global vector rows corresponding to boundary nodes to 0
    F[boundary_nodes(fs)] = 0
    # set global matrix rows corresponding to boundary nodes to 0
    A[boundary_nodes(fs), :] = 0
    # set diagonal entry on each matrix row corresponding to a boundary node to 1
    A[boundary_nodes(fs), boundary_nodes(fs)] = 1

    return A, F_hat

def boundary_nodes(fs):
    """Find the list of boundary nodes in fs. This is a
    unit-square-specific solution. A more elegant solution would employ
    the mesh topology and numbering.
    """
    eps = 1.e-10

    f = Function(fs)

    def on_boundary(x):
        """Return 1 if on the boundary, 0. otherwise."""
        if x[0] < eps or x[0] > 1 - eps or x[1] < eps or x[1] > 1 - eps:
            return 1.
        else:
            return 0.

    f.interpolate(on_boundary)

    return np.flatnonzero(f.values)

def solve_mastery(resolution, analytic=False, return_error=False):
    """This function should solve the mastery problem with the given resolution. It
    should return both the solution :class:`~fe_utils.function_spaces.Function` and
    the :math:`L^2` error in the solution.

    If ``analytic`` is ``True`` then it should not solve the equation
    but instead return the analytic solution. If ``return_error`` is
    true then the difference between the analytic solution and the
    numerical solution should be returned in place of the solution.
    """
    
    # Set up the mesh, finite element and function space required.
    mesh = UnitSquareMesh(resolution, resolution)
    se = LagrangeElement(ReferenceTriangle, 2)
    ve = VectorFiniteElement(se)
    fs = FunctionSpace(mesh, ve)

    # Create a function to hold the analytic solution for comparison purposes.
    analytic_answer = Function(fs)
    analytic_answer.interpolate(lambda x: (2*pi*(1 - cos(2*pi*x[0]))*sin(2*pi*x[1]),
                                          -2*pi*(1 - cos(2*pi*x[1]))*sin(2*pi*x[0])))

    # If the analytic answer has been requested then bail out now.
    if analytic:
        return analytic_answer, 0.0

    # Create the right hand side function and populate it with the
    # correct values.
    f = Function(fs)
    f.interpolate(lambda x: (8*pi**3*cos(2*pi*x[0])*sin(2*pi*x[1]) - 8*pi**3*(1 - cos(2*pi*x[0]))*sin(2*pi*x[1]),
                             8*pi**3*cos(2*pi*x[1])*sin(2*pi*x[0]) - 8*pi**3*(1 - cos(2*pi*x[1]))*sin(2*pi*x[0])))

    # Assemble the finite element system.
    A, F = assemble(fs, f)

    # Create the function to hold the solution.
    w = Function(fs)
    u = Function(fs)
    p = Function(fs)

    # number of edges and vertices
    v = mesh.entity_counts[0]
    e = mesh.entity_counts[1]
    m = 2*(v+e)
    n = v

    # Cast the matrix to a sparse format and use a sparse solver for
    # the linear system. This is vastly faster than the dense
    # alternative.
    A = sp.csc_matrix(A)
    lu = splinalg.splu(A)
    w.values = lu.solve(F)
    u.values = w.values[:m]
    p.values = w.values[m:]

    # Compute the L^2 error in the solution for testing purposes.
    error = errornorm(analytic_answer, u) + errornorm(analytic_answer, p)

    if return_error:
        u.values -= analytic_answer.values

    # Return the solution and the error in the solution.
    return (u, p), error

if __name__ == "__main__":

    parser = ArgumentParser(
        description="""Solve the mastery problem.""")
    parser.add_argument("--analytic", action="store_true",
                        help="Plot the analytic solution instead of solving the finite element problem.")
    parser.add_argument("--error", action="store_true",
                        help="Plot the error instead of the solution.")
    parser.add_argument("resolution", type=int, nargs=1,
                        help="The number of cells in each direction on the mesh.")
    args = parser.parse_args()
    resolution = args.resolution[0]
    analytic = args.analytic
    plot_error = args.error

    u, error = solve_mastery(resolution, analytic, plot_error)

    u.plot()
