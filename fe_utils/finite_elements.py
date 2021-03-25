# Cause division to always mean floating point division.
from __future__ import division
import numpy as np
from .reference_elements import ReferenceInterval, ReferenceTriangle

np.seterr(invalid='ignore', divide='ignore')


def lagrange_points(cell, degree):
    """Construct the locations of the equispaced Lagrange nodes on cell.

    :param cell: the :class:`~.reference_elements.ReferenceCell`
    :param degree: the degree of polynomials for which to construct nodes.

    :returns: a rank 2 :class:`~numpy.array` whose rows are the
        coordinates of the nodes.

    The implementation of this function is left as an :ref:`exercise
    <ex-lagrange-points>`.

    """

    if cell.dim == 1: #1-dimensional cell

        lagrange_p = np.array([i / degree for i in range(degree + 1)])
        lagrange_p = lagrange_p.reshape([degree+1, 1])

    elif cell.dim == 2: #2-dimensional cell

        # produces the nodes in entity order
        lagrange_p = [ [j / degree, i / degree] for i in range(degree + 1) for j in range(degree - i+ 1)]

        # convert to numpy array
        lagrange_p = np.array(lagrange_p)

    return lagrange_p


def vandermonde_matrix(cell, degree, points, grad=False):
    """Construct the generalised Vandermonde matrix for polynomials of the
    specified degree on the cell provided.

    :param cell: the :class:`~.reference_elements.ReferenceCell`
    :param degree: the degree of polynomials for which to construct the matrix.
    :param points: a list of coordinate tuples corresponding to the points.
    :param grad: whether to evaluate the Vandermonde matrix or its gradient.

    :returns: the generalised :ref:`Vandermonde matrix <sec-vandermonde>`

    The implementation of this function is left as an :ref:`exercise
    <ex-vandermonde>`.
    """
    
    # evaluate Vandermonde matrix 
    if grad == False:

        # 1-dimensional cell
        if cell.dim == 1: 

            # reshape the points
            points = points.reshape(-1,)

            # construct Vandermonde matrix
            van = [[x**i  for i in range(degree + 1)] for x in points]

            # convert to numpy array
            return np.array(van)
            
        # 2-dimensional cell
        elif cell.dim == 2: 
            # construct matrix
            van = [[coord[0]**(i-j) * coord[1]**j 
                    for i in range(degree+1) for j in range(i+1) ] for coord in points]

            # convert to numpy array
            return np.array(van)
    
    # evaluate gradient of Vandermonde matrix
    else:

        # 1-dimensional cell
        if cell.dim == 1: 

            # reshape the points
            points = points.reshape(-1,)

            # construct gradient
            gradient = [[[(i) * (x**(max(i-1,0)))] for i in range(degree + 1)] for x in points]

            # convert to numpy array
            return np.array(gradient)

        # 2-dimensional cell
        elif cell.dim == 2: 

            # construct gradient
            gradient = [[[(i-j) * coord[0]**max(0, i-j-1) * coord[1]**j, j 
                          * coord[0]**(i-j) * coord[1]**max(0, j-1)] 
                          for i in range(degree+1) for j in range(i+1) ] 
                          for coord in points]
        
            # convert nan to 0
            gradient  = np.nan_to_num( np.array(gradient))
        
            # return gradient
            return gradient


class FiniteElement(object):
    def __init__(self, cell, degree, nodes, entity_nodes=None):
        """A finite element defined over cell.

        :param cell: the :class:`~.reference_elements.ReferenceCell`
            over which the element is defined.
        :param degree: the
            polynomial degree of the element. We assume the element
            spans the complete polynomial space.
        :param nodes: a list of coordinate tuples corresponding to
            the nodes of the element.
        :param entity_nodes: a dictionary of dictionaries such that
            entity_nodes[d][i] is the list of nodes associated with entity `(d, i)`.

        Most of the implementation of this class is left as exercises.
        """

        #: The :class:`~.reference_elements.ReferenceCell`
        #: over which the element is defined.
        self.cell = cell
        #: The polynomial degree of the element. We assume the element
        #: spans the complete polynomial space.
        self.degree = degree
        #: The list of coordinate tuples corresponding to the nodes of
        #: the element.
        self.nodes = nodes
        #: A dictionary of dictionaries such that ``entity_nodes[d][i]``
        #: is the list of nodes associated with entity `(d, i)`.
        self.entity_nodes = entity_nodes

        if entity_nodes:
            #: ``nodes_per_entity[d]`` is the number of nodes
            #: associated with an entity of dimension d.
            self.nodes_per_entity = np.array([len(entity_nodes[d][0])
                                              for d in range(cell.dim+1)])

        # Replace this exception with some code which sets
        # self.basis_coefs
        # to an array of polynomial coefficients defining the basis functions.
        basis_coefs = np.linalg.inv( vandermonde_matrix(cell, degree, nodes) )
        self.basis_coefs = basis_coefs

        #: The number of nodes in this element.
        self.node_count = nodes.shape[0]

    def tabulate(self, points, grad=False):
        """Evaluate the basis functions of this finite element at the points
        provided.

        :param points: a list of coordinate tuples at which to
            tabulate the basis.
        :param grad: whether to return the tabulation of the basis or the
            tabulation of the gradient of the basis.

        :result: an array containing the value of each basis function
            at each point. If `grad` is `True`, the gradient vector of
            each basis vector at each point is returned as a rank 3
            array. The shape of the array is (points, nodes) if
            ``grad`` is ``False`` and (points, nodes, dim) if ``grad``
            is ``True``.

        The implementation of this method is left as an :ref:`exercise
        <ex-tabulate>`.

        """
        # return tabulation of the basis
        if not grad:
            result = np.dot(vandermonde_matrix(self.cell, self.degree, points, grad), self.basis_coefs)
            return result

        # return tabulation of the gradient of the basis
        else: 
            result = np.einsum("ijk,jl->ilk", vandermonde_matrix(self.cell, self.degree, points, grad), self.basis_coefs) 
            return result

    def interpolate(self, fn):
        """Interpolate fn onto this finite element by evaluating it
        at each of the nodes.

        :param fn: A function ``fn(X)`` which takes a coordinate
           vector and returns a scalar value.

        :returns: A vector containing the value of ``fn`` at each node
           of this element.

        The implementation of this method is left as an :ref:`exercise
        <ex-interpolate>`.

        """

        # evaluate fn at each nodes
        result = [fn(x) for x in self.nodes]

        # convert to numpy array
        return np.array(result)

    def __repr__(self):
        return "%s(%s, %s)" % (self.__class__.__name__,
                               self.cell,
                               self.degree)


class LagrangeElement(FiniteElement):
    def __init__(self, cell, degree):
        """An equispaced Lagrange finite element.

        :param cell: the :class:`~.reference_elements.ReferenceCell`
            over which the element is defined.
        :param degree: the
            polynomial degree of the element. We assume the element
            spans the complete polynomial space.

        The implementation of this class is left as an :ref:`exercise
        """

        # Use lagrange_points to obtain the set of nodes.  Once you
        # have obtained nodes, the following line will call the
        # __init__ method on the FiniteElement class to set up the
        # basis coefficients.
        
        nodes = lagrange_points(cell, degree)

        # initialize entity nodes
        #entity_nodes = {d: {0: np.zeros(np.math.comb(degree-1, d))} for d in range(cell.dim+1)}
        entity_nodes = {}
        
        # 1 dimesnion entity nodes
        if cell.dim == 1:
            # 2 points
            entity_nodes[0] = {0: [0], 1: [nodes.shape[0]-1]}

            # 1 edge
            entity_nodes[1] = {0: [n+1 for n in range(nodes.shape[0]-2)] }

        # 2 dimension entity nodes    
        elif cell.dim == 2:

            # 3 points 
            entity_nodes[0] = {0: [0], 1: [degree], 2: [nodes.shape[0]-1]}

            # 3 edges
            entity_nodes[1] = {i: [] for i in range(3)}
            
            # 1 faces
            entity_nodes[2] = {0: []}
            
            # loop over all points
            for n in range(nodes.shape[0]):
                # check if it is a vertex
                if n != 0 and n != degree and n != nodes.shape[0]-1:
                    # assume it is on the face
                    face = True

                    for e in range(3):
                        if cell.point_in_entity(nodes[n], (1, e)):
                            # not on the face
                            face = False
                            # insert to the edge
                            entity_nodes[1][e] = entity_nodes[1][e] + [n]

                    if face and cell.point_in_entity(nodes[n], (2, 0)):
                        # insert to the face
                        entity_nodes[2][0] = entity_nodes[2][0] + [n]
    
        # set up basis coefficients
        super(LagrangeElement, self).__init__(cell, degree, nodes, entity_nodes = entity_nodes)


class VectorFiniteElement(object):
    def __init__(self, FiniteElement):
        """
        Construct the corresponding vector element to the FiniteElement.

        """
        self.finite_element = FiniteElement
        # set cell and degree
        self.cell = FiniteElement.cell
        self.degree = FiniteElement.degree
        # topological dimension
        self.d = self.cell.dim
        # set entity_nodes
        self.entity_nodes = FiniteElement.entity_nodes
        for i in range(self.d+1):
            for k, v in self.entity_nodes[i].items():
                n1 = lambda x: 2*x
                n2 = lambda x: 2*x+1
                self.entity_nodes[i][k] = [f(node) for node in v for f in (n1,n2)]
        # set nodes_per_entity
        self.nodes_per_entity = self.d * FiniteElement.nodes_per_entity
        # set nodes
        self.nodes = np.repeat(FiniteElement.nodes, self.d, axis=0)
        # set correct canonical basis
        self.node_weights = np.array([[1, 0] if i%self.d==0 else [0, 1] for i in range(self.nodes.shape[0])])

    def tabulate(self, points, grad=False):
        """Evaluate the basis functions of this finite element at the points
        provided.

        :param points: a list of coordinate tuples at which to
            tabulate the basis.
        :param grad: whether to return the tabulation of the basis or the
            tabulation of the gradient of the basis.

        :result: an array containing the value of each basis function
            at each point. If `grad` is `True`, the gradient vector of
            each basis vector at each point is returned as a rank 3
            array. The shape of the array is (points, nodes) if
            ``grad`` is ``False`` and (points, nodes, dim) if ``grad``
            is ``True``.

        The implementation of this method is left as an :ref:`exercise
        <ex-tabulate>`.

        """
        # standard basis in R^2
        e0 = np.array([1, 0])
        e1 = np.array([0, 1])
        e = [e0, e1]

        # return tabulation of the basis
        if not grad:
            result = self.finite_element.tabulate(self, points, grad)
            result_vector = np.array([result[i, j//2] * e[i%2]
                                    for i in range(result.shape[0])
                                    for j in range(self.d*result.shape[1])])
            return result_vector

        # return tabulation of the gradient of the basis
        else: 
            result = self.finite_element.tabulate(self, points, grad)
            result_vector = np.array([result[i, j//2, k] * e[i%2]
                                    for i in range(result.shape[0])
                                    for j in range(self.d*result.shape[1])
                                    for k in range(result.shape[2])])
            return result_vector
