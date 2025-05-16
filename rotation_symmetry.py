#!/usr/bin/env python

from ase.build import molecule
import numpy as np
import scipy as sp
from pprint import pprint
from collections import Counter
from functools import reduce, cached_property
import math
from itertools import combinations
from ase import Atom

class RotationSymmetry:
    """Calculates rotation symmetry, symmetry number, and geometry of
    molecules.
    
    Cannot tell the difference between rigid and non-rigid symmetries. E.g.,
    a glycerol molecule whose OH groups line up perfectly will give symmetry
    number 2 
       
    Inputs:
       
    atoms : an ASE atoms object
    nb_nearest_neighbors : integer
        maximum number of nearest neighbors to search through to find rotation
        axes. higher numbers slows down calculation for large, highly symmetric
        molecules, e.g. fullerenes
    tol : float
        tolerance for atomic distances
    hard_tol : float
        tolerance for other things. might not be needed
    """
    
    def __init__(self, atoms, num_nearest_neighbors=9,
                 tol=0.075, hard_tol=1e-5):
        self.atoms = self.center_on_center_of_mass(atoms)
        self.num_nearest_neighbors = num_nearest_neighbors
        self.tol = tol
        self.hard_tol = hard_tol
        
        self.pos = atoms.get_positions()
        self._pos_backup = np.copy(self.pos)
        self.symbols = atoms.get_chemical_symbols()        
        self.masses = atoms.get_masses()
        self.mass_check()
        self.com = atoms.get_center_of_mass()
        
    
    @property
    def symmetry_number(self):
        """Returns the rotation symmetry number, sigma"""
        
        return len(self.rotation_mappings)
        
    
    @property
    def rotation_mappings(self):
        """Returns rotation mappings, every rotation permutation of atom
        indices"""
        
        return self._mappings_and_axes[0]
    
    
    @property
    def axes_and_orders(self):
        """Returns tuples of axis, a 1D numpy.ndarray, and and order, the
        integer n for a Cn axis"""
        
        return self._mappings_and_axes[1]
    
        
    @property
    def geometry(self):
        """Returns the geometry of the molecule"""
        
        geometry, vector = self.get_geometry_and_orthogonal_vector()
        return geometry
    
    
    def get_rotation_images(self, interpolation=6, orders=None, dummy_element=None,):
        """Returns images showing rotation axes
        
        Inputs:
        
        interpolation : integer
            number of images between every valid rotation
        orders : integer, list of integers or None
            rotation orders to show. default is showing all
        dummy_element: string or None
            dummy atoms are inserted to make axis clearer
        """
        
        if not isinstance(interpolation, int) or interpolation < 0:
            raise ValueError('Interpolation must be non-negative')
            
        if isinstance(orders, int):
            orders = [orders]
            
        dummy_dist = max(np.linalg.norm(self.pos, axis=1)) + 3.0
            
        images = []
        #pprint(self._axes_and_orders)
        for axis, order in self.axes_and_orders:
            if orders is None or order in orders:
                rot_atoms = self.atoms.copy()
                if dummy_element is not None:
                    rot_atoms.append(Atom(dummy_element, axis * dummy_dist))
                    rot_atoms.append(Atom(dummy_element, -axis * dummy_dist))
                turns = order * (interpolation + 1)
                angle = 360 / turns # degrees
                for i in range(turns - 1):
                    rot_atoms.rotate(axis, angle)
                    images.append(rot_atoms.copy())
        return images

    @cached_property
    def _mappings_and_axes(self):
        mappings_and_axes = self._raw_calc_rotation_mappings()
        mappings, axes_and_orders = self._process_mappings_and_axes(mappings_and_axes)
        return mappings, axes_and_orders


    def _process_mappings_and_axes(self, mappings_and_axes, verbose=True):
        """Removes duplicates of mappings and associated axis and then
        separates the mappings from the axes"""
        
        rotation_mappings = []
        axes_and_orders = []
        
        seen = set()
        for mapping, axis in mappings_and_axes:
            sorted_mapping = tuple(sorted(mapping))
            if sorted_mapping not in seen:
                seen.add(sorted_mapping)
                rotation_mappings.append(mapping)
                if axis is not None:
                    axes_and_orders.append((axis, len(mapping) + 1))
        
        rotation_mappings = np.vstack(rotation_mappings)
        num_calc_mappings = len(rotation_mappings)
        rotation_mappings = np.unique(rotation_mappings, axis=0)
        num_unique_mappings = len(rotation_mappings)
        if num_calc_mappings != num_unique_mappings and verbose:
            print('Warning! Rotation mapping over between distinct ' + \
                  'rotations. Consider increasing tolerance')
        return rotation_mappings, axes_and_orders


    def _raw_calc_rotation_mappings(self):
        """Calculates rotation mappings by searching all possibles axes"""
        
        id_map = [tuple(range(len(self.atoms)))]
        mappings_and_axes = [(id_map, None)]
        geometry, vector = self.get_geometry_and_orthogonal_vector()
        
        if geometry == 'monoatomic':
            return mappings_and_axes
        if geometry == 'linear':
            mappings = self.calc_rotation_mappings(axis=vector, orders=2)
            if mappings is not None:
                mappings_and_axes.append((mappings, vector))
            return mappings_and_axes
        
        # If element centers of mass don't align, rotation axes are restricted
        com_geometry, axis_com = self.get_center_of_mass_geometry()
        
        if com_geometry == 'off_line':
            return mappings_and_axes
        if com_geometry == 'on_line':
            mappings = self.calc_rotation_mappings(axis=axis_com, orders=None)
            if mappings is not None:
                mappings_and_axes.append((mappings, axis_com))
            return mappings_and_axes
        
        checked_axes = []
        
        # Search all axes going through an atom
        for position in self.pos:
            added = self.add_if_valid_and_new(position, checked_axes)

            if not added:
                continue
            mappings = self.calc_rotation_mappings(axis=position, orders=None)
            if mappings is not None:
                mappings_and_axes.append((mappings, position))   
        
        # Rotations can only take place if exchanged atoms have the same
        # distance to the center of mass (and are of the same element)
        #
        # Make groups of atoms allowed to exchange
        # If there is an atom in the center of mass, it is excluded from
        # any group as it is always trivially mapped on itself
        
        atom_groups = self.group_positions_by_symbol_and_norm()
        position_groups = atom_groups.values()
        
        # Crucially, as we have checked all axes through atoms, all remaining
        # valid axes don't go through an atom in atom groups. That means that
        # any remaining rotations MUST permute all atoms in an atom group, as
        # the only way an atom is mapped on itself and not any other is if it's
        # on the rotation axis
        
        # In principle, we could restrict ourself to check axes for the
        # smallest atom group. However, if that group has 2 atoms, it only
        # restricts rotations to the plane perpendicular to the line between
        # them. If there are several 2-atom groups, they may restrict even more
        
        line_vecs = [p[1] - p[0] for p in position_groups if len(p) == 2]
        axis_space = self.get_orthogonal_vectors(line_vecs)
        
        if len(axis_space) == 0:
            return mappings_and_axes
        if len(axis_space) == 1:
            axis = axis_space[0]
            added = self.add_if_valid_and_new(axis, checked_axes)
            if added:
                mappings = self.calc_rotation_mappings(axis, order=2)
                if mappings is not None:
                    mappings_and_axes.append((mappings, axis))
            return mappings_and_axes

        # There is also a restriction on axes for 2D axis space, where all
        # 2-groups are on a line like C2H6, but should be only marginally
        # time-saving, not crucial, thus not implemented
        
        # Now, we can get all remaining axes from the smallest atom group
        # with at least 3 atoms
        
        sizes = [len(p) for p in position_groups]
        possible_orders = self.get_common_divisors(sizes)
        higher_orders = [o for o in possible_orders if o > 2]
        
        triples_and_greater = [p for p in position_groups if len(p) > 2]
        if len(triples_and_greater) == 0:
            raise Exception('There should exist an atom group with size > 3 ')
        positions = min(triples_and_greater, key=len)
        natoms = len(positions)
        
        tree = sp.spatial.KDTree(positions)
        number = min(natoms, self.num_nearest_neighbors + 1)
        distances, indices = tree.query(positions, k=number)
        
        # Look for regular polygons in planes, defined by triples of atoms
        # in the studied group, to find valid axes
        
        if len(higher_orders) > 0:
            triples = []
            for i in range(natoms):
                neighbors = indices[i, 1:]  # skip self (index 0)
                for j, k in combinations(neighbors, 2):
                    triples.append((i, j, k))
            triples = np.unique(np.array(triples), axis=0)
            for triple in triples:
                axis, orders = self.get_Cn_axes_from_triple(triple, positions,
                                                            indices,
                                                            higher_orders)
                added = self.add_if_valid_and_new(axis, checked_axes)
                if not added:
                    continue
                
                mappings = self.calc_rotation_mappings(axis, orders)
                if mappings is None:
                    continue
                mappings_and_axes.append((mappings, axis))
            
        # Look for axes through midpoints of pairs
        if 2 in possible_orders:
            pairs = [(i, j) for i in range(natoms) \
                      for j in indices[i, 1:] if i < j]
            for pair in pairs:
                i, j = pair
                midpoint = 0.5 * (positions[i] + positions[j])
                
                added = self.add_if_valid_and_new(midpoint, checked_axes)
                if not added:
                    continue
   
                mappings = self.calc_rotation_mappings(midpoint, orders=2)
                if mappings is None:
                    continue
                mappings_and_axes.append((mappings, midpoint))

        return mappings_and_axes
    
    
    def center_on_center_of_mass(self, atoms):
        """Center atoms on center of mass
        
        Inputs:
        
        atoms : an ASE atoms object
            the atoms to be centered
        """
        
        pos_start = atoms.get_positions()
        com = atoms.get_center_of_mass()
        atoms.set_positions(pos_start - com, apply_constraint=False)
        return atoms
        
    
    def mass_check(self):
        """Checks that atoms of elements have the same mass, else exception"""
        unique_symbols = np.unique(self.symbols)
        for symbol in unique_symbols:
            mask = [symbol == s for s in self.symbols]
            masses_for_symbol = self.masses[mask]

            ref_mass = masses_for_symbol[0]
            if not np.allclose(masses_for_symbol, ref_mass,
                               atol=self.hard_tol):
                print('Masses not equal for',symbol)
                print('Current implementation requires equal mass')
                raise Exception('Unique masses required')
        return


    def get_geometry_and_orthogonal_vector(self):
        """Calculates geometry ('monatomic', 'linear', 'nonlinear')
        and one orthogonal vector if it exists (else None)
        """
        
        natoms = len(self.atoms)
        if natoms == 0:
            raise ValueError('Empty atoms object has no valid geometry')
        if natoms == 1:
            return 'monatomic', None
            
        null_vectors = self.get_orthogonal_vectors(self.pos)
        if len(null_vectors) == 2:
            return 'linear', null_vectors[0]
        return 'nonlinear', None
        
    
    def calc_rotation_mappings(self, axis, orders=None):
        """Calculates rotation mappings around axis with given rotation orders
        
        Inputs:
        
        axis : numpy.ndarray
            rotation axis.
        orders : list of integers
            orders, e.g. 3 for a C3 axis. They are tested from high to low
        """
        
        norm = np.linalg.norm(axis)
        
        if orders is None:
            symbols_offline = np.array(self.symbols)[~self.points_on_line(axis)]
            counts = Counter(symbols_offline)
            orders = self.get_common_divisors(list(counts.values()))            
        elif isinstance(orders, int):
            orders = [orders]

        orders.sort(reverse=True)
        for order in orders:
            angle = 2 * np.pi / order

            R = sp.spatial.transform.Rotation.from_rotvec(angle * axis / norm)
            R = R.as_matrix().T

            kdtree = sp.spatial.KDTree(self.pos)        
            mappings = []
            
            pos_rotated = self.pos.copy()
            for i in range(order - 1):
                pos_rotated = pos_rotated @ R
                distances, indices = kdtree.query(pos_rotated)

                # The error compounds from calculating axis and rotating
                # Higher tolerance
                tol = self.tol * 2
                
                if np.all(distances <= tol):
                    # Must map on same element
                    symbols_rotated = [self.symbols[i] for i in indices]
                    if self.symbols == symbols_rotated:
                        mappings.append(tuple(indices))
                    else:
                        break
                else:
                    #print(order)
                    break
            
            #if len(mappings) != 0:
            if len(mappings) == order - 1:
                # If we find a high order that works, we can discard all lower
                return mappings
        return None


    def add_if_valid_and_new(self, vec, vecs):
        """Check if vec is not None, non-zero and not parallel to any vector
        in vecs. If valid and new, append to vecs and return True. Otherwise
        False
        
        Inputs:
        
        vec : numpy.ndarray (1D)
            candidate vector
        vecs : list of numpy.ndarray
            already checked vectors
            
        Returns:
        
        bool : True if vec was added, False otherwise
        
        """
        
        if vec is None:
            return False
        norm = np.linalg.norm(vec)
        if norm < self.tol:
            return False
        norm_vec = vec / norm
        if len(vecs) == 0:
            vecs.append(norm_vec)
            return True
        norms = np.linalg.norm(vecs, axis=1)
        crosses = np.cross(vecs, norm_vec)
        sin_angles = np.linalg.norm(crosses, axis=1) / norms
        
        # Tolerance is lower, since it's better not to miss any vectors
        if np.all(sin_angles > self.tol * 0.01):
            vecs.append(norm_vec)
            return True
        return False


    def get_center_of_mass_geometry(self):
        """Returns the geometry the sums of positions for each element (same
        as elemental centers of mass times number of atom of the element)
        """
    
        # Calculate elemental centers of mass
        centers = {}
        numbers = {}
        pos_sums = []
        for element in np.unique(self.symbols):
            inds = [s == element for s in self.symbols]
            pos_sum = np.sum(self.pos[inds], axis=0)
            pos_sums.append(pos_sum)
            centers[element] = pos_sum
            numbers[element] = len(inds)  
        
        norm = np.linalg.norm(pos_sums, axis=1)
        if np.all(norm < self.tol):
            return 'identical', None
        
        null_space = self.get_orthogonal_vectors(pos_sums)

        if len(null_space) == 2:
            return 'on_line', np.cross(*null_space)
        if len(null_space) < 2:
            return 'off_line', None
        
        raise Exception('Dimension of null space: ' + str(len(null_space)))
        
        
    def points_on_line(self, vector):
        """Returns mask of point indices on the line
        
        Inputs:
        
        vector : numpy.ndarray
            defines the line points are on or off
        """
        
        if np.linalg.norm(vector) < self.tol:
            raise ValueError("Direction vector cannot be zero.")
        nvector = vector / np.linalg.norm(vector)
            
        dists_origin = np.linalg.norm(self.pos, axis=1)
        
        crosses = np.cross(self.pos, nvector)
        cross_norms = np.linalg.norm(crosses, axis=1)
        
        sin_angles = np.zeros_like(dists_origin)
        nonzero = dists_origin > self.tol
        sin_angles[nonzero] = cross_norms[nonzero] / dists_origin[nonzero]
        
        mask = sin_angles <= self.tol
        return mask
        
        
    def get_common_divisors(self, values):
        """Returns common divisors
        
        Inputs:
        
        values : list of integers
        """
        
        gcd = reduce(math.gcd, values)
        divisors = []
        for i in range(2, gcd + 1):
            if gcd % i == 0:
                divisors.append(i)
        divisors.reverse()
        return divisors
        
    def group_positions_by_symbol_and_norm(self):
        """Returns a dict where the keys are tuples of element and distance
        from center of mass and the values arrays of positions
        """
        positions = self.pos.tolist()
        dists = np.linalg.norm(self.pos, axis=1).tolist()
        symbols = self.symbols.copy()
        dps_triples = sorted(list(zip(dists, positions, symbols)))
        
        curr_norms = {}
        groups = {}
        for dist, position, symbol in dps_triples:
            if dist < self.tol:
                continue
            if symbol not in curr_norms or \
               dist - curr_norms[symbol] > self.tol:
                curr_norms[symbol] = dist
                groups[(symbol, curr_norms[symbol])] = [position]
            else:
                groups[(symbol, curr_norms[symbol])].append(position)
        for group in groups:
            groups[group] = np.array(groups[group])

        return groups
        
        
    def get_orthogonal_vectors(self, vectors):
        """Returns the null space (numpy.ndarray) of the input. If the input
        spans R3, it returns None
        
        Inputs
        
        vectors : numpy.ndarray or list
        """
        
        if len(vectors) == 0:
            return np.eye(3)
        vectors = np.array(vectors)

        ndim = vectors.shape[1]
        rank = np.linalg.matrix_rank(vectors, tol=self.tol)
        
        if rank == ndim:
            # Full rank: no nonzero orthogonal vector
            return []
        # Find the null space
        ns = sp.linalg.null_space(vectors, rcond=self.tol)
        nullity = ns.shape[1] # Number of independent orthogonal vectors
        if ns.size == 0:
            return []
        return ns.T
        
        
    def get_Cn_axes_from_triple(self, triple, positions, neighbors,
                                possible_orders):
        """Returns axis and possible rotation orders from positions of three
        atoms and near neighbors, if on a circle orthogonal to a valid axis
        
        Inputs:
        
        triple : list of integers
            indices of the defining atoms' positions
        positions : np.ndarray
            positions of atoms of the same element. should have same distance
            from center of mass
        neigbors : list of integers
            nearest neighbor lists of the atoms at positions
        possible_orders : list of integers
            possible rotation orders. will most likely be narrowed down by the
            function
        """
        
        pos_triple = positions[triple]
        
        vec0 = pos_triple[1] - pos_triple[0]
        vec1 = pos_triple[2] - pos_triple[0]
        
        normal = np.cross(vec0, vec1)
        norm_normal = np.linalg.norm(normal)
        if norm_normal < self.tol:
            return None, []
            
        normal /= norm_normal
        
        # Find where cross from origin cuts plane
        # Plane defined by n * (r - p0) = 0, r point in plane
        # Line defined by t * n, t parameter
        
        t = np.dot(normal, pos_triple[0])
        cut = t * normal
        
        cutvecs = pos_triple - cut
        
        # Triple on circle?
        radii = np.linalg.norm(cutvecs, axis=1)
        radius = radii[0]
        if not np.all(np.isclose(radii, radius, atol=self.tol)):
            return None, []

        # Find more points on the circle
        counter = 0
        max_iter = len(positions)
        in_plane_inds = list(triple)
        searched_inds = in_plane_inds.copy()
        while in_plane_inds and counter <= max_iter:
            # Pop an index on circle...
            i = in_plane_inds[-1]
            in_plane_inds = in_plane_inds[:-1]
            for j in neighbors[i, 1:]:
                # ...and search its neighbors
                if j in searched_inds:
                    continue
                searched_inds.append(j)
                cutvec = positions[j] - cut
                jnorm = np.linalg.norm(cutvec)
                if not np.isclose(jnorm, radius, atol=self.tol):
                    # Incorrect distance
                    continue
                jcross = np.cross(cutvec, cutvecs[0])
                jcross_norm = np.linalg.norm(jcross)
                if jcross_norm <= self.tol:
                    # On line with a point from cut point
                    # This should never happen
                    continue
                # In plane?
                meta_cross = np.cross(normal, jcross)
                meta_cross_norm = np.linalg.norm(meta_cross)
                if meta_cross_norm < self.tol:
                    in_plane_inds.append(j)
                
            counter += 1
        assert counter <= max_iter
        # Could check if even spaced, but not sure if necessary
        # Just rotating these atoms could be useful
        
        orders = [o for o in possible_orders if len(in_plane_inds) % o == 0]
        return normal, orders
        
