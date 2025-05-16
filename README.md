# rotation_symmetry
Class for calculating rotation symmetry data, e.g. linearity or symmetry number for partition function

By Mikael Valter-Lithander

It searches through all possible rotation axes and finds all possible allowed permutations from rotations. The symmetry number is equal to the number of rotation permutations.

USAGE

Write a script, import RotationSymmetry and create an object with e.g. ```rs = RotationSymmetry(atoms)``` where ```atoms``` is an Atoms object from ASE. Then call e.g. ```rs.symmetry_number``` or ```rs.geometry```. ```get_rotation_images()``` will return a list of atoms showing the rotation axes.

Note that the default tolerance is quite high. My experience is that it's better to be high, because two high will likely crash or give noticably strange results, while too low gives reasonable but incorrect low symmetries that can go under the radar.

Note also that there is no way for the program to know if a symmetry is rigid or not. E.g. glycerol can be in a configuration where all OH groups line up perfectly and the program would determine the symmetry number as 2, while it should be 1.
