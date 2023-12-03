# distutils: language = c++

cimport numpy as np
from libc.stdlib cimport malloc
from libcpp.vector cimport vector

import cython
import numpy as np


cdef struct Oct:
    Oct** children
    double x
    double y
    double z
    np.int32_t ind

cdef enum CellType:
    LEAF = 0
    BRANCH = 1

ctypedef int (*function_type)(Oct* oct, CellType cell_type)


cdef class OctTree:
    cdef Oct* root
    cdef int count

    def __init__(self):
        self.root = <Oct*>malloc(sizeof(Oct))
        self.root.x = 0.5
        self.root.y = 0.5
        self.root.z = 0.5
        self.root.children = NULL
        self.root.ind = -1
        self.count = 1

    @cython.boundscheck(False)
    cdef Oct* add(self, const double[::1] x, const int level):
        cdef float dx = 0.5

        cdef size_t ilvl, i, ind
        cdef np.int8_t ix, iy, iz

        cdef Oct* node = self.root
        cdef Oct* child = self.root

        for ilvl in range(level):
            ix = x[0] > node.x
            iy = x[1] > node.y
            iz = x[2] > node.z
            ind = 4 * ix + 2 * iy + iz

            if node.children == NULL:
                node.children = <Oct **> malloc(8 * sizeof(Oct*))
                for i in range(8):
                    node.children[i] = NULL

            if node.children[ind] == NULL:
                # Create a new node
                child = <Oct*>malloc(sizeof(Oct))
                child.children = NULL
                child.x = node.x + 0.5 * dx * (2 * ix - 1)
                child.y = node.y + 0.5 * dx * (2 * iy - 1)
                child.z = node.z + 0.5 * dx * (2 * iz - 1)
                child.ind = -1

                self.count += 1

                node.children[ind] = child

            dx *= 0.5

            node = node.children[ind]

        # if not np.isclose(node.x, x[0]) or not np.isclose(node.y, x[1]) or not np.isclose(node.z, x[2]):
        #     raise ValueError(
        #         "Node xc does not match. Expected "
        #         f"{x[0]}, {x[1]}, {x[2]}, got {node.x}, {node.y}, {node.z} @ level {level}"
        #     )
        return node

    @classmethod
    @cython.boundscheck(False)
    def from_list(cls, const double[:, ::1] Xc, const int[::1] levels):
        cdef int i

        cdef Oct* node

        cdef OctTree tree = cls()
        for i in range(Xc.shape[0]):
            node = tree.add(Xc[i], levels[i])
            node.ind = i

        return tree

    cdef void depth_first_refmask(self, Oct* node, vector[np.uint8_t]& ref_mask, vector[np.int32_t]& leaf_order):
        cdef int i
        cdef Oct* child
        ret = 0
        for i in range(8):
            child = node.children[i]
            if child is NULL:
                leaf_order.push_back(-1)
                ref_mask.push_back(False)
                continue

            if child.children:   # Child has children
                ref_mask.push_back(True)
                self.depth_first_refmask(child, ref_mask, leaf_order)
            else:                # Child is a leaf
                leaf_order.push_back(child.ind)
                ref_mask.push_back(False)

    def get_refmask(self):
        cdef vector[np.uint8_t] ref_mask
        cdef vector[np.int32_t] leaf_order

        # Preallocate memory
        ref_mask.reserve(self.count)
        leaf_order.reserve(self.count)

        self.depth_first_refmask(self.root, ref_mask, leaf_order)

        # Copy back data
        cdef np.uint8_t[:] ref_mask_view = np.zeros(1 + ref_mask.size(), dtype=np.uint8)
        cdef np.int32_t[:] leaf_order_view = np.zeros(leaf_order.size(), dtype=np.int32)

        cdef size_t i
        ref_mask_view[0] = 8
        for i in range(ref_mask.size()):
            ref_mask_view[i + 1] = ref_mask[i] * 8
        for i in range(leaf_order.size()):
            leaf_order_view[i] = leaf_order[i]
        return np.asarray(ref_mask_view), np.asarray(leaf_order_view)
