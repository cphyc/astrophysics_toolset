# distutils: language = c++
# distutils: extra_compile_args=["-std=c++11"]

import numpy as np

cimport numpy as np
cimport openmp
from cython cimport integral
from cython.operator cimport dereference as deref
from libc.stdlib cimport free, malloc
from libcpp.map cimport map
from libcpp.pair cimport pair
from libcpp.queue cimport queue
from libcpp.unordered_map cimport unordered_map
from libcpp.vector cimport vector

from cython.parallel import prange

cimport cython

from .hilbert cimport hilbert3d_single


cdef extern from *:
    ctypedef int int128_t "__int128_t"

cdef struct Oct:
    np.int64_t file_ind       # on file index
    np.int64_t domain_ind     # original domain
    np.int64_t owning_cpu     # _id_ of the file containing the oct
    np.int64_t new_domain_ind # new domain
    np.uint8_t flag1[8]       # attribute for selection
    np.uint8_t flag2[8]       # temporary flag2
    np.float64_t hilbert_key[8]
    np.int32_t refmap[8]

    Oct* parent
    np.uint8_t icell            # index in parent cell
    Oct* children[8]
    Oct* neighbours[6]          # pointer to the *parent* oct of the neighbour
    np.uint8_t ineighbours[6]   # index of the neighbour cell in its parent oct

cdef struct OctInfo:
    Oct* oct
    np.uint64_t ipos[3]
    int ilvl
    np.uint8_t icell

@cython.boundscheck(False)
@cython.wraparound(False)
cdef inline void setup_oct(Oct *o, Oct *parent, np.uint8_t icell) nogil:
    cdef int i
    for i in range(8):
        o.children[i] = NULL
    for i in range(6):
        o.neighbours[i] = NULL
        o.ineighbours[i] = 0
    o.parent = parent
    o.icell = icell
    o.file_ind = -1
    o.domain_ind = -1
    o.owning_cpu = -1
    o.new_domain_ind = -1
    for i in range(8):
        o.hilbert_key[i] = -1
        o.flag1[i] = 0
        o.flag2[i] = 0
        o.refmap[i] = -1

#############################################################
# Mapping functions
#############################################################
cdef inline np.int64_t encode_mapping(np.int32_t file_ind, np.uint64_t domain_ind) nogil:
    return <np.int64_t>(
        ((<np.uint64_t>file_ind) << 20) + (<np.uint64_t>domain_ind)
    )


#############################################################
# Visitors
#############################################################
cdef class Visitor:
    # cdef np.int64_t ipos[3]
    # cdef int ilvl
    # cdef np.uint8_t icell
    cdef int levelmax, levelmin
    cdef int bit_length
    cdef bint parallel_friendly
    def __cinit__(self):
        self.parallel_friendly = False

    cdef void visit(self, Oct* o, OctInfo* oi) nogil:
        pass

cdef class ClearPaint(Visitor):
    @cython.boundscheck(False)
    @cython.wraparound(False)
    cdef void visit(self, Oct* o, OctInfo* oi) nogil:
        cdef int i
        for i in range(8):
            o.flag1[i] = 0
            o.flag2[i] = 0


cdef class ClearFlag2(Visitor):
    @cython.boundscheck(False)
    @cython.wraparound(False)
    cdef void visit(self, Oct* o, OctInfo* oi) nogil:
        cdef int i
        for i in range(8):
            o.flag2[i] = 0

cdef class PrintVisitor(Visitor):
    cdef bint print_neighbours
    @cython.boundscheck(False)
    @cython.wraparound(False)
    cdef void visit(self, Oct* o, OctInfo* oi) nogil:
        cdef int i
        cdef np.uint8_t ineigh
        cdef Oct* no
        with gil:
            if o.parent != NULL:
                print('\t' * oi.ilvl, f'{o.file_ind} ← {o.parent.file_ind}\t{oi.icell}', end='')
            else:
                print('\t' * oi.ilvl, f'{o.file_ind} ← NULL\t{oi.icell}', end='')

            if self.print_neighbours:
                print(' | ', end='')
                for i in range(6):
                    no = o.neighbours[i]
                    ineigh = o.ineighbours[i]
                    if no != NULL:
                        print(f'{no.file_ind: 7d}[{ineigh}]', end='  ')
                    else:
                        print('_'*10, end='  ')
            print()

cdef class CheckTreeVisitor(Visitor):
    cdef bint error
    def __cinit__(self):
        self.error = False

    cdef void visit(self, Oct* o, OctInfo* oi) nogil:
        cdef int i
        cdef np.uint8_t ineigh
        cdef Oct* no
        # Check parents
        if oi.ilvl > 1:
            if o.parent == NULL:
                self.error = True
                with gil: print('{o.file_ind} is orphan')

        # Check neighbours
        for i in range(6):
            if o.neighbours[i] == NULL:
                self.error = True
                with gil: print('{o.file_ind} has no uncle in direction {i}')

# cdef class MarkDomainVisitor(Visitor):
#     """Mark all cells contained in an oct in the current domain"""
#     cdef np.int64_t idomain

#     def __init__(self, int idomain):
#         self.idomain = idomain

#     @cython.boundscheck(False)
#     @cython.wraparound(False)
#     cdef void visit(self, Oct* o, OctInfo* oi) nogil:
#         cdef Oct *parent

#         assert False
#         # Select all cells in the oct
#         if o.new_domain_ind == self.idomain:
#             # FIXME o.flag1 |= 0b1111_1111

#             # Mark the parents
#             parent = o.parent
#             while parent != NULL:
#                 # FIXME parent.flag1 |= 0b1 << o.icell
#                 o = parent
#                 parent = parent.parent

cdef class ParentVisitor(Visitor):
    cdef Oct* push(self, Oct* o, np.uint8_t icell, queue[OctInfo*] &q, int flag) nogil:
        cdef OctInfo* oi
        if o.flag1[icell] == 0:
            o.flag1[icell] = 1  # select the cell
            oi = <OctInfo*> malloc(sizeof(OctInfo))
            oi.oct = o
            oi.icell = icell
            q.push(oi)
            with gil:
                print(f'\tAdding {o.file_ind}[{icell}] (={o.children[icell].file_ind}) to the queue (flag={flag})')
        return o.children[icell]

    cdef void visit(self, Oct* o, OctInfo* oi0) nogil:
        cdef queue[OctInfo*] q
        cdef np.uint8_t icell
        cdef OctInfo* oi
        cdef Oct* o1
        cdef Oct* o2
        cdef Oct* o3
        cdef int i, j, k

        # Skip root
        if o.parent == NULL:
            return

        self.push(o.parent, o.icell, q, 1)

        while not q.empty():
            oi = q.front()
            q.pop()

            # icell is the index of last visited cell, i.e.
            # the cell is o.children[icell]
            o = oi.oct
            icell = oi.icell
            free(oi)

            # Nothing to do anymore if at root level
            if o.parent == NULL:
                continue

            # Add the parent cell
            self.push(o.parent, o.icell, q, 2)

            # Add the parent's neighbours
            # NOTE: the AMR structure ensures that the 3x3x3 cube
            #       of parents exist, so no need to check for NULL pointers
            for i in range(6):
                o1 = self.push(
                    o.parent.neighbours[i],
                    o.parent.ineighbours[i],
                    q,
                    0
                )
                for j in range(i+1, 6):
                    if (i // 2) == (j // 2):
                        continue
                    o2 = self.push(
                        o1.neighbours[j],
                        o1.ineighbours[j],
                        q,
                        0
                    )
                    for k in range(j+1, 6):
                        if (i // 2) == (k // 2) or (j // 2) == (k // 2):
                            continue
                        o3 = self.push(
                            o2.neighbours[k],
                            o2.ineighbours[k],
                            q,
                            0
                        )


cdef class UncleAuntVisitor(Visitor):
    cdef void visit(self, Oct *o, OctInfo* oi) nogil:
        cdef int i
        cdef Oct* no
        cdef Oct* no2
        cdef np.uint8_t ino, ino2
        cdef queue[Oct*] q
        cdef queue[np.uint8_t] qi

        # Mark the uncle/aunts
        for i in range(6):
            no = o.neighbours[i]
            ino = o.ineighbours[i]
            if no == NULL:
                with gil:
                    raise Exception('Should not happen!')

            if no.flag1[ino] == 0:
                q.push(no)
                qi.push(ino)

        while not q.empty():
            no = q.front()
            ino = qi.front()
            q.pop()
            qi.pop()
            no.flag1[ino] = 1

            # Because of refinement rules, we know that at one level higher,
            # the grid has all its 26 neighbours, let's add all of them
            if no.parent != NULL:
                for i in range(6):
                    no2 = no.neighbours[i]
                    ino2 = no.ineighbours[i]
                    if no2.flag1[ino2] == 0:
                        with gil:
                            print(f'{no2.file_ind} was missed!')
                        q.push(no2)
                        qi.push(ino2)

cdef class HilbertVisitor(Visitor):
    """Mark all cells contained in an oct in the current domain"""
    cdef int128_t bk_low, bk_up

    def __cinit__(self):
        self.parallel_friendly = True

    cdef void visit(self, Oct *o, OctInfo* oi) nogil:
        cdef np.uint64_t lvl
        cdef np.float64_t hk
        cdef int128_t ishift, order_min, order_max
        cdef int i

        lvl = oi.ilvl
        ishift = 3 * (self.bit_length - lvl + 1)

        for i in range(8):
            hk = o.hilbert_key[i]
            order_min = (<int128_t> hk) >> ishift
            order_max = (order_min + 1) << ishift
            order_min <<= ishift

            o.flag1[i] = (order_max > self.bk_low) & (order_min < self.bk_up)


# cdef class HilbertDomainVisitor(Visitor):
#     """Mark all cells contained in an oct in the current domain"""
#     cdef np.uint64_t bk_min, bk_max

#     def __init__(self, np.uint64_t bk_min, np.uint64_t bk_max):
#         self.bk_min = bk_min
#         self.bk_max = bk_max

#     @cython.boundscheck(False)
#     @cython.wraparound(False)
#     cdef void visit(self, Oct* o, OctInfo* oi) nogil:
#         cdef int i, j, k, di, dlvl, ishift
#         cdef np.uint8_t icell
#         cdef np.int64_t ipos_child[3]
#         cdef np.uint64_t order_min, order_max
#         cdef np.uint64_t vlow, vmax
#         cdef bint ok

#         dlvl = self.levelmax - self.ilvl + 1
#         di = 1 << (dlvl - 1)
#         ishift = dlvl * 3

#         cdef bint dbg = ((o.new_domain_ind >= 3) and (o.new_domain_ind <= 5) and (self.ilvl >= 10)) or self.ilvl == 4

#         vlow = 1<<63
#         vmax = 0
#         # Setup children positions
#         for icell in range(8):
#             i = (icell >> 0) & 0b1
#             j = (icell >> 1) & 0b1
#             k = (icell >> 2) & 0b1
#             ipos_child[0] = (self.ipos[0] + di*(2*i-1)) >> dlvl
#             ipos_child[1] = (self.ipos[1] + di*(2*j-1)) >> dlvl
#             ipos_child[2] = (self.ipos[2] + di*(2*k-1)) >> dlvl

#             order_min = hilbert3d_single(ipos_child, self.ilvl)
#             order_max = order_min + 1

#             order_min <<= ishift
#             order_max <<= ishift

#             vlow = min(order_min, vlow)
#             vmax = max(order_max, vmax)

#             ok = (order_max > self.bk_min) & (order_min < self.bk_max)
#             if ok:
#                 o.flag1 |= 1 << icell

#             if False or dbg or not (vlow <= o.hilbert_key <= vmax):
#                 print()
#                 pre = '  '*self.ilvl
#                 print(pre, 'ilvl=%s, icell=%s' % (self.ilvl, icell))
#                 print(pre, self.ipos[0], self.ipos[1], self.ipos[2])
#                 print(pre, (self.ipos[0] + di*(2*i-1)),
#                       (self.ipos[1] + di*(2*j-1)),
#                       (self.ipos[2] + di*(2*k-1)))
#                 print(pre, ipos_child[0], ipos_child[1], ipos_child[2])
#                 print(pre, self.bk_max/2.**(3*self.bit_length),
#                       self.bk_min/2.**(3*self.bit_length))
#                 print(pre, order_min/2.**(3*self.bit_length),
#                       order_max/2.**(3*self.bit_length), ok)

#         if False or dbg or not (vlow <= o.hilbert_key <= vmax):
#             print(pre, f'{o.file_ind}[{o.new_domain_ind}] ({o.hilbert_key / 2.**(3*self.bit_length)})')
#             print(pre, f'vlow={vlow}; vmed={o.hilbert_key}; vmax={vmax}; {vlow<o.hilbert_key}, {o.hilbert_key<vmax}')

# cdef class CountNeighbourFlaggedOctVisitor(Visitor):
#     cdef int n_neigh
#     cdef int _DIRECTIONS[3]

#     cdef bint helper(self, Oct* o, int* directions, int depth, bint dbg):
#         cdef Oct* no
#         cdef np.uint8_t ino
#         cdef int ret

#         if depth == 3:
#             return False

#         for i in range(2):
#             no = o.neighbours[directions[0]*2+i]
#             ino = o.ineighbours[directions[0]*2+i]

#             no = no.children[ino]

#             if no == NULL:
#                 continue

#             if dbg:
#                 print(f'\tjumped from {o.file_ind} to {no.file_ind}')

#             if no.flag1 > 0 or self.helper(no, &directions[1], depth+1, dbg):
#                 return True

#         return False


#     cdef void visit(self, Oct* o, OctInfo* oi) nogil:
#         cdef int i, j, k
#         cdef int directions[3]
#         cdef bint dbg = (o.file_ind == 24994) and (o.domain_ind == 1)

#         if self.ilvl == 1:
#             o.flag1 = 1
#             return

#         for i in range(3):
#             directions[0] = i
#             for j in range(3):
#                 if j == i: continue
#                 directions[1] = j
#                 for k in range(3):
#                     if k == i or k == j: continue
#                     directions[2] = k

#                     if dbg:
#                         print(f'{o.file_ind}[{o.domain_ind}]')
#                         print(f'Exploring {directions[0]} {directions[1]} {directions[2]}')
#                     if self.helper(o, directions, 0, dbg):
#                         o.flag1 = 1
#                         o.parent.flag1 = 1
#                         return

# cdef class CountNeighbourFlaggedOctVisitor(Visitor):
#     cdef int n_neigh
#     @cython.cdivision(True)
#     cdef void visit(self, Oct* o, OctInfo* oi) nogil:
#         cdef int i
#         cdef Oct *no
#         cdef np.uint8_t ino

#         cdef int count = 0


#         if self.ilvl == 1:
#             for i in range(8):
#                 o.flag2[i] = 1
#             return

#         for i in range(6):
#             no = o.neighbours[i]
#             ino = o.ineighbours[i]

#             if no == NULL:
#                 with gil:
#                     print(f'This should not happen...@lvl={self.ilvl}')
#                     raise Exception()

#             if no.children[ino] != NULL:
#                 no = no.children[ino]
#                 if no.flag1[ino] > 0:
#                     count += 1
#                 if count >= self.n_neigh:
#                     break

#         if count >= self.n_neigh:
#             # FIXME o.flag2 = 1
#             pass
#         else:
#             # FIXME o.flag2 = 0
#             pass

cdef class CountNeighbourCellFlaggedVisitor(Visitor):
    """Select all cells in a domain + the ones directly adjacent to them."""
    cdef int n_neigh
    cdef np.uint8_t[:, ::1] neigh_grid, neigh_cell

    def __cinit__(self):
        cdef np.uint8_t _ = 6
        self.neigh_grid = np.array(
            [
                [0, _, 0, _, 0, _, 0, _], # -x
                [_, 1, _, 1, _, 1, _, 1], # +x
                [2, 2, _, _, 2, 2, _, _], # -y
                [_, _, 3, 3, _, _, 3, 3], # +y
                [4, 4, 4, 4, _, _, _, _], # -z
                [_, _, _, _, 5, 5, 5, 5]  # +z
            ], order='F', dtype=np.uint8
        ).T
        _ = 8
        self.neigh_cell = np.array(
            [
                [1, 0, 3, 2, 5, 4, 7, 6], # -x
                [1, 0, 3, 2, 5, 4, 7, 6], # +x
                [2, 3, 0, 1, 6, 7, 4, 5], # -y
                [2, 3, 0, 1, 6, 7, 4, 5], # +y
                [4, 5, 6, 7, 0, 1, 2, 3], # -z
                [4, 5, 6, 7, 0, 1, 2, 3]  # +z
            ], order='F', dtype=np.uint8
        ).T

        self.parallel_friendly = True

    @cython.boundscheck(False)
    @cython.wraparound(False)
    @cython.cdivision(True)
    cdef void visit(self, Oct* o, OctInfo* oi) nogil:
        cdef Oct* parent
        cdef Oct* neigh
        cdef np.uint8_t neigh_icell
        cdef int icell, idim, itmp, count

        # Loop over cells
        for icell in range(8):
            count = 0

            # Find all neighbour cells
            for idim in range(6):
                itmp = self.neigh_grid[icell, idim]
                if itmp == 6:  # 6 is myself!
                    neigh = o
                else:
                    # Get the uncle/aunt of oct ...
                    neigh = o.neighbours[itmp]
                    if neigh == NULL:
                        continue
                    # ... and now the sibling of the oct
                    neigh = neigh.children[o.ineighbours[itmp]]

                # Add 8 to account for bit shifting
                neigh_icell = self.neigh_cell[icell, idim]

                # Neighbouring cell
                if neigh != NULL:
                    if neigh.flag1[neigh_icell] == 1:
                        count +=1

                    if count >= self.n_neigh:
                        o.flag2[icell] = 1
                        break

        # Tag oct upwards
        parent = o.parent
        while parent != NULL and (parent.flag2[o.icell] == 0):
            parent.flag2[o.icell] = 1
            o = parent
            parent = parent.parent

cdef class SetFlag1(Visitor):
    def __cinit__(self):
        self.parallel_friendly = True

    @cython.boundscheck(False)
    @cython.wraparound(False)
    cdef void visit(self, Oct* o, OctInfo* oi) nogil:
        cdef int i
        for i in range(8):
            o.flag1[i] = 1

cdef class SetFlag2FromFlag1(Visitor):
    def __cinit__(self):
        self.parallel_friendly = True

    @cython.boundscheck(False)
    @cython.wraparound(False)
    cdef void visit(self, Oct* o, OctInfo* oi) nogil:
        cdef int i
        for i in range(8):
            if o.flag2[i] == 1:
                o.flag1[i] = 1

cdef class CountVisitor(Visitor):
    cdef int count
    def __init__(self, int initial_value=0):
        self.count = initial_value

    @cython.boundscheck(False)
    @cython.wraparound(False)
    cdef void visit(self, Oct* o, OctInfo* oi) nogil:
        self.count += 1

cdef class ExtractionVisitor(Visitor):
    cdef np.int32_t[::1] file_ind
    cdef np.int32_t[::1] domain_ind
    cdef np.int32_t[::1] owning_cpu
    cdef np.int32_t[::1] new_domain_ind
    cdef np.int32_t[::1] lvl
    cdef np.int64_t[:, :, ::1] nbor
    cdef np.int64_t[:, ::1] son
    cdef np.int64_t[:, ::1] parent
    cdef np.int32_t[:, ::1] refmap

    cdef int ind_glob

    def __init__(self):
        self.ind_glob = 0

    @cython.boundscheck(False)
    @cython.wraparound(False)
    cdef void visit(self, Oct* o, OctInfo* oi) nogil:
        cdef int i, ishift
        cdef np.int64_t ii, jj, kk
        cdef np.uint8_t icell
        cdef Oct* no

        self.file_ind[self.ind_glob] = o.file_ind
        self.domain_ind[self.ind_glob] = o.domain_ind
        self.owning_cpu[self.ind_glob] = o.owning_cpu
        self.new_domain_ind[self.ind_glob] = o.new_domain_ind
        self.lvl[self.ind_glob] = oi.ilvl

        # Fill neighbours
        for i in range(6):
            no = o.neighbours[i]
            icell = o.ineighbours[i]
            if no != NULL:
                self.nbor[self.ind_glob, i, 0] = encode_mapping(no.file_ind, no.owning_cpu)
                self.nbor[self.ind_glob, i, 1] = <np.int64_t> icell
                if self.nbor[self.ind_glob, i, 0] < 0:
                    raise Exception(
                        'This should not happen when encoding',
                        no.file_ind, no.domain_ind, encode_mapping(no.file_ind, no.owning_cpu)
                    )
        # Fill son
        for i in range(8):
            no = o.children[i]
            if no != NULL:
                self.son[self.ind_glob, i] = encode_mapping(no.file_ind, no.owning_cpu)
                # if self.ilvl <= 2:
                #     print(f'{o.file_ind}.{i} -> {no.file_ind}[{no.owning_cpu}]')

        # Fill parent
        no = o.parent
        if no != NULL:  # only for root
            self.parent[self.ind_glob, 0] = encode_mapping(no.file_ind, no.owning_cpu)
            self.parent[self.ind_glob, 1] = o.icell

        # Fill refmap
        for i in range(8):
            self.refmap[self.ind_glob, i] = o.refmap[i]

        self.ind_glob += 1

#############################################################
# Selectors
#############################################################
cdef class Octree

cdef inline integral find(integral i, integral j, integral k):
    return i + 2*j + 4*k

cdef inline integral cind(integral i, integral j, integral k):
    return 4*i + 2*j + k

cdef class Selector:
    cdef Octree octree
    cdef int levelmin, levelmax
    cdef int bit_length # number of bits per dimension in hilbert key

    def __init__(self, Octree octree):
        self.octree = octree
        self.levelmin = self.octree.levelmin
        self.levelmax = self.octree.levelmax
        self.bit_length = self.levelmax + 1

    @cython.cdivision(True)
    cdef void visit_all_octs(self, Visitor visitor, str traversal='depth_first'):
        cdef int ilvl
        cdef Oct* root
        cdef np.uint64_t ipos[3]
        cdef np.uint64_t di
        root = <Oct*> self.octree.root
        di = 1 << self.levelmax
        # The position is between 0 and 2 * 2**levelmax
        ipos[0] = di
        ipos[1] = di
        ipos[2] = di
        ilvl = 1

        visitor.levelmin = self.levelmin
        visitor.levelmax = self.levelmax
        visitor.bit_length = self.bit_length

        if traversal == 'depth_first':
            self.recursively_visit_all_octs(root, ipos, ilvl, visitor, di / 2, icell=0)
        elif traversal == 'breadth_first':
            self.breadth_first_visit_all_octs(root, ipos, ilvl, visitor)
        else:
            raise NotImplementedError(traversal)

    @cython.cdivision(True)
    cdef void recursively_visit_all_octs(self, Oct* o, np.uint64_t ipos[3], const int ilvl, Visitor visitor, const np.uint64_t di, np.uint8_t icell):
        cdef int i, j, k
        cdef OctInfo oi
        if o == NULL:
            return

        if self.select(o, ipos, ilvl, icell):
            # Visit the oct
            oi.ilvl = <int> ilvl
            oi.ipos = ipos
            oi.icell = icell
            if debug: print('  '*ilvl, f'DF: Visiting {o.file_ind}')
            visitor.visit(o, &oi)

            # Visit its children
            for k in range(2):
                ipos[2] += (2*k - 1) * di
                for j in range(2):
                    ipos[1] += (2*j - 1) * di
                    for i in range(2):
                        ipos[0] += (2*i - 1) * di
                        icell = find(i, j, k)
                        self.recursively_visit_all_octs(o.children[icell], ipos, ilvl+1, visitor, di / 2, icell)
                        ipos[0] -= (2*i - 1) * di
                    ipos[1] -= (2*j - 1) * di
                ipos[2] -= (2*k - 1) * di

    cdef void empty_queue(self, vector[OctInfo*] &q2, Visitor visitor) nogil:
        cdef int ii
        cdef OctInfo* oi
        for ii in prange(q2.size()):
            oi = q2[ii]
            visitor.visit(oi.oct, oi)
            free(oi)
        q2.clear()

    @cython.cdivision(True)
    cdef void breadth_first_visit_all_octs(self, Oct* root, np.uint64_t ipos[3], int ilvl, Visitor visitor):
        cdef queue[OctInfo*] q
        cdef vector[OctInfo*] q2
        cdef OctInfo *oi
        cdef OctInfo *oi2
        cdef Oct* o
        cdef int di, i, j, k, icell

        cdef size_t CHUNK_SIZE = max(1, 128 * openmp.omp_get_num_threads())

        oi = <OctInfo*> malloc(sizeof(OctInfo))
        oi.oct = root
        oi.ipos = ipos
        oi.ilvl = ilvl

        q.push(oi)
        if visitor.parallel_friendly:
            q2.reserve(CHUNK_SIZE)

        while not q.empty():
            oi = q.front()
            q.pop()

            if not self.select(oi.oct, oi.ipos, oi.ilvl, oi.icell):
                free(oi)
                continue

            if debug: print('  '*oi.ilvl, f'BF: Visiting {oi.oct.file_ind}')

            di = 1 << (self.levelmax-oi.ilvl)
            # Visit its children
            for k in range(2):
                oi.ipos[2] += (2*k - 1) * di
                for j in range(2):
                    oi.ipos[1] += (2*j - 1) * di
                    for i in range(2):
                        oi.ipos[0] += (2*i - 1) * di
                        icell = find(i, j, k)
                        o = oi.oct.children[icell]
                        if o == NULL:
                            continue
                        oi2 = <OctInfo*> malloc(sizeof(OctInfo))
                        oi2.oct = o
                        oi2.ipos[0] = oi.ipos[0]
                        oi2.ipos[1] = oi.ipos[1]
                        oi2.ipos[2] = oi.ipos[2]
                        oi2.ilvl = oi.ilvl + 1
                        oi2.icell = icell
                        q.push(oi2)
                        oi.ipos[0] -= (2*i - 1) * di
                    oi.ipos[1] -= (2*j - 1) * di
                oi.ipos[2] -= (2*k - 1) * di

            if visitor.parallel_friendly:
                q2.push_back(oi)
                if q2.size() == CHUNK_SIZE:
                    self.empty_queue(q2, visitor)
            else:
                visitor.visit(oi.oct, oi)
                free(oi)

        if visitor.parallel_friendly:
            if q2.size() > 0:
                self.empty_queue(q2, visitor)


    cdef bint select(self, Oct* o, const np.uint64_t ipos[3], const int ilvl, const np.uint8_t icell):
        raise NotImplementedError

# Select all octs
cdef class AllOctsSelector(Selector):
    cdef bint select(self, Oct* o, const np.uint64_t ipos[3], const int ilvl, const np.uint8_t icell):
        return o != NULL

cdef class AllCellsSelector(Selector):
    cdef bint select(self, Oct* o, const np.uint64_t ipos[3], const int ilvl, const np.uint8_t icell):
        return True

cdef class LevelSelector(Selector):
    cdef int lvl
    def __init__(self, Octree octree, int lvl):
        super(LevelSelector, self).__init__(octree)
        self.lvl = lvl

    cdef bint select(self, Oct* o, const np.uint64_t ipos[3], const int ilvl, const np.uint8_t icell):
        return ilvl <= self.lvl


cdef class FlaggedOctSelector(Selector):
    """Traverse all octs that have one flag1 set"""
    cdef bint select(self, Oct* o, const np.uint64_t ipos[3], const int ilvl, const np.uint8_t icell):
        if ilvl == 1:   # special case for coarse lvl
            return True

        cdef int i
        for i in range(8):
            if o.flag1[i] == 1:
                return True
        else:
            return False

cdef class FlaggedParentOctSelector(Selector):
    """Traverse all octs using the parent flag1"""
    cdef bint select(self, Oct* o, const np.uint64_t ipos[3], const int ilvl, const np.uint8_t icell):
        if ilvl == 1:   # special case for coarse lvl
            return True
        return o.parent.flag1[icell] > 0

# # Select all octs that may any key in range provided
# cdef class HilbertSelector(Selector):
#     cdef np.uint64_t key_lower
#     cdef np.uint64_t key_upper
#     def __init__(self, Octree octree, np.uint64_t key_lower, np.uint64_t key_upper):
#         super(HilbertSelector, self).__init__(octree)
#         self.key_lower = key_lower
#         self.key_upper = key_upper

#     cdef bint select(self, Oct* o, const np.uint64_t ipos[3], const int ilvl, const np.uint8_t icell):
#         # Compute hilbert key
#         cdef np.uint64_t order_min = o.hilbert_key
#         cdef np.uint64_t order_max

#         cdef int ishift = (self.bit_length - ilvl + 1)*3

#         order_min >>= ishift
#         order_max = order_min + 1

#         order_min <<= ishift
#         order_max <<= ishift

#         # print('  '*ilvl, f'Selecting? {o.file_ind} {order_min} {self.key_lower} | {order_max} {self.key_upper}', (order_max > self.key_lower) & (order_min < self.key_upper))

#         return (order_max > self.key_lower) & (order_min < self.key_upper)

#############################################################
# Octree implementation
#############################################################
cdef bint debug = False
cdef class Octree:
    cdef Oct* root
    cdef int levelmax
    cdef int levelmin
    cdef int _ntot
    cdef int old_ncpu
    cdef int new_ncpu

    def __cinit__(self):
        if debug: print("Malloc'ing root")
        self.root = <Oct*> malloc(sizeof(Oct))
        setup_oct(self.root, parent=NULL, icell=8)
        # Set root neighbours
        for i in range(6):
            self.root.neighbours[i] = self.root
            self.root.ineighbours[i] = 0

    def __init__(self, int levelmin, int levelmax, int old_ncpu, int new_ncpu):
        self.levelmin = levelmin
        self.levelmax = levelmax
        self._ntot = 8  # Count the 8 cells in the root oct
        self.old_ncpu = old_ncpu
        self.new_ncpu = new_ncpu

    @cython.boundscheck(False)
    @cython.cdivision(True)
    @cython.wraparound(False)
    def add(self, const np.int64_t[:, ::1] ipos,
                  const np.int64_t[::1] file_ind,
                  const np.int64_t[::1] domain_ind,
                  const np.int64_t[::1] new_domain_ind,
                  const np.int64_t[::1] owning_cpu,
                  const np.float64_t[:, ::1] hilbert_key,
                  const np.int32_t[:, ::1] refmap,
                  const np.int64_t[::1] lvl):
        cdef int N, i, ilvl
        cdef np.uint8_t ichild
        cdef Oct* node
        cdef int nbefore = self.ntot

        N = ipos.shape[0]

        for i in range(N):
            node = self.get(&ipos[i, 0], lvl[i], create_child=True)
            if node == NULL:
                print('THIS SHOULD NOT HAPPEN')
                raise Exception()
            node.file_ind = file_ind[i]
            node.domain_ind = domain_ind[i]
            node.owning_cpu = owning_cpu[i]
            node.new_domain_ind = new_domain_ind[i]

            for j in range(8):
                node.hilbert_key[j] = hilbert_key[i, j]
                if node.refmap[j] != -1 and node.refmap[j] != refmap[i, j]:
                    print(f'Changing {node.refmap[j]}->{refmap[i,j]} for oct#{file_ind[i]}')
                node.refmap[j] = refmap[i, j]

        return self.ntot - nbefore

    cdef Oct* get_child(self, Oct* parent, const np.uint8_t ichild, bint create_child) nogil:
        cdef Oct* o

        o = parent.children[ichild]
        # Create child if doesn't exist
        if o == NULL:
            if not create_child:
                return NULL

            if debug:
                with gil:
                    print(f'Creating child <Oct #{parent.file_ind}>.children[{ichild}]')
            o = <Oct*> malloc(sizeof(Oct))
            self._ntot += 1

            parent.children[ichild] = o
            setup_oct(o, parent, ichild)

        return o


    cdef Oct* get(
            self,
            const np.int64_t* ipos,
            const np.int64_t lvl,
            bint create_child=False,
            np.uint8_t* ichild_ret=NULL,
            bint return_parent=False
        ) nogil:
        """Get an oct from the tree.

        Parameters
        ----------
        ipos : the integer position (between 0 and 2**self.bit_length)
        lvl : the level at which to find the oct
        create_child :
            If True, create missing nodes in the tree
        ichild_ret :
            The index of the last child when going down the tree (i.e.
            the index of the deepest oct (if any) in its parent.
            If the deepest oct does not exist, this qty is computed
        return_parent :
            If True, stop the search one level higher.
        """
        cdef int ilvl
        cdef Oct* o = self.root
        cdef Oct* parent = NULL # parent oct
        cdef np.uint8_t ichild = 8

        if debug:
            with gil:
                print(f'Getting {ipos[0]} {ipos[1]} {ipos[2]} @ lvl={lvl}')

        for ilvl in range(lvl-1):
            # Compute index of child
            ichild  = (ipos[0] >> (self.levelmax-ilvl-0)) & 0b001
            ichild |= (ipos[1] >> (self.levelmax-ilvl-1)) & 0b010
            ichild |= (ipos[2] >> (self.levelmax-ilvl-2)) & 0b100

            if debug:
                with gil:
                    print('\t'*ilvl, f'ilvl={ilvl}\tichild={ichild}\t{create_child}')

            parent = o
            o = self.get_child(o, ichild, create_child)
            if o == NULL:
                break

        # Store the index of the last child
        if ichild_ret != NULL:
            ichild_ret[0] = ichild
        if return_parent:
            return parent

        return o

    @cython.boundscheck(False)
    @cython.cdivision(True)
    @cython.wraparound(False)
    def set_neighbours(self, const np.int64_t[:, ::1] ipos, const np.int64_t[:, :, ::1] neigh_pos, const np.int64_t[::1] ilvl):
        cdef int i, j, N = ipos.shape[0]
        cdef Oct* o
        cdef Oct* neigh
        cdef np.uint8_t ichild

        for i in range(N):
            o = self.get(&ipos[i, 0], ilvl[i])
            if o == NULL:
                raise Exception(
                    'Could not find oct at position %s,%s,%s, level=%s.' %
                    (ipos[i, 0], ipos[i, 1], ipos[i, 2], ilvl[i])
                )
            for j in range(6):
                neigh = self.get(
                    &neigh_pos[i, j, 0],
                    ilvl[i],
                    create_child=False,
                    ichild_ret=&ichild,
                    return_parent=True
                )
                if neigh == NULL:
                    continue

                o.neighbours[j] = neigh
                o.ineighbours[j] = ichild

    def clear_paint(self):
        cdef AllOctsSelector oct_selector = AllOctsSelector(self)
        cdef ClearPaint cp = ClearPaint()
        oct_selector.visit_all_octs(cp)

    @cython.boundscheck(False)
    def select(self, np.int64_t[:, ::1] ipos, np.int64_t[::1] ilvl):
        '''Select *cells* in the tree at provided positions'''
        cdef int N = len(ipos)
        cdef int i, count = 0
        cdef np.uint8_t ichild

        cdef Oct* o

        for i in prange(N, nogil=True):
            o = self.get(&ipos[i, 0], ilvl[i], False,
                         ichild_ret=&ichild, return_parent=True)
            if o == NULL:
                with gil:
                    print()
                    raise Exception(
                        'Cannot find oct at position %s, %s, %s.'
                        'This should not happen!' %
                        (ipos[i, 0], ipos[i, 1], ipos[i, 2])
                    )

            # Set flag1 for cell
            if o.flag1[ichild] == 0:
                o.flag1[ichild] = 1
                count += 1

            # Set flag1 for parent
            if o.parent != NULL:
                o.parent.flag1[o.icell] |= 0b1
        return count

    def count_octs(self):
        cdef AllOctsSelector all_octs = AllOctsSelector(self)
        cdef CountVisitor counter = CountVisitor(0)
        all_octs.visit_all_octs(counter)

        return counter.count

    def select_level(self, int level_max):
        cdef LevelSelector sel = LevelSelector(self, level_max)
        cdef SetFlag1 set_flag1 = SetFlag1()

        sel.visit_all_octs(set_flag1, traversal='breadth_first')

    def select_hilbert(self, bk_low, bk_up):
        cdef AllOctsSelector all_octs = AllOctsSelector(self)
        cdef HilbertVisitor select_hilbert = HilbertVisitor()

        select_hilbert.bk_low = bk_low
        select_hilbert.bk_up = bk_up

        print(f'\thilbert key at boundary: {select_hilbert.bk_low} {select_hilbert.bk_up}')
        all_octs.visit_all_octs(select_hilbert, traversal='breadth_first')

    @cython.boundscheck(False)
    @cython.wraparound(False)
    def expand_boundaries(
            self,
            int idomain,
            int nexpand=1
    ):
        '''Yield the file and domain indices sorted by level'''
        cdef AllOctsSelector all_octs = AllOctsSelector(self)
        cdef AllCellsSelector all_cells = AllCellsSelector(self)
        cdef FlaggedOctSelector selected_octs = FlaggedOctSelector(self)
        cdef FlaggedParentOctSelector selected_parent_octs = FlaggedParentOctSelector(self)
        cdef int Noct, idim, i

        # Mark all cells contained in an oct in the domain
        # cdef MarkDomainVisitor mark_domain = MarkDomainVisitor(idomain)
        # all_octs.visit_all_octs(mark_domain, traversal='breadth_first')

        # Expand boundaries
        # flag1: already contains marked cells
        # flag2: used as temp array
        # cdef CountNeighbourFlaggedOctVisitor count_neigh = CountNeighbourFlaggedOctVisitor()
        # cdef SetFlag2FromFlag1 copy_flag2_in_flag1 = SetFlag2FromFlag1()
        # for i in range(nexpand):
        #     for idim in range(3):
        #         print('Expanding - pass %s:%s' % (i, idim))
        #         count_neigh.n_neigh = idim + 1
        #         all_octs.visit_all_octs(count_neigh)
        #         all_octs.visit_all_octs(copy_flag2_in_flag1)

        # Make sure the parents are selected
        print('Selecting parents')
        cdef ParentVisitor select_parents = ParentVisitor()
        cdef SetFlag2FromFlag1 copy_flag2_in_flag1 = SetFlag2FromFlag1()
        selected_octs.visit_all_octs(select_parents, traversal='breadth_first')
        all_octs.visit_all_octs(copy_flag2_in_flag1, traversal='breadth_first')

        cdef CountNeighbourCellFlaggedVisitor count_neigh = CountNeighbourCellFlaggedVisitor()
        for i in range(nexpand):
            for idim in range(3):
                print('Expanding - pass %s:%s' % (i, idim))
                count_neigh.n_neigh = idim + 1
                all_octs.visit_all_octs(count_neigh, traversal='breadth_first')
                all_octs.visit_all_octs(copy_flag2_in_flag1, traversal='breadth_first')

        cdef ClearFlag2 clear_flag2 = ClearFlag2()
        all_octs.visit_all_octs(clear_flag2)

        # Make sure all parents + uncles/aunts are selected
        # print('Selecting uncles')
        cdef UncleAuntVisitor select_uncle = UncleAuntVisitor()
        selected_octs.visit_all_octs(select_uncle, traversal='breadth_first')
        all_octs.visit_all_octs(copy_flag2_in_flag1, traversal='breadth_first')

        # print('Ensure AMR structure is complete')
        selected_octs.visit_all_octs(select_parents, traversal='breadth_first')
        all_octs.visit_all_octs(copy_flag2_in_flag1)

    @cython.wraparound(False)
    @cython.boundscheck(False)
    def extract_data(self):
        """Assuming all the relevant cells have been flagged in flag1,
        build and return the AMR structure"""
        cdef int Noct
        cdef FlaggedOctSelector selected_octs = FlaggedOctSelector(self)
        # Count number of selected octs
        cdef CountVisitor counter = CountVisitor(0)
        selected_octs.visit_all_octs(counter)

        Noct = counter.count

        # Extract indices
        cdef ExtractionVisitor extract = ExtractionVisitor()

        cdef np.ndarray[np.int32_t, ndim=1] file_ind_arr = np.full(Noct, -1, np.int32)
        cdef np.ndarray[np.int32_t, ndim=1] domain_ind_arr = np.full(Noct, -1, np.int32)
        cdef np.ndarray[np.int32_t, ndim=1] owning_cpu_arr = np.full(Noct, -1, np.int32)
        cdef np.ndarray[np.int32_t, ndim=1] new_domain_ind_arr = np.full(Noct, -1, np.int32)
        cdef np.ndarray[np.int32_t, ndim=1] lvl_arr = np.full(Noct, -1, np.int32)
        cdef np.ndarray[np.int64_t, ndim=3] nbor_arr = np.full((Noct, 6, 2), 0, np.int64)
        cdef np.ndarray[np.int64_t, ndim=2] son_arr = np.full((Noct, 8), 0, np.int64)
        cdef np.ndarray[np.int64_t, ndim=2] parent_arr = np.full((Noct, 2), 0, np.int64)
        cdef np.ndarray[np.int32_t, ndim=2] refmap_arr = np.full((Noct, 8), -1, np.int32)
        cdef np.int32_t[::1] file_ind = file_ind_arr
        cdef np.int32_t[::1] domain_ind = domain_ind_arr
        cdef np.int32_t[::1] owning_cpu = owning_cpu_arr
        cdef np.int32_t[::1] new_domain_ind = new_domain_ind_arr
        cdef np.int32_t[::1] lvl = lvl_arr
        cdef np.int64_t[:, :, ::1] nbor = nbor_arr
        cdef np.int64_t[:, ::1] son = son_arr
        cdef np.int64_t[:, ::1] parent = parent_arr
        cdef np.int32_t[:, ::1] refmap = refmap_arr

        extract.file_ind = file_ind
        extract.domain_ind = domain_ind
        extract.owning_cpu = owning_cpu
        extract.new_domain_ind = new_domain_ind
        extract.lvl = lvl
        extract.nbor = nbor
        extract.son = son
        extract.parent = parent
        extract.refmap = refmap
        print('Extracting data')
        selected_octs.visit_all_octs(extract, traversal='breadth_first')

        # At this point we have
        # - file_ind   : the position within the old file
        # - owning_cpu : the old CPU domains to read
        # - new_domain_ind : the new CPU domain
        # - lvl    : the level of the oct

        # We need now to reorder the domains in each lvl
        cdef int ilvl, i0, i
        cdef np.int64_t[::1] order
        i0 = 0
        i = 0

        def _sort(arr):
            arr[i0:i] = arr[i0:i][order]

        for ilvl in range(1, self.levelmax+1):
            while i < Noct and lvl[i] == ilvl:
                i += 1

            order = np.argsort(new_domain_ind_arr[i0:i], kind='stable')
            _sort(file_ind_arr)
            _sort(domain_ind_arr)
            _sort(owning_cpu_arr)
            _sort(new_domain_ind_arr)
            _sort(nbor_arr)
            _sort(son_arr)
            _sort(parent_arr)
            _sort(refmap_arr)

            # No need to do this for lvl ind, already sorted
            i0 = i

        son = son_arr
        parent = parent_arr
        nbor = nbor_arr

        # Create map from global position to local one
        cdef unordered_map[np.uint64_t, np.int64_t] global_to_local
        cdef np.int64_t key
        print('Global to loc')
        for i in range(Noct):
            key = encode_mapping(file_ind[i], owning_cpu[i])
            global_to_local[key] = i + 1

        # Global indices -> local indices
        print('Inverting indices')
        cdef unordered_map[np.uint64_t, np.int64_t].iterator it, end
        cdef bint dbg = True
        end = global_to_local.end()
        for i in range(Noct):
            if i == 0:
                # Special case for root:
                for j in range(6):
                    nbor[i, j, 0] = 0
                    nbor[i, j, 1] = 0
            else:
                for j in range(6):
                    it = global_to_local.find(nbor[i, j, 0])
                    if it != end:
                        nbor[i, j, 0] = deref(it).second
                    else:
                        # print(f'Could not set {i}.nbor[{j}] ({nbor[i,j,0]})')
                        nbor[i, j, 0] = -1  # compensate for ncoarse
                        nbor[i, j, 1] = 0
            for j in range(8):
                it = global_to_local.find(son[i, j])
                if it != end:
                    son[i, j] = deref(it).second
                else:
                    son[i, j] = 0
            it = global_to_local.find(parent[i, 0])
            if it != end:
                parent[i, 0] = deref(it).second
            else:
                parent[i, 0] = 0
                if i > 0:   # normal for root
                    raise Exception(f'Could not find parent, this should not happen! (i={i})')

        # Create AMR structure
        print('Creating AMR structure for %s octs' % Noct)
        cdef int icpu
        cdef np.ndarray[np.int32_t, ndim=2] headl_arr = np.zeros((self.levelmax, self.new_ncpu), dtype=np.int32)
        cdef np.ndarray[np.int32_t, ndim=2] taill_arr = np.zeros((self.levelmax, self.new_ncpu), dtype=np.int32)
        cdef np.ndarray[np.int32_t, ndim=2] numbl_arr = np.zeros((self.levelmax, self.new_ncpu), dtype=np.int32)

        cdef np.int32_t[:, ::1] headl = headl_arr
        cdef np.int32_t[:, ::1] taill = taill_arr
        cdef np.int32_t[:, ::1] numbl = numbl_arr
        cdef np.int32_t[::1] inext = np.zeros(Noct, np.int32)
        cdef np.int32_t[::1] iprev = np.zeros(Noct, np.int32)

        icpu = 0
        # Compute headl,taill, numbl
        for i in range(Noct):
            ilvl = lvl[i]
            # Set head at beginning new level
            icpu = new_domain_ind[i]
            if headl[ilvl-1, icpu-1] == 0:
                headl[ilvl-1, icpu-1] = i + 1
            taill[ilvl-1, icpu-1] = i + 1
            numbl[ilvl-1, icpu-1] += 1

            # Set prev array
            iprev[i] = i
            if i == 0 or lvl[i-1] < ilvl or new_domain_ind[i-1] != icpu:
                iprev[i] = 0

            # Set next array
            inext[i] = i + 2
            if i == Noct-1 or lvl[i+1] > ilvl or new_domain_ind[i+1] != icpu:
                inext[i] = 0


        return dict(
            file_ind=file_ind_arr,
            domain_ind=domain_ind_arr,
            owning_cpu=owning_cpu_arr,
            new_domain_ind=new_domain_ind_arr,
            lvl=lvl_arr,
            nbor=nbor_arr,
            headl=headl_arr,
            taill=taill_arr,
            numbl=numbl_arr,
            son=son_arr,
            parent=parent_arr,
            next=np.asarray(inext),
            prev=np.asarray(iprev),
            refmap=refmap_arr
        )

    @property
    def ntot(self):
        return self._ntot # self.count(self.root)

    cdef int count(self, Oct* o):
        cdef int N = 8
        for i in range(8):
            if o.children[i] != NULL:
                N += self.count(o.children[i])
        return N

    def __dealloc__(self):
        if debug: print('Deallocating')
        # TODO: go through the tree and deallocate everything
        self.mem_free(self.root)

    cdef void mem_free(self, Oct* node):
        cdef int i

        if node.children == NULL:
            return

        for i in range(8):
            if node.children[i] != NULL:
                self.mem_free(node.children[i])
                free(node.children[i])

    def print_tree(self, int lvl_max, bint print_neighbours=False):
        """ Print the tree"""
        cdef LevelSelector sel = LevelSelector(self, lvl_max)
        cdef PrintVisitor visit = PrintVisitor(self)
        visit.print_neighbours = print_neighbours

        sel.visit_all_octs(visit)


    def check_tree(self, int lvl_max=999):
        cdef LevelSelector sel = LevelSelector(self, lvl_max)
        cdef CheckTreeVisitor check = CheckTreeVisitor()

        sel.visit_all_octs(check)

        if check.error:
            raise Exception('Errors were found in the tree!')
