# distutils: language = c++
import numpy as np
cimport numpy as np

from libc.stdlib cimport malloc, free
from libcpp.queue cimport queue
from libcpp.unordered_map cimport unordered_map
from libcpp.map cimport map

from libcpp.pair cimport pair
from cython cimport integral
from cython.operator cimport dereference as deref
cimport cython

from .hilbert cimport hilbert3d_single

cdef struct Oct:
    np.int64_t file_ind       # on file index
    np.int64_t domain_ind     # original domain
    np.int64_t new_domain_ind # new domain
    np.int64_t flag1         # attribute for selection
    np.uint8_t flag2           # temporary flag2
    np.uint64_t hilbert_key

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
    o.parent = parent
    o.icell = icell
    o.file_ind = -1
    o.domain_ind = -1
    o.new_domain_ind = -1
    o.hilbert_key = -1
    o.flag1 = 0
    o.flag2 = 0

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
    cdef np.int64_t ipos[3]
    cdef int ilvl
    cdef int levelmax, levelmin
    cdef int bit_length
    cdef np.uint8_t icell
    cdef void visit(self, Oct* o):
        pass

cdef class ClearPaint(Visitor):
    @cython.boundscheck(False)
    @cython.wraparound(False)
    cdef void visit(self, Oct* o):
        o.flag1 = 0
        o.flag2 = 0

cdef class PrintVisitor(Visitor):
    cdef bint print_neighbours
    @cython.boundscheck(False)
    @cython.wraparound(False)
    cdef void visit(self, Oct* o):
        cdef int i
        cdef np.uint8_t ineigh
        cdef Oct* no
        if o.parent != NULL:
            print('\t' * self.ilvl, f'{o.file_ind} ← {o.parent.file_ind}\t{self.icell}', end='')
        else:
            print('\t' * self.ilvl, f'{o.file_ind} ← NULL\t{self.icell}', end='')

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


cdef class MarkDomainVisitor(Visitor):
    """Mark all cells contained in an oct in the current domain"""
    cdef np.int64_t idomain

    def __init__(self, int idomain):
        self.idomain = idomain

    @cython.boundscheck(False)
    @cython.wraparound(False)
    cdef void visit(self, Oct* o):
        cdef Oct *parent

        # Select all cells in the oct
        if o.new_domain_ind == self.idomain:
            o.flag1 |= 0b1111_1111

            # Mark the parents
            parent = o.parent
            while parent != NULL:
                parent.flag1 |= 0b1 << o.icell
                o = parent
                parent = parent.parent

cdef class HilbertDomainVisitor(Visitor):
    """Mark all cells contained in an oct in the current domain"""
    cdef np.uint64_t bk_min, bk_max

    def __init__(self, np.uint64_t bk_min, np.uint64_t bk_max):
        self.bk_min = bk_min
        self.bk_max = bk_max

    @cython.boundscheck(False)
    @cython.wraparound(False)
    cdef void visit(self, Oct* o):
        cdef int i, j, k, di, dlvl, ishift
        cdef np.uint8_t icell
        cdef np.int64_t ipos_child[3]
        cdef np.uint64_t order_min, order_max
        cdef bint ok

        dlvl = self.levelmax - self.ilvl + 1
        di = 1 << (dlvl - 1)
        ishift = dlvl * 3

        # Setup children positions
        for icell in range(8):
            i = (icell >> 0) & 0b1
            j = (icell >> 1) & 0b1
            k = (icell >> 2) & 0b1
            ipos_child[0] = (self.ipos[0] + di*(2*i-1)) >> dlvl
            ipos_child[1] = (self.ipos[1] + di*(2*j-1)) >> dlvl
            ipos_child[2] = (self.ipos[2] + di*(2*k-1)) >> dlvl

            order_min = hilbert3d_single(ipos_child, self.ilvl)
            order_max = order_min + 1

            order_min <<= ishift
            order_max <<= ishift

            ok = (order_max > self.bk_min) & (order_min < self.bk_max)
            o.flag1 |= (<np.uint8_t>ok) << icell
            
            if self.ilvl < 3:
                print()
                pre = '  '*self.ilvl
                print(pre, 'ilvl=%s, icell=%s' % (self.ilvl, icell))
                print(pre, self.ipos[0], self.ipos[1], self.ipos[2])
                print(pre, (self.ipos[0] + di*(2*i-1)),
                      (self.ipos[1] + di*(2*j-1)),
                      (self.ipos[2] + di*(2*k-1)))
                print(pre, ipos_child[0], ipos_child[1], ipos_child[2])
                print(pre, self.bk_max/2.**(3*self.bit_length),
                      self.bk_min/2.**(3*self.bit_length))
                print(pre, order_min/2.**(3*self.bit_length),
                      order_max/2.**(3*self.bit_length), ok)

cdef class CountNeighbourFlaggedOctVisitor(Visitor):
    cdef int n_neigh
    @cython.cdivision(True)
    cdef void visit(self, Oct* o):
        cdef int i
        cdef Oct *no
        cdef np.uint8_t ino

        # cdef bint dbg = (o.new_domain_ind >= 8 and o.new_domain_ind <= 10 and self.ilvl >= 12)

        if self.ilvl == 1:
            o.flag2 = 1
            return

        for i in range(6):
            no = o.neighbours[i]
            ino = o.ineighbours[i]

            if no == NULL:
                print(f'This should not happen...@lvl={self.ilvl}')
                raise Exception()
            if no.children[ino] != NULL and no.children[ino].flag1 > 0:
                no = no.children[ino]
                # if dbg: print('  '*self.ilvl, f'{o.file_ind}[{o.new_domain_ind}]: marked neigh in dir {i} -> {no.file_ind}[{no.new_domain_ind}]')
                o.flag2 += 1

        # if dbg: print('  '*self.ilvl, f'{o.file_ind}[{o.new_domain_ind}]: {o.flag2}')

        if o.flag2 >= self.n_neigh:
            # if dbg: print('  '*self.ilvl, f'{o.file_ind}[{o.new_domain_ind}]: {o.flag2}')
            o.flag2 = 1
        else:
            o.flag2 = 0

cdef class FlagParents(Visitor):
    cdef void visit(self, Oct* o):
        if o.flag1 == 0:
            return
        cdef Oct* parent

        parent = o.parent
        while parent != NULL and parent.flag1 == 0:
            print('Flagging parent!')
            parent.flag1 = 1
            parent = parent.parent

cdef class MarkNeighbourCellVisitor(Visitor):
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

    # @cython.boundscheck(False)
    # @cython.wraparound(False)
    @cython.cdivision(True)
    cdef void visit(self, Oct* o):
        cdef Oct* parent
        cdef Oct* neigh
        cdef np.uint8_t neigh_icell
        cdef int icell, idim, itmp, count
        cdef np.int64_t flag1, flag2, neigh_flag2

        flag2 = o.flag2
        flag1 = o.flag1

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
                    neigh_flag2 = ((neigh.flag1 >> neigh_icell) >> 8) & 0b1
                    if neigh_flag2 == 1:
                        count += 1

                    if count >= self.n_neigh:
                        flag1 |= (0b1 << icell)
                        break

        # Store flags
        # TODO: fix this
        o.flag2 = flag2


cdef class _FlagResetterVisitor(Visitor):
    cdef int idomain
    """Reset flag1 in octs in other domains"""
    def __init__(self, int idomain):
        self.idomain = idomain

cdef class ResetFlag1OtherDomain(_FlagResetterVisitor):
    cdef void visit(self, Oct* o):
        if o.new_domain_ind != self.idomain:
            o.flag1 = 0

cdef class ResetFlag2OtherDomain(_FlagResetterVisitor):
    cdef void visit(self, Oct* o):
        if o.new_domain_ind != self.idomain:
            o.flag2 = 0

cdef class SetFlag2FromFlag1(Visitor):
    @cython.boundscheck(False)
    @cython.wraparound(False)
    cdef void visit(self, Oct* o):
        # cdef bint dbg = (o.new_domain_ind >= 8 and o.new_domain_ind <= 10 and self.ilvl >= 12)
        # if dbg: print('  '*self.ilvl, f'{o.file_ind}[{o.new_domain_ind}]: setting flag1={o.flag1} | flag2={o.flag2}')

        o.flag1 |= o.flag2


cdef class CountVisitor(Visitor):
    cdef int count
    def __init__(self, int initial_value=0):
        self.count = initial_value

    @cython.boundscheck(False)
    @cython.wraparound(False)
    cdef void visit(self, Oct* o):
        self.count += 1

cdef class ExtractionVisitor(Visitor):
    cdef np.int32_t[::1] file_ind
    cdef np.int32_t[::1] domain_ind
    cdef np.int32_t[::1] new_domain_ind
    cdef np.int32_t[::1] lvl
    cdef np.int64_t[:, :, ::1] nbor
    cdef np.int64_t[:, ::1] son
    cdef np.int64_t[:, ::1] parent

    cdef int ind_glob

    def __init__(self):
        self.ind_glob = 0

    @cython.boundscheck(False)
    @cython.wraparound(False)
    cdef void visit(self, Oct* o):
        cdef int i, ishift
        cdef np.int64_t ii, jj, kk
        cdef np.uint8_t icell
        cdef Oct* no

        self.file_ind[self.ind_glob] = o.file_ind
        self.domain_ind[self.ind_glob] = o.domain_ind
        self.new_domain_ind[self.ind_glob] = o.new_domain_ind
        self.lvl[self.ind_glob] = self.ilvl

        # Fill neighbours
        for i in range(6):
            no = o.neighbours[i]
            icell = o.ineighbours[i]
            if no != NULL:
                self.nbor[self.ind_glob, i, 0] = encode_mapping(no.file_ind, no.domain_ind)
                self.nbor[self.ind_glob, i, 1] = <np.int64_t> icell
                if self.nbor[self.ind_glob, i, 0] < 0:
                    raise Exception('This should not happen when encoding', no.file_ind, no.domain_ind, encode_mapping(no.file_ind, no.domain_ind))
        # Fill son
        for i in range(8):
            no = o.children[i]
            if no != NULL:
                self.son[self.ind_glob, i] = encode_mapping(no.file_ind, no.domain_ind)

        # Fill parent
        no = o.parent
        if no != NULL:  # only for root
            self.parent[self.ind_glob, 0] = encode_mapping(no.file_ind, no.domain_ind)
            self.parent[self.ind_glob, 1] = self.icell

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
        di = 2**self.levelmax
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
        if o == NULL:
            return

        if self.select(o, ipos, ilvl, icell):
            # Visit the oct
            visitor.ilvl = <int> ilvl
            visitor.ipos = ipos
            visitor.icell = icell
            if debug: print('  '*ilvl, f'DF: Visiting {o.file_ind}')
            visitor.visit(o)

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

    @cython.cdivision(True)
    cdef void breadth_first_visit_all_octs(self, Oct* root, np.uint64_t ipos[3], int ilvl, Visitor visitor):
        cdef queue[OctInfo*] q
        cdef OctInfo *oi
        cdef OctInfo *oi2
        cdef Oct* o
        cdef int di, i, j, k, icell

        oi = <OctInfo*> malloc(sizeof(OctInfo))
        oi.oct = root
        oi.ipos = ipos
        oi.ilvl = ilvl

        q.push(oi)

        while not q.empty():
            oi = q.front()
            q.pop()

            if not self.select(oi.oct, oi.ipos, oi.ilvl, oi.icell):
                continue

            visitor.ilvl = oi.ilvl
            visitor.ipos = oi.ipos
            visitor.icell = oi.icell
            if debug: print('  '*oi.ilvl, f'BF: Visiting {oi.oct.file_ind}')
            visitor.visit(oi.oct)

            di = 2**(self.levelmax-oi.ilvl)
            # Visit its children
            for k in range(2):
                oi.ipos[2] += (2*k - 1) * di
                for j in range(2):
                    oi.ipos[1] += (2*j - 1) * di
                    for i in range(2):
                        icell = find(i, j, k)
                        o = oi.oct.children[icell]
                        if o == NULL:
                            continue
                        oi.ipos[0] += (2*i - 1) * di
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
            free(oi)


    cdef bint select(self, Oct* o, const np.uint64_t ipos[3], const int ilvl, const np.uint8_t icell):
        raise NotImplementedError

# Select all octs
cdef class AlwaysSelector(Selector):
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
    """Traverse all octs using the parent flag1"""
    cdef bint select(self, Oct* o, const np.uint64_t ipos[3], const int ilvl, const np.uint8_t icell):
        if ilvl == 1:   # special case for coarse lvl
            return True
        return o.flag1 > 0

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
        setup_oct(self.root, NULL, 8)

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
                  const np.uint64_t[::1] hilbert_key,
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
            node.new_domain_ind = new_domain_ind[i]
            node.hilbert_key = hilbert_key[i]
            node.flag1 = 0

        return self.ntot - nbefore

    cdef Oct* get_child(self, Oct* parent, const np.uint8_t ichild, bint create_child):
        cdef Oct* o

        o = parent.children[ichild]
        # Create child if doesn't exist
        if o == NULL:
            if not create_child:
                return NULL

            if debug: print(f'Creating child <Oct #{parent.file_ind}>.children[{ichild}]')
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
        ):
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

        if debug: print(f'Getting {ipos[0]} {ipos[1]} {ipos[2]} @ lvl={lvl}')

        for ilvl in range(lvl-1):
            # Compute index of child
            ichild  = (ipos[0] >> (self.levelmax-ilvl-2)) & 0b100
            ichild |= (ipos[1] >> (self.levelmax-ilvl-1)) & 0b010
            ichild |= (ipos[2] >> (self.levelmax-ilvl-0)) & 0b001

            if debug: print('\t'*ilvl, f'ilvl={ilvl}\tichild={ichild}\t{create_child}')

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
                print('This should not happen.')
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
        cdef AlwaysSelector sel = AlwaysSelector(self)
        cdef ClearPaint cp = ClearPaint()
        sel.visit_all_octs(cp)

    # def select(self, np.int64_t[:, ::1] ipos, np.int64_t[::1] ilvl):
    #     '''Set the lower 8 bits of all *cells* at provided positions'''
    #     cdef int N = len(ipos)
    #     cdef int i
    #     cdef np.uint8_t ichild

    #     cdef Oct* o
    #     # global debug
    #     # debug = True
    #     for i in range(N):
    #         o = self.get(&ipos[i, 0], ilvl[i], False,
    #                      ichild_ret=&ichild, return_parent=True)
    #         if o == NULL:
    #             print('This should not happen!')
    #             raise Exception()

    #         o.flag1 |= 0b1<<(<np.int64_t>ichild)

    #         if ilvl[i] < 4:
    #             print('  '*ilvl[i], f'{ilvl[i]} | {o.file_ind}.{ichild} (flag1={o.flag1:8b})\t{ipos[i,0]:10d} {ipos[i,1]:10d} {ipos[i,2]:10d}')

    #     # debug = False

    # @cython.boundscheck(False)
    # @cython.wraparound(False)
    def domain_info(
            self, 
            int idomain,
            np.uint64_t bound_key_min,
            np.uint64_t bound_key_max
    ):
        '''Yield the file and domain indices sorted by level'''
        cdef AlwaysSelector always_sel = AlwaysSelector(self)
        cdef FlaggedOctSelector selected_octs = FlaggedOctSelector(self)
        cdef int Noct, idim

        # Reset flags
        self.clear_paint()

        # Mark all cells contained in an oct in the domain
        cdef MarkDomainVisitor mark_domain = MarkDomainVisitor(idomain)
        # cdef HilbertDomainVisitor mark_hilbert = HilbertDomainVisitor(bound_key_min, bound_key_max)
        always_sel.visit_all_octs(mark_domain, traversal='breadth_first')
        # always_sel.visit_all_octs(mark_hilbert, traversal='breadth_first')

        # Mark all cells 

        # Expand boundaries
        # flag1: already contains marked cells
        # flag2: used as temp array
        cdef CountNeighbourFlaggedOctVisitor count_neigh = CountNeighbourFlaggedOctVisitor()
        cdef ResetFlag2OtherDomain reset_flag2_other_dom = ResetFlag2OtherDomain(idomain)
        cdef SetFlag2FromFlag1 copy_flag2_in_flag1 = SetFlag2FromFlag1()
        for idim in range(3):
            print('Expanding - pass %s' % idim)
            count_neigh.n_neigh = idim + 1
            always_sel.visit_all_octs(reset_flag2_other_dom)
            always_sel.visit_all_octs(count_neigh, traversal='breadth_first')
            always_sel.visit_all_octs(copy_flag2_in_flag1)

        # NOTE: we need to visit all octs since we're expanding
        # cdef MarkNeighbourVisitor mark_neighbours = MarkNeighbourVisitor()
        # cdef ResetFlag2OtherDomain reset_flag2 = ResetFlag2OtherDomain(idomain)
        # cdef ResetFlag1OtherDomain reset_flag1 = ResetFlag1OtherDomain(idomain)
        # cdef SetFlag2FromFlag1 copy_flag2_in_flag1 = SetFlag2FromFlag1()
        # for idim in range(3):
        #     # Expand boundaries by one pixel (storing flagged cells in upper bits)
        #     # only selecting cells surrounded by this many marked cells
        #     mark_neighbours.n_neigh = idim + 1
        #     always_sel.visit_all_octs(reset_flag1)
        #     always_sel.visit_all_octs(mark_neighbours) # , traversal='breadth_first')
        #     # Store flagged cells (in upper bits) into lower bits
        #     always_sel.visit_all_octs(copy_flag2_in_flag1)

        # Count number of selected octs - at this point
        # each oct.flag1 contains in its upper 8 bits the content of flag2
        cdef CountVisitor counter = CountVisitor(0)
        selected_octs.visit_all_octs(counter)

        Noct = counter.count

        # Extract indices
        cdef ExtractionVisitor extract = ExtractionVisitor()

        cdef np.ndarray[np.int32_t, ndim=1] file_ind_arr = np.full(Noct, -1, np.int32)
        cdef np.ndarray[np.int32_t, ndim=1] domain_ind_arr = np.full(Noct, -1, np.int32)
        cdef np.ndarray[np.int32_t, ndim=1] new_domain_ind_arr = np.full(Noct, -1, np.int32)
        cdef np.ndarray[np.int32_t, ndim=1] lvl_arr = np.full(Noct, -1, np.int32)
        # These need to be int64 because of the mapping used
        cdef np.ndarray[np.int64_t, ndim=3] nbor_arr = np.full((Noct, 6, 2), 0, np.int64)
        cdef np.ndarray[np.int64_t, ndim=2] son_arr = np.full((Noct, 8), 0, np.int64)
        cdef np.ndarray[np.int64_t, ndim=2] parent_arr = np.full((Noct, 2), 0, np.int64)

        cdef np.int32_t[::1] file_ind = file_ind_arr
        cdef np.int32_t[::1] domain_ind = domain_ind_arr
        cdef np.int32_t[::1] new_domain_ind = new_domain_ind_arr
        cdef np.int32_t[::1] lvl = lvl_arr
        cdef np.int64_t[:, :, ::1] nbor = nbor_arr
        cdef np.int64_t[:, ::1] son = son_arr
        cdef np.int64_t[:, ::1] parent = parent_arr

        extract.file_ind = file_ind
        extract.domain_ind = domain_ind
        extract.new_domain_ind = new_domain_ind
        extract.lvl = lvl
        extract.nbor = nbor
        extract.son = son
        extract.parent = parent
        selected_octs.visit_all_octs(extract, traversal='breadth_first')

        # At this point we have
        # - domain_ind : the old CPU domains to read
        # - file_ind   : the position within the file
        # - new_domain_ind : the new CPU domain
        # - lvl    : the level of the oct

        # We need now to reorder the domains in each lvl
        cdef int ilvl, i, i0
        cdef np.int64_t[::1] order
        i0 = 0
        i = 0
        for ilvl in range(1, self.levelmax+1):
            while i < Noct and lvl[i] == ilvl:
                i += 1

            order = np.argsort(new_domain_ind_arr[i0:i], kind='stable') + i0
            file_ind_arr[i0:i] = file_ind_arr[order]
            domain_ind_arr[i0:i] = domain_ind_arr[order]
            new_domain_ind_arr[i0:i] = new_domain_ind_arr[order]
            nbor_arr[i0:i] = nbor_arr[order]
            son_arr[i0:i] = son_arr[order]
            parent_arr[i0:i] = parent_arr[order]

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
            # 40 bits should be enough
            key = encode_mapping(file_ind[i], domain_ind[i])

            # print('Inserting mapping[%s] = %s' % (key, i+1))
            global_to_local[key] = i + 1

        # Global indices -> local indices
        print('Inverting indices')
        cdef unordered_map[np.uint64_t, np.int64_t].iterator it, end
        end = global_to_local.end()
        for i in range(Noct):
            for j in range(6):
                it = global_to_local.find(nbor[i, j, 0])
                if it != end:
                    nbor[i, j, 0] = deref(it).second
                else:
                    nbor[i, j, 0] = 0
                    nbor[i, j, 1] = 1
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
                parent[i, 1] = 0

        # Create AMR structure
        print('Creating AMR structure for %s octs' % Noct)
        cdef int icpu
        cdef np.ndarray[np.int32_t, ndim=2] headl_arr = np.zeros((self.levelmax, self.new_ncpu), dtype=np.int32)
        cdef np.ndarray[np.int32_t, ndim=2] taill_arr = np.zeros((self.levelmax, self.new_ncpu), dtype=np.int32)
        cdef np.ndarray[np.int32_t, ndim=2] numbl_arr = np.zeros((self.levelmax, self.new_ncpu), dtype=np.int32)

        cdef np.int32_t[:, ::1] headl = headl_arr
        cdef np.int32_t[:, ::1] taill = taill_arr
        cdef np.int32_t[:, ::1] numbl = numbl_arr

        ilvl = 0
        icpu = 0
        # Compute headl,taill, numbl
        for i in range(Noct):
            ilvl = lvl[i]
            icpu = new_domain_ind[i]
            if headl[ilvl-1, icpu-1] == 0:
                headl[ilvl-1, icpu-1] = i + 1
            taill[ilvl-1, icpu-1] = i + 1
            numbl[ilvl-1, icpu-1] += 1

        return dict(
            file_ind=file_ind_arr,
            old_domain_ind=domain_ind_arr,
            new_domain_ind=new_domain_ind_arr,
            lvl=lvl_arr,
            nbor=nbor_arr,
            headl=headl_arr,
            taill=taill_arr,
            numbl=numbl_arr,
            son=son_arr,
            parent=parent_arr
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
