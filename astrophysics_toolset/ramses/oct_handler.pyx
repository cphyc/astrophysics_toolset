# distutils: language = c++

import numpy as np
cimport numpy as np

from libc.stdlib cimport malloc, free
from libcpp.queue cimport queue
from cython cimport integral
cimport cython

cdef struct Oct:
    np.int64_t file_ind     # index with respect to the order in which it was
                            # added
    np.int64_t domain_ind   # index within the global set of domains
    np.int64_t new_domain_ind
    np.uint8_t colour        # temporary attribute
    np.uint64_t hilbert_key # hilbert key
    Oct* parent
    Oct* children[8]
    Oct* neighbours[6]

cdef struct OctInfo:
    Oct* oct
    np.uint64_t ipos[3]
    int ilvl


#############################################################
# Visitors
#############################################################
cdef class Visitor:
    cdef void visit(self, Oct* o):
        pass

cdef class ClearPaint(Visitor):
    @cython.boundscheck(False)
    @cython.wraparound(False)
    cdef void visit(self, Oct* o):
        o.colour = 0

cdef class DomainVisitor(Visitor):
    """Select all cells in a domain + the ones directly adjacent to them."""
    cdef int _nselected, idim, _nneigh, _other
    def __init__(self, int idim):
        self.idim = idim
        self._nselected = 0
        self._nneigh = 0
        self._other = 0

    @property
    def nselected(self):
        return self._nselected

    @cython.boundscheck(False)
    @cython.wraparound(False)
    cdef void visit(self, Oct* o):
        cdef Oct* parent
        cdef Oct* neigh
        cdef int i

        if o.colour == 0:
            self._nselected += 1
        o.colour = 1

        # Tag neighbour octs in given direction (if they exist)
        for i in range(2):
            neigh = o.neighbours[i + 2*self.idim]
            if neigh != NULL:
                if neigh.colour == 0:
                    self._nselected += 1
                    self._nneigh += 1
                neigh.colour = 1

cdef class CountVisitor(Visitor):
    cdef int count
    def __init__(self):
        self.count = 0

    @cython.boundscheck(False)
    @cython.wraparound(False)
    cdef void visit(self, Oct* o):
        if o.colour > 0:
            self.count += 1

cdef class IndVisitor(Visitor):
    cdef np.int64_t[::1] file_ind
    cdef np.int64_t[::1] domain_ind
    cdef int ind_glob

    def __init__(self):
        self.ind_glob = 0

    @cython.boundscheck(False)
    @cython.wraparound(False)
    cdef void visit(self, Oct* o):
        if o.colour > 0:
            self.file_ind[self.ind_glob] = o.file_ind
            self.domain_ind[self.ind_glob] = o.domain_ind
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
    cdef int levelmax
    cdef int bit_length # number of bits per dimension in hilbert key

    def __init__(self, Octree octree):
        self.octree = octree
        self.levelmax = self.octree.levelmax
        self.bit_length = self.levelmax + 1

    @cython.cdivision(True)
    cdef void visit_all_octs(self, Visitor visitor, str traversal = 'depth_first'):
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

        print(f'Root has file_ind={root.file_ind}')
        if traversal == 'depth_first':
            self.recursively_visit_all_octs(root, ipos, ilvl, visitor, di / 2)
        elif traversal == 'breadth_first':
            self.breadth_first_visit_all_octs(root, ipos, ilvl, visitor)

    @cython.cdivision(True)
    cdef void recursively_visit_all_octs(self, Oct* o, np.uint64_t ipos[3], int ilvl, Visitor visitor, np.uint64_t di):
        cdef int i, j, k
        if o == NULL:
            return

        if self.select(o, ipos, ilvl):
            # Visit the oct
            visitor.visit(o)

            # Visit its children
            for k in range(2):
                ipos[2] += (2*k - 1) * di
                for j in range(2):
                    ipos[1] += (2*j - 1) * di
                    for i in range(2):
                        ipos[0] += (2*i - 1) * di
                        # print(' '*ilvl, f'ilvl={ilvl}, {ipos[0]}, {ipos[1]}, {ipos[2]} di={di}')
                        self.recursively_visit_all_octs(o.children[find(i, j, k)], ipos, ilvl+1, visitor, di / 2)
                        ipos[0] -= (2*i - 1) * di
                    ipos[1] -= (2*j - 1) * di
                ipos[2] -= (2*k - 1) * di

    @cython.cdivision(True)
    cdef void breadth_first_visit_all_octs(self, Oct* root, np.uint64_t ipos[3], int ilvl, Visitor visitor):
        cdef queue[OctInfo*] q
        cdef OctInfo *oi
        cdef OctInfo *oi2
        cdef Oct* o
        cdef int di, i, j, k

        oi = <OctInfo*> malloc(sizeof(OctInfo))
        oi.oct = root
        oi.ipos = ipos
        oi.ilvl = ilvl

        q.push(oi)

        while not q.empty():
            oi = q.front()
            q.pop()

            if not self.select(oi.oct, oi.ipos, oi.ilvl):
                continue

            visitor.visit(oi.oct)

            di = 2**(self.levelmax-oi.ilvl)
            # Visit its children
            for k in range(2):
                oi.ipos[2] += (2*k - 1) * di
                for j in range(2):
                    oi.ipos[1] += (2*j - 1) * di
                    for i in range(2):
                        o = oi.oct.children[find(i, j, k)]
                        if o == NULL:
                            continue
                        oi.ipos[0] += (2*i - 1) * di
                        oi2 = <OctInfo*> malloc(sizeof(OctInfo))
                        oi2.oct = o
                        oi2.ipos[0] = oi.ipos[0]
                        oi2.ipos[1] = oi.ipos[1]
                        oi2.ipos[2] = oi.ipos[2]
                        q.push(oi2)
                        oi.ipos[0] -= (2*i - 1) * di
                    oi.ipos[1] -= (2*j - 1) * di
                oi.ipos[2] -= (2*k - 1) * di
            free(oi)


    cdef bint select(self, Oct* o, const np.uint64_t ipos[3], const int ilvl):
        raise NotImplementedError

# Select all octs
cdef class AlwaysSelector(Selector):
    cdef bint select(self, Oct* o, const np.uint64_t ipos[3], const int ilvl):
        return True

# Select oct which have been painted
cdef class PaintSelector(Selector):
    cdef bint select(self, Oct* o, const np.uint64_t ipos[3], const int ilvl):
        return o.colour > 0

# Select all octs that may any key in range provided
cdef class HilbertSelector(Selector):
    cdef np.uint64_t key_lower
    cdef np.uint64_t key_upper
    def __init__(self, Octree octree, np.uint64_t key_lower, np.uint64_t key_upper):
        super(HilbertSelector, self).__init__(octree)
        self.key_lower = key_lower
        self.key_upper = key_upper

    cdef bint select(self, Oct* o, const np.uint64_t ipos[3], const int ilvl):
        # Compute hilbert key
        cdef np.uint64_t order_min = o.hilbert_key
        cdef np.uint64_t order_max

        cdef int ishift = (self.bit_length - ilvl + 1)*3

        order_min >>= ishift
        order_max = order_min + 1

        order_min <<= ishift
        order_max <<= ishift

        print('  '*ilvl, f'Selecting? {o.file_ind} {order_min} {self.key_lower} | {order_max} {self.key_upper}', (order_max > self.key_lower) & (order_min < self.key_upper))

        return (order_max > self.key_lower) & (order_min < self.key_upper)

#############################################################
# Octree implementation
#############################################################
cdef bint debug = False
cdef class Octree:
    cdef Oct* root
    cdef int levelmax
    cdef int _ntot

    def __cinit__(self):
        if debug: print("Malloc'ing root")
        self.root = <Oct*> malloc(sizeof(Oct))
        for i in range(8):
            self.root.children[i] = NULL
        for i in range(6):
            self.root.neighbours[i] = NULL
        self.root.new_domain_ind = 0
        self.root.file_ind = 0
        self.root.domain_ind = 0
        self.root.colour = 0
        self.root.parent = NULL

    def __init__(self, int levelmax):
        self.levelmax = levelmax
        self._ntot = 8  # Count the 8 cells in the root oct

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
            node = self.get(&ipos[i, 0], lvl[i], True)
            if node == NULL:
                print('THIS SHOULD NOT HAPPEN')
                return
            node.file_ind = file_ind[i]
            node.domain_ind = domain_ind[i]
            node.new_domain_ind = new_domain_ind[i]
            node.hilbert_key = hilbert_key[i]
            node.colour = 0

        return self.ntot - nbefore

    cdef Oct* get_child(self, Oct* parent, const np.uint8_t ichild, const bint create_child):
        cdef Oct* oct

        oct = parent.children[ichild]
        # print('In get_child', ichild, oct == NULL)

        # Create child if doesn't exist
        if oct == NULL:
            if not create_child:
                return NULL

            oct = <Oct*> malloc(sizeof(Oct))
            self._ntot += 1

            parent.children[ichild] = oct
            for i in range(8):
                oct.children[i] = NULL
            for i in range(6):
                oct.neighbours[i] = NULL
            oct.parent = parent

        return oct


    cdef Oct* get(self, const np.int64_t* ipos, const np.int64_t lvl, const bint create_child):
        cdef int ilvl
        cdef Oct* node = self.root
        cdef np.uint8_t ichild

        # print('Getting', ipos[0], ipos[1], ipos[2])

        for ilvl in range(lvl-1):
            # Compute index of child
            ichild  = (ipos[0] >> (self.levelmax-ilvl-3)) & 0b100
            ichild |= (ipos[1] >> (self.levelmax-ilvl-2)) & 0b010
            ichild |= (ipos[2] >> (self.levelmax-ilvl-1)) & 0b001

            # print(f'ilvl={ilvl}\tichild={ichild}\t{create_child}')

            node = self.get_child(node, ichild, create_child)
            if node == NULL:
                break

        return node

    @cython.boundscheck(False)
    @cython.cdivision(True)
    @cython.wraparound(False)
    def set_neighbours(self, const np.int64_t[:, ::1] ipos, const np.int64_t[:, :, ::1] neigh_pos, const np.int64_t[::1] ilvl):
        cdef int i, j, N = ipos.shape[0]
        cdef Oct* o
        cdef Oct* neigh

        for i in range(N):
            o = self.get(&ipos[i, 0], ilvl[i], False)
            if o == NULL:
                print('This should not happen.')
            for j in range(6):
                neigh = self.get(&ipos[i, 0], ilvl[i], False)
                if neigh == NULL:
                    continue

                o.neighbours[j] = neigh

    def clear_paint(self):
        cdef AlwaysSelector sel = AlwaysSelector(self)
        cdef ClearPaint cp = ClearPaint()
        sel.visit_all_octs(cp)
        print('Cleared paint!')

    def select(self, np.int64_t[:, ::1] ipos, np.int64_t[::1] ilvl):
        '''Select all octs at positions'''
        cdef int N = len(ipos)
        cdef int i

        cdef Oct* o

        for i in range(N):
            o = self.get(&ipos[i, 0], ilvl[i], False)
            if o == NULL:
                print('This should not happen!')

            # Select oct and its parents
            while o != NULL and o.colour == 0:
                o.colour = 1
                o = o.parent

    def iter_selected(self):
        '''Yield the file and domain indices sorted by level'''
        cdef PaintSelector sel = PaintSelector(self)
        cdef int Noct

        # Count number of selected octs
        cdef CountVisitor counter = CountVisitor()
        sel.visit_all_octs(counter)

        Noct = sel.count

        # Extract indices
        cdef IndVisitor extract = IndVisitor()
        extract.file_ind = np.full(Noct, -1, np.int64)
        extract.domain_ind = np.full(Noct, -1, np.int64)
        sel.visit_all_octs(extract, traversal='breadth_first')

    @property
    def ntot(self):
        return self._ntot # self.count(self.root)

    cdef int count(self, Oct* oct):
        cdef int N = 8
        for i in range(8):
            if oct.children[i] != NULL:
                N += self.count(oct.children[i])
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

    def traverse_tree(self, method):
        if method == 'depth_first':
            raise NotImplementedError
        elif method == 'breadth_first':
            self._breadth_traversal()
