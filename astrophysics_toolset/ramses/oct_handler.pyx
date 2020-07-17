import numpy as np
cimport numpy as np

from libc.stdlib cimport malloc, free
from cython cimport integral
cimport cython

ctypedef fused myint:
    np.int8_t
    np.int16_t
    np.int32_t
    np.int64_t

ctypedef fused myuint:
    np.uint8_t
    np.uint16_t
    np.uint32_t
    np.uint64_t

cdef struct Oct
cdef struct Oct:
    np.int64_t file_ind     # index with respect to the order in which it was
                            # added
    np.int64_t domain_ind   # index within the global set of domains
    np.int64_t new_domain_ind
    np.uint8_t color        # temporary attribute
    Oct* parent
    Oct* children[8]
    Oct* neighbours[6]


ctypedef void (*f_type)(Oct*)

cdef class Selector:
    cdef void visit(self, Oct* o):
        pass

cdef class DomainSelector(Selector):
    """Select all cells in a domain + the ones directly adjacent to them."""
    cdef int domain_ind, _nselected
    def __init__(self, int domain_ind):
        self.domain_ind = domain_ind
        self._nselected = 0

    @property
    def nselected(self):
        return self._nselected

    cdef void paint_parent(self, Oct* o, int color):
        cdef Oct* parent
        parent = o
        while parent != NULL and parent.color == 0:
            parent.color = color
            self._nselected += 1
            parent = parent.parent

    cdef void visit(self, Oct* o):
        cdef Oct* parent
        cdef Oct* neigh

        if o.new_domain_ind == self.domain_ind:
            # Select all parent octs
            self.paint_parent(o, 1)

            # Also tag neighbour octs (if they exist)
            for i in range(6):
                neigh = o.neighbours[i]
                if neigh != NULL:
                    self.paint_parent(o, 2)

cdef class ExtractSelector(Selector):
    cdef np.int64_t[::1] domain_ind
    cdef np.int64_t[::1] file_ind
    cdef int iloc
    def __init__(self):
        self.iloc = 0

    @cython.boundscheck(False)
    @cython.wraparound(False)
    cdef void visit(self, Oct* o):
        if o.color > 0:
            self.domain_ind[self.iloc] = o.domain_ind
            self.file_ind[self.iloc] = o.file_ind
            self.iloc += 1

        o.color = 0

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
        self.root.color = 0
        self.root.parent = NULL

    def __init__(self, int levelmax):
        self.levelmax = levelmax
        self._ntot = 8  # Count the 8 cells in the root oct

    @cython.boundscheck(False)
    @cython.cdivision(True)
    @cython.wraparound(False)
    def add(self, const np.int64_t[:, ::1] ipos,
                  const np.int64_t[:] file_ind,
                  const np.int64_t[:] domain_ind,
                  const np.int64_t[:] new_domain_ind,
                  const np.int64_t[:] lvl):
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
            node.color = 0

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

        return node

    cdef void recursively_visit_all_octs(self, Oct* o, Selector sel, bint depth_first=True):
        cdef int i

        for i in range(8):
            if not depth_first:
                sel.visit(o)
            if o.children[i] != NULL:
                self.recursively_visit_all_octs(o.children[i], sel, depth_first)
            if depth_first:
                sel.visit(o)

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

    def select_domain_and_parents(self, int domain_ind):
        cdef DomainSelector sel = DomainSelector(domain_ind)
        self.recursively_visit_all_octs(self.root, sel)

        cdef ExtractSelector extract = ExtractSelector()
        cdef np.ndarray[np.int64_t, ndim=1] dom_ind = np.zeros(sel.nselected, dtype=np.int64)
        cdef np.ndarray[np.int64_t, ndim=1] file_ind = np.zeros(sel.nselected, dtype=np.int64)
        extract.domain_ind = dom_ind
        extract.file_ind = file_ind

        self.recursively_visit_all_octs(self.root, extract)
        return dom_ind, file_ind

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