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

cdef struct Oct:
    np.int64_t file_ind       # on file index
    np.int64_t domain_ind     # original domain
    np.int64_t new_domain_ind # new domain
    np.int64_t colour         # attribute for selection
    np.uint64_t hilbert_key

    Oct* parent
    Oct* children[8]
    Oct* neighbours[6]

cdef struct OctInfo:
    Oct* oct
    np.uint64_t ipos[3]
    int ilvl


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
    cdef int ilvl, levelmax, bit_length
    cdef void visit(self, Oct* o):
        pass

cdef class ClearPaint(Visitor):
    @cython.boundscheck(False)
    @cython.wraparound(False)
    cdef void visit(self, Oct* o):
        o.colour = 0

cdef class PrintVisitor(Visitor):
    cdef bint print_neighbours
    @cython.boundscheck(False)
    @cython.wraparound(False)
    cdef void visit(self, Oct* o):
        cdef int i
        cdef Oct* no
        if o.parent != NULL:
            print('\t' * self.ilvl, f'{o.file_ind} ← {o.parent.file_ind}', end='')
        else:
            print('\t' * self.ilvl, f'{o.file_ind} ← NULL', end='')

        if self.print_neighbours:
            print(' | ', end='')
            for i in range(6):
                no = o.neighbours[i]
                if no != NULL:
                    print(f'{no.file_ind: 7d}', end='  ')
                else:
                    print('_'*7, end='  ')
        print()


cdef class DomainVisitor(Visitor):
    """Select all cells in a domain + the ones directly adjacent to them."""
    cdef int idim

    @cython.boundscheck(False)
    @cython.wraparound(False)
    cdef void visit(self, Oct* o):
        cdef Oct* parent
        cdef Oct* neigh
        cdef int i

        o.colour = 1

        # Tag neighbour octs in given direction (if they exist)
        for i in range(2):
            neigh = o.neighbours[i + 2*self.idim]
            if neigh != NULL:
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
    cdef np.int32_t[::1] file_ind
    cdef np.int32_t[::1] domain_ind
    cdef np.int32_t[::1] new_domain_ind
    cdef np.int32_t[::1] lvl
    cdef np.int64_t[:, ::1] nbor
    cdef np.int64_t[:, ::1] son
    cdef np.int64_t[:, ::1] parent

    cdef int ind_glob

    def __init__(self):
        self.ind_glob = 0

    # @cython.boundscheck(False)
    # @cython.wraparound(False)
    cdef void visit(self, Oct* o):
        cdef int i, ishift
        cdef np.int64_t ii, jj, kk
        cdef Oct* no
        if o.colour > 0:
            self.file_ind[self.ind_glob] = o.file_ind
            self.domain_ind[self.ind_glob] = o.domain_ind
            self.new_domain_ind[self.ind_glob] = o.new_domain_ind
            self.lvl[self.ind_glob] = self.ilvl

            # Fill neighbours
            for i in range(6):
                no = o.neighbours[i]
                if no != NULL:
                    self.nbor[self.ind_glob, i] = encode_mapping(no.file_ind, no.domain_ind)
                    if self.nbor[self.ind_glob, i] < 0:
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
                # Second item contains the location of the parent cell in its oct
                ishift = self.bit_length - self.ilvl + 2

                self.parent[self.ind_glob, 1] = find(
                    (self.ipos[0] >> ishift) & 0b1,
                    (self.ipos[1] >> ishift) & 0b1,
                    (self.ipos[2] >> ishift) & 0b1
                )

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

        visitor.levelmax = self.levelmax
        visitor.bit_length = self.bit_length

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
            visitor.ilvl = ilvl
            visitor.ipos = ipos
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

            visitor.ilvl = oi.ilvl
            visitor.ipos = oi.ipos
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
                        oi2.ilvl = oi.ilvl + 1
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

cdef class LevelSelector(Selector):
    cdef int lvl
    def __init__(self, Octree octree, int lvl):
        super(LevelSelector, self).__init__(octree)
        self.lvl = lvl

    cdef bint select(self, Oct* o, const np.uint64_t ipos[3], const int ilvl):
        return ilvl <= self.lvl


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
    cdef int old_ncpu
    cdef int new_ncpu


    def __cinit__(self):
        if debug: print("Malloc'ing root")
        self.root = <Oct*> malloc(sizeof(Oct))
        for i in range(8):
            self.root.children[i] = NULL
        for i in range(6):
            self.root.neighbours[i] = NULL
        self.root.new_domain_ind = -1
        self.root.file_ind = -1
        self.root.domain_ind = -1
        self.root.colour = -1
        self.root.parent = NULL

    def __init__(self, int levelmax, int old_ncpu, int new_ncpu):
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
            node.colour = 0

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
            for i in range(8):
                o.children[i] = NULL
            for i in range(6):
                o.neighbours[i] = NULL
            o.parent = parent
            o.file_ind = -1
            o.domain_ind = -1
            o.new_domain_ind = -1
            o.hilbert_key = -1
            o.colour = -1

        return o


    cdef Oct* get(self, const np.int64_t* ipos, const np.int64_t lvl, bint create_child=False):
        cdef int ilvl
        cdef Oct* node = self.root
        cdef np.uint8_t ichild

        if debug: print('Getting', ipos[0], ipos[1], ipos[2])

        for ilvl in range(lvl-1):
            # Compute index of child
            ichild  = (ipos[0] >> (self.levelmax-ilvl-2)) & 0b100
            ichild |= (ipos[1] >> (self.levelmax-ilvl-1)) & 0b010
            ichild |= (ipos[2] >> (self.levelmax-ilvl-0)) & 0b001

            if debug: print('\t'*ilvl, f'ilvl={ilvl}\tichild={ichild}\t{create_child}')

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
            o = self.get(&ipos[i, 0], ilvl[i])
            if o == NULL:
                print('This should not happen.')
            for j in range(6):
                neigh = self.get(&neigh_pos[i, j, 0], ilvl[i])
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
            o = self.get(&ipos[i, 0], ilvl[i])
            if o == NULL:
                print('This should not happen!')

            # Select oct and its parents
            while o != NULL and o.colour == 0:
                o.colour = 1
                o = o.parent

    # @cython.boundscheck(False)
    # @cython.wraparound(False)
    def domain_info(self):
        '''Yield the file and domain indices sorted by level'''
        cdef PaintSelector sel = PaintSelector(self)
        cdef int Noct, idim

        # Expand boundaries in each direction
        cdef DomainVisitor domvis = DomainVisitor()

        for idim in range(3):
            domvis.idim = idim
            sel.visit_all_octs(domvis, traversal='breadth_first')

        # Count number of selected octs
        cdef CountVisitor counter = CountVisitor()
        sel.visit_all_octs(counter)

        Noct = counter.count

        # Extract indices
        cdef IndVisitor extract = IndVisitor()

        cdef np.ndarray[np.int32_t, ndim=1] file_ind_arr = np.full(Noct, -1, np.int32)
        cdef np.ndarray[np.int32_t, ndim=1] domain_ind_arr = np.full(Noct, -1, np.int32)
        cdef np.ndarray[np.int32_t, ndim=1] new_domain_ind_arr = np.full(Noct, -1, np.int32)
        cdef np.ndarray[np.int32_t, ndim=1] lvl_arr = np.full(Noct, -1, np.int32)
        # These need to be int64 because of the mapping used
        cdef np.ndarray[np.int64_t, ndim=2] nbor_arr = np.full((Noct, 6), 0, np.int64)
        cdef np.ndarray[np.int64_t, ndim=2] son_arr = np.full((Noct, 8), 0, np.int64)
        cdef np.ndarray[np.int64_t, ndim=2] parent_arr = np.full((Noct, 2), 0, np.int64)

        cdef np.int32_t[::1] file_ind = file_ind_arr
        cdef np.int32_t[::1] domain_ind = domain_ind_arr
        cdef np.int32_t[::1] new_domain_ind = new_domain_ind_arr
        cdef np.int32_t[::1] lvl = lvl_arr
        cdef np.int64_t[:, ::1] nbor = nbor_arr
        cdef np.int64_t[:, ::1] son = son_arr
        cdef np.int64_t[:, ::1] parent = parent_arr

        extract.file_ind = file_ind
        extract.domain_ind = domain_ind
        extract.new_domain_ind = new_domain_ind
        extract.lvl = lvl
        extract.nbor = nbor
        extract.son = son
        extract.parent = parent
        sel.visit_all_octs(extract, traversal='breadth_first')

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
                it = global_to_local.find(nbor[i, j])
                if it != end:
                    nbor[i, j] = deref(it).second
                    if nbor[i, j] < 0:
                        print('This should not happen!', i, j, nbor[i, j])
                        return {}
            for j in range(8):
                it = global_to_local.find(son[i, j])
                if it != end:
                    son[i, j] = deref(it).second
            it = global_to_local.find(parent[i, 0])
            if it != end:
                parent[i, 0] = deref(it).second

        # Create AMR structure
        print('Creating AMR structure')
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
