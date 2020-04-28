# This is a generated file! Please edit source .ksy file and use kaitai-struct-compiler to rebuild

from pkg_resources import parse_version
from kaitaistruct import __version__ as ks_version, KaitaiStruct, KaitaiStream, BytesIO


if parse_version(ks_version) < parse_version('0.7'):
    raise Exception("Incompatible Kaitai Struct Python API: 0.7 or later is required, but you have %s" % (ks_version))

class Disperse(KaitaiStruct):
    """
    .. seealso::
       Source - http://www2.iap.fr/users/sousbie/web/html/indexf83e.html?post/NDskl-format
    """
    def __init__(self, _io, _parent=None, _root=None):
        self._io = _io
        self._parent = _parent
        self._root = _root if _root else self
        self._read()

    def _read(self):
        self.header = self._root.Header(self._io, self, self._root)
        self.data = self._root.Data(self._io, self, self._root)

    class SegDataStruct(KaitaiStruct):
        def __init__(self, _io, _parent=None, _root=None):
            self._io = _io
            self._parent = _parent
            self._root = _root if _root else self
            self._read()

        def _read(self):
            self.next_node = self._io.read_u4le()
            self.next_seg = self._io.read_u4le()


    class Data(KaitaiStruct):
        def __init__(self, _io, _parent=None, _root=None):
            self._io = _io
            self._parent = _parent
            self._root = _root if _root else self
            self._read()

        def _read(self):
            self.seg_pos = [None] * (((2 * self._root.header.ndims) * self._root.header.nsegs))
            for i in range(((2 * self._root.header.ndims) * self._root.header.nsegs)):
                self.seg_pos[i] = self._io.read_f4le()

            self._unnamed1 = self._io.read_u4le()
            self._unnamed2 = self._io.read_u4le()
            self.nodes_pos = [None] * ((self._root.header.ndims * self._root.header.nnodes))
            for i in range((self._root.header.ndims * self._root.header.nnodes)):
                self.nodes_pos[i] = self._io.read_f4le()

            self._unnamed4 = self._io.read_u4le()
            self._unnamed5 = self._io.read_u4le()
            self.segdata = [None] * ((self._root.header.nsegs * self._root.header.nsegdata))
            for i in range((self._root.header.nsegs * self._root.header.nsegdata)):
                self.segdata[i] = self._io.read_f8le()

            self._unnamed7 = self._io.read_u4le()
            self._unnamed8 = self._io.read_u4le()
            self.nodesdata = [None] * ((self._root.header.nnodes * self._root.header.nnodedata))
            for i in range((self._root.header.nnodes * self._root.header.nnodedata)):
                self.nodesdata[i] = self._io.read_f8le()

            self._unnamed10 = self._io.read_u4le()
            self._unnamed11 = self._io.read_u4le()
            self.nodesdata_int = [None] * (self._root.header.nnodes)
            for i in range(self._root.header.nnodes):
                self.nodesdata_int[i] = self._root.NodeStruct(self._io, self, self._root)

            self._unnamed13 = self._io.read_u4le()
            self._unnamed14 = self._io.read_u4le()
            self.segdata_int = [None] * (self._root.header.nsegs)
            for i in range(self._root.header.nsegs):
                self.segdata_int[i] = self._root.SegStruct(self._io, self, self._root)

            self._unnamed16 = self._io.read_u4le()


    class Header(KaitaiStruct):
        def __init__(self, _io, _parent=None, _root=None):
            self._io = _io
            self._parent = _parent
            self._root = _root if _root else self
            self._read()

        def _read(self):
            self._unnamed0 = self._io.ensure_fixed_contents(b"\x10\x00\x00\x00")
            self._unnamed1 = self._io.ensure_fixed_contents(b"\x4E\x44\x53\x4B\x45\x4C\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00")
            self._unnamed2 = self._io.ensure_fixed_contents(b"\x10\x00\x00\x00\x00\x00\x00\x00")
            self.comment = (self._io.read_bytes(80)).decode(u"ASCII")
            self.ndims = self._io.read_u4le()
            self.dims = [None] * (self.ndims)
            for i in range(self.ndims):
                self.dims[i] = self._io.read_u4le()

            self._unnamed6 = [None] * ((20 - self.ndims))
            for i in range((20 - self.ndims)):
                self._unnamed6[i] = self._io.read_u4le()

            self.x0 = [None] * (self.ndims)
            for i in range(self.ndims):
                self.x0[i] = self._io.read_f8le()

            self._unnamed8 = [None] * ((20 - self.ndims))
            for i in range((20 - self.ndims)):
                self._unnamed8[i] = self._io.read_f8le()

            self.delta = [None] * (self.ndims)
            for i in range(self.ndims):
                self.delta[i] = self._io.read_f8le()

            self._unnamed10 = [None] * ((20 - self.ndims))
            for i in range((20 - self.ndims)):
                self._unnamed10[i] = self._io.read_f8le()

            self.nsegs = self._io.read_u4le()
            self.nnodes = self._io.read_u4le()
            self.nsegdata = self._io.read_u4le()
            self.nnodedata = self._io.read_u4le()
            self._unnamed15 = self._io.read_u4le()
            self._unnamed16 = self._io.read_u4le()
            self.segdata_info = [None] * (self.nsegdata)
            for i in range(self.nsegdata):
                self.segdata_info[i] = (self._io.read_bytes(20)).decode(u"ASCII")

            self._unnamed18 = self._io.read_u4le()
            self._unnamed19 = self._io.read_u4le()
            self.nodedata_info = [None] * (self.nnodedata)
            for i in range(self.nnodedata):
                self.nodedata_info[i] = (self._io.read_bytes(20)).decode(u"ASCII")

            self._unnamed21 = self._io.read_u4le()
            self._unnamed22 = self._io.read_u4le()


    class NodeStruct(KaitaiStruct):
        def __init__(self, _io, _parent=None, _root=None):
            self._io = _io
            self._parent = _parent
            self._root = _root if _root else self
            self._read()

        def _read(self):
            self.pos_index = self._io.read_u4le()
            self.flags = self._io.read_u4le()
            self.nnext = self._io.read_u4le()
            self.type = self._io.read_u4le()
            self.index = self._io.read_u4le()
            self.nsegs = [None] * (self.nnext)
            for i in range(self.nnext):
                self.nsegs[i] = self._io.read_u4le()

            self.seg_data = [None] * (self.nnext)
            for i in range(self.nnext):
                self.seg_data[i] = self._root.SegDataStruct(self._io, self, self._root)



    class SegStruct(KaitaiStruct):
        def __init__(self, _io, _parent=None, _root=None):
            self._io = _io
            self._parent = _parent
            self._root = _root if _root else self
            self._read()

        def _read(self):
            self.pos_index = self._io.read_u4le()
            self.nodes = [None] * (2)
            for i in range(2):
                self.nodes[i] = self._io.read_u4le()

            self.flags = self._io.read_u4le()
            self.index = self._io.read_u4le()
            self.next_seg = self._io.read_u4le()
            self.prev_seg = self._io.read_u4le()



