meta:
  id: disperse
  file-extension: NDskl
  endian: le
  license: CC0-1.0
  encoding: ASCII
doc-ref: http://www2.iap.fr/users/sousbie/web/html/indexf83e.html?post/NDskl-format
seq:
  - id: header
    type: header
  - id: data
    type: data
types:
  header:
    seq:
      - contents: [16, 0, 0, 0]
        size: 4
      - # Magic
        contents: ['N', 'D', 'S', 'K', 'E', 'L', 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]
      - contents: [16, 0, 0, 0, 0, 0, 0, 0]
        size: 8
      - id: comment
        type: str
        size: 80
      - id: ndims
        type: u4
      - id: dims
        type: u4
        repeat: expr
        repeat-expr: ndims
      - # Pad to length of 20
        type: u4
        repeat: expr
        repeat-expr: 20-ndims
      - id: x0
        type: f8
        repeat: expr
        repeat-expr: ndims
      - # Pad to length of 20
        type: f8
        repeat: expr
        repeat-expr: 20-ndims
      - id: delta
        type: f8
        repeat: expr
        repeat-expr: ndims
      - # Pad to length of 20
        type: f8
        repeat: expr
        repeat-expr: 20-ndims
      - id: nsegs
        type: u4
      - id: nnodes
        type: u4
      - id: nsegdata
        type: u4
      - id: nnodedata
        type: u4
      - # Dummy
        type: u4
      - # Dummy
        type: u4
      - id: segdata_info
        type: str
        size: 20
        repeat: expr
        repeat-expr: nsegdata
      - # Dummy
        type: u4
      - # Dummy
        type: u4
      - id: nodedata_info
        type: str
        size: 20
        repeat: expr
        repeat-expr: nnodedata
      - # Dummy
        type: u4
      - # Dummy
        type: u4
  data:
    seq:
      - id: seg_pos
        type: f4
        repeat: expr
        repeat-expr: 2*_root.header.ndims*_root.header.nsegs
      - # Dummy
        type: u4
      - # Dummy
        type: u4
      - id: nodes_pos
        type: f4
        repeat: expr
        repeat-expr: _root.header.ndims*_root.header.nnodes
      - # Dummy
        type: u4
      - # Dummy
        type: u4
      - id: segdata
        type: f8
        repeat: expr
        repeat-expr: _root.header.nsegs*_root.header.nsegdata
      - # Dummy
        type: u4
      - # Dummy
        type: u4
      - id: nodesdata
        type: f8
        repeat: expr
        repeat-expr: _root.header.nnodes*_root.header.nnodedata
      - # Dummy
        type: u4
      - # Dummy
        type: u4
      - id: nodesdata_int
        type: node_struct
        repeat: expr
        repeat-expr: _root.header.nnodes
      - # Dummy
        type: u4
      - # Dummy
        type: u4
      - id: segdata_int
        type: seg_struct
        repeat: expr
        repeat-expr: _root.header.nsegs
      - # Dummy
        type: u4
  node_struct:
    seq:
      - id: pos_index
        type: u4
      - id: flags
        type: u4
        doc: Non-null if on boundary
      - id: nnext
        type: u4
        doc: Number of arcs connected
      - id: type
        type: u4
      - id: index
        type: u4
        doc: Critical index
      - id: nsegs
        type: u4
        repeat: expr
        repeat-expr: nnext
      - id: seg_data
        type: seg_data_struct
        repeat: expr
        repeat-expr: nnext
  seg_data_struct:
    seq:
      - id: next_node
        type: u4
      - id: next_seg
        type: u4
  seg_struct:
    seq:
      - id: pos_index
        type: u4
      - id: nodes
        type: u4
        repeat: expr
        repeat-expr: 2
        doc: Index of the nodes at the extremity of the arc. Segment is oriented from nodes[0] toward nodes[1]
      - id: flags
        type: u4
        doc: Non-null if on boundary
      - id: index
        type: u4
        doc: Index of the segment in the Seg array
      - id: next_seg
        type: u4
        doc: Index of the next segment in the arc, -1 if extremity
      - id: prev_seg
        type: u4
        doc: Index of the previous segment in the arc, -1 if extremity