'''
A minimal glTF/GLB reader, just enough to answer the geometry questions that
setting up the 3D viewer requires: where a part sits in the scene, and which
axis a shaft rotates about.

This is setup-time code only. The actual rendering is done by three.js in the
browser (see viewer3d.py / arm_viewer.js), which does its own parsing.
'''

import json
import re
import struct
from typing import NamedTuple

import numpy as np

# glTF componentType -> numpy dtype, and type -> number of components
_DTYPE = {5120: 'i1', 5121: 'u1', 5122: 'i2', 5123: 'u2', 5125: 'u4', 5126: 'f4'}
_NCOMP = {'SCALAR': 1, 'VEC2': 2, 'VEC3': 3, 'VEC4': 4, 'MAT4': 16}


class Glb(NamedTuple):
    gltf: dict   # the JSON chunk
    buf: bytes   # the BIN chunk


class ShaftAxis(NamedTuple):
    '''The axis of rotation of a roughly cylindrical part.'''
    point: np.ndarray      # a point on the axis, in world coordinates (m)
    direction: np.ndarray  # unit vector along the axis, in world coordinates
    length: float          # extent of the part along the axis (m)
    max_radius: float      # largest distance from the axis to any vertex (m)


def load(path):
    '''Read a binary .glb into its JSON and BIN chunks.'''
    with open(path, 'rb') as f:
        magic, version, total = struct.unpack('<III', f.read(12))
        if magic.to_bytes(4, 'little') != b'glTF':
            raise ValueError(f'{path} is not a binary glTF (.glb)')
        if version != 2:
            raise ValueError(f'{path} is glTF version {version}, expected 2')
        gltf, buf = None, b''
        while f.tell() < total:
            length, kind = struct.unpack('<II', f.read(8))
            chunk = f.read(length)
            kind = kind.to_bytes(4, 'little')
            if kind == b'JSON':
                gltf = json.loads(chunk)
            elif kind.startswith(b'BIN'):
                buf = chunk
    return Glb(gltf, buf)


def accessor(glb, index):
    '''Read accessor `index` as an (count, ncomp) array.'''
    acc = glb.gltf['accessors'][index]
    view = glb.gltf['bufferViews'][acc['bufferView']]
    dtype = np.dtype(_DTYPE[acc['componentType']])
    ncomp = _NCOMP[acc['type']]
    offset = view.get('byteOffset', 0) + acc.get('byteOffset', 0)
    packed = dtype.itemsize * ncomp
    stride = view.get('byteStride') or packed
    if stride == packed:
        flat = np.frombuffer(glb.buf, dtype=dtype, count=acc['count'] * ncomp, offset=offset)
        return flat.reshape(acc['count'], ncomp)
    # Interleaved: pull the rows out byte-wise, then reinterpret the leading columns
    rows = np.frombuffer(glb.buf, dtype=np.uint8, count=stride * acc['count'], offset=offset)
    rows = rows.reshape(acc['count'], stride)[:, :packed]
    return np.ascontiguousarray(rows).view(dtype).reshape(acc['count'], ncomp)


def parents(glb):
    '''Map node index -> parent node index (roots are absent).'''
    out = {}
    for i, node in enumerate(glb.gltf['nodes']):
        for child in node.get('children', []):
            out[child] = i
    return out


def local_matrix(node):
    '''The node's local transform as a 4x4 matrix.'''
    if 'matrix' in node:
        # glTF stores matrices column-major
        return np.array(node['matrix'], dtype=float).reshape(4, 4).T
    M = np.eye(4)
    if 'rotation' in node:
        x, y, z, w = node['rotation']
        M[:3, :3] = np.array([
            [1 - 2 * (y * y + z * z), 2 * (x * y - z * w), 2 * (x * z + y * w)],
            [2 * (x * y + z * w), 1 - 2 * (x * x + z * z), 2 * (y * z - x * w)],
            [2 * (x * z - y * w), 2 * (y * z + x * w), 1 - 2 * (x * x + y * y)],
        ])
    if 'scale' in node:
        M[:3, :3] = M[:3, :3] @ np.diag(node['scale'])
    if 'translation' in node:
        M[:3, 3] = node['translation']
    return M


def world_matrix(glb, index, parent_of=None):
    '''The node's transform all the way up to the scene root.'''
    parent_of = parent_of if parent_of is not None else parents(glb)
    chain, i = [], index
    while True:
        chain.append(i)
        if i not in parent_of:
            break
        i = parent_of[i]
    M = np.eye(4)
    for i in reversed(chain):
        M = M @ local_matrix(glb.gltf['nodes'][i])
    return M


def _normalize(name):
    '''Collapse a node name the way three.js's name sanitizing does, so that
    names match whether they came from the file or from the loaded scene.'''
    return re.sub(r'[\s._:/\[\]]', '', name).lower()


def find_node(glb, name):
    '''Index of the node called `name`. Exact matches win over substring ones.'''
    want = _normalize(name)
    exact, partial = [], []
    for i, node in enumerate(glb.gltf['nodes']):
        got = _normalize(node.get('name', ''))
        if got == want:
            exact.append(i)
        elif want and want in got:
            partial.append(i)
    hits = exact or partial
    if not hits:
        raise KeyError(f'no node named {name!r}')
    return hits[0]


def subtree(glb, index):
    '''Indices of `index` and everything below it.'''
    out = [index]
    for child in glb.gltf['nodes'][index].get('children', []):
        out += subtree(glb, child)
    return out


def world_points(glb, index, parent_of=None):
    '''Every vertex of `index` and its descendants, in world coordinates.'''
    parent_of = parent_of if parent_of is not None else parents(glb)
    chunks = []
    for i in subtree(glb, index):
        node = glb.gltf['nodes'][i]
        if 'mesh' not in node:
            continue
        M = world_matrix(glb, i, parent_of)
        for prim in glb.gltf['meshes'][node['mesh']]['primitives']:
            v = accessor(glb, prim['attributes']['POSITION']).astype(float)
            chunks.append((np.c_[v, np.ones(len(v))] @ M.T)[:, :3])
    if not chunks:
        raise ValueError(f'node {index} has no geometry under it')
    return np.vstack(chunks)


def fit_shaft_axis(glb, name):
    '''Fit the axis of rotation of a shaft-like part.

    A shaft is long and thin, so its vertices spread far along the axis and
    barely at all across it: the first principal component of the vertex cloud
    is the axis, and the centroid is a point on it.
    '''
    P = world_points(glb, find_node(glb, name))
    centroid = P.mean(axis=0)
    _, sv, basis = np.linalg.svd(P - centroid, full_matrices=False)
    direction = basis[0]
    # The sign of a principal component is arbitrary; pick one deterministically
    if direction[np.argmax(np.abs(direction))] < 0:
        direction = -direction
    along = (P - centroid) @ direction
    radial = np.linalg.norm((P - centroid) - np.outer(along, direction), axis=1)
    if sv[1] > 0.2 * sv[0]:
        raise ValueError(
            f'{name!r} is not shaft-like enough to fit an axis to '
            f'(singular values {sv.round(4)}); pass the axis in by hand')
    return ShaftAxis(centroid, direction, float(np.ptp(along)), float(radial.max()))


def summary(glb):
    '''Counts, for a quick look at how heavy a file is to render.'''
    meshes = glb.gltf['meshes']
    prims = [p for m in meshes for p in m['primitives']]
    tris = sum(glb.gltf['accessors'][p['indices']]['count'] // 3
               for p in prims if 'indices' in p)
    verts = sum(glb.gltf['accessors'][p['attributes']['POSITION']]['count'] for p in prims)
    return {'nodes': len(glb.gltf['nodes']), 'meshes': len(meshes),
            'primitives': len(prims), 'triangles': tris, 'vertices': verts,
            'materials': len(glb.gltf.get('materials', []))}


def part_center(glb, name):
    '''Centre of a part's bounding box, in world coordinates.

    The bounding box rather than the vertex mean, so that a densely tessellated
    corner doesn't drag the result towards itself.
    '''
    P = world_points(glb, find_node(glb, name))
    return (P.min(axis=0) + P.max(axis=0)) / 2
