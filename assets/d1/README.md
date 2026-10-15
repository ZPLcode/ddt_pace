# D1 URDF

The D1 asset is copied from `go2_rl_robotlab/resources/d1/`.
`robot.urdf` preserves its joint frames, limits, masses, centers of mass, inertia,
and geometry. Mesh package URIs are replaced with relative `meshes/` paths so
the asset can be loaded from this repository.

`manifest.json` records the source URDF hash, the repository URDF hash, and mesh
hashes. PACE checkpoints also record the selected URDF hash; `fit.py --checkpoint`
rejects checkpoints fitted with another asset.

Regenerate the hardware geometry header after changing the asset:

```bash
python assets/d1/generate_geometry.py
```

The generator requires NumPy, SciPy and trimesh. Collected data and checkpoints
stay outside this asset directory.
