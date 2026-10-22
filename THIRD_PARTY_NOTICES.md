# Third-party notices

The project does not claim copyright in CFMS-34, BGE or its dependencies. Dataset citation and scope are documented in [licensing](docs/licensing.md). Research dates and source revision are retained there.

Direct package metadata inspected in the existing CPU research environment on 2026-10-22:

| Dependency | Inspected version | Upstream licence family |
| --- | --- | --- |
| NumPy | 1.26.4 | BSD-3-Clause; wheel bundles additional libraries/notices |
| SciPy | 1.12.0 | BSD-3-Clause; wheel bundles BLAS and compiler runtimes with separate terms |
| scikit-learn | 1.8.0 | BSD-3-Clause |
| joblib | 1.4.2 | BSD-3-Clause |
| FastAPI | 0.128.0 | MIT |
| Uvicorn | 0.40.0 | BSD-3-Clause |
| Pydantic | 2.12.3 | MIT |
| psutil | 7.0.0 | BSD-3-Clause |
| jsonschema | 4.26.0 | MIT |
| pytest (test only) | 9.0.2 | MIT |
| HTTPX (test only) | 0.28.1 | BSD-3-Clause |

These are not a complete transitive binary SBOM. No dependency source or wheels are redistributed in the repositories/releases. Use the distribution's own licence files and metadata; when redistributing a container or offline environment, inventory every transitive package and native library, preserve notices, and meet any source/relocation obligations. No licence compatibility conclusion for an unbuilt downstream distribution is implied.
