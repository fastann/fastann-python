mod util;
use fa;
use fa::core::ann_index::ANNIndex;
use fa::core::arguments;
use fa::core::metrics;
use fa::core::node;
use pyo3::conversion::FromPyObject;
use pyo3::prelude::*;
use pyo3::types::PyList;
use pyo3::wrap_pyfunction;

#[pyclass]
struct ANNNode {
    #[pyo3(get)]
    vectors: Vec<f32>, // the vectors;
    #[pyo3(get)]
    idx: String, // data id, it can be any type;
}

fn transform(src: &[(node::Node<f32, String>, f32)]) -> Vec<(ANNNode, f32)> {
    let mut dst = Vec::new();
    for i in src.iter() {
        dst.push((
            ANNNode {
                vectors: i.0.vectors().clone(),
                idx: i.0.idx().unwrap(),
            },
            i.1.clone(),
        ));
    }
    dst
}

#[pyclass]
struct BruteForceIndex {
    _idx: Box<fa::bf::bf::BruteForceIndex<f32, String>>, // use f32 and string, because pyo3 don't support generic
}

#[pyclass]
struct BPForestIndex {
    _idx: Box<fa::bpforest::bpforest::BinaryProjectionForestIndex<f32, String>>,
}

#[pyclass]
struct HnswIndex {
    _idx: Box<fa::hnsw::hnsw::HnswIndex<f32, String>>,
}

#[pyclass]
struct PQIndex {
    _idx: Box<fa::pq::pq::PQIndex<f32, String>>,
}

#[pymethods]
impl BruteForceIndex {
    #[new]
    fn new() -> Self {
        BruteForceIndex {
            _idx: Box::new(fa::bf::bf::BruteForceIndex::<f32, String>::new()),
        }
    }
}

#[pymethods]
impl BPForestIndex {
    #[new]
    fn new(dimension: usize, tree_num: i32, search_k: i32) -> Self {
        BPForestIndex {
            _idx: Box::new(fa::bpforest::bpforest::BinaryProjectionForestIndex::<
                f32,
                String,
            >::new(dimension, tree_num, search_k)),
        }
    }
}

#[pymethods]
impl HnswIndex {
    #[new]
    fn new(
        demension: usize,
        max_item: usize,
        n_neigh: usize,
        n_neigh0: usize,
        max_level: usize,
        metri: String,
        ef: usize,
        has_deletion: bool,
    ) -> Self {
        HnswIndex {
            _idx: Box::new(fa::hnsw::hnsw::HnswIndex::<f32, String>::new(
                demension,
                max_item,
                n_neigh,
                n_neigh0,
                max_level,
                util::util::metrics_transform(&metri),
                ef,
                has_deletion,
            )),
        }
    }
}

#[pymethods]
impl PQIndex {
    #[new]
    fn new(
        demension: usize,
        n_sub: usize,
        sub_bits: usize,
        train_epoch: usize,
        metri: String,
    ) -> Self {
        PQIndex {
            _idx: Box::new(fa::pq::pq::PQIndex::<f32, String>::new(
                demension,
                n_sub,
                sub_bits,
                train_epoch,
                util::util::metrics_transform(&metri),
            )),
        }
    }
}

#[macro_export]
macro_rules! inherit_ann_index_method {
    (  $ann_idx:ident  ) => {
        impl $ann_idx {
            fn add_node(&mut self, item: &node::Node<f32, String>) -> PyResult<bool> {
                return match self._idx.add_node(item) {
                    Ok(()) => Ok(true),
                    Err(e) => Ok(false), //TODO
                };
            }
            fn node_search_k(
                &self,
                item: &node::Node<f32, String>,
                k: usize,
            ) -> PyResult<Vec<(ANNNode, f32)>> {
                Ok(transform(&self._idx.node_search_k(
                    item,
                    k,
                    &arguments::Arguments::new(),
                ))) //TODO: wrap argument
            }
        }

        #[pymethods]
        impl $ann_idx {
            fn construct(&mut self, s: String) -> PyResult<bool> {
                self._idx
                    .construct(util::util::metrics_transform(&s))
                    .unwrap();
                Ok(true)
            }
            fn add_without_idx(&mut self, pyvs: &PyList) -> PyResult<bool> {
                let mut vs = Vec::new();
                for i in pyvs.iter() {
                    vs.push(i.extract::<f32>().unwrap())
                }
                let n = node::Node::new(&vs);
                self.add_node(&n)
            }
            fn add(&mut self, pyvs: &PyList, idx: String) -> PyResult<bool> {
                let mut vs = Vec::new();
                for i in pyvs.iter() {
                    vs.push(i.extract::<f32>().unwrap())
                }
                let n = node::Node::new_with_idx(&vs, idx);
                self.add_node(&n)
            }

            fn search_k(&self, pyvs: &PyList, k: usize) -> PyResult<Vec<(ANNNode, f32)>> {
                let mut vs = Vec::new();
                for i in pyvs.iter() {
                    vs.push(i.extract::<f32>().unwrap())
                }
                let n = node::Node::new(&vs);
                self.node_search_k(&n, k)
            }

            fn name(&self) -> PyResult<String> {
                Ok(stringify!($ann_idx).to_string())
            }
        }
    };
}

inherit_ann_index_method!(BruteForceIndex);
inherit_ann_index_method!(BPForestIndex);
inherit_ann_index_method!(HnswIndex);
inherit_ann_index_method!(PQIndex);

/// A Python module implemented in Rust.
#[pymodule]
fn fastann(py: Python, m: &PyModule) -> PyResult<()> {
    m.add_class::<BruteForceIndex>()?;
    m.add_class::<BPForestIndex>()?;
    m.add_class::<HnswIndex>()?;
    m.add_class::<PQIndex>()?;
    Ok(())
}
