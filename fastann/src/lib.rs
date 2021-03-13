mod util;
use fa;
use fa::core::ann_index::ANNIndex;
use fa::core::ann_index::SerializableIndex;
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
                idx: (*i.0.idx().as_ref().unwrap()).to_string(),
            },
            i.1.clone(),
        ));
    }
    dst
}

#[macro_export]
macro_rules! inherit_ann_index_method {
    (  $ann_idx:ident,$type_expr: ty) => {
        #[pyclass]
        struct $ann_idx {
            _idx: Box<$type_expr>,
        }

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
                    &arguments::Args::new(),
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

            #[staticmethod]
            fn load(path: String) -> Self {
                $ann_idx {
                    _idx: Box::new(<$type_expr>::load(&path, &arguments::Args::new()).unwrap()),
                }
            }

            fn dump(&mut self, path: String) {
                self._idx.dump(&path, &arguments::Args::new());
            }
        }
    };
}

inherit_ann_index_method!(BruteForceIndex, fa::bf::bf::BruteForceIndex<f32,String>);
inherit_ann_index_method!(BPForestIndex, fa::bpforest::bpforest::BinaryProjectionForestIndex<f32, String>);
inherit_ann_index_method!(HNSWIndex, fa::hnsw::hnsw::HNSWIndex<f32, String>);
inherit_ann_index_method!(PQIndex, fa::pq::pq::PQIndex<f32, String>);
inherit_ann_index_method!(SatelliteSystemGraphIndex, fa::mrng::ssg::SatelliteSystemGraphIndex<f32, String>);

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
impl HNSWIndex {
    #[new]
    fn new(
        dimension: usize,
        max_item: usize,
        n_neigh: usize,
        n_neigh0: usize,
        max_level: usize,
        ef: usize,
        has_deletion: bool,
    ) -> Self {
        HNSWIndex {
            _idx: Box::new(fa::hnsw::hnsw::HNSWIndex::<f32, String>::new(
                dimension,
                max_item,
                n_neigh,
                n_neigh0,
                max_level,
                ef,
                has_deletion,
            )),
        }
    }
}

#[pymethods]
impl PQIndex {
    #[new]
    fn new(dimension: usize, n_sub: usize, sub_bits: usize, train_epoch: usize) -> Self {
        PQIndex {
            _idx: Box::new(fa::pq::pq::PQIndex::<f32, String>::new(
                dimension,
                n_sub,
                sub_bits,
                train_epoch,
            )),
        }
    }
}

#[pymethods]
impl SatelliteSystemGraphIndex {
    #[new]
    fn new(        dimension: usize,
        neighbor_size: usize,
        init_k: usize,
        index_size: usize,
        angle: f32,
        root_size: usize,) -> Self {
        SatelliteSystemGraphIndex {
            _idx: Box::new(fa::mrng::ssg::SatelliteSystemGraphIndex::<f32, String>::new(dimension, neighbor_size, init_k, index_size, angle, root_size)),
        }
    }
}

/// A Python module implemented in Rust.
#[pymodule]
fn fastann(py: Python, m: &PyModule) -> PyResult<()> {
    m.add_class::<BruteForceIndex>()?;
    m.add_class::<BPForestIndex>()?;
    m.add_class::<HNSWIndex>()?;
    m.add_class::<PQIndex>()?;
    m.add_class::<SatelliteSystemGraphIndex>()?;
    Ok(())
}
