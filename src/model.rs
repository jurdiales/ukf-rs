use crate::types::{ControlVector, StateVector, MeasurementVector};

pub trait KFModel<const N: usize, const M: usize, const L: usize> {
    /// State transition function
    fn f(&self, x: &StateVector<N>, u: Option<&ControlVector<L>>) -> StateVector<N>;
    fn g(&self, x: &StateVector<N>) -> MeasurementVector<M>;
}

#[derive(Clone)]
pub struct CTCA<const N: usize> {

}

impl<const N: usize, const M: usize, const L: usize> KFModel<N, M, L> for CTCA<N> {
    fn f(&self, x: &StateVector<N>, u: Option<&ControlVector<L>>) -> StateVector<N> {
        let xx = x.clone();
        xx
    }

    fn g(&self, x: &StateVector<N>) -> MeasurementVector<M> {
        MeasurementVector::<M>::zeros()
    }
}
