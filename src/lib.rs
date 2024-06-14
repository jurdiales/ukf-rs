// #![feature(generic_const_exprs)]
// #![feature(adt_const_params)]
// #![allow(incomplete_features)]

mod types;
mod model;
use crate::types::{Kfloat, Covariances};
use model::KFModel;
use nalgebra::{SMatrix, SVector};
use types::{FilterState, MeasurementVector, SigmaPoints, StateVector};

/// Unscented Kalman Filter structure implementation.
pub struct UnscentedKalmanFilter<'a, const N: usize, const M: usize, const L: usize>
// where [(); 2 * N + 1]: Sized,
{
    /// Kalman Filter predicted state vector
    pub prediction: StateVector<N>,
    /// Kalman Filter estimated state vector
    pub estimation: StateVector<N>,
    /// Last measurement given to the Kalman Filter
    pub measurement: MeasurementVector<M>,
    /// Kalman Filter innovation vector
    pub innovation: MeasurementVector<M>,
    predicted_measurement: MeasurementVector<M>,
    /// Kalman Filter last 
    step: FilterState,
    /// Sigma Points variables
    sg: SigmaPoints<N>,
    // Filter internal covariance matrices
    cov: Covariances<N, M, L>,
    // Model used: any type that implements 
    model: Box<dyn KFModel<N, M, L> + 'a>   // TODO: find a way to do this without the Box indirection, the problem is that with the KFModel<> the compiler cannot know the
                                            // size of the variable. It need to be Sized, but this has problems 
}

impl<'a, const N: usize, const M: usize, const L: usize> UnscentedKalmanFilter<'a, N, M, L> {
    /// Constant indicating the number of sigma points
    const K: usize = 2 * N + 1;

    /// Unscented Kalman Filter constructor. Creates a new Unscented Kalman Filter given a model and some parameters needed to generate the sigma points.
    pub fn new(model: impl KFModel<N, M, L> + 'a, alpha: Kfloat, beta: Option<Kfloat>, kappa: Option<Kfloat>) -> Self {
        UnscentedKalmanFilter {
            prediction: StateVector::<N>::zeros(),
            estimation: StateVector::<N>::zeros(),
            measurement: MeasurementVector::<M>::zeros(),
            innovation: MeasurementVector::<M>::zeros(),
            predicted_measurement: MeasurementVector::<M>::zeros(),
            step: FilterState::INITIAL,
            sg: SigmaPoints::new(alpha, beta, kappa),
            cov: Covariances::new(),
            model: Box::new(model)
        }
    }

    /// Initializes the state and covariance matrices of the filter.
    pub fn init(&mut self, x0: SVector<Kfloat, N>,
                p0: SMatrix<Kfloat, N, N>,
                q0: SMatrix<Kfloat, N, N>,
                r0: SMatrix<Kfloat, M, M>) {
        // set initial state values
        self.prediction.copy_from(&x0);
        self.estimation.copy_from(&x0);

        // set initial covariance values
        self.cov.p.copy_from(&p0);
        self.cov.q.copy_from(&q0);
        self.cov.r.copy_from(&r0);
    }

    /// Kalman Filter prediction step.
    pub fn predict(&mut self, u: Option<&StateVector<L>>) {
        // 1. Get system input control vector, if any
        // let control = u.unwrap_or(StateVector::<L>::zeros());

        // 2. Compute predicted sigma points
        for i in 0..Self::K {
            let sg_col: StateVector<N> = self.sg.sigma_points.column(i).into();
            let new_sg_state = self.model.f(&sg_col, u);
            self.sg.sigma_points.set_column(i, &new_sg_state);
        }

        // 3. Calculate mean of the predicted sigma points, which in turn represents the predicted estate
        self.prediction = StateVector::<N>::zeros();
        for i in 0..Self::K {
            self.prediction += self.sg.wm[i] * self.sg.sigma_points.column(i);
        }

        // 4. Compute the a-priori state covariance matrix
        self.cov.p.copy_from(&self.cov.q);
        for i in 0..Self::K {
            let a: StateVector<N> = self.sg.sigma_points.column(i) - self.prediction;
            let b = a.transpose();
            self.cov.p += self.sg.wc[i] * (a * b);
        }

        // 5. Update sigma points (recalculate them to incorporate the effect of process noise)
        self.sg.compute(&self.prediction, &self.cov.p);
        self.step = FilterState::PREDICTION;
    }

    /// Kalman Filter innovation step. Calculates innovation covariance matrix and innovation vector.
    pub fn innovation(&mut self, y: &MeasurementVector<M>) {
        // check last filter state to avoid to compute the innovation covariance several times when dealing with
        // multi-object tracking
        if self.step == FilterState::INNOVATION {
            self.measurement.copy_from(y);
            self.innovation = self.measurement - self.predicted_measurement;
            return;
        }

        // 1. Get and store the new measurement
        self.measurement.copy_from(y);

        // 2. Compute innovation covariance matrix

    }

    pub fn update(&mut self) {
        self.step = FilterState::UPDATE;
    }

    /// Computes innovation covariance and inverse innovation covariance matrices.
    pub fn compute_innovation_covariance(&self) {
        // 1. Compute output sigma points
    }

    /// Filter state estimation.
    pub fn x(&self) -> &StateVector<N> {
        &self.estimation
    }
}



pub fn add(left: usize, right: usize) -> usize {
    left + right
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn it_works() {
        let result = add(2, 2);
        assert_eq!(result, 4);
    }
}