mod types;
mod model;
use crate::types::{Kfloat, Covariances};
use model::KFModel;
use nalgebra::SMatrix;
use types::{ControlVector, CovMatrix, FilterState, MeasurementVector, SigmaPoints, StateVector};

/// Unscented Kalman Filter structure implementation.
/// 
/// This version of the Kalman Filter uses a set of sigma points to propagate means and covariances through system and
/// measurement functions. Unlike the Kalman Filter, the state and measurement functions can be non-linear. This is 
/// the additive noise version and it is specially designed to handle cases in which several consecutive prediction 
/// or estimation steps are executed.
pub struct UnscentedKalmanFilter<'a, const N: usize, const M: usize, const L: usize>
{
    /// Current system state (prediction or estimation depending on last step)
    pub x: StateVector<N>,
    /// Kalman Filter predicted state vector
    pub prediction: StateVector<N>,
    /// Kalman Filter estimated state vector
    pub estimation: StateVector<N>,
    /// Last measurement given to the Kalman Filter
    pub y: MeasurementVector<M>,
    /// Kalman Filter innovation vector
    pub innovation: MeasurementVector<M>,
    /// Control vector
    pub control: ControlVector<L>,
    // ----------------------------------------------------------------------- //
    /// A priori prediction
    y_pred: MeasurementVector<M>,
    /// Kalman Filter last 
    step: FilterState,
    // ----------------------------------------------------------------------- //
    /// Sigma Points variables
    sg: SigmaPoints<N, M>,
    // Filter internal covariance matrices
    cov: Covariances<N, M, L>,
    // Model used: any type that implements the KFModel trait
    model: Box<dyn KFModel<N, M, L> + 'a>   // TODO: find a way to do this without the Box indirection, 
                                            // the problem is that with the KFModel<> the compiler cannot know the
                                            // size of the variable. It need to be Sized, but this has problems 
}

impl<'a, const N: usize, const M: usize, const L: usize> UnscentedKalmanFilter<'a, N, M, L> {
    /// Constant indicating the number of sigma points
    const K: usize = 2 * N + 1;

    /// Unscented Kalman Filter constructor. Creates a new Unscented Kalman Filter given a model and some parameters needed to generate the sigma points.
    pub fn new(model: impl KFModel<N, M, L> + 'a, alpha: Kfloat, beta: Option<Kfloat>, kappa: Option<Kfloat>) -> Self {
        UnscentedKalmanFilter {
            x: StateVector::<N>::zeros(),
            prediction: StateVector::<N>::zeros(),
            estimation: StateVector::<N>::zeros(),
            y: MeasurementVector::<M>::zeros(),
            innovation: MeasurementVector::<M>::zeros(),
            control: ControlVector::<L>::zeros(),
            // ----------------------------------------- //
            y_pred: MeasurementVector::<M>::zeros(),
            step: FilterState::INITIAL,
            // ----------------------------------------- //
            sg: SigmaPoints::new(alpha, beta, kappa),
            cov: Covariances::new(),
            model: Box::new(model)
        }
    }

    /// Sets the initial state and covariance matrices of the filter.
    pub fn init(&mut self, x0: &StateVector<N>, p0: &CovMatrix<N>, q0: &CovMatrix<N>, r0: &SMatrix<Kfloat, M, M>) {
        // 1. set initial state values
        self.x.copy_from(x0);
        self.prediction.copy_from(x0);
        self.estimation.copy_from(x0);

        // 2. set initial covariance values
        self.cov.p.copy_from(p0);
        self.cov.q.copy_from(q0);
        self.cov.r.copy_from(r0);

        // 3. get initial sigma points from initial state
        self.sg.compute(&self.x, &self.cov.p);
    }

    /// Kalman Filter prediction step. Calculates the predicted state.
    pub fn predict(&mut self, u: Option<&StateVector<L>>) {
        // 1. Get system input control vector, if any
        self.control.copy_from(u.unwrap_or(&ControlVector::<L>::zeros()));

        // 2. Compute predicted sigma points
        for i in 0..Self::K {
            let sg_col: StateVector<N> = self.sg.sigma_points.column(i).into();
            let new_sg_state = self.model.f(&sg_col, u);
            self.sg.sigma_points.set_column(i, &new_sg_state);
        }

        // 3. Calculate mean of the predicted sigma points, which in turn represents the predicted estate
        self.x = StateVector::<N>::zeros();
        for i in 0..Self::K {
            self.x += (self.sg.wm[i] as Kfloat) * self.sg.sigma_points.column(i);
        }

        // 4. Compute the a-priori state covariance matrix P
        self.cov.p.copy_from(&self.cov.q);
        for i in 0..Self::K {
            let a: StateVector<N> = self.sg.sigma_points.column(i) - self.x;
            let b = a.transpose();
            self.cov.p += (self.sg.wc[i] as Kfloat) * (a * b);
        }

        // 5. Update sigma points (recalculate them to incorporate the effect of process noise)
        self.sg.compute(&self.x, &self.cov.p);

        // Update internal variables
        self.prediction.copy_from(&self.x);
        self.step = FilterState::PREDICTION;
    }

    /// Kalman Filter innovation step. Calculates innovation covariance matrix and innovation vector.
    pub fn innovate(&mut self, y: &MeasurementVector<M>) {
        // check last filter state to avoid to compute the innovation covariance matrix several times when 
        // dealing with multi-object or multi-sensor tracking
        if self.step == FilterState::INNOVATION {
            self.y.copy_from(y);
            self.innovation = self.y - self.y_pred;
            return;
        }

        // 1. Get and store the new measurement
        self.y.copy_from(y);

        // 2. Compute innovation covariance matrix
        self.compute_innovation_covariance();

        // 3. Get innovation vector (also called pre-residual)
        self.innovation = self.y - self.y_pred;
        self.step = FilterState::INNOVATION;
    }

    pub fn correct(&mut self) {
        // this function cannot be called two or more consecutive times
        if self.step == FilterState::UPDATE {
            panic!("Consecutive calls to correct() not allowed. Use update() instead");
        }

        // 1. Compute cross-covariance matrix
        let mut p_xy = SMatrix::<Kfloat, N, M>::zeros();
        for i in 0..Self::K {
            let a: StateVector<N> = self.sg.sigma_points.column(i) - self.x;
            let b = (self.sg.output_sigma_points.column(i) - self.y_pred).transpose();
            p_xy += (self.sg.wc[i] as Kfloat) * (a * b);
        }

        // 2. Compute Kalman Filter gain
        let k = p_xy * self.cov.si;

        // 3. State stimation
        self.x = self.x + k * (self.y - self.y_pred);

        // 4. Update a posteriori state error covariance matrix
        self.cov.p = self.cov.p - (k * self.cov.s * k.transpose());

        // 5. update sigma points with new state and its covariance
        self.sg.compute(&self.estimation, &self.cov.p);

        // Update internal variables
        self.estimation.copy_from(&self.x);
        self.step = FilterState::UPDATE;
    }

    /// Kalman Filter update step. It executes innovation and correction steps.
    pub fn update(&mut self, y: &MeasurementVector<M>) {
        self.innovate(y);
        self.correct();
    }

    /// Computes innovation covariance and inverse innovation covariance matrices for the Kalman Filter innovation step.
    pub fn compute_innovation_covariance(&mut self) {
        // 1. Compute output sigma points
        self.sg.clear_output_sigma_points();
        for i in 0..Self::K {
            self.sg.output_sigma_points.set_column(i, &self.model.g(&self.sg.sigma_points.column(i).into()));
        }

        // 2. Compute output mean (predicted measurement)
        self.y_pred = MeasurementVector::<M>::zeros();
        for i in 0..Self::K {
            self.y_pred += (self.sg.wm[i] as Kfloat) * self.sg.output_sigma_points.column(i);
        }

        // 3. Update innovation covariance and its square root matrices
        self.cov.s.copy_from(&self.cov.r);
        for i in 0..Self::K {
            let a: MeasurementVector<M> = self.sg.output_sigma_points.column(i) - self.y_pred;
            let b = a.transpose();
            self.cov.s += (self.sg.wc[i] as Kfloat) * (a * b);
        }
        self.cov.si = self.cov.s.try_inverse().unwrap();    // TODO: this can panic
    }
}
