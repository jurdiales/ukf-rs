use nalgebra::{Const, DVector, Dyn, OMatrix, SMatrix, SVector};

/// Type used for all Kalman Filter floating point variables
pub type Kfloat = f32;
/// Type for state vectors of dimension N
pub type StateVector<const N: usize> = SVector<Kfloat, N>;
/// Type for control vectors of dimension L
pub type ControlVector<const L: usize> = SVector<Kfloat, L>;
/// Type for measurement vectors of dimension M
pub type MeasurementVector<const M: usize> = SVector<Kfloat, M>;
/// Type for covariance matrices of size (N x N)
pub type CovMatrix<const N: usize> = SMatrix<Kfloat, N, N>;

// ------------------------------------------------------------------------------------------ //

#[derive(PartialEq)]
pub enum FilterState {
    INITIAL,
    PREDICTION,
    INNOVATION,
    UPDATE
}

// ****************************************************************************************** //
// Kalman Filter covariance matrices
// ****************************************************************************************** //

/// Struct to store the covariance matrices used in the Kalman filter.
pub struct Covariances<const N: usize, const M: usize, const L: usize> {
    /// State error covariance matrix P
    pub p: CovMatrix<N>,
    /// Process noise covariance matrix Q
    pub q: CovMatrix<N>,
    /// Measurement noise covariance matrix R
    pub r: SMatrix<Kfloat, M, M>,
    /// Innovation covariance matrix S
    pub s: SMatrix<Kfloat, M, M>,
    /// Innovation covariance matrix inverse SI
    pub si: SMatrix<Kfloat, M, M>,
}

impl<const N: usize, const M: usize, const L: usize> Covariances<N, M, L> {
    /// Zero initialization of all the matrices.
    pub fn new() -> Self {
        Covariances {
            p: SMatrix::<Kfloat, N, N>::zeros(),
            q: SMatrix::<Kfloat, N, N>::zeros(),
            r: SMatrix::<Kfloat, M, M>::zeros(),
            s: SMatrix::<Kfloat, M, M>::zeros(),
            si: SMatrix::<Kfloat, M, M>::zeros(),
        }
    }
}

// ****************************************************************************************** //
// Sigma Points store and computation
// ****************************************************************************************** //

pub struct SigmaPoints<const N: usize, const M: usize> {
    pub gamma: Kfloat,
    /// Set of sigma points of the Kalman Filter
    pub sigma_points: OMatrix<Kfloat, Const<N>, Dyn>,           // TODO: SMatrix<Kfloat, N, {2 * N + 1} when generic_const_exprs
    pub output_sigma_points: OMatrix<Kfloat, Const<M>, Dyn>,    // TODO: SMatrix<Kfloat, M, {2 * N + 1} when generic_const_exprs
    pub wm: DVector<Kfloat>,                                    // TODO: wm: [Kfloat; 2 * N + 1] when generic_const_exprs
    pub wc: DVector<Kfloat>,                                    // TODO: wm: [Kfloat; 2 * N + 1] when generic_const_exprs
}

impl<const N: usize, const M: usize> SigmaPoints<N, M> {
    /// Constant indicating the number of sigma points
    const K: usize = 2 * N + 1;

    /// Initializes a new set of sigma points.
    /// 
    /// Alpha determines the spread of the sigma points around the mean of the state and is usually set to a small positive
    /// value like 1e-3. Beta and kappa are optional values with with default values of 2.0 and 0.0 respectively. Beta is 
    /// used to incorporate prior knowledge of the distribution of x and for gaussian distributions the vaule of 2.0 is optimal.
    pub fn new(alpha: Kfloat, beta: Option<Kfloat>, kappa: Option<Kfloat>) -> Self {
        // compute compound scaling parameters
        let lambda: Kfloat = alpha.powi(2) * ((N as Kfloat) + kappa.unwrap_or(0.0)) - (N as Kfloat);
        let gamma: Kfloat = Kfloat::sqrt((N as Kfloat) + lambda);

        let mut wm: DVector<Kfloat> = DVector::zeros(Self::K);        // TODO: SVector<Kfloat, {UnscentedKalmanFilter::K}>
        let mut wc: DVector<Kfloat> = DVector::zeros(Self::K);        // TODO: SVector<Kfloat, {UnscentedKalmanFilter::K}>
        
        // compute weights for central sigma point
        wm[0] = lambda / ((N as Kfloat) + lambda);
        wc[0] = lambda / ((N as Kfloat) + lambda) + (1.0 - alpha.powi(2) + beta.unwrap_or(2.0));

        // compute weights for other sigma points
        for i in 1..Self::K {
            wm[i] = 0.5 / ((N as Kfloat) + lambda);
            wc[i] = 0.5 / ((N as Kfloat) + lambda);
        }

        SigmaPoints {
            gamma,
            sigma_points: OMatrix::<Kfloat, Const<N>, Dyn>::zeros(Self::K),
            output_sigma_points: OMatrix::<Kfloat, Const<M>, Dyn>::zeros(Self::K),
            wm,
            wc
        }
    }

    /// Compute sigma points from state vector and state noise covariance matrix.
    pub fn compute(&mut self, x: &StateVector<N>, p: &CovMatrix<N>) {
        // compute P matrix square root using cholesky decomposition
        let sp: CovMatrix<N> = p.cholesky().unwrap().l();  // TODO: match patter for cholesky
        
        self.sigma_points.set_column(0, x);
        for i in 1..N+1 {
            let col = self.gamma * sp.column(i-1);
            self.sigma_points.set_column(i, &(x + col));
            self.sigma_points.set_column(i + N, &(x - col));
        }
    }

    pub fn clear_output_sigma_points(&mut self) {
        self.output_sigma_points = OMatrix::<Kfloat, Const<M>, Dyn>::zeros(Self::K);
    }
}