pub mod discrete_ebm;
pub mod ebm;
pub mod ising;

// Re-export everything in one place so consumers can pull modules from `thrml::models::*`.
pub use discrete_ebm::*;
pub use ebm::*;
pub use ising::*;
