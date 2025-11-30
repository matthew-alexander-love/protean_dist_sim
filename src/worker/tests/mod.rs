//! Tests for worker module
//!
//! Contains tests for Worker gRPC endpoints and ActorProtean control messages.

#[cfg(test)]
mod helpers;

#[cfg(test)]
mod worker_tests;

#[cfg(test)]
mod actor_tests;

// Integration and churn tests are disabled - they require multi-worker setup
// with actual network connections and coordinator
// #[cfg(test)]
// mod integration_tests;

// #[cfg(test)]
// mod churn_tests;
