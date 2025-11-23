# Multi-stage Dockerfile for protean_dist_sim coordinator and worker binaries
# Optimized for production deployment on GCP

# ============================================================================
# Stage 1: Builder - Compile Rust binaries
# ============================================================================
FROM rust:1.86-slim-bookworm AS builder

# Install build dependencies
RUN apt-get update && apt-get install -y \
    pkg-config \
    libssl-dev \
    protobuf-compiler \
    && rm -rf /var/lib/apt/lists/*

# Create app directory
WORKDIR /app

# Copy workspace files
COPY . .

# Build protean library first (dependency)
WORKDIR /app/protean
RUN cargo build --release

# Build dist_sim binaries
WORKDIR /app/protean_dist_sim
RUN cargo build --release --bin coordinator --bin worker

# ============================================================================
# Stage 2: Coordinator Runtime Image
# ============================================================================
FROM debian:bookworm-slim AS coordinator

# Install runtime dependencies
RUN apt-get update && apt-get install -y \
    ca-certificates \
    && rm -rf /var/lib/apt/lists/*

# Create app user for security
RUN useradd -m -u 1000 appuser

# Create directories for data and output
RUN mkdir -p /app/data /app/config /app/output && \
    chown -R appuser:appuser /app

# Copy coordinator binary from builder
COPY --from=builder /app/protean_dist_sim/target/release/coordinator /app/coordinator

# Set ownership
RUN chown appuser:appuser /app/coordinator

# Switch to non-root user
USER appuser
WORKDIR /app

# Expose coordinator gRPC port
EXPOSE 50050

# Health check
HEALTHCHECK --interval=30s --timeout=10s --start-period=5s --retries=3 \
    CMD timeout 5 bash -c '</dev/tcp/localhost/50050' || exit 1

# Default command (expects config and test-plan to be mounted)
ENTRYPOINT ["/app/coordinator"]
CMD ["--config", "/app/config/coordinator_config.yaml", "--test-plan", "/app/config/test_plan.yaml"]

# ============================================================================
# Stage 3: Worker Runtime Image
# ============================================================================
FROM debian:bookworm-slim AS worker

# Install runtime dependencies
RUN apt-get update && apt-get install -y \
    ca-certificates \
    && rm -rf /var/lib/apt/lists/*

# Create app user for security
RUN useradd -m -u 1000 appuser

# Create app directory
RUN mkdir -p /app && chown -R appuser:appuser /app

# Copy worker binary from builder
COPY --from=builder /app/protean_dist_sim/target/release/worker /app/worker

# Set ownership
RUN chown appuser:appuser /app/worker

# Switch to non-root user
USER appuser
WORKDIR /app

# Expose worker gRPC port (will be overridden in docker-compose)
EXPOSE 50051

# Health check
HEALTHCHECK --interval=30s --timeout=10s --start-period=5s --retries=3 \
    CMD timeout 5 bash -c '</dev/tcp/localhost/50051' || exit 1

# Default command (expects environment variables to be set)
ENTRYPOINT ["/app/worker"]
