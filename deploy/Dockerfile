FROM rust:1.97-slim AS build
WORKDIR /src
COPY Cargo.toml Cargo.lock ./
COPY src src
COPY weights weights
COPY migrations migrations
RUN cargo build --release --bin pullsift

FROM debian:bookworm-slim
RUN apt-get update && apt-get install -y --no-install-recommends ca-certificates \
    && rm -rf /var/lib/apt/lists/*
COPY --from=build /src/target/release/pullsift /usr/local/bin/pullsift
USER 65534:65534
EXPOSE 8080
ENTRYPOINT ["pullsift"]
