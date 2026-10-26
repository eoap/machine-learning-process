# Stage 1: Build stage
FROM rockylinux:9.3-minimal AS build

# Refresh the OS trust store before downloading build tools or runtimes.
RUN microdnf update -y && microdnf install -y ca-certificates && \
    update-ca-trust && microdnf clean all

# Seed Hatch virtual environments from a verified patched pip wheel.
ADD --checksum=sha256:fd7f4dfa8190e014013b9f08f3e973f4e4c26ffd179a204895533a7254813a70 https://files.pythonhosted.org/packages/3f/06/3e4420302363fbb71be3c7f35ce042f54bfca330e3e66b3a2e3c4cca579f/pip-26.2.2-py3-none-any.whl /opt/python-bootstrap/pip-26.2.2-py3-none-any.whl
ENV VIRTUALENV_PIP=26.2.2 \
    VIRTUALENV_EXTRA_SEARCH_DIR=/opt/python-bootstrap \
    VIRTUALENV_DOWNLOAD=false \
    UV_SYSTEM_CERTS=true

# Install necessary build tools
RUN microdnf install -y curl tar

# Download the hatch tar.gz file from GitHub
RUN curl -L https://github.com/pypa/hatch/releases/download/hatch-v1.14.0/hatch-x86_64-unknown-linux-gnu.tar.gz -o /tmp/hatch-x86_64-unknown-linux-gnu.tar.gz

# Extract the hatch binary
RUN tar -xzf /tmp/hatch-x86_64-unknown-linux-gnu.tar.gz -C /tmp/

# Stage 2: Final stage
FROM rockylinux:9.3-minimal

# Refresh the OS trust store before downloading build tools or runtimes.
RUN microdnf update -y && microdnf install -y ca-certificates && \
    update-ca-trust && microdnf clean all

# Seed Hatch virtual environments from a verified patched pip wheel.
ADD --checksum=sha256:fd7f4dfa8190e014013b9f08f3e973f4e4c26ffd179a204895533a7254813a70 https://files.pythonhosted.org/packages/3f/06/3e4420302363fbb71be3c7f35ce042f54bfca330e3e66b3a2e3c4cca579f/pip-26.2.2-py3-none-any.whl /opt/python-bootstrap/pip-26.2.2-py3-none-any.whl
ENV VIRTUALENV_PIP=26.2.2 \
    VIRTUALENV_EXTRA_SEARCH_DIR=/opt/python-bootstrap \
    VIRTUALENV_DOWNLOAD=false \
    UV_SYSTEM_CERTS=true

# Set up a default user and home directory
ENV HOME=/home/neo

# Install essential libraries including expat and python3 without `config` commands
RUN microdnf install -y \
    expat \
    libpq \
    curl \
    git \
    wget \
    tar \
    && microdnf install -y python3 \
    && microdnf clean all

# Create a user with UID 1001, group root, and a home directory
RUN useradd -u 1001 -g 100 -m -d ${HOME} -s /sbin/nologin \
         -c "Default Neo User" neo && \
    mkdir -p /code /prod ${HOME}/.cache /home/neo/.local/share/hatch && \
    chown -R neo:100 /code ${HOME} /home/neo/.local/share/hatch && \
    chmod -R 775 /code ${HOME}

# Copy the hatch binary from the build stage
COPY --from=build /tmp/hatch /usr/bin/hatch

# Ensure the hatch binary is executable
RUN chmod +x /usr/bin/hatch

# Switch to the non-root user
USER neo

# Copy the application files into the /code directory
COPY --chown=1001:100 . /code

WORKDIR /code

# Set up virtual environment paths
ENV VIRTUAL_ENV=/code/envs/make_inference
ENV PATH="$VIRTUAL_ENV/bin:$PATH"
ENV HATCH_PYTHON_VARIANT_LINUX=v2

# Prune any existing environments and create a new production environment
RUN hatch env prune && \
    hatch env create prod && \
    hatch run prod:make-inference --help