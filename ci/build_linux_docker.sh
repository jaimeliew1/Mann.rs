sudo docker run --rm -v "$PWD":/io -w /io quay.io/pypa/manylinux_2_28_x86_64 /bin/bash ci/build_linux.sh
